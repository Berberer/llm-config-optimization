"""Host hardware detection for the inference experiment layer.

Detection resolves the host to a :class:`~llm_config_optimization.inference.types.HardwareTarget`
and a usable device count. It is deliberately injectable: :func:`detect_hardware`
takes the vendor probes as an argument, so callers and tests can supply fakes and
no backend ever probes the host itself.

The module also owns the host's accelerator-memory probes. They observe one
vendor's tooling, parse its captured output, and attribute a VRAM number to a
container by cgroup membership. A probe is best-effort: a missing tool, a failed
query, or an unreadable process file leaves a value absent rather than failing
the run.
"""

import csv
import io
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .types import DetectedHardware, HardwareTarget, VramSource

_PROBE_TIMEOUT_SECONDS = 10
_MIB = 1024 * 1024
_KFD_DEVICE = Path("/dev/kfd")
_DRM_DIR = Path("/dev/dri")
_PROC = Path("/proc")
_EMPTY_PROCESSES: Mapping[int, int] = MappingProxyType({})
_ROCM_USED_COLUMN = "VRAM Total Used Memory (B)"

ToolRunner = Callable[[Sequence[str]], str | None]


def _run_tool(command: Sequence[str]) -> str | None:
    """Run a vendor tool and return its stdout, or ``None`` when it is unusable.

    A missing binary, a non-zero exit, and a timeout all mean "no measurement":
    the probes are diagnostics and must never fail the run they observe.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


@dataclass(frozen=True, slots=True)
class HardwareProbe:
    """A probe for one vendor's accelerator stack.

    ``probe`` returns the number of usable devices, or ``None`` when the stack
    is absent or broken. A vendor whose tooling is present but cannot enumerate
    its devices is reported as unusable so detection falls back to CPU.
    """

    target: HardwareTarget
    probe: Callable[[], int | None]

    def __call__(self) -> int | None:
        return self.probe()


def _probe_nvidia() -> int | None:
    """Return the CUDA device count, or ``None`` if the driver is unusable.

    A missing ``nvidia-smi``, a non-zero exit, or a timeout all mean the CUDA
    stack is absent or broken, which detection treats as "not CUDA".
    """
    output = _run_tool(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
    if output is None:
        return None
    count = sum(1 for line in output.splitlines() if line.strip())
    return count or None


def _probe_amd() -> int | None:
    """Return the ROCm device count, or ``None`` if the stack is absent or partial.

    A container run needs only the host's KFD device and DRM render nodes: the
    ROCm userspace lives in the image, so host tooling such as ``rocm-smi`` is
    deliberately not consulted (it can fail for packaging reasons, for example a
    missing unversioned ``libdrm_amdgpu.so`` dev symlink, while the runtime is
    perfectly usable). A KFD node without any render node is a partial stack and
    is reported as unusable rather than as ROCm.
    """
    if not _KFD_DEVICE.exists():
        return None
    count = sum(1 for _ in _DRM_DIR.glob("renderD*"))
    return count or None


DEFAULT_PROBES: tuple[HardwareProbe, ...] = (
    HardwareProbe(HardwareTarget.CUDA, _probe_nvidia),
    HardwareProbe(HardwareTarget.ROCM, _probe_amd),
)


def detect_hardware(
    override: HardwareTarget | None = None,
    *,
    probes: Sequence[HardwareProbe] = DEFAULT_PROBES,
) -> DetectedHardware:
    """Resolve the host's accelerator target and usable device count.

    Vendor probes run in order and the first usable one wins; when none is
    usable the host is treated as CPU with a single device. An explicit
    ``override`` forces the target, keeping the device count of that vendor's
    probe when it can still enumerate devices and defaulting to one otherwise.
    """
    if override is not None:
        return DetectedHardware(override, _device_count_for(override, probes))
    for probe in probes:
        count = probe()
        if count:
            return DetectedHardware(probe.target, count)
    return DetectedHardware(HardwareTarget.CPU, 1)


def _device_count_for(target: HardwareTarget, probes: Sequence[HardwareProbe]) -> int:
    """Count devices for an explicitly requested target, defaulting to one."""
    if target == HardwareTarget.CPU:
        return 1
    for probe in probes:
        if probe.target != target:
            continue
        return probe() or 1
    return 1


@dataclass(frozen=True, slots=True)
class GpuMemory:
    """One best-effort observation of accelerator memory.

    ``used_bytes`` is the memory the vendor's tool reports in use across the
    devices it enumerates. ``processes`` maps each process the vendor accounts
    memory to — its host PID — to that process's byte count. A failed query
    leaves its field absent or empty instead of raising, because a memory
    measurement is a diagnostic that must never fail a trial.
    """

    used_bytes: int | None = None
    processes: Mapping[int, int] = _EMPTY_PROCESSES

    def __post_init__(self) -> None:
        """Store ``processes`` as a read-only view so the observation is stable."""
        object.__setattr__(self, "processes", MappingProxyType(dict(self.processes)))


@dataclass(frozen=True, slots=True)
class MemoryProbe:
    """A memory probe for one vendor's accelerator stack.

    ``probe`` returns one :class:`GpuMemory` observation. It is injectable for
    the same reason the device-count probes are: the orchestrator and tests
    supply fakes, so no backend shells out to a vendor tool.
    """

    target: HardwareTarget
    probe: Callable[[], GpuMemory]

    def __call__(self) -> GpuMemory:
        return self.probe()


@dataclass(frozen=True, slots=True)
class VramMeasurement:
    """One attributed VRAM value and the source that produced it.

    ``used_bytes`` is ``None`` when nothing could be measured. A measured zero
    stays ``0`` so "used nothing" remains distinguishable from "unmeasured".
    """

    used_bytes: int | None = None
    source: VramSource | None = None


def _parse_nvidia_used_memory(output: str) -> int | None:
    """Parse ``nvidia-smi --query-gpu=memory.used`` rows from MiB into bytes.

    Every reported device contributes to the total, so a multi-device run is
    measured as one aggregate rather than a per-device breakdown. A missing or
    unparsable row makes the whole observation unknown instead of silently
    understating the total.
    """
    total = 0
    seen = False
    for line in output.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            total += int(value) * _MIB
        except ValueError:
            return None
        seen = True
    return total if seen else None


def _parse_nvidia_processes(output: str) -> Mapping[int, int]:
    """Parse ``nvidia-smi --query-compute-apps`` rows from MiB into bytes.

    Rows the tool cannot account for — ``[N/A]`` and similar placeholders — are
    skipped, and a duplicate PID (one row per device) is summed, because the
    mapping describes the process's total footprint.
    """
    usage: dict[int, int] = {}
    for line in output.splitlines():
        pid_text, separator, used_text = line.partition(",")
        if not separator:
            continue
        try:
            pid = int(pid_text.strip())
            used = int(used_text.strip()) * _MIB
        except ValueError:
            continue
        usage[pid] = usage.get(pid, 0) + used
    return MappingProxyType(usage)


def _parse_rocm_used_memory(output: str) -> int | None:
    """Parse ``rocm-smi --showmeminfo vram --csv`` bytes into one aggregate.

    The column is located by name rather than position, so a rocm-smi version
    that adds columns keeps working. As with the NVIDIA parser, an unreadable
    row makes the whole observation unknown rather than understating it.
    """
    rows = [row for row in csv.reader(io.StringIO(output)) if any(cell.strip() for cell in row)]
    if not rows:
        return None
    header = [cell.strip() for cell in rows[0]]
    try:
        index = header.index(_ROCM_USED_COLUMN)
    except ValueError:
        return None
    total = 0
    seen = False
    for row in rows[1:]:
        try:
            total += int(row[index].strip())
        except IndexError, ValueError:
            return None
        seen = True
    return total if seen else None


def _parse_rocm_processes(output: str) -> Mapping[int, int]:
    """Parse ``rocm-smi --showpids`` rows into a process-to-bytes mapping.

    The table's columns are ``PID PROCESS NAME GPU(s) VRAM USED SDMA USED CU
    OCCUPANCY``. The VRAM figure is read from the right so a process name
    containing spaces stays part of the name, and rows reporting ``UNKNOWN``
    VRAM are skipped because the tool could not attribute them.
    """
    usage: dict[int, int] = {}
    for line in output.splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        try:
            pid = int(tokens[0])
            used = int(tokens[-3])
        except ValueError:
            continue
        usage[pid] = usage.get(pid, 0) + used
    return MappingProxyType(usage)


def _process_cgroup(pid: int) -> str | None:
    """Return a process's cgroup memberships, or ``None`` when unreadable.

    The cgroup text names the runtime's container scope when the process
    belongs to a container, which is how accelerator memory is attributed to
    the trial that caused it.
    """
    try:
        return (_PROC / str(pid) / "cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def attribute_vram(
    memory: GpuMemory,
    *,
    container_id: str,
    baseline_bytes: int | None,
    cgroup_of: Callable[[int], str | None] = _process_cgroup,
) -> VramMeasurement:
    """Attribute ``memory`` to a container, preferring its own processes.

    The per-process values whose cgroup names ``container_id`` are summed, so
    another workload on a shared device does not inflate the trial's number.
    When no process can be attributed, the device-level used memory minus the
    idle ``baseline_bytes`` — clamped at zero — is the fallback, and
    :attr:`VramMeasurement.source` records which route produced the value.
    Without an attributed process and without both device values, the
    measurement is unknown. An empty ``container_id`` attributes no process,
    because a substring test against an empty id would match every one.
    """
    attributed = 0
    found = False
    if container_id:
        for pid, used_bytes in memory.processes.items():
            cgroup = cgroup_of(pid)
            if cgroup is None or container_id not in cgroup:
                continue
            attributed += used_bytes
            found = True
    if found:
        return VramMeasurement(attributed, VramSource.PROCESS)
    if memory.used_bytes is None or baseline_bytes is None:
        return VramMeasurement()
    return VramMeasurement(max(0, memory.used_bytes - baseline_bytes), VramSource.DEVICE)


def _probe_nvidia_memory(run: ToolRunner = _run_tool) -> GpuMemory:
    """Observe CUDA memory through ``nvidia-smi``, best-effort per field."""
    used = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    processes = run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    )
    return GpuMemory(
        used_bytes=None if used is None else _parse_nvidia_used_memory(used),
        processes={} if processes is None else _parse_nvidia_processes(processes),
    )


def _probe_amd_memory(run: ToolRunner = _run_tool) -> GpuMemory:
    """Observe ROCm memory through ``rocm-smi``, best-effort per field."""
    used = run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    processes = run(["rocm-smi", "--showpids"])
    return GpuMemory(
        used_bytes=None if used is None else _parse_rocm_used_memory(used),
        processes={} if processes is None else _parse_rocm_processes(processes),
    )


DEFAULT_MEMORY_PROBES: tuple[MemoryProbe, ...] = (
    MemoryProbe(HardwareTarget.CUDA, _probe_nvidia_memory),
    MemoryProbe(HardwareTarget.ROCM, _probe_amd_memory),
)


def memory_probe_for(
    target: HardwareTarget,
    *,
    probes: Sequence[MemoryProbe] = DEFAULT_MEMORY_PROBES,
) -> MemoryProbe | None:
    """Select the memory probe for ``target``, or ``None`` when it has none.

    A CPU target has no accelerator memory to measure and never gets a probe,
    even one injected under the CPU target. A target without a matching probe —
    a future accelerator class, or a deliberately empty probe set — is left
    unmeasured rather than borrowing another vendor's tooling.
    """
    if target == HardwareTarget.CPU:
        return None
    for probe in probes:
        if probe.target == target:
            return probe
    return None
