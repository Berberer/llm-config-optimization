"""Host hardware detection for the inference experiment layer.

Detection resolves the host to a :class:`~llm_config_optimization.inference.types.HardwareTarget`
and a usable device count. It is deliberately injectable: :func:`detect_hardware`
takes the vendor probes as an argument, so callers and tests can supply fakes and
no backend ever probes the host itself.
"""

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .types import DetectedHardware, HardwareTarget

_PROBE_TIMEOUT_SECONDS = 10
_KFD_DEVICE = Path("/dev/kfd")
_DRM_DIR = Path("/dev/dri")


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
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    count = sum(1 for line in completed.stdout.splitlines() if line.strip())
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
