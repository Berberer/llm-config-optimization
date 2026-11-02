"""Unit tests for host hardware detection, driven by injected fake probes.

Detection's observable contract is the accelerator target and usable device
count it resolves for a given set of vendor probes. These tests inject fakes
that report CUDA, ROCm, CPU, broken, and multi-device hosts — no real hardware
is required — and assert the resolved result, including the fallback to CPU and
the explicit-override path.

The memory-probe tests feed captured vendor output to the pure parsers, select
the probe for a detected target, and assert the cgroup attribution rule. No
vendor tool is executed and no accelerator is required.
"""

from collections.abc import Callable, Sequence

from llm_config_optimization.inference import DetectedHardware, HardwareTarget, detect_hardware
from llm_config_optimization.inference.hardware import (
    GpuMemory,
    HardwareProbe,
    MemoryProbe,
    VramMeasurement,
    VramSource,
    _parse_nvidia_processes,
    _parse_nvidia_used_memory,
    _parse_rocm_processes,
    _parse_rocm_used_memory,
    _probe_amd_memory,
    _probe_nvidia_memory,
    attribute_vram,
    memory_probe_for,
)

MIB = 1024 * 1024


def _probe(target: HardwareTarget, count: int | None) -> HardwareProbe:
    """Build a fake vendor probe that reports ``count`` devices."""
    return HardwareProbe(target, lambda: count)


def test_cuda_host_resolves_to_cuda_with_its_device_count() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, 2),
        _probe(HardwareTarget.ROCM, 1),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.CUDA, 2)


def test_rocm_host_resolves_to_rocm_when_cuda_is_absent() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, None),
        _probe(HardwareTarget.ROCM, 4),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.ROCM, 4)


def test_multi_device_host_reports_every_device() -> None:
    probes = (_probe(HardwareTarget.CUDA, 8),)

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.CUDA, 8)


def test_broken_cuda_stack_falls_back_to_the_next_vendor() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, None),
        _probe(HardwareTarget.ROCM, 1),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.ROCM, 1)


def test_vendor_reporting_no_devices_is_treated_as_unusable() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, 0),
        _probe(HardwareTarget.ROCM, 2),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.ROCM, 2)


def test_no_usable_vendor_falls_back_to_a_single_cpu_device() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, None),
        _probe(HardwareTarget.ROCM, None),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.CPU, 1)


def test_empty_probe_set_falls_back_to_cpu() -> None:
    assert detect_hardware(probes=()) == DetectedHardware(HardwareTarget.CPU, 1)


def test_probes_run_in_order_and_stop_at_the_first_usable_vendor() -> None:
    calls: list[str] = []

    def cuda() -> int | None:
        calls.append("cuda")
        return 2

    def rocm() -> int | None:
        calls.append("rocm")
        return 8

    probes = (
        HardwareProbe(HardwareTarget.CUDA, cuda),
        HardwareProbe(HardwareTarget.ROCM, rocm),
    )

    assert detect_hardware(probes=probes) == DetectedHardware(HardwareTarget.CUDA, 2)
    assert calls == ["cuda"]


def test_override_forces_the_target_and_keeps_the_probe_device_count() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, 2),
        _probe(HardwareTarget.ROCM, 8),
    )

    assert detect_hardware(HardwareTarget.ROCM, probes=probes) == DetectedHardware(
        HardwareTarget.ROCM, 8
    )


def test_override_to_an_unprobed_target_defaults_to_one_device() -> None:
    probes = (_probe(HardwareTarget.CUDA, 4),)

    assert detect_hardware(HardwareTarget.ROCM, probes=probes) == DetectedHardware(
        HardwareTarget.ROCM, 1
    )


def test_override_to_a_broken_vendor_defaults_to_one_device() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, None),
        _probe(HardwareTarget.ROCM, 8),
    )

    assert detect_hardware(HardwareTarget.CUDA, probes=probes) == DetectedHardware(
        HardwareTarget.CUDA, 1
    )


def test_cpu_override_always_reports_a_single_device() -> None:
    probes = (
        _probe(HardwareTarget.CUDA, 4),
        _probe(HardwareTarget.ROCM, 8),
    )

    assert detect_hardware(HardwareTarget.CPU, probes=probes) == DetectedHardware(
        HardwareTarget.CPU, 1
    )


def test_nvidia_used_memory_sums_every_reported_device_in_bytes() -> None:
    assert _parse_nvidia_used_memory("3584\n1024\n") == (3584 + 1024) * MIB


def test_nvidia_used_memory_reports_unknown_for_a_placeholder_row() -> None:
    assert _parse_nvidia_used_memory("3584\n[N/A]\n") is None


def test_nvidia_used_memory_reports_unknown_for_empty_output() -> None:
    assert _parse_nvidia_used_memory("") is None


def test_nvidia_processes_sum_duplicate_pids_and_skip_placeholders() -> None:
    captured = "1234, 512\n5678, 2048\n1234, 256\n9999, [N/A]\n"

    assert _parse_nvidia_processes(captured) == {
        1234: (512 + 256) * MIB,
        5678: 2048 * MIB,
    }


def test_rocm_used_memory_reads_the_named_column() -> None:
    captured = (
        "device,Card Model,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card0,AMD Radeon RX 7900 XT,17163091968,108814336\n"
        "card1,AMD Radeon RX 7900 XT,17163091968,2048\n"
    )

    assert _parse_rocm_used_memory(captured) == 108814336 + 2048


def test_rocm_used_memory_reports_unknown_without_the_used_column() -> None:
    assert _parse_rocm_used_memory("device,VRAM Total Memory (B)\ncard0,17163091968\n") is None


def test_rocm_processes_parse_the_table_and_skip_unknown_rows() -> None:
    captured = """\
============================ ROCm System Management Interface ============================
==================================== KFD Processes ======================================
KFD process information:
PID   PROCESS NAME   GPU(s)   VRAM USED   SDMA USED   CU OCCUPANCY
2254756 work 1 356581376 0 9
2254757 python worker 0 UNKNOWN UNKNOWN UNKNOWN
2254758 Web Content 1 1048576 0 0
==========================================================================================
================================== End of ROCm SMI Log ==================================
"""

    assert _parse_rocm_processes(captured) == {
        2254756: 356581376,
        2254758: 1048576,
    }


def test_nvidia_memory_probe_parses_both_queries() -> None:
    commands: list[list[str]] = []

    def run(command: Sequence[str]) -> str | None:
        commands.append(list(command))
        if command[1] == "--query-gpu=memory.used":
            return "3584\n"
        return "1234, 512\n"

    memory = _probe_nvidia_memory(run)

    assert memory == GpuMemory(used_bytes=3584 * MIB, processes={1234: 512 * MIB})
    assert commands == [
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ]


def test_amd_memory_probe_parses_both_queries() -> None:
    captured = {
        "vram": (
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,17163091968,108814336\n"
        ),
        "pids": "PID PROCESS NAME GPU(s) VRAM USED SDMA USED CU OCCUPANCY\n"
        "2254756 work 1 356581376 0 9\n",
    }

    def run(command: Sequence[str]) -> str | None:
        if "--showmeminfo" in command:
            return captured["vram"]
        return captured["pids"]

    memory = _probe_amd_memory(run)

    assert memory == GpuMemory(used_bytes=108814336, processes={2254756: 356581376})


def test_memory_probes_report_unknown_when_the_tooling_is_missing() -> None:
    assert _probe_nvidia_memory(lambda command: None) == GpuMemory()
    assert _probe_amd_memory(lambda command: None) == GpuMemory()


def test_memory_probe_selection_matches_the_detected_target() -> None:
    cuda = MemoryProbe(HardwareTarget.CUDA, lambda: GpuMemory(used_bytes=1))
    rocm = MemoryProbe(HardwareTarget.ROCM, lambda: GpuMemory(used_bytes=2))
    probes = (cuda, rocm)

    assert memory_probe_for(HardwareTarget.CUDA, probes=probes) is cuda
    assert memory_probe_for(HardwareTarget.ROCM, probes=probes) is rocm


def test_default_memory_probes_cover_both_gpu_vendors() -> None:
    cuda = memory_probe_for(HardwareTarget.CUDA)
    rocm = memory_probe_for(HardwareTarget.ROCM)

    assert cuda is not None
    assert rocm is not None
    assert cuda.target == HardwareTarget.CUDA
    assert rocm.target == HardwareTarget.ROCM


def test_cpu_target_has_no_memory_probe() -> None:
    cpu = MemoryProbe(HardwareTarget.CPU, lambda: GpuMemory(used_bytes=1))

    assert memory_probe_for(HardwareTarget.CPU, probes=(cpu,)) is None


def test_unmatched_target_has_no_memory_probe() -> None:
    cuda = MemoryProbe(HardwareTarget.CUDA, lambda: GpuMemory(used_bytes=1))

    assert memory_probe_for(HardwareTarget.ROCM, probes=(cuda,)) is None


def _cgroups(by_pid: dict[int, str]) -> Callable[[int], str | None]:
    """Build a cgroup reader from an in-memory PID-to-cgroup mapping."""
    return by_pid.get


def test_vram_attribution_prefers_the_containers_own_processes() -> None:
    memory = GpuMemory(
        used_bytes=10 * MIB,
        processes={111: 2 * MIB, 222: 3 * MIB, 333: 4 * MIB},
    )
    cgroups = {
        111: "0::/machine.slice/libpod-abc.scope/container",
        222: "0::/user.slice/session-1.scope",
        333: "0::/machine.slice/libpod-abc.scope/container",
    }

    measurement = attribute_vram(
        memory,
        container_id="abc",
        baseline_bytes=1 * MIB,
        cgroup_of=_cgroups(cgroups),
    )

    assert measurement == VramMeasurement(used_bytes=6 * MIB, source=VramSource.PROCESS)


def test_vram_attribution_falls_back_to_the_device_delta() -> None:
    memory = GpuMemory(used_bytes=10 * MIB, processes={222: 3 * MIB})

    measurement = attribute_vram(
        memory,
        container_id="abc",
        baseline_bytes=4 * MIB,
        cgroup_of=_cgroups({222: "0::/user.slice/session-1.scope"}),
    )

    assert measurement == VramMeasurement(used_bytes=6 * MIB, source=VramSource.DEVICE)


def test_vram_attribution_clamps_a_device_delta_below_the_baseline_at_zero() -> None:
    measurement = attribute_vram(
        GpuMemory(used_bytes=2 * MIB),
        container_id="abc",
        baseline_bytes=4 * MIB,
        cgroup_of=_cgroups({}),
    )

    assert measurement == VramMeasurement(used_bytes=0, source=VramSource.DEVICE)


def test_vram_attribution_reports_unknown_without_a_baseline() -> None:
    measurement = attribute_vram(
        GpuMemory(used_bytes=10 * MIB),
        container_id="abc",
        baseline_bytes=None,
        cgroup_of=_cgroups({}),
    )

    assert measurement == VramMeasurement()


def test_vram_attribution_reports_unknown_without_a_device_value() -> None:
    measurement = attribute_vram(
        GpuMemory(),
        container_id="abc",
        baseline_bytes=4 * MIB,
        cgroup_of=_cgroups({}),
    )

    assert measurement == VramMeasurement()


def test_vram_attribution_keeps_a_measured_zero_over_the_fallback() -> None:
    memory = GpuMemory(used_bytes=10 * MIB, processes={111: 0})

    measurement = attribute_vram(
        memory,
        container_id="abc",
        baseline_bytes=0,
        cgroup_of=_cgroups({111: "0::/machine.slice/libpod-abc.scope"}),
    )

    assert measurement == VramMeasurement(used_bytes=0, source=VramSource.PROCESS)


def test_vram_attribution_treats_an_unreadable_cgroup_as_unattributed() -> None:
    measurement = attribute_vram(
        GpuMemory(used_bytes=10 * MIB, processes={111: 3 * MIB}),
        container_id="abc",
        baseline_bytes=4 * MIB,
        cgroup_of=_cgroups({}),
    )

    assert measurement == VramMeasurement(used_bytes=6 * MIB, source=VramSource.DEVICE)


def test_vram_attribution_without_a_container_id_attributes_no_process() -> None:
    """An empty id would substring-match every cgroup, so it attributes nothing."""
    measurement = attribute_vram(
        GpuMemory(used_bytes=10 * MIB, processes={111: 3 * MIB}),
        container_id="",
        baseline_bytes=4 * MIB,
        cgroup_of=_cgroups({111: "0::/machine.slice/libpod-abc.scope"}),
    )

    assert measurement == VramMeasurement(used_bytes=6 * MIB, source=VramSource.DEVICE)
