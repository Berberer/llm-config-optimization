"""Unit tests for host hardware detection, driven by injected fake probes.

Detection's observable contract is the accelerator target and usable device
count it resolves for a given set of vendor probes. These tests inject fakes
that report CUDA, ROCm, CPU, broken, and multi-device hosts — no real hardware
is required — and assert the resolved result, including the fallback to CPU and
the explicit-override path.
"""

from llm_optimization.inference import DetectedHardware, HardwareTarget, detect_hardware
from llm_optimization.inference.hardware import HardwareProbe


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
