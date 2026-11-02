"""Unit tests for the background resource sampler.

The sampler's observable contract is what it records and what it reports for a
window: the peak over the samples taken after a marker, the whole-container
peak, the raw sample series with each VRAM figure's source, and the absence of
a value when nothing could be measured. Its lifecycle contract is that sampling
starts when asked and that nothing is recorded after it is stopped.

The tests inject the RAM reader, the VRAM probe, the clock, the pacing callable,
and the cgroup reader, so no daemon, GPU, vendor tool, or real cadence wait is
required. A gated fake sleep lets a test advance the sampler one observation at
a time.
"""

import threading
from collections.abc import Callable, Sequence

from llm_config_optimization.inference import (
    GpuMemory,
    HardwareTarget,
    MemoryProbe,
    ResourceSampler,
    ResourceUsage,
    ThreadedResourceSampler,
    VramSource,
)

_WAIT_SECONDS = 5.0


class _Feed:
    """A value source that repeats its last value once its values run out."""

    def __init__(self, values: Sequence[int | None]) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> int | None:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


class _Gate:
    """A pacing callable that blocks until the test advances it.

    It records the intervals the sampler asks for and signals when the sampler
    has taken a sample, which happens immediately before every wait. Once
    :meth:`shutdown` is called it never blocks again, so stopping the sampler
    cannot deadlock on the fake sleep.
    """

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.stopping = False
        self._entered = threading.Semaphore(0)
        self._release = threading.Event()

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.stopping:
            return
        self._entered.release()
        self._release.wait()
        self._release.clear()

    def wait_for_sample(self) -> None:
        assert self._entered.acquire(timeout=_WAIT_SECONDS), "the sampler took no sample"

    def advance(self) -> None:
        self._release.set()
        self.wait_for_sample()

    def shutdown(self) -> None:
        self.stopping = True
        self._release.set()


def _process_probe(values: Sequence[int | None], pid: int = 4242) -> MemoryProbe:
    """Build a VRAM probe that reports a container process's memory."""
    feed = _Feed(values)

    def observe() -> GpuMemory:
        value = feed()
        return GpuMemory() if value is None else GpuMemory(processes={pid: value})

    return MemoryProbe(HardwareTarget.CUDA, observe)


def _sampler(
    gate: _Gate,
    *,
    probe: MemoryProbe | None = None,
    ram_reader: Callable[[], int | None] | None = None,
    baseline_vram_bytes: int | None = None,
    clock: Callable[[], float] | None = None,
) -> ThreadedResourceSampler:
    """Build a sampler whose cadence and cgroup lookup are fully injected."""
    return ThreadedResourceSampler(
        ram_reader=ram_reader if ram_reader is not None else lambda: 10,
        container_id="container-abc",
        baseline_vram_bytes=baseline_vram_bytes,
        probe=probe,
        clock=clock if clock is not None else lambda: 0.0,
        sleep=gate,
        cgroup_of=lambda _pid: "/machine.slice/libpod-container-abc.scope",
    )


def test_window_peak_covers_only_the_samples_after_its_marker() -> None:
    gate = _Gate()
    times = _Feed([100, 101, 102])
    sampler = _sampler(
        gate,
        probe=_process_probe([900, 400, 200]),
        ram_reader=_Feed([90, 20, 15]),
        clock=lambda: float(times()),
    )
    sampler.start()
    gate.wait_for_sample()

    marker = sampler.mark()
    gate.advance()
    gate.advance()

    window = sampler.peak(marker)
    lifetime = sampler.lifetime_peak()
    samples = sampler.samples(marker)

    gate.shutdown()
    sampler.stop()

    assert window == ResourceUsage(peak_vram_bytes=400, peak_ram_bytes=20)
    assert lifetime == ResourceUsage(peak_vram_bytes=900, peak_ram_bytes=90)
    assert [sample.vram_bytes for sample in samples] == [400, 200]
    assert [sample.ram_bytes for sample in samples] == [20, 15]
    assert [sample.timestamp for sample in samples] == [101.0, 102.0]
    assert {sample.vram_source for sample in samples} == {VramSource.PROCESS}


def test_window_that_saw_no_sample_reports_unknown() -> None:
    gate = _Gate()
    sampler = _sampler(gate, probe=_process_probe([900]))
    sampler.start()
    gate.wait_for_sample()

    marker = sampler.mark()

    assert sampler.samples(marker) == ()
    assert sampler.peak(marker) == ResourceUsage()
    assert sampler.peak(marker).peak_vram_bytes is None
    assert sampler.peak(marker).peak_ram_bytes is None

    gate.shutdown()
    sampler.stop()


def test_failed_probe_and_ram_read_record_unknown_instead_of_raising() -> None:
    gate = _Gate()

    def failing_probe() -> GpuMemory:
        raise RuntimeError("vendor tool failed")

    def failing_ram() -> int | None:
        raise OSError("container stats unavailable")

    sampler = ThreadedResourceSampler(
        ram_reader=failing_ram,
        container_id="container-abc",
        probe=MemoryProbe(HardwareTarget.CUDA, failing_probe),
        sleep=gate,
    )
    marker = sampler.mark()
    sampler.start()
    gate.wait_for_sample()

    (sample,) = sampler.samples(marker)

    assert sample.vram_bytes is None
    assert sample.vram_source is None
    assert sample.ram_bytes is None
    assert sampler.peak(marker) == ResourceUsage()

    gate.shutdown()
    sampler.stop()


def test_samples_record_the_device_source_baseline_and_ram() -> None:
    gate = _Gate()
    probe = MemoryProbe(HardwareTarget.CUDA, lambda: GpuMemory(used_bytes=5_000))
    sampler = ThreadedResourceSampler(
        ram_reader=lambda: 7,
        container_id="container-abc",
        baseline_vram_bytes=2_000,
        probe=probe,
        sleep=gate,
        cgroup_of=lambda _pid: None,
    )
    marker = sampler.mark()
    sampler.start()
    gate.wait_for_sample()

    (sample,) = sampler.samples(marker)

    gate.shutdown()
    sampler.stop()

    assert sampler.baseline_vram_bytes == 2_000
    assert sample.vram_bytes == 3_000
    assert sample.vram_source is VramSource.DEVICE
    assert sample.ram_bytes == 7


def test_sampler_without_a_probe_still_reports_ram() -> None:
    gate = _Gate()
    sampler = ThreadedResourceSampler(
        ram_reader=lambda: 4_096,
        container_id="container-abc",
        probe=None,
        sleep=gate,
    )
    marker = sampler.mark()
    sampler.start()
    gate.wait_for_sample()

    (sample,) = sampler.samples(marker)
    peak = sampler.peak(marker)

    gate.shutdown()
    sampler.stop()

    assert sample.vram_bytes is None
    assert sample.vram_source is None
    assert sample.ram_bytes == 4_096
    assert peak == ResourceUsage(peak_ram_bytes=4_096)


def test_measured_zero_is_recorded_as_zero_rather_than_unknown() -> None:
    gate = _Gate()
    sampler = _sampler(
        gate,
        probe=_process_probe([0]),
        ram_reader=lambda: 0,
    )
    marker = sampler.mark()
    sampler.start()
    gate.wait_for_sample()

    (sample,) = sampler.samples(marker)
    peak = sampler.peak(marker)

    gate.shutdown()
    sampler.stop()

    assert sample.vram_bytes == 0
    assert sample.vram_source is VramSource.PROCESS
    assert sample.ram_bytes == 0
    assert peak == ResourceUsage(peak_vram_bytes=0, peak_ram_bytes=0)


def test_stopping_the_sampler_ends_sampling() -> None:
    gate = _Gate()
    sampler = _sampler(gate, probe=_process_probe([900]))
    sampler.start()
    gate.wait_for_sample()

    gate.shutdown()
    sampler.stop()
    recorded_at_stop = sampler.mark()
    calls_at_stop = len(gate.calls)
    threading.Event().wait(0.02)

    assert sampler.mark() == recorded_at_stop
    assert len(gate.calls) == calls_at_stop


def test_threaded_sampler_satisfies_the_sampler_protocol() -> None:
    sampler = ThreadedResourceSampler(ram_reader=lambda: None, container_id="container-abc")

    assert isinstance(sampler, ResourceSampler)
    assert sampler.samples(sampler.mark()) == ()
