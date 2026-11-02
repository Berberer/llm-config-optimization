"""Best-effort memory sampling for a running server container.

The host benchmark client knows how to measure a server's throughput; this
module adds the other half of a trial's cost — how much VRAM and RAM the
configuration holds while it serves. A sampler observes the container on a
fixed one-second cadence and answers peak queries over the samples it kept, so
the client can bracket a measured run and the orchestrator can read the
whole-container peak.

Sampling is deliberately engine-neutral and best-effort. The VRAM probe comes
from :mod:`~llm_config_optimization.inference.hardware` and the RAM reader is
injected by the caller (in the orchestrator, the running container handle's
cgroup memory read). A missing probe, a failed vendor query, or an unreadable
container stat leaves a field absent rather than failing the trial, and the
sampler stops with the container so no background process polls the host after
a trial ends.
"""

import threading
import time
from collections.abc import Callable, Sequence

from .hardware import MemoryProbe, VramMeasurement, attribute_vram
from .types import ResourceMarker, ResourceSample, ResourceUsage

__all__ = ["ThreadedResourceSampler"]

SAMPLE_INTERVAL_SECONDS = 1.0
"""How long the sampler waits between observations, in seconds.

One second is the accuracy/overhead compromise: a vendor query costs a
fraction of a second on both toolchains, and the allocations that matter — a
model's weights, a KV cache sized at startup — are held rather than fleeting.
A sub-cadence spike can be missed.
"""


def _peak(values: Sequence[int | None]) -> int | None:
    """Largest of the values that are present, or ``None`` when none is.

    A peak takes a maximum rather than a mean, and an absent measurement is
    ignored rather than treated as zero, so one failed field cannot drag the
    peak of another sample down.
    """
    present = [value for value in values if value is not None]
    return max(present) if present else None


class ThreadedResourceSampler:
    """Sample a running container's memory from a background thread.

    The sampler is constructed with everything host-specific injected: the
    container's RAM reader, the vendor VRAM probe selected for the detected
    target, the idle device baseline taken before the container started, and
    the container id used for cgroup attribution. Its clock and pacing are
    injected too, so tests drive it without a daemon, a GPU, or real time.

    :meth:`start` begins sampling immediately — the model-load allocation must
    be observed, not just the serving steady state — and :meth:`stop` ends the
    background observation. Samples are kept in order; :meth:`mark` returns an
    opaque position in that series and :meth:`peak` reads the maximum over the
    samples at or after it. :meth:`samples` exposes the raw records, including
    each VRAM figure's source, so a run's number can be audited later.

    A sampler with no VRAM probe (a CPU target) reports only RAM. Every failure
    path is silent: a probe that raises, a RAM read that fails, or a probe that
    reports nothing contributes an absent field instead of an error.
    """

    def __init__(
        self,
        *,
        ram_reader: Callable[[], int | None],
        container_id: str,
        baseline_vram_bytes: int | None = None,
        probe: MemoryProbe | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
        cgroup_of: Callable[[int], str | None] | None = None,
    ) -> None:
        self._ram_reader = ram_reader
        self._container_id = container_id
        self._baseline_vram_bytes = baseline_vram_bytes
        self._probe = probe
        self._clock = clock
        self._sleep = sleep
        self._cgroup_of = cgroup_of
        self._samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def baseline_vram_bytes(self) -> int | None:
        """The idle device memory the device-level fallback subtracts.

        The baseline is captured before the container starts, so a device-level
        VRAM sample is the trial's own growth over an idle device. It is kept
        beside the samples because a device-level number includes whatever else
        uses the accelerator and is only interpretable against that starting
        point.
        """
        return self._baseline_vram_bytes

    def start(self) -> None:
        """Begin sampling on a background thread.

        A sampler that is already running is left alone, so a caller that
        starts it twice does not spawn a second observer.
        """
        if self._thread is not None:
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the background thread to finish.

        The sampling thread is joined, so a caller can rely on no further
        sample being recorded after this returns. Stopping an already stopped
        sampler is harmless.
        """
        self._stopped.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def mark(self) -> ResourceMarker:
        """Record a point in the observation series and return its marker.

        Every sample recorded from now on belongs to the window this marker
        opens; samples already taken do not. The position is taken under the
        sampler's lock, so a sample being appended concurrently falls cleanly on
        one side of the mark.
        """
        with self._lock:
            return len(self._samples)

    def samples(self, since: ResourceMarker) -> tuple[ResourceSample, ...]:
        """The raw observations recorded at or after ``since``, oldest first.

        The returned tuple is a snapshot, so a concurrent sample cannot change
        a record a caller is auditing.
        """
        with self._lock:
            return tuple(self._samples[since:])

    def peak(self, since: ResourceMarker) -> ResourceUsage:
        """The peak of the samples recorded at or after ``since``.

        A window that saw no sample, or whose samples measured nothing,
        reports an all-``None`` usage rather than a zero.
        """
        samples = self.samples(since)
        return ResourceUsage(
            peak_vram_bytes=_peak([sample.vram_bytes for sample in samples]),
            peak_ram_bytes=_peak([sample.ram_bytes for sample in samples]),
        )

    def lifetime_peak(self) -> ResourceUsage:
        """The peak over every sample the sampler recorded.

        This is the whole-container figure: it spans model load, the health
        wait, warmup, and the measured runs, so it is the memory a configuration
        needs just to start.
        """
        return self.peak(0)

    def _run(self) -> None:
        """Sample until stopped, pacing each observation by one interval."""
        while not self._stopped.is_set():
            self._record()
            if self._stopped.is_set():
                return
            self._pause()

    def _pause(self) -> None:
        """Wait one cadence, returning early when the sampler is stopped.

        The default wait is the sampler's own stop event, so stopping does not
        have to wait out the interval. An injected ``sleep`` replaces it for a
        test that wants to drive the cadence itself.
        """
        if self._sleep is None:
            self._stopped.wait(SAMPLE_INTERVAL_SECONDS)
        else:
            self._sleep(SAMPLE_INTERVAL_SECONDS)

    def _record(self) -> None:
        """Append one best-effort observation of the container's memory."""
        timestamp = self._clock()
        vram = self._measure_vram()
        ram = self._measure_ram()
        with self._lock:
            self._samples.append(
                ResourceSample(
                    timestamp=timestamp,
                    vram_bytes=vram.used_bytes,
                    vram_source=vram.source,
                    ram_bytes=ram,
                )
            )

    def _measure_vram(self) -> VramMeasurement:
        """Probe and attribute VRAM, reporting unknown on any failure.

        A CPU target has no probe, so the field stays absent. The probe is
        guarded because it is the boundary to vendor tooling: a missing tool,
        an unreadable process table, or an injected failure is a diagnostic
        gap, never a reason to end the trial.
        """
        if self._probe is None:
            return VramMeasurement()
        try:
            memory = self._probe()
            if self._cgroup_of is None:
                return attribute_vram(
                    memory,
                    container_id=self._container_id,
                    baseline_bytes=self._baseline_vram_bytes,
                )
            return attribute_vram(
                memory,
                container_id=self._container_id,
                baseline_bytes=self._baseline_vram_bytes,
                cgroup_of=self._cgroup_of,
            )
        except Exception:
            return VramMeasurement()

    def _measure_ram(self) -> int | None:
        """Read the container's cgroup memory, reporting unknown on any failure."""
        try:
            return self._ram_reader()
        except Exception:
            return None
