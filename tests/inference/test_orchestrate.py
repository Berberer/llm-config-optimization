"""Unit tests for experiment orchestration.

``run_experiment``'s observable contract is the collaboration it drives: detect
the host's hardware once, ask the backend to plan a server run, start that
server container, measure it with the host benchmark client, stop it, and hand
the measured outcome and captured server result back to the backend for parsing
— in that order, with the detected hardware, the experiment, and the resolved
host URL passed through. A failed start, measurement, or stop becomes an
``ExperimentError`` that preserves the cause and the raw result, and a failed
measurement keeps the engine's root-cause line alongside a bounded traceback
tail.

These tests inject a fake hardware detector, a fake backend, a fake benchmark
client, and the shared fake Podman client, so no daemon, inference engine, or
specific hardware is required.

Memory sampling is wired into the same lifecycle: the idle device baseline is
read before the container exists, sampling begins once it is up, the sampler is
handed to the benchmark client so each measured run can be bracketed, and it is
stopped on every exit path — success, failure, and interrupt — before the
result is returned. The host sampler itself is replaced here, so no test runs
``nvidia-smi`` or ``rocm-smi``, and a test that cares about the numbers injects
its own sampler.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Self

import pytest
from podman.errors import APIError

from fakes import FakeBackend, FakeBenchmarkClient, FakePodmanClient, _frame
from llm_config_optimization.inference import (
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    GpuMemory,
    HardwareTarget,
    InferenceBackend,
    LLMConfig,
    MemoryProbe,
    ResourceMarker,
    ResourceSample,
    ResourceSampler,
    ResourceUsage,
    RunMetrics,
    SamplerFactory,
    ServerSpec,
    measure_dataset_request_tokens,
    memory_probe_for,
    run_experiment,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult, ContainerRunError

HUB_MODEL = "meta-llama/Llama-3.1-8B"
_SPEC = ServerSpec(
    container=ContainerConfig(
        image="fake/engine:latest",
        command=("serve", "--model", HUB_MODEL),
        ports={"8000": None},
    ),
    health_path="/health",
    health_timeout=600.0,
    reset_paths=("/reset_prefix_cache",),
)

_ROOT_CAUSE = "AssertionError: ChunkGatedDeltaRuleFunction does not support float32"

_GIB = 1024**3
"""One gibibyte, so a memory assertion reads at the unit a user would quote."""


def _engine_output(root_cause: str, *, frames: int) -> bytes:
    """Build engine-like output: a root cause followed by a traceback."""
    traceback = "\n".join(
        f'  File "/usr/lib/python3.12/site-packages/module_{i}.py", line {i}, in <module>'
        for i in range(frames)
    )
    return f"{root_cause}\nTraceback (most recent call last):\n{traceback}".encode()


def _failure_message(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> str:
    """Drive a failed measurement with synthetic engine output and return the error."""
    events: list[str] = []
    client = RecordingPodmanClient(events, logs=_frame(2, payload))
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(
        measure_error=ExperimentError("the server never became healthy")
    )
    _detector(monkeypatch, events)
    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)
    return str(excinfo.value)


def _experiment(*, target: HardwareTarget | None = None) -> ExperimentConfig:
    """Build an experiment with an optional hardware-target override."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(target=target),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _outcome() -> BenchmarkOutcome:
    """Build the canned measured outcome the fake benchmark client returns."""
    return BenchmarkOutcome(
        runs=(
            RunMetrics(
                tokens_per_second=10.0,
                requests_per_second=1.0,
                elapsed_time=2.0,
                num_prompts=5,
                raw_json={},
            ),
        ),
        failed_runs=0,
    )


def _result() -> ExperimentResult:
    """Build the canned typed result the fake backend returns from parsing."""
    return ExperimentResult(
        runs=_outcome().runs,
        failed_runs=0,
        target=HardwareTarget.CPU,
        image=_SPEC.container.image,
        devices=(),
        raw_result=ContainerResult(exit_code=0, stdout=b"", stderr=b"", container_id="fake"),
    )


class FakeDetector:
    """Canned hardware detection that timestamps its calls and records overrides."""

    def __init__(self, hardware: DetectedHardware, events: list[str]) -> None:
        self.hardware = hardware
        self.events = events
        self.overrides: list[HardwareTarget | None] = []

    def __call__(self, override: HardwareTarget | None = None) -> DetectedHardware:
        self.events.append("detect")
        self.overrides.append(override)
        return self.hardware


class RecordingPodmanClient(FakePodmanClient):
    """A fake Podman client that timestamps container creation into ``events``."""

    def __init__(self, events: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        create = self.containers.create

        def recording_create(image: str, **create_kwargs: object) -> object:
            events.append("create")
            return create(image, **create_kwargs)

        self.containers.create = recording_create  # type: ignore[method-assign]


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    hardware: DetectedHardware | None = None,
) -> FakeDetector:
    """Install a fake hardware detector in place of the orchestrator's probe."""
    detector = FakeDetector(
        hardware if hardware is not None else DetectedHardware(HardwareTarget.ROCM, 1),
        events,
    )
    monkeypatch.setattr("llm_config_optimization.inference._orchestrate.detect_hardware", detector)
    return detector


class FakeSampler:
    """A canned resource sampler that records its lifecycle and construction.

    It stands in for the host sampler the orchestrator would otherwise build, so
    a test never runs a vendor tool. ``events`` timestamps the start and stop
    calls when a test pins the lifecycle order, ``lifetime`` is the peak the
    sampler answers with, and the arguments the orchestrator's host sampler was
    built from are kept so a test can pin what it was handed.
    """

    def __init__(
        self,
        *,
        ram_reader: Callable[[], int | None] | None = None,
        container_id: str = "",
        baseline_vram_bytes: int | None = None,
        probe: MemoryProbe | None = None,
        lifetime: ResourceUsage | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.ram_reader = ram_reader
        self.container_id = container_id
        self.baseline_vram_bytes = baseline_vram_bytes
        self.probe = probe
        self.events = events if events is not None else []
        self._lifetime = lifetime if lifetime is not None else ResourceUsage()
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1
        self.events.append("sampler_start")

    def stop(self) -> None:
        self.stops += 1
        self.events.append("sampler_stop")

    def mark(self) -> ResourceMarker:
        return 0

    def samples(self, since: ResourceMarker) -> tuple[ResourceSample, ...]:
        return ()

    def peak(self, since: ResourceMarker) -> ResourceUsage:
        return ResourceUsage()

    def lifetime_peak(self) -> ResourceUsage:
        return self._lifetime


class RecordingProbe:
    """A vendor memory probe that records its selection and its reads.

    ``targets`` keeps the target the probe was selected for, and ``events``
    timestamps each idle-device reading, so a test can pin that the baseline is
    taken from the detected target's probe before the container exists.
    """

    def __init__(self, memory: GpuMemory, events: list[str]) -> None:
        self.memory = memory
        self.events = events
        self.targets: list[HardwareTarget] = []
        self.reads = 0

    def selected_for(self, target: HardwareTarget) -> Self:
        self.targets.append(target)
        return self

    def __call__(self) -> GpuMemory:
        self.reads += 1
        self.events.append("probe")
        return self.memory


@pytest.fixture(autouse=True)
def host_samplers(monkeypatch: pytest.MonkeyPatch) -> list[FakeSampler]:
    """Replace the host sampler so no test shells out to a vendor tool.

    ``run_experiment`` builds its sampler from the detected target's vendor
    probe; a unit test must not run ``nvidia-smi`` or ``rocm-smi``, so the probe
    selection reports nothing and the sampler is a recording stand-in that
    answers no measurement. The returned list holds every sampler the default
    factory built. A test that scripted numbers supplies its own
    ``sampler_factory`` instead.
    """
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.memory_probe_for",
        lambda target: None,
    )
    samplers: list[FakeSampler] = []

    def build(**kwargs: object) -> FakeSampler:
        sampler = FakeSampler(**kwargs)  # type: ignore[arg-type]
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.ThreadedResourceSampler", build
    )
    return samplers


def _sampler_factory(sampler: FakeSampler) -> SamplerFactory:
    """Return an orchestrator sampler factory that always builds ``sampler``."""
    return lambda handle, baseline, probe: sampler


def _memory_probe(
    monkeypatch: pytest.MonkeyPatch, events: list[str], used_bytes: int
) -> RecordingProbe:
    """Install a recording vendor probe as the memory probe for any target."""
    probe = RecordingProbe(GpuMemory(used_bytes=used_bytes), events)
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.memory_probe_for", probe.selected_for
    )
    return probe


def test_detect_plan_start_measure_stop_and_parse_happen_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    detector = _detector(monkeypatch, events)
    experiment = _experiment()

    result = run_experiment(experiment, backend, client=client, benchmark_client=benchmark)

    assert events == ["detect", "plan", "create", "parse_result"]
    assert detector.overrides == [None]
    assert result == backend.result
    assert result.lifetime_resources == ResourceUsage()
    assert len(benchmark.calls) == 1
    assert client.container is not None
    assert client.container.stopped is True
    assert client.container.remove_calls == [True]


def test_each_stage_receives_the_previous_stage_output(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 4))
    experiment = _experiment()

    run_experiment(experiment, backend, client=client, benchmark_client=benchmark)

    assert backend.plan_calls == [(experiment, DetectedHardware(HardwareTarget.CUDA, 4))]
    outcome, spec, captured = backend.parse_calls[0]
    assert outcome is benchmark.outcome
    assert spec is _SPEC
    assert isinstance(captured, ContainerResult)
    assert captured.exit_code == 0
    assert client.create_calls[0][0] == _SPEC.container.image
    assert client.create_calls[0][1]["command"] == list(_SPEC.container.command)
    assert client.create_calls[0][1]["ports"] == {"8000": None}
    assert client.closed is False
    base_url, model, measured_spec, measured_benchmark = benchmark.calls[0]
    assert base_url == "http://127.0.0.1:8000"
    assert model == HUB_MODEL
    assert measured_spec is _SPEC
    assert measured_benchmark is experiment.benchmark


def test_the_started_containers_liveness_is_passed_to_the_benchmark_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark client can tell a stalled wait from a container that died."""
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events)

    run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)

    assert len(benchmark.liveness) == 1
    is_alive = benchmark.liveness[0]
    assert is_alive is not None
    assert is_alive() is True
    client.container_status = "exited"
    assert is_alive() is False


def test_sampling_starts_with_the_container_and_stops_before_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sampler brackets the container's whole life, not just the sweep's end."""
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    sampler = FakeSampler(events=events)
    _detector(monkeypatch, events)

    run_experiment(
        _experiment(),
        backend,
        client=client,
        benchmark_client=FakeBenchmarkClient(_outcome()),
        sampler_factory=_sampler_factory(sampler),
    )

    assert events == [
        "detect",
        "plan",
        "create",
        "sampler_start",
        "sampler_stop",
        "parse_result",
    ]
    assert (sampler.starts, sampler.stops) == (1, 1)
    assert client.container is not None
    assert client.container.stopped is True


def test_the_measured_runs_are_bracketed_by_the_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client receives the sampler so each measured run records its window."""
    events: list[str] = []
    sampler = FakeSampler()
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events)

    run_experiment(
        _experiment(),
        FakeBackend(_SPEC, _result(), events),
        client=RecordingPodmanClient(events),
        benchmark_client=benchmark,
        sampler_factory=_sampler_factory(sampler),
    )

    assert benchmark.samplers == [sampler]


def test_the_lifetime_peak_is_attached_to_the_parsed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole-container peak rides on the result the backend parsed."""
    events: list[str] = []
    lifetime = ResourceUsage(peak_vram_bytes=20 * _GIB, peak_ram_bytes=6 * _GIB)
    sampler = FakeSampler(lifetime=lifetime)
    backend = FakeBackend(_SPEC, _result(), events)
    _detector(monkeypatch, events)

    result = run_experiment(
        _experiment(),
        backend,
        client=RecordingPodmanClient(events),
        benchmark_client=FakeBenchmarkClient(_outcome()),
        sampler_factory=_sampler_factory(sampler),
    )

    assert result.lifetime_resources == lifetime
    assert result.runs == backend.result.runs
    assert result.target == backend.result.target
    assert result.raw_result == backend.result.raw_result


def test_the_idle_vram_baseline_is_read_before_the_container_starts(
    monkeypatch: pytest.MonkeyPatch, host_samplers: list[FakeSampler]
) -> None:
    """A device-level VRAM figure needs the device's state before the trial ran."""
    events: list[str] = []
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 1))
    probe = _memory_probe(monkeypatch, events, used_bytes=512)

    run_experiment(
        _experiment(),
        FakeBackend(_SPEC, _result(), events),
        client=RecordingPodmanClient(events),
        benchmark_client=FakeBenchmarkClient(_outcome()),
    )

    assert events[:4] == ["detect", "plan", "probe", "create"]
    assert probe.reads == 1
    assert probe.targets == [HardwareTarget.CUDA]
    (sampler,) = host_samplers
    assert sampler.baseline_vram_bytes == 512
    assert sampler.probe is probe


def test_the_host_sampler_reads_ram_through_the_running_containers_handle(
    monkeypatch: pytest.MonkeyPatch, host_samplers: list[FakeSampler]
) -> None:
    """Container memory stays the runtime's business; the sampler only asks it."""
    events: list[str] = []
    client = RecordingPodmanClient(events, stats_payload={"Stats": [{"MemUsage": 4096}]})
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 1))
    _memory_probe(monkeypatch, events, used_bytes=512)

    run_experiment(
        _experiment(),
        FakeBackend(_SPEC, _result(), events),
        client=client,
        benchmark_client=FakeBenchmarkClient(_outcome()),
    )

    (sampler,) = host_samplers
    assert sampler.ram_reader is not None
    assert sampler.ram_reader() == 4096
    assert sampler.container_id == client.container_id


def test_a_cpu_target_is_sampled_without_a_vendor_probe(
    monkeypatch: pytest.MonkeyPatch, host_samplers: list[FakeSampler]
) -> None:
    """A CPU sweep still reports RAM, and never fabricates a VRAM number."""
    events: list[str] = []
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.memory_probe_for", memory_probe_for
    )
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CPU, 1))

    run_experiment(
        _experiment(),
        FakeBackend(_SPEC, _result(), events),
        client=RecordingPodmanClient(events),
        benchmark_client=FakeBenchmarkClient(_outcome()),
    )

    (sampler,) = host_samplers
    assert sampler.probe is None
    assert sampler.baseline_vram_bytes is None


def test_a_sampler_that_cannot_measure_leaves_the_result_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken sampler is a diagnostic gap, never a failed trial."""
    events: list[str] = []

    class BrokenSampler(FakeSampler):
        def start(self) -> None:
            raise RuntimeError("no probe")

        def stop(self) -> None:
            raise RuntimeError("no probe")

        def lifetime_peak(self) -> ResourceUsage:
            raise RuntimeError("no probe")

    _detector(monkeypatch, events)

    result = run_experiment(
        _experiment(),
        FakeBackend(_SPEC, _result(), events),
        client=RecordingPodmanClient(events),
        benchmark_client=FakeBenchmarkClient(_outcome()),
        sampler_factory=_sampler_factory(BrokenSampler()),
    )

    assert result.lifetime_resources == ResourceUsage()


def test_a_failing_sampler_factory_still_tears_the_container_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building the sampler is part of the guarded lifecycle, not before it."""
    events: list[str] = []
    client = RecordingPodmanClient(events)
    _detector(monkeypatch, events)

    def failing_factory(
        handle: object, baseline_vram_bytes: int | None, probe: object
    ) -> ResourceSampler:
        raise RuntimeError("cannot build a sampler")

    with pytest.raises(RuntimeError, match="cannot build a sampler"):
        run_experiment(
            _experiment(),
            FakeBackend(_SPEC, _result(), events),
            client=client,
            benchmark_client=FakeBenchmarkClient(_outcome()),
            sampler_factory=failing_factory,
        )

    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_container_target_override_reaches_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 4))

    run_experiment(
        _experiment(target=HardwareTarget.CPU),
        backend,
        client=client,
        benchmark_client=FakeBenchmarkClient(_outcome()),
    )

    assert len(backend.plan_calls) == 1
    assert backend.plan_calls[0][1] == DetectedHardware(HardwareTarget.CUDA, 4)


def test_a_server_without_a_published_port_cannot_be_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    spec = ServerSpec(
        container=ContainerConfig(image="img"), health_path="/health", health_timeout=1.0
    )
    backend = FakeBackend(spec, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError, match="published no host port"):
        run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)

    assert benchmark.calls == []
    assert backend.parse_calls == []
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_dataset_preflight_tokenizes_once_and_stops_the_server(
    monkeypatch: pytest.MonkeyPatch, host_samplers: list[FakeSampler]
) -> None:
    """The preflight starts a one-slot server, tokenizes, and tears it down.

    Only the sweep's trials are measured; tokenizing a dataset takes no memory
    sample, so the preflight builds no sampler and brackets no run.
    """
    events: list[str] = []
    client = RecordingPodmanClient(events)
    spec = ServerSpec(
        container=ContainerConfig(
            image="fake/engine:latest",
            command=("serve",),
            ports={"8000": None},
        ),
        health_path="/health",
        health_timeout=600.0,
        tokenize_path="/tokenize",
        tokenize_prompt_field="content",
    )
    backend = FakeBackend(spec, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome(), request_tokens=321)
    experiment = replace(
        _experiment(),
        benchmark=replace(
            _experiment().benchmark, dataset_path="/data/prompts.jsonl", max_request_tokens=None
        ),
    )
    hardware = DetectedHardware(HardwareTarget.CPU, 1)

    tokens = measure_dataset_request_tokens(
        experiment, backend, hardware, client=client, benchmark_client=benchmark
    )

    assert tokens == 321
    assert events == ["plan", "create"]
    planned, _ = backend.plan_calls[0]
    assert planned.llm.max_num_seqs == 1
    assert planned.llm.speculative_method is None
    base_url, model, measured_spec, measured_benchmark = benchmark.tokenize_calls[0]
    assert base_url == "http://127.0.0.1:8000"
    assert model == HUB_MODEL
    assert measured_spec is spec
    assert measured_benchmark is experiment.benchmark
    assert benchmark.liveness[0] is not None
    assert benchmark.samplers == []
    assert host_samplers == []
    assert client.container is not None
    assert client.container.stopped is True


def test_dataset_preflight_is_skipped_without_a_dataset_or_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No dataset, or a spec without a tokenizer, starts no preflight server."""
    events: list[str] = []
    client = RecordingPodmanClient(events)
    benchmark = FakeBenchmarkClient(_outcome(), request_tokens=321)
    hardware = DetectedHardware(HardwareTarget.CPU, 1)
    backend = FakeBackend(_SPEC, _result(), events)

    assert (
        measure_dataset_request_tokens(
            _experiment(), backend, hardware, client=client, benchmark_client=benchmark
        )
        is None
    )

    experiment = replace(
        _experiment(),
        benchmark=replace(_experiment().benchmark, dataset_path="/data/prompts.jsonl"),
    )
    assert (
        measure_dataset_request_tokens(
            experiment, backend, hardware, client=client, benchmark_client=benchmark
        )
        is None
    )

    assert events == ["plan"]
    assert benchmark.tokenize_calls == []
    assert client.container is None


def test_container_start_failure_is_wrapped_without_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    start_error = APIError("failed to start container")
    client = RecordingPodmanClient(events, start_error=start_error)
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)

    error = excinfo.value
    assert isinstance(error.__cause__, ContainerRunError)
    assert error.__cause__.__cause__ is start_error
    assert error.raw_result is None
    assert benchmark.calls == []
    assert backend.parse_calls == []


def test_measurement_failure_is_wrapped_with_the_server_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    message = b"AssertionError: FlashAttention version not detected."
    client = RecordingPodmanClient(events, logs=_frame(2, message))
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(
        measure_error=ExperimentError("the server never became healthy")
    )
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)

    error = excinfo.value
    assert "the server never became healthy" in str(error)
    assert "FlashAttention version not detected." in str(error)
    assert error.raw_result is not None
    assert error.raw_result.stderr == message
    assert backend.parse_calls == []
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_measurement_failure_output_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chatty engine cannot flood the error message with its whole log."""
    message = _failure_message(monkeypatch, b"x" * 10000)

    assert "…" in message
    assert len(message) < 10000


def test_failure_excerpt_keeps_a_root_cause_before_a_long_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's cause leads the excerpt even when a long traceback follows it."""
    message = _failure_message(monkeypatch, _engine_output(_ROOT_CAUSE, frames=80))

    assert _ROOT_CAUSE in message
    assert "…" in message
    assert message.index(_ROOT_CAUSE) < message.index("…")


def test_failure_excerpt_keeps_the_earliest_root_cause_across_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frontend wrapper traceback does not hide the engine core's real cause.

    vLLM prints the engine core's traceback first and the frontend's last, with
    the frontend only restating the failure ("Engine core initialization failed.
    See root cause above"). The trailing window carries the wrapper, so the
    excerpt must prepend the earliest exception line — the engine core's — or
    the real cause is lost.
    """
    frames = "\n".join(f'  File "module_{i}.py", line {i}, in <module>' for i in range(80))
    engine_core = f"{_ROOT_CAUSE}\nTraceback (most recent call last):\n{frames}"
    wrapper = "RuntimeError: Engine core initialization failed. See root cause above"
    frontend = f"Traceback (most recent call last):\n{frames}\n{wrapper}"
    output = f"{engine_core}\n{frontend}".encode()

    message = _failure_message(monkeypatch, output)

    assert _ROOT_CAUSE in message
    assert message.index(_ROOT_CAUSE) < message.index("RuntimeError: Engine core")


def test_failure_excerpt_leaves_a_short_message_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message that fits the budget is attached verbatim, with no truncation."""
    short = "ValueError: the configured dtype is unsupported"

    message = _failure_message(monkeypatch, short.encode())

    assert message.endswith(short)
    assert "…" not in message


def test_failure_excerpt_of_empty_output_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine that captures nothing adds no excerpt to the error."""
    message = _failure_message(monkeypatch, b"")

    assert "\n" not in message
    assert "…" not in message


def test_failure_excerpt_stays_bounded_while_keeping_the_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root cause is kept without letting a chatty engine flood the message."""
    output = _engine_output(_ROOT_CAUSE, frames=300)

    message = _failure_message(monkeypatch, output)

    assert len(output) > 10000
    assert _ROOT_CAUSE in message
    assert len(message) < 3000


def test_stop_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    stop_error = APIError("stop failed")
    client = RecordingPodmanClient(events, stop_error=stop_error)
    backend = FakeBackend(_SPEC, _result(), events)
    benchmark = FakeBenchmarkClient(_outcome())
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, benchmark_client=benchmark)

    error = excinfo.value
    assert isinstance(error.__cause__, ContainerRunError)
    assert error.__cause__.__cause__ is stop_error
    assert backend.parse_calls == []
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_a_failed_measurement_still_stops_the_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial that never became usable must not leave an observer behind."""
    events: list[str] = []
    sampler = FakeSampler(events=events)
    client = RecordingPodmanClient(events)
    benchmark = FakeBenchmarkClient(
        measure_error=ExperimentError("the server never became healthy")
    )
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError, match="never became healthy"):
        run_experiment(
            _experiment(),
            FakeBackend(_SPEC, _result(), events),
            client=client,
            benchmark_client=benchmark,
            sampler_factory=_sampler_factory(sampler),
        )

    assert events == ["detect", "plan", "create", "sampler_start", "sampler_stop"]
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_an_interrupted_trial_still_stops_the_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt leaves neither a container nor a sampler polling the host."""
    events: list[str] = []
    sampler = FakeSampler()
    client = RecordingPodmanClient(events)
    _detector(monkeypatch, events)

    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            _experiment(),
            FakeBackend(_SPEC, _result(), events),
            client=client,
            benchmark_client=FakeBenchmarkClient(measure_error=KeyboardInterrupt()),
            sampler_factory=_sampler_factory(sampler),
        )

    assert (sampler.starts, sampler.stops) == (1, 1)
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_a_failed_container_stop_still_stops_the_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    sampler = FakeSampler(events=events)
    client = RecordingPodmanClient(events, stop_error=APIError("stop failed"))
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError, match="failed to stop the benchmark server"):
        run_experiment(
            _experiment(),
            FakeBackend(_SPEC, _result(), events),
            client=client,
            benchmark_client=FakeBenchmarkClient(_outcome()),
            sampler_factory=_sampler_factory(sampler),
        )

    assert sampler.stops == 1
    assert events == ["detect", "plan", "create", "sampler_start", "sampler_stop"]


def test_a_failed_container_start_builds_no_sampler(
    monkeypatch: pytest.MonkeyPatch, host_samplers: list[FakeSampler]
) -> None:
    """Nothing was started, so there is nothing to observe."""
    events: list[str] = []
    client = RecordingPodmanClient(events, start_error=APIError("failed to start container"))
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError):
        run_experiment(
            _experiment(),
            FakeBackend(_SPEC, _result(), events),
            client=client,
            benchmark_client=FakeBenchmarkClient(_outcome()),
        )

    assert host_samplers == []


def test_parse_failure_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    parse_error = ExperimentError("engine wrote nothing")
    backend = FakeBackend(_SPEC, _result(), events, parse_error=parse_error)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(
            _experiment(),
            backend,
            client=client,
            benchmark_client=FakeBenchmarkClient(_outcome()),
        )

    assert excinfo.value is parse_error
    assert events == ["detect", "plan", "create", "parse_result"]


def test_backend_defaults_to_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_SPEC, _result(), events)
    _detector(monkeypatch, events)
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.VLLMBackend", lambda: backend
    )

    result = run_experiment(
        _experiment(), client=client, benchmark_client=FakeBenchmarkClient(_outcome())
    )

    assert result == backend.result
    assert events == ["detect", "plan", "create", "parse_result"]


def test_backend_doubles_conform_to_the_inference_protocol() -> None:
    """Every backend double passed to ``run_experiment`` satisfies the protocol.

    ``run_experiment`` accepts any object with the ``InferenceBackend`` shape, so
    the double must track the protocol as it grows; a missing method would
    otherwise surface only when a test happened to call it.
    """
    assert isinstance(FakeBackend(_SPEC, _result(), []), InferenceBackend)


def test_sampler_doubles_conform_to_the_resource_sampler_protocol() -> None:
    """The sampler stand-in tracks ``ResourceSampler`` as the protocol grows."""
    assert isinstance(FakeSampler(), ResourceSampler)
