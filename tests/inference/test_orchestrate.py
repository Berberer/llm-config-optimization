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
"""

from dataclasses import replace

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
    HardwareTarget,
    InferenceBackend,
    LLMConfig,
    RunMetrics,
    ServerSpec,
    measure_dataset_request_tokens,
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
    assert result is backend.result
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preflight starts a one-slot server, tokenizes, and tears it down."""
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

    assert result is backend.result
    assert events == ["detect", "plan", "create", "parse_result"]


def test_backend_doubles_conform_to_the_inference_protocol() -> None:
    """Every backend double passed to ``run_experiment`` satisfies the protocol.

    ``run_experiment`` accepts any object with the ``InferenceBackend`` shape, so
    the double must track the protocol as it grows; a missing method would
    otherwise surface only when a test happened to call it.
    """
    assert isinstance(FakeBackend(_SPEC, _result(), []), InferenceBackend)
