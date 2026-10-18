"""Unit tests for experiment orchestration.

``run_experiment``'s observable contract is the collaboration it drives: detect
the host's hardware once, ask the backend to plan a run, execute that plan
through the runtime layer, and hand the captured outcome back to the backend for
parsing — in that order, with the detected hardware, the experiment, and the
managed output directory passed through. A failed container run becomes an
``ExperimentError`` that preserves the cause and the raw result, and its message
keeps the engine's root-cause line alongside a bounded traceback tail.

These tests inject a fake hardware detector, a fake backend, and the shared fake
Podman client, so no daemon, inference engine, or specific hardware is required.
"""

from pathlib import Path

import pytest
from podman.errors import APIError

from fakes import FakeBackend, FakeContainer, FakePodmanClient, _frame
from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    InferenceBackend,
    LLMConfig,
    RunMetrics,
    run_experiment,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult, ContainerRunError

HUB_MODEL = "meta-llama/Llama-3.1-8B"
_PLAN = ContainerConfig(
    image="fake/engine:latest",
    command=("bench", "--model", HUB_MODEL),
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
    """Drive a failed run with synthetic engine output and return the error message."""
    events: list[str] = []
    client = RecordingPodmanClient(events, exit_code=1, logs=_frame(2, payload))
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)
    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client)
    return str(excinfo.value)


def _experiment(*, target: HardwareTarget | None = None) -> ExperimentConfig:
    """Build an experiment with an optional hardware-target override."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(target=target),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _result() -> ExperimentResult:
    """Build the canned typed result the fake backend returns from parsing."""
    return ExperimentResult(
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
        target=HardwareTarget.CPU,
        image=_PLAN.image,
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


class DirectoryAwareBackend(FakeBackend):
    """A backend that fails unless the output directory exists when it is called."""

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        output_dir: Path | None = None,
    ) -> ContainerConfig:
        assert output_dir is not None
        assert output_dir.is_dir()
        return super().plan(experiment, hardware, output_dir)

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        assert output_dir is not None
        assert output_dir.is_dir()
        return super().parse_result(result, plan, output_dir)


class RecordingPodmanClient(FakePodmanClient):
    """A fake Podman client that timestamps container creation into ``events``."""

    def __init__(self, events: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        create = self.containers.create

        def recording_create(image: str, **create_kwargs: object) -> FakeContainer:
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


def test_detect_plan_run_and_parse_happen_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_PLAN, _result(), events)
    detector = _detector(monkeypatch, events)
    experiment = _experiment()

    result = run_experiment(experiment, backend, client=client, output_dir=tmp_path)

    assert events == ["detect", "plan", "create", "parse_result"]
    assert detector.overrides == [None]
    assert result is backend.result


def test_each_stage_receives_the_previous_stage_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 4))
    experiment = _experiment()

    run_experiment(experiment, backend, client=client, output_dir=tmp_path)

    assert backend.plan_calls == [(experiment, DetectedHardware(HardwareTarget.CUDA, 4), tmp_path)]
    captured, planned, output_dir = backend.parse_calls[0]
    assert isinstance(captured, ContainerResult)
    assert captured.exit_code == 0
    assert planned is _PLAN
    assert output_dir == tmp_path
    assert client.create_calls[0][0] == _PLAN.image
    assert client.create_calls[0][1]["command"] == list(_PLAN.command)
    assert client.closed is False


def test_container_target_override_reaches_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_PLAN, _result(), events)
    detector = _detector(monkeypatch, events, DetectedHardware(HardwareTarget.CUDA, 4))

    run_experiment(
        _experiment(target=HardwareTarget.CPU), backend, client=client, output_dir=tmp_path
    )

    assert detector.overrides == [HardwareTarget.CPU]
    assert len(detector.overrides) == 1
    assert backend.plan_calls[0][1] == DetectedHardware(HardwareTarget.CUDA, 4)


def test_container_failure_is_wrapped_with_cause_and_raw_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events, exit_code=3, logs=_frame(1, b"boom"))
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, output_dir=tmp_path)

    error = excinfo.value
    assert isinstance(error.__cause__, ContainerRunError)
    assert error.raw_result is not None
    assert error.raw_result.exit_code == 3
    assert error.raw_result.stdout == b"boom"
    assert HUB_MODEL in str(error)
    assert "boom" in str(error)
    assert backend.parse_calls == []
    assert events == ["detect", "plan", "create"]


def test_container_failure_message_includes_the_engine_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine's own diagnostic is surfaced, not just the exit code."""
    events: list[str] = []
    message = b"AssertionError: FlashAttention version not detected."
    client = RecordingPodmanClient(events, exit_code=1, logs=_frame(2, message))
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, output_dir=tmp_path)

    assert "FlashAttention version not detected." in str(excinfo.value)
    assert excinfo.value.raw_result is not None
    assert excinfo.value.raw_result.stderr == message


def test_container_failure_output_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chatty engine cannot flood the error message with its whole log."""
    events: list[str] = []
    client = RecordingPodmanClient(events, exit_code=1, logs=_frame(2, b"x" * 10000))
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, output_dir=tmp_path)

    assert "…" in str(excinfo.value)
    assert len(str(excinfo.value)) < 10000


def test_failure_excerpt_keeps_a_root_cause_before_a_long_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's cause leads the excerpt even when a long traceback follows it."""
    message = _failure_message(monkeypatch, _engine_output(_ROOT_CAUSE, frames=80))

    assert _ROOT_CAUSE in message
    assert "…" in message
    assert message.index(_ROOT_CAUSE) < message.index("…")


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


def test_container_start_failure_is_wrapped_without_a_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    start_error = APIError("failed to start container")
    client = RecordingPodmanClient(events, start_error=start_error)
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, output_dir=tmp_path)

    error = excinfo.value
    assert isinstance(error.__cause__, ContainerRunError)
    assert error.__cause__.__cause__ is start_error
    assert error.raw_result is None
    assert backend.parse_calls == []


def test_parse_failure_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    parse_error = ExperimentError("engine wrote nothing")
    backend = FakeBackend(_PLAN, _result(), events, parse_error=parse_error)
    _detector(monkeypatch, events)

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(_experiment(), backend, client=client, output_dir=tmp_path)

    assert excinfo.value is parse_error
    assert events == ["detect", "plan", "create", "parse_result"]


def test_explicit_output_directory_is_used_as_is_and_left_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    run_experiment(_experiment(), backend, client=client, output_dir=output_dir)

    assert backend.plan_calls[0][2] == output_dir
    assert backend.parse_calls[0][2] == output_dir
    assert output_dir.is_dir()


def test_omitted_output_directory_is_temporary_and_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = DirectoryAwareBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)

    run_experiment(_experiment(), backend, client=client)

    planned_dir = backend.plan_calls[0][2]
    assert planned_dir is not None
    assert backend.parse_calls[0][2] == planned_dir
    assert not planned_dir.exists()


def test_backend_defaults_to_vllm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    client = RecordingPodmanClient(events)
    backend = FakeBackend(_PLAN, _result(), events)
    _detector(monkeypatch, events)
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.VLLMBackend", lambda: backend
    )

    result = run_experiment(_experiment(), client=client, output_dir=tmp_path)

    assert result is backend.result
    assert events == ["detect", "plan", "create", "parse_result"]


@pytest.mark.parametrize("backend_cls", [FakeBackend, DirectoryAwareBackend])
def test_backend_doubles_conform_to_the_inference_protocol(
    backend_cls: type[FakeBackend],
) -> None:
    """Every backend double passed to ``run_experiment`` satisfies the protocol.

    ``run_experiment`` accepts any object with the ``InferenceBackend`` shape, so
    the doubles must track the protocol as it grows; a missing method would
    otherwise surface only when a test happened to call it.
    """
    assert isinstance(backend_cls(_PLAN, _result(), []), InferenceBackend)
