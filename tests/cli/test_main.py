"""Unit tests for the command-line entry point.

``main``'s observable contract is the exit code it returns and what it writes to
its output streams for a given argument vector, injected backend, and injected
container client. These tests drive the real composition root — argument
parsing, the preparation sequence, the reporter, the optimizer, and the final
summary — with the shared fake inference backend and fake Podman client, so no
daemon, GPU, or engine is involved and the whole wiring is exercised end to end.

The model argument is a real temporary directory, so the preparation stage takes
the local-checkpoint branch and never touches the network, and ``--target cpu``
forces the CPU target so hardware detection short-circuits without a probe. The
scenarios mirror the exit-code contract: a successful bounded sweep, a space the
backend rejects, a model the backend rejects, an invalid definition, a sweep that
completes no trial, an interrupt, and an explicit TUI request without a terminal.
"""

import json
from io import StringIO
from pathlib import Path

import pytest
import yaml

from fakes import FakeBackend, FakePodmanClient
from llm_config_optimization.cli import main
from llm_config_optimization.inference import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    VLLMBackend,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult
from llm_config_optimization.search_space import TUNABLE_PARAMETERS

PLAN = ContainerConfig(image="fake/engine:latest", command=("bench",))
"""The canned container plan the fake backend returns."""

RESULT = ExperimentResult(
    tokens_per_second=42.0,
    requests_per_second=4.0,
    elapsed_time=2.5,
    num_prompts=8,
    target=HardwareTarget.CPU,
    image=PLAN.image,
    devices=(),
    raw_json={"tokens_per_second": 42.0},
    raw_result=ContainerResult(exit_code=0, stdout=b"", stderr=b"", container_id="fake"),
)
"""The canned typed result the fake backend returns from parsing every trial."""


def _backend() -> FakeBackend:
    """Build a fake backend returning the canned plan and result."""
    return FakeBackend(PLAN, RESULT, [])


class UnsupportedBackend(FakeBackend):
    """A backend that reports the space as unsupported during validation."""

    def validate(self, space: object, hardware: DetectedHardware | None = None) -> tuple[str, ...]:
        return ("space.dtype: the backend does not support this dtype",)


class HardwareRecordingBackend(FakeBackend):
    """A backend that records the hardware the CLI validates the space against."""

    def __init__(self) -> None:
        super().__init__(PLAN, RESULT, [])
        self.validated_hardware: list[DetectedHardware | None] = []

    def validate(self, space: object, hardware: DetectedHardware | None = None) -> tuple[str, ...]:
        self.validated_hardware.append(hardware)
        return ()


class ModelProblemBackend(FakeBackend):
    """A backend that reports canned model-compatibility problems and records the calls.

    It records every ``(experiment, model)`` pair it is asked about, so a test
    can pin that the CLI passed the base configuration and the resolved
    checkpoint path, and it returns every canned problem so the aggregation
    into one failure can be observed.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(PLAN, RESULT, [])
        self.problems = problems
        self.model_calls: list[tuple[ExperimentConfig, Path]] = []

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> tuple[str, ...]:
        self.model_calls.append((experiment, model))
        return self.problems


class CheckpointValidatingBackend(FakeBackend):
    """A fake backend that delegates model validation to the real vLLM rule.

    Planning and result parsing stay canned, so a compatible run completes
    without a daemon or a valid benchmark payload, while the checkpoint on disk
    is inspected by :meth:`VLLMBackend.validate_model`. This is the CLI wiring
    under test; the compatibility rule itself is covered directly against the
    backend.
    """

    def __init__(self) -> None:
        super().__init__(PLAN, RESULT, [])

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> tuple[str, ...]:
        return tuple(VLLMBackend().validate_model(experiment, model))


class InterruptingBackend(FakeBackend):
    """A backend that completes one trial and then raises ``KeyboardInterrupt``.

    Raising from the second parse makes the first trial complete and the second
    fail with an interrupt, so the CLI has a best point to report and the exit
    code is the interrupted one rather than the empty-sweep one.
    """

    def __init__(self) -> None:
        super().__init__(PLAN, RESULT, [])
        self.parses = 0

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        self.parses += 1
        if self.parses > 1:
            raise KeyboardInterrupt
        return super().parse_result(result, plan, output_dir)


def _write_space(path: Path, **overrides: object) -> Path:
    """Write a complete, valid space file, mostly deferring to the engine.

    Every tunable is declared so the engine-agnostic loader accepts the file;
    ``overrides`` replace individual dimensions. The default is one fixed
    dimension, so a sweep still samples a point without depending on the
    optimizer's choices.
    """
    space: dict[str, object] = {parameter: {"default": True} for parameter in TUNABLE_PARAMETERS}
    space.update({"dtype": {"fixed": "float16"}, **overrides})
    path.write_text(yaml.safe_dump({"space": space}), encoding="utf-8")
    return path


def _write_checkpoint(model: str, method: str) -> None:
    """Write a ``config.json`` declaring ``method`` into the checkpoint directory."""
    payload = {"quantization_config": {"quant_method": method}}
    Path(model, "config.json").write_text(json.dumps(payload), encoding="utf-8")


def _invoke(
    model: str,
    space: Path,
    *extra: str,
    backend: FakeBackend | None = None,
    client: FakePodmanClient | None = None,
) -> tuple[int, str, str]:
    """Run ``main`` with injected fakes and captured streams.

    The CPU target is always passed so detection short-circuits without probing
    the host; the caller supplies the output flag and budget through ``extra``.
    Returns the exit code and the captured stdout and stderr.
    """
    out, err = StringIO(), StringIO()
    code = main(
        [model, str(space), "--target", "cpu", *extra],
        backend=_backend() if backend is None else backend,
        client=FakePodmanClient() if client is None else client,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def model(tmp_path: Path) -> str:
    """A local checkpoint path, so preparation never downloads a model."""
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    return str(checkpoint)


def test_a_successful_bounded_sweep_reports_progress_and_the_winner(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()
    client = FakePodmanClient()

    code, out, err = _invoke(
        model, space, "--n-trials", "2", "--output", "log", backend=backend, client=client
    )

    assert code == 0
    assert err == ""
    assert f"model: {model}" in out
    assert f"space: {space}" in out
    assert "target: cpu" in out
    assert "image: default" in out
    assert "budget: 2 trials" in out
    assert f"loading search space: {space}" in out
    assert "validating search space" in out
    assert "resolved target: cpu" in out
    assert f"using local model: {model}" in out
    assert f"pulling container image: {PLAN.image}" in out
    assert "preparing study" in out
    assert "trial 1/2  COMPLETE  42.00 tps  best 42.00" in out
    assert "trial 2/2  COMPLETE  42.00 tps  best 42.00" in out
    assert "best trial: 0" in out
    assert "best throughput: 42.00 tps" in out
    assert "trials run: 2" in out
    assert "winning configuration:" in out
    assert len(backend.plan_calls) == 2
    assert (PLAN.image, "missing") in client.pull_calls


def test_the_cli_gives_the_default_backend_a_temporary_cache_directory(
    tmp_path: Path, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-run compile cache is created for the default backend and removed."""
    space = _write_space(tmp_path / "space.yaml")
    created: list[FakeBackend] = []

    class CacheRecordingBackend(FakeBackend):
        def __init__(self, *, cache_dir: Path | None = None, **kwargs: object) -> None:
            super().__init__(PLAN, RESULT, [])
            self.cache_dir = cache_dir
            created.append(self)

    monkeypatch.setattr("llm_config_optimization.cli.VLLMBackend", CacheRecordingBackend)
    out, err = StringIO(), StringIO()

    code = main(
        [model, str(space), "--target", "cpu", "--n-trials", "1", "--output", "log"],
        client=FakePodmanClient(),
        stdout=out,
        stderr=err,
    )

    assert code == 0
    assert err.getvalue() == ""
    assert len(created) == 1
    cache_dir = created[0].cache_dir
    assert cache_dir is not None
    assert not cache_dir.exists()


def test_the_output_mode_defaults_to_log_without_a_terminal(tmp_path: Path, model: str) -> None:
    """With no ``--output`` flag a redirected stream gets one line per trial."""
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(model, space, "--n-trials", "1")

    assert code == 0
    assert err == ""
    assert "trial 1/1  COMPLETE  42.00 tps" in out


def test_the_never_pull_policy_skips_the_pull_and_says_so(tmp_path: Path, model: str) -> None:
    """``--pull-policy never`` must not pull and must not claim to."""
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model, space, "--n-trials", "1", "--pull-policy", "never", client=client
    )

    assert code == 0
    assert err == ""
    assert client.pull_calls == []
    assert f"using container image: {PLAN.image} (pull policy never)" in out
    assert "pulling container image" not in out


def test_a_successful_sweep_writes_the_winning_configuration(tmp_path: Path, model: str) -> None:
    space = _write_space(tmp_path / "space.yaml")
    destination = tmp_path / "best.yaml"

    code, _, err = _invoke(
        model, space, "--n-trials", "1", "--output", "log", "--output-config", str(destination)
    )

    assert code == 0
    assert err == ""
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert set(document) == {"container", "benchmark", "llm"}
    assert document["container"]["target"] == "cpu"
    assert document["container"]["image"] == PLAN.image


def test_a_space_the_backend_rejects_returns_the_unsupported_code(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")
    backend = UnsupportedBackend(PLAN, RESULT, [])

    code, out, err = _invoke(model, space, "--n-trials", "1", backend=backend)

    assert code == 4
    assert "space.dtype" in err
    assert "best trial:" not in out
    assert backend.plan_calls == []


def test_validation_receives_the_resolved_hardware(tmp_path: Path, model: str) -> None:
    """The backend is asked about the accelerator class and device count to use."""
    space = _write_space(tmp_path / "space.yaml")
    backend = HardwareRecordingBackend()

    code, _, err = _invoke(model, space, "--n-trials", "1", "--target", "rocm", backend=backend)

    assert code == 0
    assert err == ""
    assert backend.validated_hardware
    assert backend.validated_hardware[0] is not None
    assert backend.validated_hardware[0].target == HardwareTarget.ROCM


def test_an_attention_backend_unsupported_on_the_target_is_rejected_up_front(
    tmp_path: Path, model: str
) -> None:
    """A CUDA-only backend on the CPU target fails validation, not every trial."""
    space = _write_space(tmp_path / "space.yaml", attention_backend={"fixed": "flash_attention"})

    code, out, err = _invoke(model, space, "--n-trials", "1", backend=VLLMBackend())

    assert code == 4
    assert "attention_backend" in err
    assert "cpu" in err
    assert "preparing study" not in out


def test_a_tensor_parallel_size_larger_than_the_device_count_is_rejected_up_front(
    tmp_path: Path, model: str
) -> None:
    """A single-device target cannot run a two-way tensor-parallel space."""
    space = _write_space(tmp_path / "space.yaml", tensor_parallel_size={"fixed": 2})

    code, out, err = _invoke(model, space, "--n-trials", "1", backend=VLLMBackend())

    assert code == 4
    assert "tensor_parallel_size" in err
    assert "preparing study" not in out


def test_every_model_compatibility_problem_is_reported_before_the_image_pull(
    tmp_path: Path, model: str
) -> None:
    """The backend's problems are aggregated and fail the run before the pull."""
    space = _write_space(tmp_path / "space.yaml")
    backend = ModelProblemBackend("first problem", "second problem")
    client = FakePodmanClient()

    code, out, err = _invoke(model, space, "--n-trials", "1", backend=backend, client=client)

    assert code == 4
    assert "unsupported configuration" in err
    assert "  - first problem" in err
    assert "  - second problem" in err
    assert "validating model compatibility" in out
    assert "pulling container image" not in out
    assert "preparing study" not in out
    assert client.pull_calls == []
    assert len(backend.model_calls) == 1
    experiment, resolved = backend.model_calls[0]
    assert experiment.llm.quantization is None
    assert resolved == Path(model).expanduser().resolve()


def test_an_incompatible_quantization_is_rejected_before_the_image_pull(
    tmp_path: Path, model: str
) -> None:
    """An AWQ checkpoint with ``--quantization gptq`` fails before any download."""
    _write_checkpoint(model, "awq")
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        "--quantization",
        "gptq",
        backend=CheckpointValidatingBackend(),
        client=client,
    )

    assert code == 4
    assert "unsupported configuration" in err
    assert "awq" in err
    assert "gptq" in err
    assert "validating model compatibility" in out
    assert "pulling container image" not in out
    assert client.pull_calls == []


def test_a_compatible_quantization_proceeds_to_the_image_pull(tmp_path: Path, model: str) -> None:
    """An AWQ checkpoint with ``--quantization awq`` is accepted and the sweep runs."""
    _write_checkpoint(model, "awq")
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        "--quantization",
        "awq",
        backend=CheckpointValidatingBackend(),
        client=client,
    )

    assert code == 0
    assert err == ""
    assert "validating model compatibility" in out
    assert f"pulling container image: {PLAN.image}" in out
    assert (PLAN.image, "missing") in client.pull_calls


def test_an_omitted_quantization_accepts_a_mismatched_checkpoint(
    tmp_path: Path, model: str
) -> None:
    """With no ``--quantization`` the engine auto-detects, so the check is a no-op."""
    _write_checkpoint(model, "awq")
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(model, space, "--n-trials", "1", backend=CheckpointValidatingBackend())

    assert code == 0
    assert err == ""
    assert "validating model compatibility" in out


def test_an_invalid_definition_returns_the_definition_code(tmp_path: Path, model: str) -> None:
    space = tmp_path / "space.yaml"
    space.write_text("space: [unclosed\n", encoding="utf-8")

    code, _, err = _invoke(model, space, "--n-trials", "1")

    assert code == 3
    assert "invalid experiment definition" in err


def test_a_missing_definition_returns_the_definition_code(tmp_path: Path, model: str) -> None:
    space = tmp_path / "absent.yaml"

    code, _, err = _invoke(model, space, "--n-trials", "1")

    assert code == 3
    assert err != ""


def test_a_sweep_that_completes_no_trial_returns_the_empty_sweep_code(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")
    backend = FakeBackend(PLAN, RESULT, [], parse_error=ExperimentError("engine never started"))

    code, _, err = _invoke(model, space, "--n-trials", "2", backend=backend)

    assert code == 5
    assert "without a completed trial" in err


def test_an_interrupt_reports_the_best_point_and_returns_the_interrupt_code(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(
        model, space, "--n-trials", "2", "--output", "log", backend=InterruptingBackend()
    )

    assert code == 130
    assert err == ""
    assert "trial 1/2  COMPLETE  42.00 tps" in out
    assert "best trial: 0" in out
    assert "best throughput: 42.00 tps" in out
    assert "trials run: 2" in out


def test_an_explicit_tui_request_without_a_terminal_is_a_usage_error(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(model, space, "--n-trials", "1", "--output", "tui")

    assert code == 2
    assert "requires a terminal" in err
    assert out == ""


def test_version_exits_zero_without_running_a_sweep() -> None:
    out, err = StringIO(), StringIO()

    code = main(
        ["--version"], backend=_backend(), client=FakePodmanClient(), stdout=out, stderr=err
    )

    assert code == 0
    assert "llm-config-optimization" in out.getvalue()
    assert err.getvalue() == ""


def test_help_is_written_to_the_injected_stdout() -> None:
    out, err = StringIO(), StringIO()

    code = main(["--help"], backend=_backend(), client=FakePodmanClient(), stdout=out, stderr=err)

    assert code == 0
    assert "MODEL" in out.getvalue()
    assert err.getvalue() == ""


def test_a_usage_error_is_reported_on_the_injected_stderr() -> None:
    out, err = StringIO(), StringIO()

    code = main(
        ["model", "space.yaml", "--unknown"],
        backend=_backend(),
        client=FakePodmanClient(),
        stdout=out,
        stderr=err,
    )

    assert code == 2
    assert "unrecognized arguments" in err.getvalue()
