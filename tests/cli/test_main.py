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
The engine-scoped option rules are covered here too: the registry selects the
llama.cpp backend for the ``llamacpp`` token, and a vLLM-only option paired with
it is a usage error rather than a value the engine ignores.
"""

import json
import struct
from io import StringIO
from pathlib import Path

import optuna
import pytest
import yaml

from fakes import FakeBackend, FakePodmanClient
from llm_config_optimization.cli import main
from llm_config_optimization.inference import (
    BenchmarkOutcome,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    LlamaCppBackend,
    ModelRequest,
    QuantizationMethod,
    RunMetrics,
    ServerSpec,
    VLLMBackend,
)
from llm_config_optimization.optimization import ObjectiveMode
from llm_config_optimization.runtime import ContainerConfig, ContainerResult
from llm_config_optimization.search_space import TUNABLE_PARAMETERS

PLAN = ServerSpec(
    container=ContainerConfig(image="fake/engine:latest", command=("serve",), ports={"8000": None}),
    health_path="/health",
    health_timeout=600.0,
)
"""The canned server plan the fake backend returns."""

RESULT = ExperimentResult(
    runs=(
        RunMetrics(
            tokens_per_second=42.0,
            requests_per_second=4.0,
            elapsed_time=2.5,
            num_prompts=8,
            raw_json={"tokens_per_second": 42.0},
            prefill_tokens_per_second=1000.0,
            decode_tokens_per_second=50.0,
        ),
    ),
    failed_runs=0,
    target=HardwareTarget.CPU,
    image=PLAN.container.image,
    devices=(),
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


class UnresolvableModelBackend(FakeBackend):
    """A backend whose model-argument resolution rejects the model.

    The real llama.cpp backend raises :class:`ExperimentError` for a directory,
    a repository with no GGUF file, and an ambiguous ``:QUANT`` selector; the
    composition root must map every such rejection rather than letting it
    escape as an unexpected error. Planning and parsing stay canned and are
    never reached.
    """

    def __init__(self) -> None:
        super().__init__(PLAN, RESULT, [])

    def model_request(self, model: str) -> ModelRequest:
        raise ExperimentError("the repository holds several GGUF files; select one with :QUANT")


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


class GgufValidatingBackend(FakeBackend):
    """A fake backend that delegates model validation to the real llama.cpp rule.

    Planning and result parsing stay canned, so a compatible run completes
    without a daemon or a valid benchmark payload, while the checkpoint on disk
    is inspected by :meth:`LlamaCppBackend.validate_model`. This is the CLI
    wiring under test; the GGUF rule itself is covered directly against the
    backend.
    """

    def __init__(self) -> None:
        super().__init__(PLAN, RESULT, [])

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> tuple[str, ...]:
        return tuple(LlamaCppBackend().validate_model(experiment, model))


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
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        self.parses += 1
        if self.parses > 1:
            raise KeyboardInterrupt
        return super().parse_result(outcome, spec, result)


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
    engine: str = "vllm",
) -> tuple[int, str, str]:
    """Run ``main`` with injected fakes and captured streams.

    The CPU target is always passed so detection short-circuits without probing
    the host; the caller supplies the output flag and budget through ``extra``
    and the ``ENGINE`` token through ``engine``, which defaults to ``vllm``.
    Returns the exit code and the captured stdout and stderr.
    """
    out, err = StringIO(), StringIO()
    code = main(
        [model, str(space), engine, "--target", "cpu", *extra],
        backend=_backend() if backend is None else backend,
        client=FakePodmanClient() if client is None else client,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


class TtyStringIO(StringIO):
    """A captured stream that reports itself as a terminal, so TUI mode is selected."""

    def isatty(self) -> bool:
        return True


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
    assert "engine: vllm" in out
    assert "target: cpu" in out
    assert "image: default" in out
    assert "budget: 2 trials" in out
    assert f"loading search space: {space}" in out
    assert "validating search space" in out
    assert "resolved target: cpu" in out
    assert f"using local model: {model}" in out
    assert f"pulling container image: {PLAN.container.image}" in out
    assert "preparing study" in out
    assert "trial 1/2  COMPLETE  prefill 1000.00 tps  decode 50.00 tps" in out
    assert "trial 2/2  COMPLETE  prefill 1000.00 tps  decode 50.00 tps" in out
    assert "recommended trial: 0" in out
    assert "recommended rates: prefill 1000.00 tps  decode 50.00 tps" in out
    assert "pareto front:" in out
    assert "trials run: 2" in out
    assert "winning configuration:" in out
    assert len(backend.plan_calls) == 2
    assert (PLAN.container.image, "missing") in client.pull_calls


def test_the_default_study_name_carries_the_objective_mode(tmp_path: Path, model: str) -> None:
    """A zero-config resume cannot collide with a study scored under another mode."""
    space = _write_space(tmp_path / "space.yaml")
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    code, out, err = _invoke(
        model, space, "--n-trials", "1", "--output", "log", "--storage", storage
    )

    assert code == 0
    assert err == ""
    assert "preparing study" in out
    recorded = optuna.load_study(
        study_name=f"{model.replace('/', '_')}_{ObjectiveMode.MULTI.value}",
        storage=storage,
    )
    assert recorded.user_attrs["objective_mode"] == ObjectiveMode.MULTI.value


def test_an_explicit_objective_selects_a_scalar_study(tmp_path: Path, model: str) -> None:
    """``--objective combined`` runs the historical single-objective sweep."""
    space = _write_space(tmp_path / "space.yaml")
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        "--output",
        "log",
        "--storage",
        storage,
        "--objective",
        "combined",
    )

    assert code == 0
    assert err == ""
    assert "trial 1/1  COMPLETE  42.00 tps  prefill 1000.00 tps  decode 50.00 tps" in out
    assert "best trial: 0" in out
    assert "best throughput: 42.00 tps" in out
    recorded = optuna.load_study(
        study_name=f"{model.replace('/', '_')}_{ObjectiveMode.COMBINED.value}",
        storage=storage,
    )
    assert recorded.user_attrs["objective_mode"] == ObjectiveMode.COMBINED.value


def test_a_weighted_sweep_scores_the_weighted_harmonic_rate(tmp_path: Path, model: str) -> None:
    """The weights default to the benchmark mix, so no arithmetic is needed."""
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(
        model, space, "--n-trials", "1", "--output", "log", "--objective", "weighted"
    )

    assert code == 0
    assert err == ""
    # (1024 + 128) / (1024/1000 + 128/50) for the canned 1000/50 phase rates.
    assert "trial 1/1  COMPLETE  321.43 tps" in out
    assert "best throughput: 321.43 tps" in out


def test_explicit_weights_override_the_benchmark_mix(tmp_path: Path, model: str) -> None:
    """A zero prefill weight degenerates the score to the decode rate."""
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        "--output",
        "log",
        "--objective",
        "weighted",
        "--prefill-weight",
        "0",
        "--decode-weight",
        "1",
    )

    assert code == 0
    assert err == ""
    assert "best throughput: 50.00 tps" in out


@pytest.mark.parametrize(
    "extra",
    [
        ["--objective", "multi", "--prefill-weight", "0", "--decode-weight", "0"],
        ["--objective", "weighted", "--prefill-weight", "-1"],
        ["--objective", "combined", "--prefill-weight", "nan"],
    ],
    ids=["both-zero", "negative", "non-finite"],
)
def test_invalid_objective_weights_are_a_usage_error_before_any_preparation(
    tmp_path: Path, model: str, extra: list[str]
) -> None:
    """A weight outside the model fails before the space, hardware, or image work."""
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()
    client = FakePodmanClient()

    code, out, err = _invoke(
        model, space, "--n-trials", "1", *extra, backend=backend, client=client
    )

    assert code == 2
    assert out == ""
    assert "error:" in err
    assert backend.model_request_calls == []
    assert backend.plan_calls == []
    assert client.pull_calls == []


def test_the_llamacpp_engine_is_selected_through_the_registry(
    tmp_path: Path, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry maps the ``llamacpp`` token to the llama.cpp backend.

    The backend class is replaced with the shared fakes, so the selection, the
    startup header, and the sweep are exercised without a GGUF checkpoint or an
    engine; the recorded instance pins that the factory built the llama.cpp
    backend for this run.
    """
    space = _write_space(tmp_path / "space.yaml")
    created: list[FakeBackend] = []

    class RecordingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__(PLAN, RESULT, [])
            created.append(self)

    monkeypatch.setattr("llm_config_optimization.cli.LlamaCppBackend", RecordingBackend)
    out, err = StringIO(), StringIO()

    code = main(
        [model, str(space), "llamacpp", "--target", "cpu", "--n-trials", "1", "--output", "log"],
        client=FakePodmanClient(),
        stdout=out,
        stderr=err,
    )

    assert code == 0
    assert err.getvalue() == ""
    assert len(created) == 1
    assert "engine: llamacpp" in out.getvalue()
    assert "trial 1/1  COMPLETE  prefill 1000.00 tps  decode 50.00 tps" in out.getvalue()


@pytest.mark.parametrize(
    "extra",
    [
        ["--quantization", "awq"],
        ["--gpu-memory-utilization", "0.9"],
    ],
    ids=["quantization", "gpu-memory-utilization"],
)
def test_a_vllm_only_option_is_rejected_for_llamacpp_as_a_usage_error(
    tmp_path: Path, model: str, extra: list[str]
) -> None:
    """A vLLM-only option paired with llama.cpp fails before any preparation.

    llama.cpp has no ``--quantization`` flag (the GGUF file carries its
    quantization) and no ``--gpu-memory-utilization`` knob, so both are usage
    errors rather than values the backend silently ignores. Nothing is loaded,
    planned, or pulled.
    """
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()
    client = FakePodmanClient()
    out, err = StringIO(), StringIO()

    code = main(
        [model, str(space), "llamacpp", "--target", "cpu", "--n-trials", "1", *extra],
        backend=backend,
        client=client,
        stdout=out,
        stderr=err,
    )

    assert code == 2
    assert out.getvalue() == ""
    assert extra[0] in err.getvalue()
    assert "llamacpp" in err.getvalue()
    assert backend.plan_calls == []
    assert client.pull_calls == []


def test_the_vllm_only_options_are_preserved_for_vllm(tmp_path: Path, model: str) -> None:
    """The same options still reach a vLLM run unchanged."""
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()

    code, _, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        "--quantization",
        "awq",
        "--gpu-memory-utilization",
        "0.9",
        backend=backend,
    )

    assert code == 0
    assert err == ""
    planned, _ = backend.plan_calls[0]
    assert planned.llm.quantization is QuantizationMethod.AWQ
    assert planned.llm.gpu_memory_utilization == 0.9


def test_the_resolved_checkpoint_reaches_the_planned_run(tmp_path: Path, model: str) -> None:
    """The backend hook decides the local path, and the run is planned from it."""
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()

    code, _, err = _invoke(model, space, "--n-trials", "1", backend=backend)

    assert code == 0
    assert err == ""
    assert backend.model_request_calls == [model]
    planned, _ = backend.plan_calls[0]
    assert planned.benchmark.resolved_path == Path(model).expanduser().resolve()
    assert planned.benchmark.model == model


def test_an_unresolvable_model_argument_is_an_unsupported_model(tmp_path: Path, model: str) -> None:
    """A backend rejection while resolving MODEL maps to the unsupported code.

    llama.cpp rejects a directory, a repository with no GGUF file, and an
    ambiguous ``:QUANT`` selector from ``model_request``. Those are expected
    input errors, so they must return the unsupported-model exit code before any
    download, plan, or pull instead of falling through to the unexpected-error
    code.
    """
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model, space, "--n-trials", "1", backend=UnresolvableModelBackend(), client=client
    )

    assert code == 4
    assert "unsupported model" in err
    assert "several GGUF files" in err
    assert "preparing study" not in out
    assert client.pull_calls == []


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
        [model, str(space), "vllm", "--target", "cpu", "--n-trials", "1", "--output", "log"],
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
    assert "trial 1/1  COMPLETE  prefill 1000.00 tps  decode 50.00 tps" in out


def test_the_budget_reports_the_sweep_timeout(tmp_path: Path, model: str) -> None:
    """The renamed ``--sweep-timeout`` flag is the budget the header reports."""
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(model, space, "--n-trials", "2", "--sweep-timeout", "30")

    assert code == 0
    assert err == ""
    assert "budget: 2 trials, 30s" in out


@pytest.mark.parametrize(
    "extra",
    [
        ["--repetitions", "0"],
        ["--warmup-runs", "-1"],
        ["--startup-timeout", "0"],
    ],
    ids=["repetitions", "warmup", "startup-timeout"],
)
def test_an_out_of_range_measurement_control_is_a_usage_error(
    tmp_path: Path, model: str, extra: list[str]
) -> None:
    """A bad measurement control fails before the space is loaded or a trial runs."""
    space = _write_space(tmp_path / "space.yaml")
    backend = _backend()
    client = FakePodmanClient()

    code, out, err = _invoke(
        model, space, "--n-trials", "1", *extra, backend=backend, client=client
    )

    assert code == 2
    assert out == ""
    assert "error:" in err
    assert backend.plan_calls == []
    assert client.pull_calls == []


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
    assert f"using container image: {PLAN.container.image} (pull policy never)" in out
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
    assert document["container"]["image"] == PLAN.container.image


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


def test_the_loader_requires_only_the_backends_applicable_parameters(
    tmp_path: Path, model: str
) -> None:
    """A space declaring exactly a reduced applicable set is accepted."""
    required = {"max_model_len", "max_num_seqs"}
    space = tmp_path / "space.yaml"
    space.write_text(
        yaml.safe_dump({"space": {name: {"default": True} for name in required}}),
        encoding="utf-8",
    )
    backend = FakeBackend(PLAN, RESULT, [], applicable_parameters=required)

    code, _, err = _invoke(model, space, "--n-trials", "1", backend=backend)

    assert code == 0
    assert err == ""


def test_a_space_missing_a_backend_required_tunable_is_a_definition_error(
    tmp_path: Path, model: str
) -> None:
    """A parameter the selected engine requires but the file omits fails before any trial."""
    required = {"max_model_len", "max_num_seqs"}
    space = tmp_path / "space.yaml"
    space.write_text(
        yaml.safe_dump({"space": {"max_model_len": {"default": True}}}),
        encoding="utf-8",
    )
    backend = FakeBackend(PLAN, RESULT, [], applicable_parameters=required)

    code, _, err = _invoke(model, space, "--n-trials", "1", backend=backend)

    assert code == 3
    assert "space.max_num_seqs" in err
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


def test_a_space_llama_cpp_does_not_forward_is_rejected_up_front(
    tmp_path: Path, model: str
) -> None:
    """A declared ``dtype`` is a parameter llama.cpp has no flag for."""
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        backend=LlamaCppBackend(),
        client=client,
        engine="llamacpp",
    )

    assert code == 4
    assert "space.dtype" in err
    assert "does not forward" in err
    assert "preparing study" not in out
    assert client.pull_calls == []


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


def test_a_non_gguf_checkpoint_is_rejected_before_the_image_pull(
    tmp_path: Path, model: str
) -> None:
    """The real llama.cpp rule rejects the directory the fixture points at."""
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        model,
        space,
        "--n-trials",
        "1",
        backend=GgufValidatingBackend(),
        client=client,
        engine="llamacpp",
    )

    assert code == 4
    assert "unsupported configuration" in err
    assert "not a GGUF file" in err
    assert "validating model compatibility" in out
    assert "pulling container image" not in out
    assert client.pull_calls == []


def test_a_gguf_checkpoint_proceeds_to_the_image_pull(tmp_path: Path) -> None:
    """A readable GGUF checkpoint with no configured quantization is accepted."""
    checkpoint = tmp_path / "model.gguf"
    checkpoint.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    space = _write_space(tmp_path / "space.yaml")
    client = FakePodmanClient()

    code, out, err = _invoke(
        str(checkpoint),
        space,
        "--n-trials",
        "1",
        backend=GgufValidatingBackend(),
        client=client,
        engine="llamacpp",
    )

    assert code == 0
    assert err == ""
    assert "validating model compatibility" in out
    assert f"pulling container image: {PLAN.container.image}" in out
    assert (PLAN.container.image, "missing") in client.pull_calls


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
    assert f"pulling container image: {PLAN.container.image}" in out
    assert (PLAN.container.image, "missing") in client.pull_calls


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


@pytest.mark.parametrize("engine", ["vllm", "llamacpp"])
def test_a_sweep_that_completes_no_trial_returns_the_empty_sweep_code(
    tmp_path: Path, model: str, engine: str
) -> None:
    """The exit-code contract is engine-independent; only the scoped flags differ."""
    space = _write_space(tmp_path / "space.yaml")
    backend = FakeBackend(PLAN, RESULT, [], parse_error=ExperimentError("engine never started"))

    code, _, err = _invoke(model, space, "--n-trials", "2", backend=backend, engine=engine)

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
    assert "trial 1/2  COMPLETE  prefill 1000.00 tps  decode 50.00 tps" in out
    assert "recommended trial: 0" in out
    assert "recommended rates: prefill 1000.00 tps  decode 50.00 tps" in out
    assert "trials run: 2" in out


def test_an_explicit_tui_request_without_a_terminal_is_a_usage_error(
    tmp_path: Path, model: str
) -> None:
    space = _write_space(tmp_path / "space.yaml")

    code, out, err = _invoke(model, space, "--n-trials", "1", "--output", "tui")

    assert code == 2
    assert "requires a terminal" in err
    assert out == ""


def test_tui_mode_prints_preparation_before_the_first_frame(tmp_path: Path, model: str) -> None:
    """The cold start is explained on screen, and no frame is drawn until a trial runs."""
    space = _write_space(tmp_path / "space.yaml")
    out, err = TtyStringIO(), StringIO()

    code = main(
        [model, str(space), "vllm", "--target", "cpu", "--n-trials", "1", "--output", "tui"],
        backend=_backend(),
        client=FakePodmanClient(),
        stdout=out,
        stderr=err,
    )

    output = out.getvalue()
    assert code == 0
    assert err.getvalue() == ""
    preparation = output.index("preparing experiments")
    assert output.index("preparing study") < preparation
    assert "prefill vs decode" not in output[:preparation]
    assert output.index("prefill vs decode") > preparation


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
        ["model", "space.yaml", "vllm", "--unknown"],
        backend=_backend(),
        client=FakePodmanClient(),
        stdout=out,
        stderr=err,
    )

    assert code == 2
    assert "unrecognized arguments" in err.getvalue()
