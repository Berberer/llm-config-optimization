"""Unit tests for the optimization study and trial lifecycle.

``optimize``'s observable contract is the study it drives and the result it
returns: it samples points of a space, runs each through the inference layer,
records what ran on the trial, keeps going when a trial fails, and reports the
winning point reconstructed into a complete experiment. A sweep that completes
no trial raises ``OptimizationError`` carrying the study, and passing a storage
URL persists the study so a later call resumes it by name.

These tests drive the real orchestration with the shared fake inference backend
and fake Podman client, so no daemon, GPU, or engine is involved. The base
experiment forces the CPU target, which short-circuits hardware detection
without a probe. They mirror the inference orchestration tests' use of the
shared doubles and the search-space tests' engine-neutral spaces.
"""

import json
from pathlib import Path

import optuna
import pytest

from fakes import FakeBackend, FakePodmanClient
from llm_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    LLMConfig,
)
from llm_optimization.optimization import OptimizationError, OptimizationResult, optimize
from llm_optimization.runtime import ContainerConfig, ContainerResult
from llm_optimization.search_space import Categorical, Dimension, ExperimentSpace, Interval

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
"""The canned typed result the fake backend returns from parsing."""

BASE = ExperimentConfig(
    container=ContainerRuntimeConfig(target=HardwareTarget.CPU),
    benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B", num_prompts=8),
    llm=LLMConfig(seed=11, gpu_memory_utilization=0.5),
)
"""A base experiment whose container target short-circuits hardware detection."""

SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16))),
        Dimension("max_model_len", Interval(1024, 4096)),
    )
)
"""A small space with a categorical and an interval dimension to sample."""


def _backend(*, parse_error: Exception | None = None) -> FakeBackend:
    """Build a fake backend returning the canned plan and result."""
    return FakeBackend(PLAN, RESULT, [], parse_error=parse_error)


class FailsOnceBackend(FakeBackend):
    """A backend that fails its first parse and succeeds afterwards.

    It proves the study continues past a failed trial rather than aborting the
    sweep, without depending on which point the sampler happens to pick.
    """

    def __init__(self, plan_config: ContainerConfig, result: ExperimentResult) -> None:
        super().__init__(plan_config, result, [])
        self.parses = 0

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        self.parses += 1
        if self.parses == 1:
            raise ExperimentError("simulated out-of-memory")
        return super().parse_result(result, plan, output_dir)


def test_optimize_returns_the_best_trial_reconstructed_into_a_config() -> None:
    result = optimize(
        SPACE, BASE, n_trials=3, seed=0, backend=_backend(), client=FakePodmanClient()
    )

    best = result.study.best_trial

    assert isinstance(result, OptimizationResult)
    assert result.best_trial_number == best.number
    assert result.best_value == best.value == RESULT.tokens_per_second
    assert result.trial_count == 3
    assert result.best_config.container is BASE.container
    assert result.best_config.benchmark is BASE.benchmark
    assert result.best_config.llm.dtype is DType(best.params["dtype"])
    assert result.best_config.llm.max_model_len == best.params["max_model_len"]
    assert result.best_config.llm.seed == BASE.llm.seed
    assert result.best_config.llm.gpu_memory_utilization == BASE.llm.gpu_memory_utilization


def test_a_successful_trial_records_a_serializable_result_projection() -> None:
    result = optimize(
        SPACE, BASE, n_trials=1, seed=0, backend=_backend(), client=FakePodmanClient()
    )

    (trial,) = result.study.trials
    recorded = trial.user_attrs["result"]

    assert recorded == {
        "tokens_per_second": 42.0,
        "requests_per_second": 4.0,
        "elapsed_time": 2.5,
        "target": "cpu",
        "image": PLAN.image,
        "devices": [],
    }
    assert json.loads(json.dumps(recorded)) == recorded


def test_a_failed_trial_is_marked_failed_and_the_study_continues() -> None:
    backend = FailsOnceBackend(PLAN, RESULT)

    result = optimize(SPACE, BASE, n_trials=2, seed=0, backend=backend, client=FakePodmanClient())

    assert [trial.state for trial in result.study.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert result.study.trials[0].user_attrs["error"] == "simulated out-of-memory"
    assert result.trial_count == 2


def test_no_completed_trial_raises_optimization_error_with_the_study() -> None:
    backend = _backend(parse_error=ExperimentError("engine never started"))

    with pytest.raises(OptimizationError) as excinfo:
        optimize(SPACE, BASE, n_trials=2, seed=0, backend=backend, client=FakePodmanClient())

    error = excinfo.value
    assert error.study is not None
    assert len(error.study.trials) == 2
    assert all(trial.state is optuna.trial.TrialState.FAIL for trial in error.study.trials)


def test_without_storage_each_sweep_is_independent() -> None:
    """The in-memory default persists nothing, so a repeated name starts fresh."""
    first = optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
    )
    second = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
    )

    assert first.trial_count == 2
    assert second.trial_count == 1


def test_a_persisted_study_resumes_by_name(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    first = optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        storage=storage,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
    )
    second = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        storage=storage,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
    )

    assert first.trial_count == 2
    assert second.trial_count == 3
    assert second.study.study_name == "sweep"
    assert (tmp_path / "study.db").exists()


def test_a_seed_makes_a_sweep_reproducible() -> None:
    first = optimize(
        SPACE, BASE, n_trials=5, seed=123, backend=_backend(), client=FakePodmanClient()
    )
    second = optimize(
        SPACE, BASE, n_trials=5, seed=123, backend=_backend(), client=FakePodmanClient()
    )

    assert [trial.params for trial in first.study.trials] == [
        trial.params for trial in second.study.trials
    ]


@pytest.mark.parametrize(("passed", "expected"), [(None, 1), (4, 4)], ids=["default", "explicit"])
def test_n_jobs_defaults_to_one_and_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, passed: int | None, expected: int
) -> None:
    recorded: dict[str, object] = {}
    real_optimize = optuna.Study.optimize

    def recording_optimize(self: optuna.Study, *args: object, **kwargs: object) -> object:
        recorded.update(kwargs)
        return real_optimize(self, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", recording_optimize)
    controls = {} if passed is None else {"n_jobs": passed}

    optimize(SPACE, BASE, n_trials=1, backend=_backend(), client=FakePodmanClient(), **controls)

    assert recorded["n_jobs"] == expected
