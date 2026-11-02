"""Unit tests for the optimization study and trial lifecycle.

``optimize``'s observable contract is the study it drives and the result it
returns: it samples points of a space, runs each through the inference layer,
records what ran on the trial, keeps going when a trial fails, and reports the
winning point reconstructed into a complete experiment. A sweep that completes
no trial raises ``OptimizationError`` carrying the study, and passing a storage
URL persists the study so a later call resumes it by name. An optional per-trial
callback is invoked once at the end of every trial — failed ones included — an
optional trial-start hook is invoked once at the beginning of every trial, and
a Ctrl-C during the sweep raises ``OptimizationInterrupted`` carrying the
best-so-far result, or the empty-sweep error when nothing completed.

These tests drive the real orchestration with the shared fake inference backend
and fake Podman client, so no daemon, GPU, or engine is involved. The base
experiment forces the CPU target, which short-circuits hardware detection
without a probe. They mirror the inference orchestration tests' use of the
shared doubles and the search-space tests' engine-neutral spaces.

The objective mode decides what each trial returns: the historical combined
throughput, the weighted-harmonic rate under explicit weights, or — for
``multi`` — the two phase rates on a study created with two maximize
directions. The library default is ``multi``, so the tests that exercise the
scalar lifecycle behavior select :attr:`ObjectiveMode.COMBINED` explicitly. A
mode that needs an absent phase rate fails the trial with the source-naming
error recorded on it. The result is shaped around the frontier: the scalar
modes yield one solution and the multi mode the whole Pareto front, with the
recommended point chosen by the weighted-harmonic score under the sweep's
weights. The interrupt path goes through the same projection.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import optuna
import pytest

from fakes import FakeBackend, FakePodmanClient
from llm_config_optimization.inference import (
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    LLMConfig,
    ResourceUsage,
    RunMetrics,
    ServerSpec,
)
from llm_config_optimization.optimization import (
    ObjectiveMode,
    ObjectiveWeights,
    OptimizationError,
    OptimizationInterrupted,
    OptimizationResult,
    optimize,
)
from llm_config_optimization.optimization._optuna import _project_result, make_objective
from llm_config_optimization.runtime import ContainerConfig, ContainerResult
from llm_config_optimization.search_space import Categorical, Dimension, ExperimentSpace, Interval

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
        ),
    ),
    failed_runs=0,
    target=HardwareTarget.CPU,
    image=PLAN.container.image,
    devices=(),
    raw_result=ContainerResult(exit_code=0, stdout=b"", stderr=b"", container_id="fake"),
)
"""The canned typed result the fake backend returns from parsing."""

PHASED_RESULT = ExperimentResult(
    runs=(
        RunMetrics(
            tokens_per_second=42.0,
            requests_per_second=4.0,
            elapsed_time=2.5,
            num_prompts=8,
            raw_json={},
            prefill_tokens_per_second=1000.0,
            decode_tokens_per_second=50.0,
        ),
    ),
    failed_runs=0,
    target=HardwareTarget.CPU,
    image=PLAN.container.image,
    devices=(),
    raw_result=RESULT.raw_result,
)
"""The canned result with both phase rates, for the phase-aware objective modes."""

MEASURED_RESULT = ExperimentResult(
    runs=(
        RunMetrics(
            tokens_per_second=40.0,
            requests_per_second=3.5,
            elapsed_time=2.0,
            num_prompts=8,
            raw_json={},
            resources=ResourceUsage(peak_vram_bytes=21_000_000_000, peak_ram_bytes=6_000),
        ),
        RunMetrics(
            tokens_per_second=50.0,
            requests_per_second=4.5,
            elapsed_time=3.0,
            num_prompts=8,
            raw_json={},
            resources=ResourceUsage(peak_vram_bytes=20_000_000_000),
        ),
    ),
    failed_runs=0,
    target=HardwareTarget.CPU,
    image=PLAN.container.image,
    devices=(),
    raw_result=RESULT.raw_result,
    lifetime_resources=ResourceUsage(peak_vram_bytes=24_000_000_000, peak_ram_bytes=7_000),
)
"""The canned result with a lifetime and per-run memory peak, partly measured."""

WEIGHTS = ObjectiveWeights(prefill=1024, decode=128)
"""A workload token mix of 1024 prompt and 128 generated tokens per request."""

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


def _phased(prefill: float, decode: float) -> ExperimentResult:
    """A one-run result carrying the given phase rates."""
    run = replace(
        PHASED_RESULT.runs[0],
        prefill_tokens_per_second=prefill,
        decode_tokens_per_second=decode,
    )
    return replace(PHASED_RESULT, runs=(run,))


class SequenceBackend(FakeBackend):
    """A backend returning a preset result for each successive trial.

    Every trial still runs through the real orchestration; only the parsed
    result differs, so a test can give each sampled point a distinct phase-rate
    pair and observe the frontier it produces. The last result repeats when the
    study runs more trials than results were given.
    """

    def __init__(self, results: Sequence[ExperimentResult]) -> None:
        super().__init__(PLAN, results[0], [])
        self.results = tuple(results)
        self.parses = 0

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        super().parse_result(outcome, spec, result)
        current = self.results[min(self.parses, len(self.results) - 1)]
        self.parses += 1
        return current


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
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        self.parses += 1
        if self.parses == 1:
            raise ExperimentError("simulated out-of-memory")
        return super().parse_result(outcome, spec, result)


class InterruptingBackend(FakeBackend):
    """A backend that completes ``completed`` trials and then interrupts.

    Raising ``KeyboardInterrupt`` from a parse is how Ctrl-C reaches the study
    mid-trial, so this drives the real interrupt path rather than mocking it.
    ``completed=1`` lets the first trial finish and interrupts the second, so
    there is a best point to report; ``completed=0`` interrupts the first trial,
    leaving the study with no completed trial.
    """

    def __init__(self, completed: int, result: ExperimentResult = RESULT) -> None:
        super().__init__(PLAN, result, [])
        self.completed = completed
        self.parses = 0

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        self.parses += 1
        if self.parses > self.completed:
            raise KeyboardInterrupt
        return super().parse_result(outcome, spec, result)


def test_optimize_returns_the_best_trial_reconstructed_into_a_config() -> None:
    result = optimize(
        SPACE,
        BASE,
        n_trials=3,
        seed=0,
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
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
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    (trial,) = result.study.trials
    recorded = trial.user_attrs["result"]

    assert recorded == {
        "tokens_per_second": 42.0,
        "requests_per_second": 4.0,
        "prefill_tokens_per_second": None,
        "decode_tokens_per_second": None,
        "elapsed_time": 2.5,
        "target": "cpu",
        "image": PLAN.container.image,
        "devices": [],
        "failed_runs": 0,
        "lifetime_peak_vram_bytes": None,
        "lifetime_peak_ram_bytes": None,
        "runs": [
            {
                "tokens_per_second": 42.0,
                "requests_per_second": 4.0,
                "prefill_tokens_per_second": None,
                "decode_tokens_per_second": None,
                "peak_vram_bytes": None,
                "peak_ram_bytes": None,
            }
        ],
    }
    assert json.loads(json.dumps(recorded)) == recorded


def test_a_trial_projection_keeps_the_per_run_spread_and_failure_count() -> None:
    """The projection explains a marginal pass, not only the mean it scored."""
    spread = ExperimentResult(
        runs=(
            RunMetrics(
                tokens_per_second=40.0,
                requests_per_second=3.5,
                elapsed_time=2.0,
                num_prompts=8,
                raw_json={"run": 0},
            ),
            RunMetrics(
                tokens_per_second=50.0,
                requests_per_second=4.5,
                elapsed_time=3.0,
                num_prompts=8,
                raw_json={"run": 1},
            ),
        ),
        failed_runs=1,
        target=HardwareTarget.CPU,
        image=PLAN.container.image,
        devices=(),
        raw_result=RESULT.raw_result,
    )
    backend = FakeBackend(PLAN, spread, [])

    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        backend=backend,
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    (trial,) = result.study.trials
    recorded = trial.user_attrs["result"]

    assert recorded["failed_runs"] == 1
    assert recorded["runs"] == [
        {
            "tokens_per_second": 40.0,
            "requests_per_second": 3.5,
            "prefill_tokens_per_second": None,
            "decode_tokens_per_second": None,
            "peak_vram_bytes": None,
            "peak_ram_bytes": None,
        },
        {
            "tokens_per_second": 50.0,
            "requests_per_second": 4.5,
            "prefill_tokens_per_second": None,
            "decode_tokens_per_second": None,
            "peak_vram_bytes": None,
            "peak_ram_bytes": None,
        },
    ]
    assert recorded["tokens_per_second"] == 45.0
    assert trial.value == result.best_value == 45.0
    assert json.loads(json.dumps(recorded)) == recorded


def test_a_trial_projection_records_phase_rates_per_run_and_aggregated() -> None:
    """The projection says which phase a configuration helped, and how steadily."""
    phased = ExperimentResult(
        runs=(
            RunMetrics(
                tokens_per_second=40.0,
                requests_per_second=3.5,
                elapsed_time=2.0,
                num_prompts=8,
                raw_json={"run": 0},
                prefill_tokens_per_second=1000.0,
                decode_tokens_per_second=50.0,
            ),
            RunMetrics(
                tokens_per_second=60.0,
                requests_per_second=4.5,
                elapsed_time=3.0,
                num_prompts=8,
                raw_json={"run": 1},
                prefill_tokens_per_second=2000.0,
            ),
        ),
        failed_runs=0,
        target=HardwareTarget.CPU,
        image=PLAN.container.image,
        devices=(),
        raw_result=RESULT.raw_result,
    )
    backend = FakeBackend(PLAN, phased, [])

    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        backend=backend,
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    (trial,) = result.study.trials
    recorded = trial.user_attrs["result"]

    assert recorded["runs"] == [
        {
            "tokens_per_second": 40.0,
            "requests_per_second": 3.5,
            "prefill_tokens_per_second": 1000.0,
            "decode_tokens_per_second": 50.0,
            "peak_vram_bytes": None,
            "peak_ram_bytes": None,
        },
        {
            "tokens_per_second": 60.0,
            "requests_per_second": 4.5,
            "prefill_tokens_per_second": 2000.0,
            "decode_tokens_per_second": None,
            "peak_vram_bytes": None,
            "peak_ram_bytes": None,
        },
    ]
    assert recorded["prefill_tokens_per_second"] == 1500.0
    assert recorded["decode_tokens_per_second"] == 50.0
    assert json.loads(json.dumps(recorded)) == recorded


def test_a_trial_projection_records_the_memory_peaks_as_plain_integers() -> None:
    """A persisted study keeps each configuration's footprint without scoring it."""
    recorded = _project_result(MEASURED_RESULT)

    assert recorded["lifetime_peak_vram_bytes"] == 24_000_000_000
    assert recorded["lifetime_peak_ram_bytes"] == 7_000
    assert recorded["runs"][0]["peak_vram_bytes"] == 21_000_000_000
    assert recorded["runs"][0]["peak_ram_bytes"] == 6_000
    assert recorded["runs"][1]["peak_vram_bytes"] == 20_000_000_000
    assert recorded["runs"][1]["peak_ram_bytes"] is None
    assert json.loads(json.dumps(recorded)) == recorded


def test_a_failed_trial_is_marked_failed_and_the_study_continues() -> None:
    backend = FailsOnceBackend(PLAN, RESULT)

    result = optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        backend=backend,
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert [trial.state for trial in result.study.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert result.study.trials[0].user_attrs["error"] == "simulated out-of-memory"
    assert result.trial_count == 2


def test_no_completed_trial_raises_optimization_error_with_the_study() -> None:
    backend = _backend(parse_error=ExperimentError("engine never started"))

    with pytest.raises(OptimizationError) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=2,
            seed=0,
            backend=backend,
            client=FakePodmanClient(),
            mode=ObjectiveMode.COMBINED,
        )

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
        mode=ObjectiveMode.COMBINED,
    )
    second = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert first.trial_count == 2
    assert second.trial_count == 1


def test_a_scalar_sweep_reports_the_single_winner_as_its_only_solution() -> None:
    """The frontier degenerates to the best trial in the scalar modes."""
    result = optimize(
        SPACE,
        BASE,
        n_trials=3,
        seed=0,
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert len(result.solutions) == 1
    (solution,) = result.solutions
    assert result.recommended is solution
    assert solution.trial_number == result.study.best_trial.number
    assert solution.value == result.study.best_trial.value
    assert solution.config.llm.dtype is DType(result.study.best_trial.params["dtype"])


def test_the_legacy_aliases_point_at_the_recommended_front_member() -> None:
    backend = SequenceBackend(
        (
            _phased(prefill=1000.0, decode=50.0),
            _phased(prefill=500.0, decode=200.0),
        )
    )

    result = optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        mode=ObjectiveMode.MULTI,
        weights=WEIGHTS,
        backend=backend,
        client=FakePodmanClient(),
    )

    assert result.best_config is result.recommended.config
    assert result.best_value == result.recommended.value
    assert result.best_trial_number == result.recommended.trial_number


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
        mode=ObjectiveMode.COMBINED,
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
        mode=ObjectiveMode.COMBINED,
    )

    assert first.trial_count == 2
    assert second.trial_count == 3
    assert second.study.study_name == "sweep"
    assert (tmp_path / "study.db").exists()


def test_a_new_persisted_study_records_its_mode_and_metric_schema(tmp_path: Path) -> None:
    """A study says which objective and phase-metric definitions scored it."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        storage=storage,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert result.study.user_attrs == {"objective_mode": "combined", "metric_schema_version": 1}


@pytest.mark.parametrize(
    ("created", "resumed"),
    [
        (ObjectiveMode.COMBINED, ObjectiveMode.MULTI),
        (ObjectiveMode.MULTI, ObjectiveMode.COMBINED),
        (ObjectiveMode.WEIGHTED, ObjectiveMode.COMBINED),
    ],
    ids=["combined-to-multi", "multi-to-combined", "weighted-to-combined"],
)
def test_resuming_a_study_under_a_different_objective_mode_is_refused(
    tmp_path: Path, created: ObjectiveMode, resumed: ObjectiveMode
) -> None:
    """A resume never appends trials scored by another objective.

    The refusal happens before the first trial of the resumed run, so the
    existing study keeps exactly the trials it already had, and the message
    names the recorded mode and points at a different study name.
    """
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    first = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        mode=created,
        weights=WEIGHTS,
        storage=storage,
        study_name="sweep",
        backend=FakeBackend(PLAN, PHASED_RESULT, []),
        client=FakePodmanClient(),
    )

    with pytest.raises(OptimizationError, match=created.value) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            seed=0,
            mode=resumed,
            weights=WEIGHTS,
            storage=storage,
            study_name="sweep",
            backend=_backend(),
            client=FakePodmanClient(),
        )

    assert first.trial_count == 1
    error = excinfo.value
    assert error.study is not None
    assert len(error.study.trials) == 1
    assert resumed.value in str(error)
    assert "different study name" in str(error)


def _legacy_study(storage: str) -> optuna.Study:
    """A study as the adapter persisted it before provenance existed: scored, but unmarked.

    The real objective drives the run, so the study holds a reconstructed point
    the resume can rebuild; the mode and schema attributes are simply never set,
    which is what a study persisted by an older version looks like.
    """
    study = optuna.create_study(storage=storage, study_name="sweep", direction="maximize")
    study.optimize(
        make_objective(
            SPACE, BASE, mode=ObjectiveMode.COMBINED, backend=_backend(), client=FakePodmanClient()
        ),
        n_trials=1,
    )
    return study


def test_an_unmarked_study_resumes_as_legacy_combined(tmp_path: Path) -> None:
    """A study persisted before the attribute existed keeps resuming as combined."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    _legacy_study(storage)

    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        storage=storage,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert result.trial_count == 2
    assert result.study.study_name == "sweep"


def test_an_empty_unmarked_study_adopts_the_mode_that_starts_it(tmp_path: Path) -> None:
    """A study interrupted before its first trial holds no score to compare."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    optuna.create_study(storage=storage, study_name="sweep", direction="maximize")

    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        storage=storage,
        study_name="sweep",
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    assert result.study.user_attrs["objective_mode"] == "combined"
    assert result.trial_count == 1


def test_an_unmarked_study_refuses_a_non_combined_resume(tmp_path: Path) -> None:
    """The historical scores are combined numbers, so another mode cannot join them."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    _legacy_study(storage)

    with pytest.raises(OptimizationError, match="combined") as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            seed=0,
            mode=ObjectiveMode.WEIGHTED,
            weights=WEIGHTS,
            storage=storage,
            study_name="sweep",
            backend=_backend(),
            client=FakePodmanClient(),
        )

    error = excinfo.value
    assert error.study is not None
    assert len(error.study.trials) == 1
    assert "sweep" in str(error)


def test_an_empty_study_left_by_another_mode_is_refused(tmp_path: Path) -> None:
    """Directions cannot change after creation, so a two-direction study is not combined."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    optuna.create_study(storage=storage, study_name="sweep", directions=["maximize", "maximize"])

    with pytest.raises(OptimizationError, match="direction") as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            seed=0,
            storage=storage,
            study_name="sweep",
            backend=_backend(),
            client=FakePodmanClient(),
            mode=ObjectiveMode.COMBINED,
        )

    assert "different study name" in str(excinfo.value)


def test_a_seed_makes_a_sweep_reproducible() -> None:
    first = optimize(
        SPACE,
        BASE,
        n_trials=5,
        seed=123,
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )
    second = optimize(
        SPACE,
        BASE,
        n_trials=5,
        seed=123,
        backend=_backend(),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
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

    optimize(
        SPACE,
        BASE,
        n_trials=1,
        backend=_backend(),
        client=FakePodmanClient(),
        **controls,
        mode=ObjectiveMode.COMBINED,
    )

    assert recorded["n_jobs"] == expected


def test_the_callback_fires_once_per_completed_trial() -> None:
    calls: list[tuple[optuna.Study, optuna.trial.FrozenTrial]] = []

    result = optimize(
        SPACE,
        BASE,
        n_trials=3,
        seed=0,
        backend=_backend(),
        client=FakePodmanClient(),
        callback=lambda study, trial: calls.append((study, trial)),
        mode=ObjectiveMode.COMBINED,
    )

    assert [trial.number for _, trial in calls] == [0, 1, 2]
    assert all(study is result.study for study, _ in calls)
    assert all(trial.state is optuna.trial.TrialState.COMPLETE for _, trial in calls)


def test_the_callback_fires_for_failed_trials_too() -> None:
    calls: list[optuna.trial.FrozenTrial] = []

    optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        backend=FailsOnceBackend(PLAN, RESULT),
        client=FakePodmanClient(),
        callback=lambda study, trial: calls.append(trial),
        mode=ObjectiveMode.COMBINED,
    )

    assert [trial.state for trial in calls] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]


def test_the_trial_start_hook_fires_once_per_trial() -> None:
    starts: list[None] = []

    result = optimize(
        SPACE,
        BASE,
        n_trials=3,
        seed=0,
        backend=_backend(),
        client=FakePodmanClient(),
        on_trial_start=lambda: starts.append(None),
        mode=ObjectiveMode.COMBINED,
    )

    assert len(starts) == len(result.study.trials) == 3


def test_the_trial_start_hook_fires_for_failed_trials_too() -> None:
    starts: list[None] = []

    result = optimize(
        SPACE,
        BASE,
        n_trials=2,
        seed=0,
        backend=FailsOnceBackend(PLAN, RESULT),
        client=FakePodmanClient(),
        on_trial_start=lambda: starts.append(None),
        mode=ObjectiveMode.COMBINED,
    )

    assert [trial.state for trial in result.study.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert len(starts) == 2


def test_an_interrupt_raises_the_dedicated_exception_with_the_best_result() -> None:
    with pytest.raises(OptimizationInterrupted) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=2,
            seed=0,
            backend=InterruptingBackend(completed=1),
            client=FakePodmanClient(),
            mode=ObjectiveMode.COMBINED,
        )

    error = excinfo.value
    assert isinstance(error, OptimizationError)
    assert error.result.best_value == RESULT.tokens_per_second
    assert error.result.best_trial_number == 0
    assert error.result.trial_count == 2
    assert error.study is error.result.study
    assert error.result.best_config.container is BASE.container
    assert error.result.best_config.benchmark is BASE.benchmark
    assert "interrupted" in str(error)


def test_an_interrupted_multi_sweep_reports_the_front_found_so_far() -> None:
    """The interrupt path projects the same front shape as a finished sweep."""
    with pytest.raises(OptimizationInterrupted) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=3,
            seed=0,
            mode=ObjectiveMode.MULTI,
            weights=WEIGHTS,
            backend=InterruptingBackend(completed=2, result=PHASED_RESULT),
            client=FakePodmanClient(),
        )

    result = excinfo.value.result
    assert [solution.trial_number for solution in result.solutions] == [0, 1]
    assert all(solution.value == (1000.0, 50.0) for solution in result.solutions)
    assert result.recommended.trial_number == 0
    assert result.trial_count == 3


def test_an_interrupted_sweep_with_no_completed_trial_raises_the_empty_sweep_error() -> None:
    with pytest.raises(OptimizationError) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=2,
            seed=0,
            backend=InterruptingBackend(completed=0),
            client=FakePodmanClient(),
            mode=ObjectiveMode.COMBINED,
        )

    error = excinfo.value
    assert not isinstance(error, OptimizationInterrupted)
    assert error.study is not None
    assert all(trial.state is optuna.trial.TrialState.FAIL for trial in error.study.trials)


def test_combined_mode_keeps_the_combined_throughput_as_the_scalar_objective() -> None:
    """The mode is explicit, yet still ignores the phase rates it does not need."""
    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        backend=FakeBackend(PLAN, PHASED_RESULT, []),
        client=FakePodmanClient(),
        mode=ObjectiveMode.COMBINED,
    )

    (trial,) = result.study.trials
    assert trial.value == 42.0
    assert result.best_value == trial.value


def test_weighted_mode_scores_the_weighted_harmonic_rate() -> None:
    result = optimize(
        SPACE,
        BASE,
        n_trials=1,
        seed=0,
        mode=ObjectiveMode.WEIGHTED,
        weights=WEIGHTS,
        backend=FakeBackend(PLAN, PHASED_RESULT, []),
        client=FakePodmanClient(),
    )

    (trial,) = result.study.trials
    assert trial.state is optuna.trial.TrialState.COMPLETE
    # (1024 + 128) / (1024/1000 + 128/50) for the phased result.
    assert trial.value == pytest.approx(1152.0 / 3.584)
    assert result.best_value == trial.value


def test_weighted_mode_requires_weights_before_any_trial() -> None:
    """A missing token mix is a caller error, rejected before any engine work."""
    with pytest.raises(ValueError, match="weights"):
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            mode=ObjectiveMode.WEIGHTED,
            backend=_backend(),
            client=FakePodmanClient(),
        )


def test_multi_mode_requires_weights_before_any_trial() -> None:
    """The recommended front point is scored by them, so they cannot be omitted."""
    with pytest.raises(ValueError, match="weights"):
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            mode=ObjectiveMode.MULTI,
            backend=_backend(),
            client=FakePodmanClient(),
        )


def test_the_default_mode_is_multi_and_requires_weights() -> None:
    """The library default flips with the CLI; no token mix is silently invented."""
    with pytest.raises(ValueError, match="multi"):
        optimize(SPACE, BASE, n_trials=1, backend=_backend(), client=FakePodmanClient())


def test_make_objective_defaults_to_the_multi_rate_pair() -> None:
    """The builder's default is the phase-rate pair of the multi objective."""
    objective = make_objective(
        SPACE, BASE, backend=FakeBackend(PLAN, PHASED_RESULT, []), client=FakePodmanClient()
    )
    study = optuna.create_study(directions=["maximize", "maximize"])

    study.optimize(objective, n_trials=1)

    assert study.trials[0].values == [1000.0, 50.0]


def test_multi_mode_drives_a_two_direction_study_with_tuple_objectives() -> None:
    objective = make_objective(
        SPACE,
        BASE,
        mode=ObjectiveMode.MULTI,
        backend=FakeBackend(PLAN, PHASED_RESULT, []),
        client=FakePodmanClient(),
    )
    study = optuna.create_study(directions=["maximize", "maximize"])

    study.optimize(objective, n_trials=2)

    assert list(study.directions) == [optuna.study.StudyDirection.MAXIMIZE] * 2
    assert [trial.values for trial in study.trials] == [[1000.0, 50.0], [1000.0, 50.0]]


def test_a_multi_sweep_returns_the_pareto_front_with_a_recommended_point() -> None:
    """Dominated points drop out; the pick maximizes the weighted-harmonic score."""
    backend = SequenceBackend(
        (
            _phased(prefill=1000.0, decode=50.0),
            _phased(prefill=500.0, decode=200.0),
            _phased(prefill=400.0, decode=40.0),
            _phased(prefill=200.0, decode=500.0),
        )
    )

    result = optimize(
        SPACE,
        BASE,
        n_trials=4,
        seed=0,
        mode=ObjectiveMode.MULTI,
        weights=WEIGHTS,
        backend=backend,
        client=FakePodmanClient(),
    )

    # Trial 2's (400, 40) is dominated by trial 0's (1000, 50), so it drops out.
    assert [solution.trial_number for solution in result.solutions] == [0, 1, 3]
    assert [solution.value for solution in result.solutions] == [
        (1000.0, 50.0),
        (500.0, 200.0),
        (200.0, 500.0),
    ]
    assert [trial.number for trial in result.study.best_trials] == [0, 1, 3]
    # Under the 1024/128 token mix, 1152 / (1024/500 + 128/200) beats both the
    # prefill-mostly first front member and the decode-mostly last one.
    assert result.recommended.trial_number == 1
    assert result.recommended.value == (500.0, 200.0)
    assert result.recommended in result.solutions


@pytest.mark.parametrize(
    "mode", [ObjectiveMode.MULTI, ObjectiveMode.WEIGHTED], ids=["multi", "weighted"]
)
def test_a_mode_needing_an_absent_rate_fails_the_trial_with_a_source_naming_error(
    mode: ObjectiveMode,
) -> None:
    """The failure names what the engine did not report, not a generic score error."""
    with pytest.raises(OptimizationError) as excinfo:
        optimize(
            SPACE,
            BASE,
            n_trials=1,
            seed=0,
            mode=mode,
            weights=WEIGHTS,
            backend=_backend(),
            client=FakePodmanClient(),
        )

    study = excinfo.value.study
    assert study is not None
    (trial,) = study.trials
    assert trial.state is optuna.trial.TrialState.FAIL
    message = trial.user_attrs["error"]
    assert "prefill" in message
    assert "/metrics" in message
    assert "timings" in message
