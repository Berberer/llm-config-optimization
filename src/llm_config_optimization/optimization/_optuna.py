"""Optuna backend for the optimization layer.

This module is the only place that knows Optuna's study and trial API. It maps
each engine-neutral :class:`~llm_config_optimization.search_space.types.Spec` kind onto
an Optuna suggestion, rebuilds an
:class:`~llm_config_optimization.inference.types.ExperimentConfig` from the sampled
parameters, and drives a study to find the best configuration.

The mapping is deliberately faithful to the search-space semantics: a
:class:`~llm_config_optimization.search_space.types.Fixed` dimension is submitted as a
one-choice categorical so its value is recorded on the trial, a
:class:`~llm_config_optimization.search_space.types.Categorical` dimension samples only
its declared values in declared order, which for a numeric parameter is the set
of integers it declares, an
:class:`~llm_config_optimization.search_space.types.Interval` dimension samples
integers from its inclusive bounds with the default step, and a
:class:`~llm_config_optimization.search_space.types.Default` dimension produces no
suggestion at all, so deferral to the engine survives the sweep.

Categorical choices are submitted as their string values because a persisted
study is reloaded with plain strings; :func:`reconstruct_llm` converts a sampled
string back to the field's engine-neutral enum class, so an in-memory study and
a resumed one produce identical configurations.

The value a trial contributes is chosen by the engine-neutral
:class:`~llm_config_optimization.optimization.types.ObjectiveMode`: combined
throughput, the weighted-harmonic effective rate under
:class:`~llm_config_optimization.optimization.types.ObjectiveWeights`, or — for
``MULTI`` — the prefill and decode rates as two independent objectives, which
creates a study with two maximize directions. A mode that needs a phase rate a
result does not carry fails the trial through the same
:class:`~llm_config_optimization.inference.types.ExperimentError` catch as a
failed run, so the sweep continues.

A persisted study records the objective mode and phase-metric schema version
that produced its scores, and a resume under a different mode is refused with
an :class:`~llm_config_optimization.optimization.types.OptimizationError`
naming the recorded mode, so a sweep cannot silently append incomparable trials
to an existing study. A study persisted before those attributes existed is
treated as legacy combined.

The entry point accepts an optional per-trial callback, forwarded to Optuna's own
``callbacks``, so a caller can observe each trial without reimplementing the
study loop. A Ctrl-C during the sweep is turned into
:class:`~llm_config_optimization.optimization.types.OptimizationInterrupted` carrying
the front found so far, so the work done before the interrupt is not lost.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING, cast

import optuna

from ..inference import BenchmarkClient, run_experiment
from ..inference.types import (
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    InferenceBackend,
    LLMConfig,
)
from ..search_space.registry import TUNABLE_PARAMETERS, UnionKind, expected_kind
from ..search_space.types import (
    Categorical,
    Default,
    ExperimentSpace,
    Fixed,
    ParameterValue,
)
from .types import (
    ObjectiveMode,
    ObjectiveWeights,
    OptimizationError,
    OptimizationInterrupted,
    OptimizationResult,
    OptimizationSolution,
    objective_value,
    weighted_harmonic_rate,
)

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = [
    "make_objective",
    "optimize",
    "reconstruct_config",
    "reconstruct_llm",
    "suggest_parameters",
]

_RESULT_ATTRIBUTE = "result"
_ERROR_ATTRIBUTE = "error"
_MODE_ATTRIBUTE = "objective_mode"
_METRIC_SCHEMA_ATTRIBUTE = "metric_schema_version"
_METRIC_SCHEMA_VERSION = 1
# The version of the phase-rate metric definitions a study's scores were
# measured under. A future change to how either backend defines prefill or
# decode tokens per second must bump this so a resumed study cannot silently
# compare numbers that no longer mean the same thing.


def suggest_parameters(trial: optuna.Trial, space: ExperimentSpace) -> dict[str, object]:
    """Sample one point of ``space`` through ``trial``'s suggestion API.

    Each dimension is submitted according to its spec kind: a ``Fixed`` or
    ``Categorical`` dimension becomes a categorical suggestion over the declared
    values, and an ``Interval`` dimension becomes an integer suggestion over its
    inclusive bounds. A numeric ``Categorical`` dimension's declared values are
    integers, so Optuna treats them as unordered choices and samples only those
    integers; use an ``Interval`` when the values should be modeled as an
    ordered range. A ``Default`` dimension is skipped entirely, so its
    parameter is absent from ``trial.params`` and reconstruction leaves the
    field unset.

    Categorical values are submitted as their string values rather than enum
    members, because a study persisted and reloaded returns plain strings;
    :func:`reconstruct_llm` converts them back, so the two paths cannot diverge.
    The returned mapping is keyed by the space's parameter names.
    """
    parameters: dict[str, object] = {}
    for dimension in space.dimensions:
        spec = dimension.spec
        if isinstance(spec, Default):
            continue
        if isinstance(spec, Fixed):
            parameters[dimension.name] = trial.suggest_categorical(
                dimension.name, [_choice_value(spec.value)]
            )
        elif isinstance(spec, Categorical):
            parameters[dimension.name] = trial.suggest_categorical(
                dimension.name, [_choice_value(value) for value in spec.values]
            )
        else:
            parameters[dimension.name] = trial.suggest_int(
                dimension.name, int(spec.minimum), int(spec.maximum)
            )
    return parameters


def reconstruct_llm(
    space: ExperimentSpace,
    base: LLMConfig,
    parameters: Mapping[str, object],
) -> LLMConfig:
    """Rebuild an LLM config from a sampled parameter mapping.

    This is the single reconstruction step shared by a freshly sampled trial and
    the winning point read back from the study, so the two cannot diverge. Every
    tunable field is overwritten from the space: a dimension's sampled value is
    coerced back to the field's engine-neutral kind, while a ``Default``
    dimension sets its field to ``None``. Fields the space does not cover — the
    non-tunable ``seed`` and ``gpu_memory_utilization`` — are inherited from
    ``base`` unchanged.

    ``parameters`` is keyed by the space's parameter names, matching both
    :func:`suggest_parameters` and an Optuna trial's own ``params``.
    """
    updates: dict[str, object] = {}
    for dimension in space.dimensions:
        field = TUNABLE_PARAMETERS[dimension.name]
        if isinstance(dimension.spec, Default):
            updates[field] = None
        else:
            updates[field] = _coerce_value(field, parameters[dimension.name])
    return replace(base, **updates)


def reconstruct_config(
    space: ExperimentSpace,
    base: ExperimentConfig,
    parameters: Mapping[str, object],
) -> ExperimentConfig:
    """Rebuild a full experiment config from a sampled parameter mapping.

    The container and benchmark domains are reused from ``base`` unchanged and
    only the ``llm`` domain is rebuilt through :func:`reconstruct_llm`, so the
    fixed parts of a sweep stay fixed while the tunable serving parameters come
    from the space.
    """
    return replace(base, llm=reconstruct_llm(space, base.llm, parameters))


def make_objective(
    space: ExperimentSpace,
    base: ExperimentConfig,
    *,
    mode: ObjectiveMode = ObjectiveMode.MULTI,
    weights: ObjectiveWeights | None = None,
    backend: InferenceBackend | None = None,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
    on_trial_start: Callable[[], None] | None = None,
) -> Callable[[optuna.Trial], float | tuple[float, float]]:
    """Build the Optuna objective that evaluates one point of ``space``.

    The returned callable samples the trial's point, rebuilds the experiment
    from ``base``, and runs it through
    :func:`~llm_config_optimization.inference.run_experiment` with the injected backend
    and container client. ``backend``, ``client``, and ``benchmark_client`` may
    be ``None``, in which case the orchestrator's own defaults apply.

    ``mode`` selects the value the objective reports and defaults to
    :attr:`~llm_config_optimization.optimization.types.ObjectiveMode.MULTI`,
    which returns the ``(prefill, decode)`` rate pair Optuna accepts only on a
    study created with two maximize directions. ``WEIGHTED`` returns the
    weighted-harmonic effective rate under ``weights``, and ``COMBINED`` the
    historical combined token throughput. ``weights`` is required by — and
    only meaningful for — the weighted mode, and a missing pair is rejected
    here rather than once per trial; the multi mode consumes none here, but
    :func:`optimize` must receive the same pair to pick the recommended front
    point.

    ``on_trial_start``, when given, is invoked once at the start of every trial
    — before the experiment runs — so a caller can observe an in-flight trial
    that the completion callback cannot. It takes no arguments, because the
    optimizer's trial API is not exposed through it; a reporter derives what it
    needs from its own state.

    A successful trial records a JSON-serializable projection of its result
    under the ``result`` user attribute, so a persisted study explains what
    actually ran as well as which point won. An
    :class:`~llm_config_optimization.inference.types.ExperimentError` — including
    the one an objective needing an absent phase rate raises — records its
    message under the ``error`` user attribute and is then re-raised, so the
    trial is marked failed rather than silently returning a bogus objective;
    the caller of ``study.optimize`` must pass ``catch=(ExperimentError,)`` for
    the study to continue to the next trial.
    """
    if mode is ObjectiveMode.WEIGHTED and weights is None:
        raise ValueError("the weighted objective requires weights")

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        if on_trial_start is not None:
            on_trial_start()
        parameters = suggest_parameters(trial, space)
        config = reconstruct_config(space, base, parameters)
        try:
            result = run_experiment(
                config, backend, client=client, benchmark_client=benchmark_client
            )
            trial.set_user_attr(_RESULT_ATTRIBUTE, _project_result(result))
            return objective_value(mode, result, weights)
        except ExperimentError as error:
            trial.set_user_attr(_ERROR_ATTRIBUTE, str(error))
            raise

    return objective


def optimize(
    space: ExperimentSpace,
    base: ExperimentConfig,
    *,
    n_trials: int | None = None,
    timeout: float | None = None,
    seed: int | None = None,
    mode: ObjectiveMode = ObjectiveMode.MULTI,
    weights: ObjectiveWeights | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    backend: InferenceBackend | None = None,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
    n_jobs: int = 1,
    callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None = None,
    on_trial_start: Callable[[], None] | None = None,
) -> OptimizationResult:
    """Search ``space`` for the ``base`` experiment with the best objective value.

    ``mode`` selects what each trial maximizes and defaults to
    :attr:`~llm_config_optimization.optimization.types.ObjectiveMode.MULTI`,
    which submits the prefill and decode rates as two independent objectives,
    creating the study with two maximize directions and reporting the Pareto
    front.
    :attr:`~llm_config_optimization.optimization.types.ObjectiveMode.WEIGHTED`
    maximizes the weighted-harmonic effective rate under ``weights``, and
    :attr:`~llm_config_optimization.optimization.types.ObjectiveMode.COMBINED`
    keeps the historical combined token throughput, the only mode an engine
    that reports no phase data supports. The multi and weighted modes require
    ``weights``: the weighted mode scores the result with them, and the multi
    mode uses them to pick the recommended front point. A mode that needs a
    phase rate the engine did not report fails the trial with the missing
    source recorded on it, so the sweep continues like any other failed trial.

    A study is created for the caller with Optuna's default TPE sampler, seeded
    by ``seed`` (or left non-deterministic when ``seed`` is ``None``) and always
    maximizing the objective. ``n_trials`` bounds the sweep and ``timeout``
    bounds it by wall-clock seconds; either may be ``None``, and ``None`` for
    both runs the sweep until it is interrupted. Passing ``storage`` persists
    the study to that URL
    and enables resume-on-existing, so a study of the same ``study_name`` is
    continued rather than recreated; ``n_trials`` is then a budget for *new*
    trials, not a cap on the study. A persisted study records the objective
    mode and phase-metric schema version that produced its scores, so resuming
    it under a different mode raises :class:`OptimizationError` naming the
    recorded mode instead of silently appending incomparable trials; a study
    without the attribute is treated as legacy combined. With no ``storage``
    the study lives in memory and no file is touched.

    Each trial samples a point of ``space``, rebuilds the full experiment from
    ``base``, and runs it through
    :func:`~llm_config_optimization.inference.run_experiment` with the injected
    ``backend`` and ``client``. A trial that raises
    :class:`~llm_config_optimization.inference.types.ExperimentError` is marked failed
    and the study continues, so one out-of-memory configuration does not end the
    sweep.

    ``callback``, when given, is invoked once at the end of every trial with the
    study and the completed trial, so a caller can report progress without
    duplicating the study loop; failed trials invoke it too.

    ``on_trial_start``, when given, is invoked once at the start of every trial
    — before the experiment runs — so a caller can report a trial as in flight
    rather than only once it has finished. It is argument-less and defaults to
    ``None``, leaving callers that do not report progress unchanged.

    ``n_jobs`` is passed through to Optuna unchanged and defaults to one;
    running trials in parallel requires ``storage`` and is the caller's
    responsibility to keep on disjoint devices.

    The result's frontier is the Pareto front in the multi mode and the single
    winning trial in the scalar modes. The recommended point's parameters are
    reconstructed into a complete
    :class:`~llm_config_optimization.inference.types.ExperimentConfig` through the same
    step a trial uses, so the returned configuration re-runs exactly the point
    the sweep recommends. When no trial completes, :class:`OptimizationError`
    is raised with the study attached rather than leaking Optuna's own
    exception. A Ctrl-C stops the sweep and raises
    :class:`OptimizationInterrupted` carrying the front found so far, or the
    same empty-sweep :class:`OptimizationError` when no trial completed.
    """
    if mode in (ObjectiveMode.MULTI, ObjectiveMode.WEIGHTED) and weights is None:
        raise ValueError(f"the {mode.value} objective requires weights")
    sampler = optuna.samplers.TPESampler(seed=seed)
    objective = make_objective(
        space,
        base,
        mode=mode,
        weights=weights,
        backend=backend,
        client=client,
        benchmark_client=benchmark_client,
        on_trial_start=on_trial_start,
    )
    if mode is ObjectiveMode.MULTI:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            directions=["maximize", "maximize"],
            load_if_exists=storage is not None,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="maximize",
            load_if_exists=storage is not None,
        )
    _guard_provenance(study, mode)
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=(ExperimentError,),
            callbacks=None if callback is None else [callback],
        )
    except KeyboardInterrupt:
        raise OptimizationInterrupted(
            f"optimization was interrupted after {len(study.trials)} trials",
            result=_result_from_study(study, space, base, mode=mode, weights=weights),
        ) from None
    return _result_from_study(study, space, base, mode=mode, weights=weights)


def _guard_provenance(study: optuna.Study, mode: ObjectiveMode) -> None:
    """Record a new study's provenance, or refuse a resume that would mix modes.

    A persisted study must not silently accumulate trials scored by different
    objectives, so a study created here records the mode that will produce its
    scores and the version of the phase-rate metric definitions they are
    measured under. Loading a study that recorded a different mode is a caller
    error: its values are not comparable with this run's, so the resume is
    refused with a message naming the recorded mode and pointing at a new
    study name.

    A study without the mode attribute is either brand new or was persisted
    before the attribute existed. A new one — recognised by having no trial and
    the direction count the current mode creates, which also covers an empty
    study in storage that has no score to compare — adopts the current mode.
    Anything else is treated as legacy combined: a combined resume keeps
    working while any other mode is refused, since the historical scores are
    combined numbers. A study's directions cannot change after creation, so an
    empty study left by an interrupted multi sweep is refused rather than
    handed to a scalar objective.
    """
    recorded = study.user_attrs.get(_MODE_ATTRIBUTE)
    if (
        recorded is None
        and not study.trials
        and list(study.directions) == _expected_directions(mode)
    ):
        study.set_user_attr(_MODE_ATTRIBUTE, mode.value)
        study.set_user_attr(_METRIC_SCHEMA_ATTRIBUTE, _METRIC_SCHEMA_VERSION)
        return
    if recorded is None:
        recorded = ObjectiveMode.COMBINED.value
    if recorded != mode.value:
        raise OptimizationError(
            f"study {study.study_name!r} was created with objective mode {recorded!r}, "
            f"but this run uses {mode.value!r}; resume it with --objective {recorded}, "
            "or choose a different study name",
            study=study,
        )
    if list(study.directions) != _expected_directions(mode):
        raise OptimizationError(
            f"study {study.study_name!r} has {len(study.directions)} maximize "
            f"direction(s), but the {mode.value!r} objective needs "
            f"{len(_expected_directions(mode))}; choose a different study name",
            study=study,
        )


def _expected_directions(mode: ObjectiveMode) -> list[optuna.study.StudyDirection]:
    """The maximize directions a study for ``mode`` is created with."""
    if mode is ObjectiveMode.MULTI:
        return [optuna.study.StudyDirection.MAXIMIZE] * 2
    return [optuna.study.StudyDirection.MAXIMIZE]


def _front_solutions(
    study: optuna.Study,
    space: ExperimentSpace,
    base: ExperimentConfig,
    weights: ObjectiveWeights,
) -> tuple[tuple[OptimizationSolution, ...], OptimizationSolution]:
    """Project a multi-objective study's Pareto front and pick the recommendation.

    Every Pareto-optimal trial becomes an :class:`OptimizationSolution` carrying
    the ``(prefill, decode)`` pair it submitted. The recommended point is the
    front member with the highest weighted-harmonic score under ``weights``; an
    exact tie goes to the earlier trial, so the pick does not depend on the
    front's iteration order. A completed trial always carries its objective
    values, so the missing-values branch only guards against an Optuna contract
    change; it is not a user-facing failure.
    """
    projected: list[tuple[optuna.trial.FrozenTrial, list[float]]] = []
    for trial in study.best_trials:
        values = trial.values
        if values is None:
            raise OptimizationError(
                f"front trial {trial.number} has no objective value",
                study=study,
            )
        projected.append((trial, values))
    solutions = tuple(
        OptimizationSolution(
            config=reconstruct_config(space, base, trial.params),
            value=(values[0], values[1]),
            trial_number=trial.number,
        )
        for trial, values in projected
    )
    scores = {
        trial.number: weighted_harmonic_rate(values[0], values[1], weights)
        for trial, values in projected
    }
    recommended = max(
        solutions,
        key=lambda solution: (scores[solution.trial_number], -solution.trial_number),
    )
    return solutions, recommended


def _result_from_study(
    study: optuna.Study,
    space: ExperimentSpace,
    base: ExperimentConfig,
    *,
    mode: ObjectiveMode,
    weights: ObjectiveWeights | None,
) -> OptimizationResult:
    """Project a study's frontier onto an :class:`OptimizationResult`.

    This is shared by the normal return and the interrupt path, so an
    interrupted sweep reports exactly the result a finished one would. In the
    multi mode the whole Pareto front is projected and the recommended point is
    the front member with the highest weighted-harmonic score under
    ``weights``; the scalar modes project the single best trial instead. When
    the study holds no completed trial there is nothing to report, so the
    existing empty-sweep :class:`OptimizationError` is raised.
    """
    completed = [trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise OptimizationError(
            f"optimization finished without a completed trial in {len(study.trials)} trials",
            study=study,
        )

    if mode is ObjectiveMode.MULTI:
        # ``optimize`` rejects a missing pair before any trial; this guards the
        # private contract the same way the scalar branch guards Optuna's.
        if weights is None:
            raise OptimizationError(
                "the multi objective requires weights to pick a recommended point",
                study=study,
            )
        solutions, recommended = _front_solutions(study, space, base, weights)
        return OptimizationResult(
            solutions=solutions,
            recommended=recommended,
            trial_count=len(study.trials),
            study=study,
        )

    best = study.best_trial
    best_value = best.value
    if best_value is None:
        # A completed trial always has an objective value, so this only guards
        # against an Optuna contract change; it is not a user-facing failure.
        raise OptimizationError(
            f"best trial {best.number} has no objective value",
            study=study,
        )
    recommended = OptimizationSolution(
        config=reconstruct_config(space, base, best.params),
        value=best_value,
        trial_number=best.number,
    )
    return OptimizationResult(
        solutions=(recommended,),
        recommended=recommended,
        trial_count=len(study.trials),
        study=study,
    )


def _project_result(result: ExperimentResult) -> dict[str, object]:
    """Project an experiment result onto the JSON-serializable fields to record.

    The full :class:`~llm_config_optimization.inference.types.ExperimentResult` is not
    stored because its raw container output is not JSON-serializable; the typed
    metrics and the resolved target, image, and devices are, so the study keeps
    an inspectable record of what actually ran.

    The aggregate metrics alone would hide how a trial's measured runs were
    spread, so the projection also carries each successful run's throughputs and
    the count of runs that failed. That makes a persisted study explain a
    marginal pass as clearly as a clean one, without re-running anything.

    Each run's prefill and decode rates are recorded alongside its combined
    throughput, and the aggregate pair alongside the mean throughput, so a
    persisted study says which phase of inference a configuration helped. A
    phase rate is optional on a run, so the recorded value is ``None`` where
    the engine did not report what the rate needs.
    """
    return {
        "tokens_per_second": result.tokens_per_second,
        "requests_per_second": result.requests_per_second,
        "prefill_tokens_per_second": result.prefill_tokens_per_second,
        "decode_tokens_per_second": result.decode_tokens_per_second,
        "elapsed_time": result.elapsed_time,
        "target": result.target.value,
        "image": result.image,
        "devices": list(result.devices),
        "failed_runs": result.failed_runs,
        "runs": [
            {
                "tokens_per_second": run.tokens_per_second,
                "requests_per_second": run.requests_per_second,
                "prefill_tokens_per_second": run.prefill_tokens_per_second,
                "decode_tokens_per_second": run.decode_tokens_per_second,
            }
            for run in result.runs
        ],
    }


def _choice_value(value: ParameterValue) -> str | int:
    """Render a declared value as the JSON-native choice Optuna records.

    An engine-neutral enum member becomes its string value, because a persisted
    study is reloaded with plain strings; an integer passes through unchanged.
    """
    return value.value if isinstance(value, Enum) else value


def _coerce_value(field: str, value: object) -> object:
    """Convert a sampled value back to its field's engine-neutral kind.

    A categorical field's sampled string is resolved to a member of the enum
    named by the ``LLMConfig`` field's own type annotation, so the backend's
    value maps apply exactly as they would for a hand-written config. A boolean
    field keeps its sampled boolean unchanged, and a numeric field's value is
    coerced to an integer. A union field (an exact count or an enum mode) is
    resolved against the first branch that accepts the sampled value, so a raw
    choice that arrived as a string becomes the enum member and one that
    arrived as a number stays an integer.

    The numeric branch is intentionally narrow: this function knows only enums,
    booleans, and integers, so any other kind falls through to :func:`int` and
    is silently truncated rather than rejected — ``issubclass(float, Enum)`` is
    ``False``. That is sound for the current tunable set, where every non-enum,
    non-boolean tunable is an ``int`` field and the one float field,
    ``gpu_memory_utilization``, is a non-tunable absent from
    :data:`~llm_config_optimization.search_space.registry.TUNABLE_PARAMETERS`; the YAML
    loader additionally rejects a float ``Interval``. Two paths bypass those
    guards and would need a float branch: a hand-built ``Interval`` with float
    bounds, and a resumed study whose ``best.params`` came from a float
    suggestion. Adding a float tunable therefore means dispatching on the
    resolved kind here instead of assuming an integer.
    """
    kind = expected_kind(field)
    if isinstance(kind, UnionKind):
        return _coerce_union_value(field, value, kind)
    if issubclass(kind, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{field} expects a boolean, got {type(value).__name__}")
        return value
    if issubclass(kind, Enum):
        return kind(value)
    return int(cast("str | int | float", value))


def _coerce_union_value(field: str, value: object, kind: UnionKind) -> object:
    """Resolve a sampled value against a union field's branches.

    An enum branch accepts its own string spelling, a boolean branch only a real
    boolean, and an integer branch only an integer that is not a boolean. A
    value no branch accepts is rejected rather than coerced to the wrong
    domain, so a mode name is never handed to :func:`int` and an integer is
    never reinterpreted as an enum value.
    """
    for atom in kind.kinds:
        if issubclass(atom, bool):
            if isinstance(value, bool):
                return value
        elif issubclass(atom, Enum):
            try:
                return atom(value)
            except TypeError, ValueError:
                continue
        elif isinstance(value, int) and not isinstance(value, bool):
            return value
    raise ValueError(f"{field} expects {kind}, got {type(value).__name__}")
