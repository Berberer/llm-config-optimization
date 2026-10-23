"""Engine-neutral types for the optimization layer.

These types describe the outcome of a parameter sweep without committing to any
particular optimizer. They are built by the Optuna-specific module and
re-exported from :mod:`llm_config_optimization.optimization`; this module imports no
Optuna at runtime, so a second optimizer can be added behind the same result
type without changing callers, mirroring how
:mod:`llm_config_optimization.inference.types` keeps the engine-neutral experiment
surface free of engine details.

The engine-neutral objective model — :class:`ObjectiveMode`,
:class:`ObjectiveWeights`, and the pure scoring functions
:func:`combined_score`, :func:`phase_rates`, :func:`weighted_score`, and
:func:`weighted_harmonic_rate` behind :func:`objective_value` — describes what
a sweep maximizes without knowing an optimizer or an engine. A phase rate an
objective needs but a result lacks raises
:class:`~llm_config_optimization.inference.types.ExperimentError` naming
the phase and the sources that could have reported it, so an objective never
silently falls back to a different measurement.

:class:`OptimizationResult` is the successful outcome of a sweep: the frontier
of :class:`OptimizationSolution` points found, the recommended point to re-run,
how many trials the study holds, and the raw study for full inspection.
:class:`OptimizationError`
is raised when a sweep finishes without a single completed trial; it carries
the study so the failed sweep remains inspectable. :class:`OptimizationInterrupted`
is raised when the user interrupts a sweep that did complete trials; it carries
the front found so far so the work is not discarded.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ..inference.types import ExperimentConfig, ExperimentError, ExperimentResult

if TYPE_CHECKING:
    import optuna


class ObjectiveMode(StrEnum):
    """How a sweep turns a measured result into what it maximizes.

    ``MULTI`` submits prefill and decode rates as two independent objectives,
    so the trade-off is read from the Pareto front after the sweep rather than
    committed to before it. ``WEIGHTED`` folds the two rates into one
    weighted-harmonic effective rate, and ``COMBINED`` maximizes the historical
    combined token throughput, the only mode an engine that reports no phase
    data can support. The values are the command-line spellings.
    """

    MULTI = "multi"
    WEIGHTED = "weighted"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """The token mix a weighted objective is scored under.

    The weights are token counts, not fractions: ``prefill`` is the number of
    prompt tokens and ``decode`` the number of generated tokens a request
    shaped like the deployment is expected to carry. The score in
    :func:`weighted_score` is therefore itself a tokens-per-second rate, and
    only the ratio matters — any positive scaling of the pair leaves the score
    unchanged.

    Both weights must be finite and non-negative, and at least one must be
    positive: a zero weight degenerates the score to the other phase, but a
    pair of zeros describes no request shape at all, so it is rejected here
    rather than silently scoring every point zero.
    """

    prefill: float
    decode: float

    def __post_init__(self) -> None:
        """Reject weights that cannot describe a token mix."""
        if not (math.isfinite(self.prefill) and math.isfinite(self.decode)):
            raise ValueError(
                f"objective weights must be finite, got prefill={self.prefill}, "
                f"decode={self.decode}"
            )
        if self.prefill < 0 or self.decode < 0:
            raise ValueError(
                f"objective weights must be non-negative, got prefill={self.prefill}, "
                f"decode={self.decode}"
            )
        if self.prefill == 0 and self.decode == 0:
            raise ValueError("objective weights cannot both be zero")


def _require_rate(result: ExperimentResult, phase: str) -> float:
    """Return one phase's rate, or raise naming what the engine did not report.

    The optimization layer is engine-neutral and cannot know which engine
    produced a result, so the message names both sources a backend can derive a
    phase rate from — a Prometheus ``/metrics`` snapshot or per-response
    timings — next to the phase that is missing.
    """
    if phase == "prefill":
        rate = result.prefill_tokens_per_second
    else:
        rate = result.decode_tokens_per_second
    if rate is None:
        raise ExperimentError(
            f"the {phase} tokens-per-second rate is required by the objective, "
            "but the trial reported none: the engine exposed neither a /metrics "
            "snapshot nor response timings"
        )
    return rate


def combined_score(result: ExperimentResult) -> float:
    """Return the combined token throughput a result already carries.

    This is the historical objective and the escape hatch for engines that
    report no phase data: prompt and generated tokens over elapsed wall-clock
    time with no phase split, read straight off
    :attr:`~llm_config_optimization.inference.types.ExperimentResult.tokens_per_second`.
    """
    return result.tokens_per_second


def phase_rates(result: ExperimentResult) -> tuple[float, float]:
    """Return the ``(prefill, decode)`` rates the multi objective maximizes.

    Both are required: a result that reported only one — or neither — raises
    :class:`~llm_config_optimization.inference.types.ExperimentError` naming the
    missing phase, so a sweep cannot silently maximize a different objective
    than the one selected.
    """
    return _require_rate(result, "prefill"), _require_rate(result, "decode")


def weighted_harmonic_rate(
    prefill: float,
    decode: float,
    weights: ObjectiveWeights,
) -> float:
    """Return the weighted-harmonic effective rate of a phase-rate pair.

    This is the formula behind :func:`weighted_score`, exposed without an
    :class:`~llm_config_optimization.inference.types.ExperimentResult` because
    the multi mode's recommended point is chosen by scoring raw ``(prefill,
    decode)`` pairs read back from the study.

    The score is ``(w_p + w_d) / (w_p / prefill + w_d / decode)`` for the
    prefill weight ``w_p`` and decode weight ``w_d``. Because the weights are
    token counts, the result stays in tokens per second — the throughput of a
    request shaped like the weights — which is why the harmonic mean is used
    rather than a weighted average.

    A zero weight degenerates the score exactly to the other phase's rate; a
    phase rate of zero yields a score of zero, the formula's limit.
    """
    if weights.prefill == 0:
        return decode
    if weights.decode == 0:
        return prefill
    if prefill == 0 or decode == 0:
        return 0.0
    return (weights.prefill + weights.decode) / (
        weights.prefill / prefill + weights.decode / decode
    )


def weighted_score(result: ExperimentResult, weights: ObjectiveWeights) -> float:
    """Return the weighted-harmonic effective rate of a result's phase rates.

    A convenience wrapper around :func:`weighted_harmonic_rate` that reads the
    rates off a result: a zero weight degenerates the score exactly to the
    other phase's rate and the unused rate is not required, while a rate the
    score needs missing raises
    :class:`~llm_config_optimization.inference.types.ExperimentError` through
    :func:`_require_rate`.
    """
    if weights.prefill == 0:
        return _require_rate(result, "decode")
    if weights.decode == 0:
        return _require_rate(result, "prefill")
    return weighted_harmonic_rate(
        _require_rate(result, "prefill"),
        _require_rate(result, "decode"),
        weights,
    )


def objective_value(
    mode: ObjectiveMode,
    result: ExperimentResult,
    weights: ObjectiveWeights | None = None,
) -> float | tuple[float, float]:
    """Compute the trial value the selected objective mode implies.

    ``MULTI`` yields the ``(prefill, decode)`` tuple :func:`phase_rates`
    returns, ``WEIGHTED`` the scalar :func:`weighted_score` computes under
    ``weights``, and ``COMBINED`` the scalar :func:`combined_score` passes
    through. ``weights`` is required by — and only meaningful for — the
    weighted mode; a missing pair is a caller error, since the command line
    resolves weights before any trial runs.
    """
    if mode is ObjectiveMode.MULTI:
        return phase_rates(result)
    if mode is ObjectiveMode.WEIGHTED:
        if weights is None:
            raise ValueError("the weighted objective requires weights")
        return weighted_score(result, weights)
    return combined_score(result)


@dataclass(frozen=True, slots=True)
class OptimizationSolution:
    """One point of an optimization result's frontier.

    ``config`` is the fully specified experiment to re-run, rebuilt from the
    trial's parameters through the same step a trial uses, and ``value`` is
    what the trial submitted to the study: the single scalar objective in the
    weighted and combined modes, or the ``(prefill, decode)`` pair the multi
    mode maximizes. ``trial_number`` identifies the trial within the study.

    A solution carries only the point itself. The trial's measured detail —
    its aggregate and per-run phase rates, target, image, and devices — stays
    on the study as the trial's recorded ``result`` user attribute, so a
    reporter can show more than the objective value without this type growing
    a field per metric.
    """

    config: ExperimentConfig
    value: float | tuple[float, float]
    trial_number: int


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The typed outcome of one optimization sweep.

    ``solutions`` is the sweep's frontier: every Pareto-optimal point the study
    found in the multi mode, and the single winning point in the scalar
    weighted and combined modes. ``recommended`` is the point a caller should
    re-run — in multi mode the front member with the highest weighted-harmonic
    score under the sweep's weights, and the single solution otherwise. It is
    always a member of ``solutions``.

    ``best_config``, ``best_value``, and ``best_trial_number`` are aliases of
    the recommended point kept for callers written before the frontier existed;
    ``best_value`` is a scalar in the weighted and combined modes and the
    ``(prefill, decode)`` pair in multi mode. ``trial_count`` records how many
    trials the study holds after the call, so a resumed sweep reports its whole
    history rather than only the trials this call added.

    The raw Optuna study is kept as ``study`` so the full trial history —
    parameters, objective values, and recorded outcomes — stays inspectable even
    though only the frontier is projected onto this type. It is the one field
    that is not engine-neutral, and is typed as an Optuna study because no other
    optimizer is wired up yet.
    """

    solutions: tuple[OptimizationSolution, ...]
    recommended: OptimizationSolution
    trial_count: int
    study: optuna.Study

    @property
    def best_config(self) -> ExperimentConfig:
        """The recommended point's configuration, kept as a legacy alias."""
        return self.recommended.config

    @property
    def best_value(self) -> float | tuple[float, float]:
        """The recommended point's objective value, kept as a legacy alias."""
        return self.recommended.value

    @property
    def best_trial_number(self) -> int:
        """The recommended point's trial number, kept as a legacy alias."""
        return self.recommended.trial_number


class OptimizationError(RuntimeError):
    """Raised for every failure of the optimization layer.

    A sweep that finishes without a single completed trial has no best
    configuration to return, so the adapter raises this rather than leaking
    Optuna's own exception; callers can therefore distinguish an empty study
    from a sweep that never started. The raw study is attached as ``study``
    whenever one was created, so the error recorded on each failed trial remains
    inspectable.

    This is a standalone exception rather than an
    :class:`~llm_config_optimization.inference.types.ExperimentError`: a sweep that
    produced nothing is a distinct failure from one benchmark run that failed,
    and callers can tell them apart by type.
    """

    def __init__(self, message: str, *, study: optuna.Study | None = None) -> None:
        super().__init__(message)
        self.study = study


class OptimizationInterrupted(OptimizationError):
    """Raised when a sweep is stopped by the user before it reaches its budget.

    Ctrl-C during a sweep is a distinct outcome from an empty one: at least one
    trial may have completed, so there is a frontier worth reporting. This
    exception carries that outcome as ``result`` — the same
    :class:`OptimizationResult` a finished sweep would return — so a caller can
    report and act on the front found so far instead of discarding the work.

    It subclasses :class:`OptimizationError`, so a caller that only handles the
    optimization layer's failures still catches an interrupt; the study is
    available through both ``result.study`` and the inherited ``study``.
    """

    def __init__(self, message: str, *, result: OptimizationResult) -> None:
        super().__init__(message, study=result.study)
        self.result = result
