"""Engine-neutral types for the optimization layer.

These types describe the outcome of a parameter sweep without committing to any
particular optimizer. They are built by the Optuna-specific module and
re-exported from :mod:`llm_optimization.optimization`; this module imports no
Optuna at runtime, so a second optimizer can be added behind the same result
type without changing callers, mirroring how
:mod:`llm_optimization.inference.types` keeps the engine-neutral experiment
surface free of engine details.

:class:`OptimizationResult` is the successful outcome of a sweep: the best
configuration found, its objective value and trial number, how many trials the
study holds, and the raw study for full inspection. :class:`OptimizationError`
is raised when a sweep finishes without a single completed trial; it carries
the study so the failed sweep remains inspectable.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..inference.types import ExperimentConfig

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The typed outcome of one optimization sweep.

    ``best_config`` is the fully specified experiment to re-run, rebuilt from
    the winning trial's parameters, and ``best_value`` is that trial's objective
    value (throughput, in this version). ``best_trial_number`` identifies the
    winning trial within the study and ``trial_count`` records how many trials
    the study holds after the call, so a resumed sweep reports its whole history
    rather than only the trials this call added.

    The raw Optuna study is kept as ``study`` so the full trial history —
    parameters, objective values, and recorded outcomes — stays inspectable even
    though only the best point is projected onto this type. It is the one field
    that is not engine-neutral, and is typed as an Optuna study because no other
    optimizer is wired up yet.
    """

    best_config: ExperimentConfig
    best_value: float
    best_trial_number: int
    trial_count: int
    study: optuna.Study


class OptimizationError(RuntimeError):
    """Raised for every failure of the optimization layer.

    A sweep that finishes without a single completed trial has no best
    configuration to return, so the adapter raises this rather than leaking
    Optuna's own exception; callers can therefore distinguish an empty study
    from a sweep that never started. The raw study is attached as ``study``
    whenever one was created, so the error recorded on each failed trial remains
    inspectable.

    This is a standalone exception rather than an
    :class:`~llm_optimization.inference.types.ExperimentError`: a sweep that
    produced nothing is a distinct failure from one benchmark run that failed,
    and callers can tell them apart by type.
    """

    def __init__(self, message: str, *, study: optuna.Study | None = None) -> None:
        super().__init__(message)
        self.study = study
