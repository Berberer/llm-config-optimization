"""Optimization of LLM serving parameters.

This subpackage owns the optimization topic: choosing which point of an
:class:`~llm_config_optimization.search_space.ExperimentSpace` to try next, running the
benchmark for each sampled point, and reporting the best configuration found. It
mirrors the inference layer's neutral-types-plus-backend split: the
:mod:`llm_config_optimization.optimization.types` module holds the engine-neutral
result and error types, while the Optuna-specific module owns the study and
trial logic and builds those types.

The public surface is the single entry point, its result and error types, and
the engine-neutral objective model::

    from llm_config_optimization.optimization import optimize

``optimize`` searches a space for the best-throughput configuration and returns
an :class:`OptimizationResult`; a sweep that completes no trial raises
:class:`OptimizationError`. An interrupted sweep that did complete trials
raises :class:`OptimizationInterrupted`, a subtype of :class:`OptimizationError`
that carries the frontier found so far as its :class:`OptimizationResult`. The
result's ``solutions`` is the Pareto front in the multi mode and the single
winner in the scalar modes, and its ``recommended`` point is the one a caller
should re-run. The :class:`ObjectiveMode`, :class:`ObjectiveWeights`, and
scoring functions define what a trial maximizes and are public so the command
line and reporters share one definition.
"""

from ._optuna import optimize
from .types import (
    ObjectiveMode,
    ObjectiveWeights,
    OptimizationError,
    OptimizationInterrupted,
    OptimizationResult,
    OptimizationSolution,
    combined_score,
    objective_value,
    phase_rates,
    weighted_harmonic_rate,
    weighted_score,
)

__all__ = [
    "ObjectiveMode",
    "ObjectiveWeights",
    "OptimizationError",
    "OptimizationInterrupted",
    "OptimizationResult",
    "OptimizationSolution",
    "combined_score",
    "objective_value",
    "optimize",
    "phase_rates",
    "weighted_harmonic_rate",
    "weighted_score",
]
