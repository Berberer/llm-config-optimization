"""Optimization of LLM serving parameters.

This subpackage owns the optimization topic: choosing which point of an
:class:`~llm_config_optimization.search_space.ExperimentSpace` to try next, running the
benchmark for each sampled point, and reporting the best configuration found. It
mirrors the inference layer's neutral-types-plus-backend split: the
:mod:`llm_config_optimization.optimization.types` module holds the engine-neutral
result and error types, while the Optuna-specific module owns the study and
trial logic and builds those types.

The public surface is the single entry point and its result and error types::

    from llm_config_optimization.optimization import optimize

``optimize`` searches a space for the best-throughput configuration and returns
an :class:`OptimizationResult`; a sweep that completes no trial raises
:class:`OptimizationError`. An interrupted sweep that did complete trials
raises :class:`OptimizationInterrupted`, a subtype of :class:`OptimizationError`
that carries the best-so-far :class:`OptimizationResult`.
"""

from ._optuna import optimize
from .types import OptimizationError, OptimizationInterrupted, OptimizationResult

__all__ = [
    "OptimizationError",
    "OptimizationInterrupted",
    "OptimizationResult",
    "optimize",
]
