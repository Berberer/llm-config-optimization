"""Experiment search-space definitions.

This subpackage owns the search-space definition topic: the engine-neutral types
that describe which serving parameters a sweep may vary and which values each
may take, together with the tunable registry, the YAML loader, and the validator
built on them. Executing an experiment remains the job of
:mod:`llm_optimization.inference`; a definition is checked here and handed to a
backend there.

The public surface is the whole definition workflow::

    from llm_optimization.search_space import load_experiment_space, validate_space

``load_experiment_space`` parses and engine-agnostically validates a YAML file,
and ``validate_space`` performs the explicit, backend-specific support check
before any experiment runs.
"""

from .loader import load_experiment_space, parse_experiment_space
from .registry import TUNABLE_PARAMETERS, ParameterKind, expected_kind
from .types import (
    Categorical,
    Default,
    Dimension,
    ExperimentDefinitionError,
    ExperimentSpace,
    Fixed,
    Interval,
    ParameterValue,
    Spec,
)
from .validation import validate_space

__all__ = [
    "TUNABLE_PARAMETERS",
    "Categorical",
    "Default",
    "Dimension",
    "ExperimentDefinitionError",
    "ExperimentSpace",
    "Fixed",
    "Interval",
    "ParameterKind",
    "ParameterValue",
    "Spec",
    "expected_kind",
    "load_experiment_space",
    "parse_experiment_space",
    "validate_space",
]
