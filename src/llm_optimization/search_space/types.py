"""Engine-neutral types for an experiment search-space definition.

These types describe *which* serving parameters an experiment sweep may vary and
*which* values each one is allowed to take. They are deliberately separate from
the execution types in :mod:`llm_optimization.inference.types`: an
:class:`ExperimentSpace` is the reviewable definition of a sweep, while an
:class:`~llm_optimization.inference.types.LLMConfig` is one point the sweep
selects. Like the inference surface it feeds, this module imports no engine, so
the same definition can target a different backend later.

A :class:`Dimension` names a tunable parameter and carries exactly one
:data:`Spec`:

* :class:`Fixed` pins the parameter to one value;
* :class:`Categorical` lists a subset of allowed values in declared order;
* :class:`Interval` gives an inclusive numeric range the optimizer samples from;
* :class:`Default` records the explicit decision to defer to the engine.

Every tunable parameter is present in a well-formed space, so there is no
"absent means engine default" case; deferral is expressed only through
:class:`Default`. The types are frozen and hashable, so a space can be logged,
compared, and cached.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ..inference.types import AttentionBackend, DType, KVCacheDType, QuantizationMethod

type ParameterValue = DType | QuantizationMethod | KVCacheDType | AttentionBackend | int
"""A value a tunable parameter may take.

Categorical parameters hold an engine-neutral enum member and numeric
parameters hold an integer; the validator rejects a value of the wrong kind for
its parameter.
"""


@dataclass(frozen=True, slots=True)
class Fixed:
    """Pin a parameter to a single value that must not be varied.

    A fixed dimension still contributes its value to every experiment, so it is
    recorded in the definition rather than left implicit.
    """

    value: ParameterValue


@dataclass(frozen=True, slots=True)
class Categorical:
    """A subset of allowed values for a parameter, in declared order.

    The order is preserved because it can express a preferred or baseline-first
    ordering; it is not sorted or deduplicated here — validation rejects a
    duplicate or empty list.
    """

    values: tuple[ParameterValue, ...]

    def __post_init__(self) -> None:
        """Store the values as a tuple so the spec is immutable and hashable."""
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True, slots=True)
class Interval:
    """An inclusive numeric range for a parameter.

    The range carries only its bounds: choosing where to sample is the
    optimizer's job, so there is deliberately no step or count. ``minimum`` and
    ``maximum`` are both included, and validation requires ``minimum <=
    maximum``.
    """

    minimum: int | float
    maximum: int | float


@dataclass(frozen=True, slots=True)
class Default:
    """Explicitly defer a parameter to the inference engine's own default.

    This is a recorded decision, not an omission: the parameter is declared like
    every other, but its value is left unknown by the definition on purpose.
    """


type Spec = Fixed | Categorical | Interval | Default
"""The tagged spec a :class:`Dimension` carries, one kind per dimension."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """A named tunable parameter and the spec that constrains its values."""

    name: str
    spec: Spec


@dataclass(frozen=True, slots=True)
class ExperimentSpace:
    """A complete, immutable search space over the tunable parameters.

    The space is total over the tunable set: every tunable parameter has a
    dimension, so a run's search values are fully explainable from the
    definition alone. Dimensions are expected in canonical tunable-registry
    order, which makes the parsed space deterministic.
    """

    dimensions: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        """Store the dimensions as a tuple so the space is immutable and hashable."""
        object.__setattr__(self, "dimensions", tuple(self.dimensions))


class ExperimentDefinitionError(Exception):
    """Raised when an experiment search-space definition is invalid.

    A definition is checked in full before any experiment runs, and every
    problem found is collected here rather than reported one at a time, so a
    broken definition can be corrected in a single pass. The individual problems
    are available as ``problems`` for tooling, and the rendered message lists
    them all.

    This is a standalone exception rather than an
    :class:`~llm_optimization.inference.types.ExperimentError`: a bad definition
    is a distinct failure from a benchmark run that failed, and callers can tell
    them apart by type.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        detail = "\n".join(f"  - {problem}" for problem in self.problems)
        message = (
            "invalid experiment definition"
            if not detail
            else f"invalid experiment definition:\n{detail}"
        )
        super().__init__(message)
