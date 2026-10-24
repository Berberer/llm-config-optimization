"""Shared mechanics for translating and validating engine-neutral tunables.

Each backend owns the vocabulary tables that map an engine-neutral value to its
engine's spelling. The *mechanics* around those tables are identical across
backends: walking a dimension's declared values, translating an enum member
through a value map, and reporting a member the engine cannot spell. They live
here so the rules cannot drift between engines, while every engine spelling
stays in the backend that owns it.

Engine names appear only as labels supplied by the caller, so this module
carries no vocabulary of its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ..search_space.types import Categorical, Dimension, Fixed, ParameterValue
from .types import ExperimentError


@dataclass(frozen=True, slots=True)
class Gate:
    """A tunable that is not forwarded but switches related fields on or off.

    A gate is recognised as an applicable parameter, so a declared dimension is
    accepted by validation, yet it never reaches the engine itself: the backend
    reads its value while assembling arguments and suppresses the fields the
    gate controls. The speculative mode is the only gate today: a backend reads
    it to decide whether the engine's speculative flags are emitted at all.
    """


def translate_value(
    field: str,
    value: object,
    value_maps: Mapping[str, Mapping[object, str]],
    *,
    engine: str,
) -> object:
    """Translate an engine-neutral categorical value into ``engine``'s spelling.

    Values for fields without a value map, and values that are not enum members,
    pass through unchanged. An enum member absent from its field's map is not
    supported by the engine and raises rather than being silently dropped, so a
    requested configuration is never quietly ignored. ``engine`` labels the
    backend in that error; the mappings themselves stay in the backend.
    """
    mapping = value_maps.get(field)
    if mapping is None or not isinstance(value, Enum):
        return value
    try:
        return mapping[value]
    except KeyError:
        raise ExperimentError(f"{engine} does not support {field}={value.value!r}") from None


def declared_values(dimension: Dimension) -> list[tuple[str, ParameterValue]]:
    """Pair each value a dimension declares with the problem path naming it.

    A ``Fixed`` dimension declares one value and a ``Categorical`` one several;
    a ``Default`` or ``Interval`` dimension declares none, because a deferral
    carries no value and an interval's endpoints are not the values sampled.
    The paths match the search-space definition's own addressing, so a reported
    problem points at the value the user wrote.
    """
    spec = dimension.spec
    if isinstance(spec, Fixed):
        return [(f"space.{dimension.name}.fixed", spec.value)]
    if isinstance(spec, Categorical):
        return [
            (f"space.{dimension.name}.categorical[{index}]", value)
            for index, value in enumerate(spec.values)
        ]
    return []


def unsupported_values(
    dimension: Dimension,
    field: str,
    value_maps: Mapping[str, Mapping[object, str]],
    *,
    engine: str,
) -> list[str]:
    """Report every declared enum value ``engine`` has no spelling for.

    Only enum values are checked: a numeric parameter is forwarded verbatim, so
    it has nothing to look up in a value map. A categorical field with no value
    map at all cannot translate any member and rejects every enum value.
    ``engine`` labels the backend in the problems, while the mappings remain
    the backend's own.
    """
    mapping = value_maps.get(field)
    return [
        f"{path}: {engine} does not support {field}={value.value!r}"
        for path, value in declared_values(dimension)
        if isinstance(value, Enum) and (mapping is None or value not in mapping)
    ]
