"""Shared mechanics for translating and validating engine-neutral tunables.

Each backend owns the vocabulary tables that map an engine-neutral value to its
engine's spelling. The *mechanics* around those tables are identical across
backends: walking a dimension's declared values, translating an enum member
through a value map, reporting a member the engine cannot spell, and deriving
the effective space a model-aware assessment leaves behind. They live here so
the rules cannot drift between engines, while every engine spelling stays in
the backend that owns it.

Engine names appear only as labels supplied by the caller, so this module
carries no vocabulary of its own.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import Enum

from ..search_space.types import Categorical, Dimension, ExperimentSpace, Fixed, ParameterValue
from .types import ExperimentError


@dataclass(frozen=True, slots=True)
class Toggle:
    """A boolean tunable forwarded as one of two opposite flags.

    The engine is told both sides of the switch explicitly, so ``False`` is a
    real configuration rather than an omitted default. ``enabled`` is emitted
    for ``True`` and ``disabled`` for ``False``; an unset value emits neither.
    The flag spellings stay in the backend that owns the engine.
    """

    enabled: str
    disabled: str


def toggle_arguments(value: object, target: Toggle) -> list[str]:
    """Return the flag that expresses ``value`` for a :class:`Toggle`.

    Argument assembly calls this for a declared boolean value, so the choice
    between the two flag spellings is made in one place for every backend.
    ``value`` is typed as ``object`` because it arrives from an untyped
    parameter mapping; the toggle is only used for a boolean field, and
    ``True`` is matched by identity so no other value can masquerade as on.
    """
    return [target.enabled] if value is True else [target.disabled]


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


def prune_dimensions(
    space: ExperimentSpace,
    names: Collection[str],
    *,
    reason: str,
) -> tuple[ExperimentSpace, tuple[str, ...]]:
    """Remove the named dimensions from ``space``, noting every removal.

    A backend calls this once a model-aware rule establishes that a dimension
    cannot affect the resolved run, so the caller supplies the parameters that
    rule selects and a reason naming the fact that made them irrelevant. The
    remaining dimensions keep their canonical order, and each removal is
    returned as one note, so a caller can report why part of a declared space
    was not run instead of silently sweeping something else.

    ``names`` may include parameters the space does not declare — an
    unconditional rule can name a dimension a particular space omitted — and
    those contribute no note. A space with nothing to remove is returned
    unchanged with no notes.
    """
    dropped = frozenset(names)
    removed = tuple(dimension.name for dimension in space.dimensions if dimension.name in dropped)
    if not removed:
        return space, ()
    effective = ExperimentSpace(
        dimensions=tuple(
            dimension for dimension in space.dimensions if dimension.name not in dropped
        )
    )
    return effective, tuple(f"dropped space.{name}: {reason}" for name in removed)


@dataclass(frozen=True, slots=True)
class ValueNarrowing:
    """The space a value-level rule leaves behind, and what it removed.

    ``space`` keeps only the declared values the rule retained, in declared
    order. ``notes`` names every removed value and the reason it was removed.
    ``emptied`` is ``True`` when the rule would have removed every value of the
    targeted dimension; in that case ``space`` is returned unchanged, because an
    explicitly declared dimension must not silently become an engine default —
    the caller turns the flag into a model-aware problem instead.
    """

    space: ExperimentSpace
    notes: tuple[str, ...] = ()
    emptied: bool = False


def narrow_values(
    space: ExperimentSpace,
    name: str,
    reason_for: Callable[[ParameterValue], str | None],
) -> ValueNarrowing:
    """Narrow ``name`` to the declared values ``reason_for`` accepts.

    A value-level rule runs after the whole-dimension rules: the checkpoint may
    reject only *some* of a dimension's declared values, so a categorical or
    fixed parameter keeps the rest. ``reason_for`` returns the human-readable
    reason a value cannot run, or ``None`` when it is retained. Retained values
    keep their declared order, and each removed value is returned as one note so
    the caller can report exactly what the model narrowed.

    Only a :class:`Fixed` or :class:`Categorical` dimension declares values, so
    an ``Interval`` or ``Default`` dimension — and a name the space does not
    declare — is returned unchanged. A dimension whose values would all be
    removed is also returned unchanged, with ``emptied`` set, rather than
    becoming an empty spec the optimizer cannot sample.
    """
    dimension = next((item for item in space.dimensions if item.name == name), None)
    if dimension is None or not isinstance(dimension.spec, (Fixed, Categorical)):
        return ValueNarrowing(space=space)
    assessed = [(path, value, reason_for(value)) for path, value in declared_values(dimension)]
    rejected = [(path, value, reason) for path, value, reason in assessed if reason is not None]
    if not rejected:
        return ValueNarrowing(space=space)
    retained = tuple(value for _, value, reason in assessed if reason is None)
    if not retained:
        return ValueNarrowing(space=space, emptied=True)
    narrowed = Fixed(retained[0]) if isinstance(dimension.spec, Fixed) else Categorical(retained)
    effective = ExperimentSpace(
        dimensions=tuple(
            Dimension(name=item.name, spec=narrowed) if item.name == name else item
            for item in space.dimensions
        )
    )
    notes = tuple(
        f"dropped {path} ({_value_label(value)}): {reason}"
        for path, value, reason in rejected
        if reason is not None
    )
    return ValueNarrowing(space=effective, notes=notes)


def _value_label(value: ParameterValue) -> str:
    """Render a declared value in a note, using an enum's engine-neutral spelling."""
    return value.value if isinstance(value, Enum) else str(value)
