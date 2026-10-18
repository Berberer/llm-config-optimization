"""Load and parse an experiment search-space definition.

A definition is a YAML document whose sole top-level key is ``space``, a mapping
from a tunable parameter name to that parameter's spec. :func:`load_experiment_space`
reads the YAML and hands the parsed document to :func:`parse_experiment_space`,
which turns it into a canonical-order
:class:`~llm_config_optimization.search_space.types.ExperimentSpace`.

The parser validates the definition engine-agnostically. It checks the top-level
shape, that every declared parameter is a known tunable, that each dimension
carries exactly one recognized spec tag, and that every value is of the kind its
``LLMConfig`` field expects. It also requires every parameter in the caller's
required set to be declared: a missing dimension is an error, so a value can
never silently fall back to an engine default. The required set defaults to
every registered tunable, so engine-agnostic callers and tests are unchanged;
the composition root passes the selected backend's applicable set instead, so a
definition declares exactly the parameters that engine forwards. Every problem
found is collected and reported at once through
:class:`~llm_config_optimization.search_space.types.ExperimentDefinitionError`, so a
broken definition is corrected in a single pass. Whether a backend actually
supports the declared parameters and values is a separate, explicit check that
needs a backend and is deliberately not done here.

Malformed YAML and a non-mapping document are reported through the same error
type, so a caller handles a broken file uniformly.
"""

from collections.abc import Iterable, Mapping
from dataclasses import fields
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from ..inference.types import LLMConfig
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

_SPACE_KEY = "space"
_SPEC_KINDS = ("fixed", "categorical", "interval", "default")
_SPEC_KIND_SET = frozenset(_SPEC_KINDS)
_INTERVAL_KEYS = frozenset({"min", "max"})
_LLM_CONFIG_FIELDS = frozenset(field.name for field in fields(LLMConfig))
_POSITIVE_PARAMETERS = frozenset(
    {"max_model_len", "max_num_seqs", "max_num_batched_tokens", "tensor_parallel_size"}
)


def load_experiment_space(path: Path, *, required: Iterable[str] | None = None) -> ExperimentSpace:
    """Read a YAML definition from ``path`` and parse its search space.

    The file is decoded as UTF-8 and parsed with :func:`yaml.safe_load`, so no
    arbitrary YAML tags are constructed. Malformed YAML is reported through
    :class:`~llm_config_optimization.search_space.types.ExperimentDefinitionError`, the
    same error the structural checks raise, so a broken file fails uniformly.

    ``required`` names the tunable parameters the caller's engine forwards and
    is forwarded to :func:`parse_experiment_space`; it defaults to every
    registered tunable.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ExperimentDefinitionError([f"malformed YAML: {error}"]) from error
    return parse_experiment_space(document, required=required)


def parse_experiment_space(
    mapping: object, *, required: Iterable[str] | None = None
) -> ExperimentSpace:
    """Parse and engine-agnostically validate an already-parsed definition.

    The document must be a mapping whose only key is ``space``, and ``space``
    must itself be a mapping. Every problem found anywhere in the definition is
    collected and reported at once.

    ``required`` is the set of tunable parameters that must be declared. It
    defaults to every registered tunable, so the total-space rule is unchanged
    for engine-agnostic callers; passing a backend's applicable set makes the
    rule total over exactly the parameters that engine forwards. A declared
    parameter outside the set is not a problem here: whether the engine can
    forward it is the backend's separate
    :meth:`~llm_config_optimization.inference.types.InferenceBackend.validate`
    check.
    """
    if not isinstance(mapping, Mapping):
        raise ExperimentDefinitionError(
            [f"definition must be a mapping, got {type(mapping).__name__}"]
        )

    problems = [f"unknown top-level key: {key!r}" for key in mapping if key != _SPACE_KEY]
    if _SPACE_KEY not in mapping:
        problems.append(f"missing required top-level key: {_SPACE_KEY!r}")
    elif not isinstance(mapping[_SPACE_KEY], Mapping):
        problems.append(
            f"{_SPACE_KEY!r} must be a mapping, got {type(mapping[_SPACE_KEY]).__name__}"
        )
    if problems:
        raise ExperimentDefinitionError(problems)

    return _parse_dimensions(mapping[_SPACE_KEY], _required_parameters(required))


def _required_parameters(required: Iterable[str] | None) -> frozenset[str]:
    """Resolve the required tunable set, defaulting to the whole registry."""
    if required is None:
        return frozenset(TUNABLE_PARAMETERS)
    return frozenset(required)


def _parse_dimensions(space: Mapping[object, object], required: frozenset[str]) -> ExperimentSpace:
    """Parse every declared dimension, then emit them in canonical registry order."""
    problems: list[str] = []
    parsed: dict[object, Dimension] = {}
    for name, raw_spec in space.items():
        dimension = _parse_dimension(name, raw_spec, problems)
        if dimension is not None:
            parsed[name] = dimension
    problems.extend(
        f"{_SPACE_KEY}.{parameter}: missing required tunable parameter; "
        f"declare one of {_format_kinds()}"
        for parameter in TUNABLE_PARAMETERS
        if parameter in required and parameter not in space
    )
    if problems:
        raise ExperimentDefinitionError(problems)
    dimensions = tuple(parsed[parameter] for parameter in TUNABLE_PARAMETERS if parameter in parsed)
    return ExperimentSpace(dimensions)


def _parse_dimension(name: object, raw_spec: object, problems: list[str]) -> Dimension | None:
    """Resolve a parameter name to a tunable and parse its spec."""
    path = f"{_SPACE_KEY}.{name}"
    if not isinstance(name, str) or name not in TUNABLE_PARAMETERS:
        if name in _LLM_CONFIG_FIELDS:
            problems.append(f"{path}: {name!r} is a known parameter but is not tunable")
        else:
            problems.append(f"{path}: unknown parameter {name!r}")
        return None
    kind = expected_kind(TUNABLE_PARAMETERS[name])
    spec = _parse_spec(
        path,
        raw_spec,
        kind,
        positive=name in _POSITIVE_PARAMETERS,
        problems=problems,
    )
    if spec is None:
        return None
    return Dimension(name, spec)


def _parse_spec(
    path: str,
    raw_spec: object,
    kind: ParameterKind,
    *,
    positive: bool,
    problems: list[str],
) -> Spec | None:
    """Parse one dimension's tagged spec, reporting structural and value problems."""
    if not isinstance(raw_spec, Mapping):
        problems.append(
            f"{path}: spec must be a mapping with exactly one of "
            f"{_format_kinds()}, got {type(raw_spec).__name__}"
        )
        return None

    unknown = [key for key in raw_spec if key not in _SPEC_KIND_SET]
    if unknown:
        problems.append(f"{path}: unknown spec key(s): {_format_keys(unknown)}")

    tags = [tag for tag in _SPEC_KINDS if tag in raw_spec]
    if len(tags) != 1:
        problems.append(
            f"{path}: spec must contain exactly one of {_format_kinds()}, found {len(tags)}"
        )
        return None

    tag = tags[0]
    value = raw_spec[tag]
    if tag == "fixed":
        return _parse_fixed(path, value, kind, positive=positive, problems=problems)
    if tag == "categorical":
        return _parse_categorical(path, value, kind, positive=positive, problems=problems)
    if tag == "interval":
        return _parse_interval(path, value, kind, positive=positive, problems=problems)
    return _parse_default(path, value, problems=problems)


def _parse_fixed(
    path: str,
    value: object,
    kind: ParameterKind,
    *,
    positive: bool,
    problems: list[str],
) -> Fixed | None:
    """Parse a ``fixed`` spec, which pins the parameter to one value."""
    value_path = f"{path}.fixed"
    coerced = _coerce_value(value_path, value, kind, problems)
    if coerced is None:
        return None
    if positive and isinstance(coerced, int) and coerced < 1:
        problems.append(f"{value_path}: value must be at least 1, got {coerced}")
    return Fixed(coerced)


def _parse_categorical(
    path: str,
    value: object,
    kind: ParameterKind,
    *,
    positive: bool,
    problems: list[str],
) -> Categorical | None:
    """Parse a ``categorical`` spec, preserving the declared order of values.

    For a numeric parameter the values are integers, so the same kind,
    uniqueness, and floor checks apply as for ``fixed`` and ``interval``.
    """
    value_path = f"{path}.categorical"
    if not isinstance(value, list):
        problems.append(
            f"{value_path}: expected a non-empty list of values, got {type(value).__name__}"
        )
        return None
    if not value:
        problems.append(f"{value_path}: must not be empty")
        return None

    coerced: list[ParameterValue] = []
    for index, item in enumerate(value):
        item_path = f"{value_path}[{index}]"
        result = _coerce_value(item_path, item, kind, problems)
        if result is None:
            continue
        if positive and isinstance(result, int) and result < 1:
            problems.append(f"{item_path}: value must be at least 1, got {result}")
        coerced.append(result)

    if len(set(coerced)) != len(coerced):
        problems.append(f"{value_path}: values must be unique")
    if len(coerced) != len(value):
        return None
    return Categorical(tuple(coerced))


def _parse_interval(
    path: str,
    value: object,
    kind: ParameterKind,
    *,
    positive: bool,
    problems: list[str],
) -> Interval | None:
    """Parse an ``interval`` spec, valid only for an integer parameter."""
    value_path = f"{path}.interval"
    if issubclass(kind, StrEnum):
        problems.append(f"{value_path}: not valid for a categorical parameter")
        return None
    if not isinstance(value, Mapping):
        problems.append(
            f"{value_path}: expected a mapping with 'min' and 'max', got {type(value).__name__}"
        )
        return None

    unknown = [key for key in value if key not in _INTERVAL_KEYS]
    if unknown:
        problems.append(f"{value_path}: unknown key(s): {_format_keys(unknown)}")
    missing = [key for key in ("min", "max") if key not in value]
    if missing:
        problems.append(f"{value_path}: missing required key(s): {_format_keys(missing)}")
    if unknown or missing:
        return None

    minimum = _coerce_int(f"{value_path}.min", value["min"], problems)
    maximum = _coerce_int(f"{value_path}.max", value["max"], problems)
    if minimum is None or maximum is None:
        return None
    if minimum > maximum:
        problems.append(f"{value_path}: 'min' ({minimum}) must not exceed 'max' ({maximum})")
    if positive:
        if minimum < 1:
            problems.append(f"{value_path}.min: value must be at least 1, got {minimum}")
        if maximum < 1:
            problems.append(f"{value_path}.max: value must be at least 1, got {maximum}")
    return Interval(minimum, maximum)


def _parse_default(path: str, value: object, *, problems: list[str]) -> Default | None:
    """Parse a ``default`` spec, which must be written as exactly boolean true."""
    if value is not True:
        problems.append(f"{path}.default: must be exactly true, got {value!r}")
        return None
    return Default()


def _coerce_value(
    path: str, value: object, kind: ParameterKind, problems: list[str]
) -> ParameterValue | None:
    """Coerce a declared value to its parameter's engine-neutral kind."""
    if issubclass(kind, StrEnum):
        if isinstance(value, kind):
            return cast("ParameterValue", value)
        if not isinstance(value, str):
            problems.append(
                f"{path}: expected one of {_format_values(kind)}, got {type(value).__name__}"
            )
            return None
        try:
            return cast("ParameterValue", kind(value))
        except ValueError:
            problems.append(
                f"{path}: {value!r} is not a valid {kind.__name__}; "
                f"expected one of {_format_values(kind)}"
            )
            return None
    return _coerce_int(path, value, problems)


def _coerce_int(path: str, value: object, problems: list[str]) -> int | None:
    """Coerce a declared value to an integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{path}: expected an integer, got {type(value).__name__}")
        return None
    return value


def _format_kinds() -> str:
    """Render the recognized spec tags for an error message."""
    return ", ".join(_SPEC_KINDS)


def _format_keys(keys: Iterable[object]) -> str:
    """Render mapping keys for an error message."""
    return ", ".join(repr(key) for key in keys)


def _format_values(kind: type[StrEnum]) -> str:
    """Render the accepted string values of an engine-neutral enum."""
    return ", ".join(repr(member.value) for member in kind)
