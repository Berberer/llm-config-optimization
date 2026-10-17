"""Unit tests for the search-space types' immutability and hashability contract.

A search space is a reviewable artifact: it is logged, compared, and cached, so
every type in it must be immutable after construction and hashable. These tests
assert that contract directly — each type rejects attribute assignment with
``FrozenInstanceError`` and produces a hash — that a list input is coerced to a
tuple so a caller cannot smuggle in a mutable sequence, and that categorical
order is preserved rather than sorted or deduplicated. They mirror the analogous
inference-layer test, ``tests/inference/test_types.py``, and need no daemon.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from llm_config_optimization.inference import DType
from llm_config_optimization.search_space import (
    TUNABLE_PARAMETERS,
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    parse_experiment_space,
)

SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("kv_cache_dtype", Default()),
    )
)
"""A space whose every spec kind is represented and every value is hashable."""


@pytest.mark.parametrize(
    ("value", "attribute", "replacement"),
    [
        (Fixed(DType.FLOAT16), "value", DType.BFLOAT16),
        (Categorical((DType.FLOAT16,)), "values", (DType.BFLOAT16,)),
        (Interval(1024, 8192), "minimum", 2048),
        (Interval(1024, 8192), "maximum", 4096),
        (Default(), "value", 1),
        (Dimension("dtype", Fixed(DType.FLOAT16)), "name", "quantization"),
        (Dimension("dtype", Fixed(DType.FLOAT16)), "spec", Default()),
        (SPACE, "dimensions", ()),
    ],
    ids=[
        "fixed",
        "categorical",
        "interval-minimum",
        "interval-maximum",
        "default",
        "dimension-name",
        "dimension-spec",
        "space",
    ],
)
def test_types_are_frozen_and_hashable(value: object, attribute: str, replacement: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, replacement)
    assert isinstance(hash(value), int)


def test_list_inputs_are_coerced_to_immutable_hashable_tuples() -> None:
    categorical = Categorical([DType.FLOAT16, DType.BFLOAT16])
    space = ExperimentSpace([Dimension("dtype", Fixed(DType.FLOAT16))])

    assert categorical.values == (DType.FLOAT16, DType.BFLOAT16)
    assert space.dimensions == (Dimension("dtype", Fixed(DType.FLOAT16)),)
    assert isinstance(hash(categorical), int)
    assert isinstance(hash(space), int)


def test_equal_values_hash_equally() -> None:
    first = Categorical((DType.FLOAT16, DType.BFLOAT16))
    second = Categorical((DType.FLOAT16, DType.BFLOAT16))

    assert first == second
    assert hash(first) == hash(second)


def test_replace_yields_a_distinct_hash() -> None:
    varied = replace(SPACE, dimensions=(Dimension("dtype", Fixed(DType.BFLOAT16)),))

    assert varied != SPACE
    assert hash(varied) != hash(SPACE)


def test_categorical_preserves_declared_order() -> None:
    values = (DType.BFLOAT16, DType.FLOAT16, DType.AUTO)

    assert Categorical(values).values == values


def test_parsing_preserves_categorical_order() -> None:
    declared = ["bfloat16", "float16", "auto"]
    space = parse_experiment_space(
        {
            "space": {
                **{parameter: {"default": True} for parameter in TUNABLE_PARAMETERS},
                "dtype": {"categorical": declared},
            }
        }
    )

    dtype = next(dimension for dimension in space.dimensions if dimension.name == "dtype")

    assert dtype.spec == Categorical((DType.BFLOAT16, DType.FLOAT16, DType.AUTO))
