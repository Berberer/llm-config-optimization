"""Unit tests for the shared tunable translation and validation mechanics.

Both backends build on the same helpers, so this module pins the generic
contract directly: how a declared dimension contributes problem paths, how an
engine-neutral enum is translated through a backend's map, and how an
unspellable member is reported. The engine maps themselves stay in the backend
tests; these tests use small inline maps.
"""

from types import MappingProxyType

import pytest

from llm_config_optimization.inference import DType, ExperimentError, KVCacheDType
from llm_config_optimization.inference._tunables import (
    Toggle,
    declared_values,
    toggle_arguments,
    translate_value,
    unsupported_values,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    Fixed,
    Interval,
)

_DTYPE_MAP = MappingProxyType({"dtype": MappingProxyType({DType.FLOAT16: "float16"})})


def test_a_mapped_enum_member_is_translated() -> None:
    assert translate_value("dtype", DType.FLOAT16, _DTYPE_MAP, engine="engine") == "float16"


def test_a_toggle_emits_the_enabling_flag_for_true() -> None:
    target = Toggle(enabled="--enable", disabled="--disable")

    assert toggle_arguments(True, target) == ["--enable"]


def test_a_toggle_emits_the_disabling_flag_for_false() -> None:
    """``False`` is a real configuration, not an omitted default."""
    target = Toggle(enabled="--enable", disabled="--disable")

    assert toggle_arguments(False, target) == ["--disable"]


def test_a_field_without_a_map_passes_its_value_through() -> None:
    assert translate_value("max_model_len", 4096, _DTYPE_MAP, engine="engine") == 4096


def test_a_non_enum_value_passes_through_a_mapped_field() -> None:
    assert translate_value("dtype", "float16", _DTYPE_MAP, engine="engine") == "float16"


def test_an_unmapped_enum_member_raises_with_the_engines_name() -> None:
    with pytest.raises(ExperimentError) as excinfo:
        translate_value("dtype", DType.BFLOAT16, _DTYPE_MAP, engine="test-engine")

    message = str(excinfo.value)
    assert "test-engine" in message
    assert "dtype='bfloat16'" in message


def test_a_fixed_dimension_declares_one_value_at_its_definition_path() -> None:
    dimension = Dimension("dtype", Fixed(DType.FLOAT16))

    assert declared_values(dimension) == [("space.dtype.fixed", DType.FLOAT16)]


def test_a_categorical_dimension_declares_each_value_at_its_index() -> None:
    dimension = Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16)))

    assert declared_values(dimension) == [
        ("space.dtype.categorical[0]", DType.FLOAT16),
        ("space.dtype.categorical[1]", DType.BFLOAT16),
    ]


@pytest.mark.parametrize("spec", [Default(), Interval(1, 4)], ids=["default", "interval"])
def test_a_deferral_or_interval_declares_no_values(spec: Default | Interval) -> None:
    assert declared_values(Dimension("num_speculative_tokens", spec)) == []


def test_a_spelled_value_has_no_problem() -> None:
    dimension = Dimension("dtype", Fixed(DType.FLOAT16))

    assert unsupported_values(dimension, "dtype", _DTYPE_MAP, engine="engine") == []


def test_an_unspellable_value_is_reported_at_its_path() -> None:
    dimension = Dimension("kv_cache_dtype", Categorical((KVCacheDType.FLOAT16, KVCacheDType.AUTO)))

    problems = unsupported_values(
        dimension,
        "kv_cache_dtype",
        MappingProxyType({"kv_cache_dtype": MappingProxyType({KVCacheDType.FLOAT16: "f16"})}),
        engine="test-engine",
    )

    assert problems == [
        "space.kv_cache_dtype.categorical[1]: test-engine does not support kv_cache_dtype='auto'"
    ]


def test_a_field_without_a_map_rejects_every_enum_member() -> None:
    dimension = Dimension("dtype", Fixed(DType.FLOAT16))

    assert unsupported_values(dimension, "dtype", {}, engine="engine") == [
        "space.dtype.fixed: engine does not support dtype='float16'"
    ]


def test_a_non_enum_value_skips_the_value_map_check() -> None:
    dimension = Dimension("dtype", Fixed("float16"))

    assert unsupported_values(dimension, "dtype", {}, engine="engine") == []
