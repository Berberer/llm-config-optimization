"""Unit tests for the shared tunable translation and validation mechanics.

Both backends build on the same helpers, so this module pins the generic
contract directly: how a declared dimension contributes problem paths, how an
engine-neutral enum is translated through a backend's map, how an unspellable
member is reported, and how a model-aware assessment derives the effective
space — dropping dimensions and narrowing declared values while preserving
their order. The engine maps and pruning rules themselves stay in the backend
tests; these tests use small inline maps and spaces.
"""

from types import MappingProxyType

import pytest

from llm_config_optimization.inference import DType, ExperimentError, KVCacheDType
from llm_config_optimization.inference._tunables import (
    Toggle,
    declared_values,
    narrow_values,
    prune_dimensions,
    toggle_arguments,
    translate_value,
    unsupported_values,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    Spec,
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


def test_a_one_sided_enabling_flag_emits_nothing_for_false() -> None:
    """A toggle with only an enabling flag keeps the engine's default off."""
    target = Toggle(enabled="--swa-full")

    assert toggle_arguments(True, target) == ["--swa-full"]
    assert toggle_arguments(False, target) == []


def test_a_one_sided_disabling_flag_emits_nothing_for_true() -> None:
    """A toggle with only a disabling flag keeps the engine's default on."""
    target = Toggle(disabled="--no-host")

    assert toggle_arguments(True, target) == []
    assert toggle_arguments(False, target) == ["--no-host"]


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


def _space(*dimensions: Dimension) -> ExperimentSpace:
    """Assemble a space from explicit dimensions, bypassing the file parser."""
    return ExperimentSpace(dimensions=dimensions)


def test_pruning_removes_only_the_named_dimensions_in_canonical_order() -> None:
    definition = _space(
        Dimension("max_num_seqs", Fixed(4)),
        Dimension("expert_parallel", Fixed(True)),
        Dimension("all2all_backend", Default()),
        Dimension("prefix_caching", Fixed(True)),
    )

    effective, notes = prune_dimensions(
        definition, {"expert_parallel", "all2all_backend"}, reason="the model has no experts"
    )

    assert [dimension.name for dimension in effective.dimensions] == [
        "max_num_seqs",
        "prefix_caching",
    ]
    assert notes == (
        "dropped space.expert_parallel: the model has no experts",
        "dropped space.all2all_backend: the model has no experts",
    )


def test_pruning_a_parameter_the_space_omits_changes_nothing() -> None:
    definition = _space(Dimension("max_num_seqs", Fixed(4)))

    effective, notes = prune_dimensions(definition, {"n_cpu_moe"}, reason="irrelevant")

    assert effective is definition
    assert notes == ()


def test_pruning_can_leave_the_effective_space_empty() -> None:
    definition = _space(Dimension("n_cpu_moe", Fixed(0)))

    effective, notes = prune_dimensions(definition, {"n_cpu_moe"}, reason="irrelevant")

    assert effective.dimensions == ()
    assert len(notes) == 1


def _reject_float32(value: object) -> str | None:
    """Stand in for a model-aware rule that rejects one dtype value."""
    return "the checkpoint's kernels reject it" if value is DType.FLOAT32 else None


def test_narrowing_keeps_the_retained_values_in_declared_order() -> None:
    definition = _space(
        Dimension("dtype", Categorical((DType.BFLOAT16, DType.FLOAT32, DType.FLOAT16))),
        Dimension("prefix_caching", Fixed(True)),
    )

    narrowed = narrow_values(definition, "dtype", _reject_float32)

    assert narrowed.space.dimensions == (
        Dimension("dtype", Categorical((DType.BFLOAT16, DType.FLOAT16))),
        Dimension("prefix_caching", Fixed(True)),
    )
    assert narrowed.notes == (
        "dropped space.dtype.categorical[1] (float32): the checkpoint's kernels reject it",
    )
    assert narrowed.emptied is False


def test_narrowing_a_fixed_dimension_to_nothing_reports_it_emptied() -> None:
    definition = _space(Dimension("dtype", Fixed(DType.FLOAT32)))

    narrowed = narrow_values(definition, "dtype", _reject_float32)

    assert narrowed.space is definition
    assert narrowed.notes == ()
    assert narrowed.emptied is True


@pytest.mark.parametrize("spec", [Interval(1, 2), Default()], ids=["interval", "default"])
def test_a_value_less_dimension_is_never_narrowed(spec: Spec) -> None:
    definition = _space(Dimension("max_num_seqs", spec))

    narrowed = narrow_values(definition, "max_num_seqs", _reject_float32)

    assert narrowed.space is definition
    assert narrowed.notes == ()
    assert narrowed.emptied is False


def test_narrowing_a_parameter_the_space_omits_changes_nothing() -> None:
    definition = _space(Dimension("max_num_seqs", Fixed(4)))

    narrowed = narrow_values(definition, "dtype", _reject_float32)

    assert narrowed.space is definition
    assert narrowed.notes == ()
    assert narrowed.emptied is False


def test_narrowing_that_rejects_nothing_changes_nothing() -> None:
    definition = _space(Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16))))

    narrowed = narrow_values(definition, "dtype", lambda value: None)

    assert narrowed.space is definition
    assert narrowed.notes == ()
    assert narrowed.emptied is False
