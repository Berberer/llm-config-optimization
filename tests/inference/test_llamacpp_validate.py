"""Unit tests for llama.cpp engine-support validation of a search space.

:meth:`LlamaCppBackend.validate`'s observable contract is the problems it
returns for a definition: whether every declared parameter is one llama.cpp
forwards and whether every enum value has a llama.cpp spelling. The definition
types are crafted directly, so no file parser, engine, or daemon is involved.
The tests assert on the parameter path and the kind of problem rather than on
exact wording, so messages can be reworded freely.
"""

from types import MappingProxyType

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DetectedHardware,
    DType,
    HardwareTarget,
    KVCacheDType,
    LlamaCppBackend,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
)

APPLICABLE = frozenset(
    {
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "attention_backend",
    }
)
"""Exactly the tunables llama.cpp has a ``llama-server`` flag for."""

SUPPORTED_SPACE = ExperimentSpace(
    dimensions=(
        Dimension("kv_cache_dtype", Default()),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(4)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION)),
    )
)
"""A space whose every parameter and value llama.cpp supports."""


def space(*dimensions: Dimension) -> ExperimentSpace:
    """Assemble a space from explicit dimensions, bypassing the file parser."""
    return ExperimentSpace(dimensions=dimensions)


def test_applicable_parameters_are_exactly_the_five_llama_cpp_forwards() -> None:
    """``dtype`` and ``tensor_parallel_size`` are absent from the set."""
    assert LlamaCppBackend().applicable_parameters == APPLICABLE


def test_a_fully_supported_space_has_no_problems() -> None:
    assert list(LlamaCppBackend().validate(SUPPORTED_SPACE)) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(4)),
        Dimension("max_num_batched_tokens", Default()),
        Dimension("kv_cache_dtype", Default()),
        Dimension("kv_cache_dtype", Fixed(KVCacheDType.FLOAT16)),
        Dimension("attention_backend", Categorical((AttentionBackend.FLASH_ATTENTION,))),
    ],
    ids=[
        "interval",
        "fixed-int",
        "default-int",
        "default-enum",
        "fixed-enum",
        "categorical-enum",
    ],
)
def test_each_spec_kind_is_accepted_for_a_supported_parameter(dimension: Dimension) -> None:
    assert list(LlamaCppBackend().validate(space(dimension))) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("tensor_parallel_size", Fixed(2)),
        Dimension("dtype", Default()),
    ],
    ids=["dtype-fixed", "tensor-parallel-fixed", "dtype-default"],
)
def test_a_parameter_llama_cpp_does_not_forward_is_reported(dimension: Dimension) -> None:
    """A deferral skips the value check, not the parameter check."""
    problems = LlamaCppBackend().validate(space(dimension))

    assert any(
        f"space.{dimension.name}" in problem and "does not forward" in problem
        for problem in problems
    )


@pytest.mark.parametrize(
    "value",
    [KVCacheDType.AUTO, KVCacheDType.FP8_E4M3, KVCacheDType.FP8_E5M2],
)
def test_an_unsupported_kv_cache_value_is_reported(value: KVCacheDType) -> None:
    problems = LlamaCppBackend().validate(space(Dimension("kv_cache_dtype", Fixed(value))))

    assert any(problem.startswith("space.kv_cache_dtype.fixed") for problem in problems), problems


@pytest.mark.parametrize(
    "value",
    [AttentionBackend.FLASH_INFER, AttentionBackend.XFORMERS],
)
def test_an_unsupported_attention_backend_is_reported(value: AttentionBackend) -> None:
    problems = LlamaCppBackend().validate(space(Dimension("attention_backend", Fixed(value))))

    assert any(problem.startswith("space.attention_backend.fixed") for problem in problems)


def test_an_unsupported_categorical_value_is_reported_at_its_position() -> None:
    dimension = Dimension(
        "kv_cache_dtype", Categorical((KVCacheDType.FLOAT16, KVCacheDType.FP8_E4M3))
    )

    problems = LlamaCppBackend().validate(space(dimension))

    assert any(problem.startswith("space.kv_cache_dtype.categorical[1]") for problem in problems), (
        problems
    )


def test_supported_categorical_values_are_accepted() -> None:
    dimension = Dimension(
        "kv_cache_dtype", Categorical((KVCacheDType.FLOAT16, KVCacheDType.BFLOAT16))
    )

    assert list(LlamaCppBackend().validate(space(dimension))) == []


def test_validation_accepts_hardware_without_narrowing_the_modelled_values() -> None:
    """No modelled llama.cpp value is accelerator-specific today."""
    problems = LlamaCppBackend().validate(SUPPORTED_SPACE, DetectedHardware(HardwareTarget.CPU, 1))

    assert list(problems) == []


def test_a_parameter_absent_from_the_value_maps_rejects_every_enum_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field with no curated spelling cannot translate any member."""
    monkeypatch.setattr(
        "llm_config_optimization.inference._llamacpp.VALUE_MAPS", MappingProxyType({})
    )

    problems = LlamaCppBackend().validate(
        space(Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION)))
    )

    assert any(problem.startswith("space.attention_backend.fixed") for problem in problems)
