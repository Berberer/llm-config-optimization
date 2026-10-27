"""Unit tests for llama.cpp engine-support validation of a search space.

:meth:`LlamaCppBackend.validate`'s observable contract is the problems it
returns for a definition: whether every declared parameter is one llama.cpp
forwards and whether every enum value has a llama.cpp spelling. The definition
types are crafted directly, so no file parser, engine, or daemon is involved.
The tests assert on the parameter path and the kind of problem rather than on
exact wording, so messages can be reworded freely.
"""

from collections.abc import Sequence
from types import MappingProxyType

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DetectedHardware,
    DType,
    GpuLayersMode,
    HardwareTarget,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    SpeculativeMethod,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
)
from llm_config_optimization.search_space.registry import SPECULATIVE_PARAMETERS

APPLICABLE = frozenset(
    {
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "attention_backend",
        "prefix_caching",
        "n_gpu_layers",
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
        Dimension("prefix_caching", Fixed(True)),
        Dimension("n_gpu_layers", Fixed(GpuLayersMode.AUTO)),
    )
)
"""A space whose every parameter and value llama.cpp supports."""


def space(*dimensions: Dimension) -> ExperimentSpace:
    """Assemble a space from explicit dimensions, bypassing the file parser."""
    return ExperimentSpace(dimensions=dimensions)


def validate(
    definition: ExperimentSpace, hardware: DetectedHardware | None = None
) -> Sequence[str]:
    """Validate ``definition`` against llama.cpp with a configuration that has no setup."""
    return LlamaCppBackend().validate(definition, LLMConfig(), hardware)


def test_applicable_parameters_are_exactly_the_tunables_llama_cpp_forwards() -> None:
    """``dtype``, the scheduling caps, and ``tensor_parallel_size`` are absent."""
    assert LlamaCppBackend().applicable_parameters(LLMConfig()) == APPLICABLE


@pytest.mark.parametrize(
    "llm",
    [
        LLMConfig(speculative_method=SpeculativeMethod.MTP),
        LLMConfig(draft_model="org/draft.gguf"),
    ],
    ids=["method", "draft-model"],
)
def test_a_configured_speculative_setup_makes_the_gated_tunables_applicable(
    llm: LLMConfig,
) -> None:
    assert LlamaCppBackend().applicable_parameters(llm) == APPLICABLE | SPECULATIVE_PARAMETERS


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("speculative_mode", Fixed("enabled")),
        Dimension("num_speculative_tokens", Fixed(4)),
    ],
    ids=["mode", "depth"],
)
def test_a_gated_tunable_without_a_configured_setup_is_reported(
    dimension: Dimension,
) -> None:
    problems = validate(space(dimension))

    assert any(
        f"space.{dimension.name}" in problem and "does not forward" in problem
        for problem in problems
    ), problems


def test_a_fully_supported_space_has_no_problems() -> None:
    assert list(validate(SUPPORTED_SPACE)) == []


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
    assert list(validate(space(dimension))) == []


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
    problems = validate(space(dimension))

    assert any(
        f"space.{dimension.name}" in problem and "does not forward" in problem
        for problem in problems
    )


@pytest.mark.parametrize(
    "value",
    [KVCacheDType.AUTO, KVCacheDType.FP8_E4M3, KVCacheDType.FP8_E5M2],
)
def test_an_unsupported_kv_cache_value_is_reported(value: KVCacheDType) -> None:
    problems = validate(space(Dimension("kv_cache_dtype", Fixed(value))))

    assert any(problem.startswith("space.kv_cache_dtype.fixed") for problem in problems), problems


@pytest.mark.parametrize(
    "value",
    [AttentionBackend.FLASH_INFER, AttentionBackend.XFORMERS],
)
def test_an_unsupported_attention_backend_is_reported(value: AttentionBackend) -> None:
    problems = validate(space(Dimension("attention_backend", Fixed(value))))

    assert any(problem.startswith("space.attention_backend.fixed") for problem in problems)


@pytest.mark.parametrize(
    "value",
    [AttentionBackend.AUTO, AttentionBackend.FLASH_ATTENTION, AttentionBackend.OFF],
)
def test_supported_attention_backends_are_accepted(value: AttentionBackend) -> None:
    assert list(validate(space(Dimension("attention_backend", Fixed(value))))) == []


def test_an_unsupported_categorical_value_is_reported_at_its_position() -> None:
    dimension = Dimension(
        "kv_cache_dtype", Categorical((KVCacheDType.FLOAT16, KVCacheDType.FP8_E4M3))
    )

    problems = validate(space(dimension))

    assert any(problem.startswith("space.kv_cache_dtype.categorical[1]") for problem in problems), (
        problems
    )


@pytest.mark.parametrize(
    "value",
    [
        KVCacheDType.FLOAT16,
        KVCacheDType.BFLOAT16,
        KVCacheDType.FLOAT32,
        KVCacheDType.Q8_0,
        KVCacheDType.Q4_0,
        KVCacheDType.Q4_1,
        KVCacheDType.IQ4_NL,
        KVCacheDType.Q5_0,
        KVCacheDType.Q5_1,
    ],
)
def test_supported_kv_cache_values_are_accepted(value: KVCacheDType) -> None:
    assert list(validate(space(Dimension("kv_cache_dtype", Fixed(value))))) == []


def test_supported_categorical_values_are_accepted() -> None:
    dimension = Dimension(
        "kv_cache_dtype",
        Categorical((KVCacheDType.FLOAT16, KVCacheDType.BFLOAT16, KVCacheDType.Q4_0)),
    )

    assert list(validate(space(dimension))) == []


@pytest.mark.parametrize(
    "cache",
    [
        Fixed(KVCacheDType.Q5_0),
        Categorical((KVCacheDType.FLOAT16, KVCacheDType.Q4_0)),
    ],
    ids=["fixed", "categorical"],
)
def test_quantized_cache_with_flash_attention_off_is_reported(cache: Fixed | Categorical) -> None:
    problems = validate(
        space(
            Dimension("kv_cache_dtype", cache),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        )
    )

    assert any(
        problem.startswith("space.attention_backend")
        and "requires flash attention" in problem
        and "quantized" in problem
        for problem in problems
    ), problems


def test_a_categorical_attention_dimension_including_off_is_reported() -> None:
    problems = validate(
        space(
            Dimension("kv_cache_dtype", Categorical((KVCacheDType.Q8_0,))),
            Dimension(
                "attention_backend",
                Categorical(
                    (AttentionBackend.AUTO, AttentionBackend.FLASH_ATTENTION, AttentionBackend.OFF)
                ),
            ),
        )
    )

    assert any(
        problem.startswith("space.attention_backend.categorical[2]")
        and "requires flash attention" in problem
        for problem in problems
    ), problems


@pytest.mark.parametrize(
    "cache",
    [
        Fixed(KVCacheDType.FLOAT16),
        Fixed(KVCacheDType.BFLOAT16),
        Fixed(KVCacheDType.FLOAT32),
    ],
    ids=["float16", "bfloat16", "float32"],
)
def test_float_cache_with_flash_attention_off_is_accepted(cache: Fixed) -> None:
    problems = validate(
        space(
            Dimension("kv_cache_dtype", cache),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        )
    )

    assert list(problems) == []


@pytest.mark.parametrize(
    "attention",
    [AttentionBackend.AUTO, AttentionBackend.FLASH_ATTENTION],
    ids=["auto", "flash-attention"],
)
def test_quantized_cache_with_flash_or_auto_attention_is_accepted(
    attention: AttentionBackend,
) -> None:
    problems = validate(
        space(
            Dimension("kv_cache_dtype", Fixed(KVCacheDType.Q4_0)),
            Dimension("attention_backend", Fixed(attention)),
        )
    )

    assert list(problems) == []


@pytest.mark.parametrize(
    "dimensions",
    [
        (
            Dimension("kv_cache_dtype", Default()),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        ),
        (
            Dimension("kv_cache_dtype", Fixed(KVCacheDType.Q5_0)),
            Dimension("attention_backend", Default()),
        ),
    ],
    ids=["default-cache", "default-attention"],
)
def test_a_default_dimension_does_not_trigger_the_conflict(
    dimensions: tuple[Dimension, Dimension],
) -> None:
    assert list(validate(space(*dimensions))) == []


def test_a_fully_supported_space_with_auto_layers_is_accepted_on_a_cpu_target() -> None:
    problems = validate(SUPPORTED_SPACE, DetectedHardware(HardwareTarget.CPU, 1))

    assert list(problems) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("n_gpu_layers", Fixed(GpuLayersMode.ALL)),
        Dimension("n_gpu_layers", Fixed(8)),
        Dimension("n_gpu_layers", Categorical((GpuLayersMode.AUTO, 0, 8))),
        Dimension("n_gpu_layers", Interval(0, 8)),
    ],
    ids=["all", "positive-count", "mixed-categorical", "positive-interval"],
)
def test_gpu_offload_is_rejected_on_a_cpu_target(dimension: Dimension) -> None:
    problems = validate(space(dimension), DetectedHardware(HardwareTarget.CPU, 1))

    assert any(
        problem.startswith(f"space.{dimension.name}") and "CPU target" in problem
        for problem in problems
    ), problems


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("n_gpu_layers", Fixed(GpuLayersMode.AUTO)),
        Dimension("n_gpu_layers", Fixed(0)),
        Dimension("n_gpu_layers", Categorical((GpuLayersMode.AUTO, 0))),
        Dimension("n_gpu_layers", Interval(0, 0)),
        Dimension("n_gpu_layers", Default()),
    ],
    ids=["auto", "zero", "mixed-categorical", "zero-interval", "default"],
)
def test_host_agnostic_gpu_offload_values_are_accepted_on_a_cpu_target(
    dimension: Dimension,
) -> None:
    problems = validate(space(dimension), DetectedHardware(HardwareTarget.CPU, 1))

    assert list(problems) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("n_gpu_layers", Fixed(GpuLayersMode.ALL)),
        Dimension("n_gpu_layers", Fixed(8)),
        Dimension("n_gpu_layers", Categorical((GpuLayersMode.AUTO, 0, 8))),
        Dimension("n_gpu_layers", Interval(0, 8)),
    ],
    ids=["all", "positive-count", "mixed-categorical", "positive-interval"],
)
def test_gpu_offload_is_accepted_on_a_gpu_target(dimension: Dimension) -> None:
    problems = validate(space(dimension), DetectedHardware(HardwareTarget.CUDA, 1))

    assert list(problems) == []


def test_a_parameter_absent_from_the_value_maps_rejects_every_enum_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field with no curated spelling cannot translate any member."""
    monkeypatch.setattr(
        "llm_config_optimization.inference._llamacpp.VALUE_MAPS", MappingProxyType({})
    )

    problems = validate(
        space(Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION)))
    )

    assert any(problem.startswith("space.attention_backend.fixed") for problem in problems)
