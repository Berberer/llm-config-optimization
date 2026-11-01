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
    CpuMoeMode,
    DetectedHardware,
    DType,
    GpuLayersMode,
    HardwareTarget,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    LoadMode,
    SpeculativeMethod,
)
from llm_config_optimization.inference._llamacpp import PARAMETER_TARGETS
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    Spec,
)
from llm_config_optimization.search_space.registry import (
    DRAFT_MODEL_PARAMETERS,
    SPECULATIVE_PARAMETERS,
)

APPLICABLE = frozenset(
    {
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "ubatch_size",
        "attention_backend",
        "prefix_caching",
        "n_gpu_layers",
        "n_cpu_moe",
        "n_cpu_ffn",
        "kv_offload",
        "op_offload",
        "host_buffer",
        "repack",
        "swa_full",
        "load_mode",
        "cache_ram",
        "ctx_checkpoints",
        "checkpoint_min_step",
        "kv_unified",
        "continuous_batching",
        "cache_idle_slots",
        "threads",
        "threads_batch",
        "poll",
        "prio",
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
        Dimension("n_cpu_moe", Categorical((0, 8, CpuMoeMode.ALL))),
        Dimension("n_cpu_ffn", Fixed(8)),
        Dimension("kv_offload", Fixed(True)),
        Dimension("op_offload", Fixed(False)),
        Dimension("host_buffer", Fixed(False)),
        Dimension("repack", Fixed(True)),
        Dimension("swa_full", Fixed(True)),
        Dimension("load_mode", Fixed(LoadMode.MMAP)),
        Dimension("cache_ram", Fixed(1024)),
        Dimension("ctx_checkpoints", Fixed(4)),
        Dimension("checkpoint_min_step", Fixed(0)),
        Dimension("kv_unified", Fixed(True)),
        Dimension("continuous_batching", Fixed(True)),
        Dimension("cache_idle_slots", Fixed(False)),
        Dimension("threads", Fixed(8)),
        Dimension("threads_batch", Fixed(4)),
        Dimension("poll", Fixed(50)),
        Dimension("prio", Fixed(2)),
    )
)
"""A space whose every parameter and value llama.cpp supports."""


def space(*dimensions: Dimension) -> ExperimentSpace:
    """Assemble a space from explicit dimensions, bypassing the file parser."""
    return ExperimentSpace(dimensions=dimensions)


def validate(
    definition: ExperimentSpace,
    hardware: DetectedHardware | None = None,
    llm: LLMConfig | None = None,
) -> Sequence[str]:
    """Validate ``definition`` with no setup, or with the supplied configuration."""
    return LlamaCppBackend().validate(definition, LLMConfig() if llm is None else llm, hardware)


DRAFT_LLM = LLMConfig(draft_model="org/draft.gguf")
"""The configuration that makes the draft-only tunables applicable."""


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
    expected = APPLICABLE | SPECULATIVE_PARAMETERS
    if llm.draft_model is not None:
        expected |= DRAFT_MODEL_PARAMETERS & frozenset(PARAMETER_TARGETS)

    assert LlamaCppBackend().applicable_parameters(llm) == expected


def test_draft_only_tunables_are_withheld_without_a_draft() -> None:
    """No draft-only dimension is required when the run names no draft model."""
    assert DRAFT_MODEL_PARAMETERS.isdisjoint(LlamaCppBackend().applicable_parameters(LLMConfig()))


def test_a_draft_makes_only_the_forwarded_draft_tunables_applicable() -> None:
    """The gate adds exactly the draft knobs llama.cpp has a flag for."""
    forwarded = frozenset(DRAFT_MODEL_PARAMETERS & frozenset(PARAMETER_TARGETS))
    llm = LLMConfig(draft_model="org/draft.gguf")

    assert LlamaCppBackend().applicable_parameters(llm) == (
        APPLICABLE | SPECULATIVE_PARAMETERS | forwarded
    )


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
    ("field", "value"),
    [
        ("poll", 0),
        ("poll", 100),
        ("prio", -1),
        ("prio", 3),
        ("cache_ram", -1),
        ("cache_ram", 0),
        ("cache_ram", 4096),
    ],
    ids=[
        "poll-floor",
        "poll-ceiling",
        "prio-floor",
        "prio-ceiling",
        "cache-ram-no-limit",
        "cache-ram-off",
        "cache-ram-limit",
    ],
)
def test_a_numeric_value_on_its_permitted_boundary_is_accepted(field: str, value: int) -> None:
    assert list(validate(space(Dimension(field, Fixed(value))))) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("poll", -1),
        ("poll", 101),
        ("prio", -2),
        ("prio", 4),
        ("cache_ram", -2),
    ],
    ids=["poll-below", "poll-above", "prio-below", "prio-above", "cache-ram-below"],
)
def test_a_numeric_value_outside_its_permitted_range_is_reported(field: str, value: int) -> None:
    problems = validate(space(Dimension(field, Fixed(value))))

    assert any(
        problem.startswith(f"space.{field}.fixed") and str(value) in problem for problem in problems
    ), problems


def test_a_two_sided_problem_names_the_offending_value_and_permitted_range() -> None:
    problems = validate(space(Dimension("poll", Fixed(101))))

    assert any("101" in problem and "0..100" in problem for problem in problems), problems


def test_a_one_sided_problem_names_its_lower_bound() -> None:
    problems = validate(space(Dimension("cache_ram", Fixed(-2))))

    assert any("-2" in problem and ">= -1" in problem for problem in problems), problems


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("poll", Interval(0, 101)),
        Dimension("prio", Interval(-2, 3)),
        Dimension("cache_ram", Interval(-2, 4096)),
    ],
    ids=["poll-above", "prio-below", "cache-ram-below"],
)
def test_an_interval_reaching_outside_the_permitted_range_is_reported(
    dimension: Dimension,
) -> None:
    problems = validate(space(dimension))

    assert any(problem.startswith(f"space.{dimension.name}.interval") for problem in problems), (
        problems
    )


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("poll", Interval(0, 100)),
        Dimension("prio", Interval(-1, 3)),
        Dimension("cache_ram", Interval(-1, 4096)),
    ],
    ids=["poll", "prio", "cache-ram"],
)
def test_an_interval_within_the_permitted_range_is_accepted(dimension: Dimension) -> None:
    assert list(validate(space(dimension))) == []


def test_a_categorical_value_outside_its_permitted_range_is_reported_at_its_position() -> None:
    problems = validate(space(Dimension("prio", Categorical((-2, 0, 4)))))

    assert any(problem.startswith("space.prio.categorical[0]") for problem in problems), problems
    assert any(problem.startswith("space.prio.categorical[2]") for problem in problems), problems


@pytest.mark.parametrize("field", ["poll", "prio", "cache_ram"])
def test_a_default_numeric_dimension_defers_to_the_engine(field: str) -> None:
    assert list(validate(space(Dimension(field, Default())))) == []


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0], ids=["zero", "half", "one"])
def test_a_split_probability_on_its_permitted_boundary_is_accepted(value: float) -> None:
    definition = space(Dimension("spec_draft_p_split", Fixed(value)))

    assert list(validate(definition, llm=DRAFT_LLM)) == []


@pytest.mark.parametrize("value", [-0.1, 1.1], ids=["below", "above"])
def test_a_split_probability_outside_its_permitted_range_is_reported(value: float) -> None:
    definition = space(Dimension("spec_draft_p_split", Fixed(value)))

    problems = validate(definition, llm=DRAFT_LLM)

    assert any(
        problem.startswith("space.spec_draft_p_split.fixed") and str(value) in problem
        for problem in problems
    ), problems
    assert any("0.0..1.0" in problem for problem in problems), problems


def test_a_split_probability_interval_reaching_outside_its_range_is_reported() -> None:
    definition = space(Dimension("spec_draft_p_split", Interval(0.0, 1.5)))

    problems = validate(definition, llm=DRAFT_LLM)

    assert any(problem.startswith("space.spec_draft_p_split.interval") for problem in problems), (
        problems
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
    "spec",
    [
        Fixed(0),
        Fixed(8),
        Fixed(CpuMoeMode.ALL),
        Categorical((0, 8, CpuMoeMode.ALL)),
        Interval(0, 8),
        Default(),
    ],
    ids=["zero", "count", "all-mode", "mixed-categorical", "interval", "default"],
)
def test_every_cpu_expert_offload_spec_is_accepted(spec: Spec) -> None:
    """Both union branches are forwardable, so every spec kind is supported."""
    assert list(validate(space(Dimension("n_cpu_moe", spec)))) == []


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


@pytest.mark.parametrize(
    "cache",
    [
        Fixed(KVCacheDType.Q5_0),
        Categorical((KVCacheDType.FLOAT16, KVCacheDType.Q4_0)),
    ],
    ids=["fixed", "categorical"],
)
def test_quantized_draft_cache_with_flash_attention_off_is_reported(
    cache: Fixed | Categorical,
) -> None:
    problems = validate(
        space(
            Dimension("draft_kv_cache_dtype", cache),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        ),
        llm=DRAFT_LLM,
    )

    assert any(
        problem.startswith("space.attention_backend")
        and "requires flash attention" in problem
        and "draft" in problem
        and "draft_kv_cache_dtype" in problem
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
def test_float_draft_cache_with_flash_attention_off_is_accepted(cache: Fixed) -> None:
    problems = validate(
        space(
            Dimension("draft_kv_cache_dtype", cache),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        ),
        llm=DRAFT_LLM,
    )

    assert list(problems) == []


@pytest.mark.parametrize(
    "attention",
    [AttentionBackend.AUTO, AttentionBackend.FLASH_ATTENTION],
    ids=["auto", "flash-attention"],
)
def test_quantized_draft_cache_with_flash_or_auto_attention_is_accepted(
    attention: AttentionBackend,
) -> None:
    problems = validate(
        space(
            Dimension("draft_kv_cache_dtype", Fixed(KVCacheDType.Q4_0)),
            Dimension("attention_backend", Fixed(attention)),
        ),
        llm=DRAFT_LLM,
    )

    assert list(problems) == []


def test_quantized_draft_cache_without_a_draft_is_only_a_support_problem() -> None:
    """A run without a draft never builds the draft cache, so the pair cannot fail."""
    problems = validate(
        space(
            Dimension("draft_kv_cache_dtype", Fixed(KVCacheDType.Q5_0)),
            Dimension("attention_backend", Fixed(AttentionBackend.OFF)),
        )
    )

    assert not any("requires flash attention" in problem for problem in problems), problems
    assert any(
        problem.startswith("space.draft_kv_cache_dtype") and "does not forward" in problem
        for problem in problems
    ), problems


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
