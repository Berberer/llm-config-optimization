"""Unit tests for llama.cpp's parameter mapping and argument assembly.

The observable contract is the argument list the translation assembles from an
engine-neutral config: which flags appear, which engine spelling each value
becomes, and which enum members are rejected. The typed config is used
throughout, because the translation forwards the fields it maps and ignores the
ones llama.cpp does not model.
"""

from dataclasses import fields

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DType,
    ExperimentError,
    KVCacheDType,
    LLMConfig,
    QuantizationMethod,
)
from llm_config_optimization.inference._llamacpp import (
    PARAMETER_TARGETS,
    VALUE_MAPS,
    serve_arguments,
)
from llm_config_optimization.search_space.registry import TUNABLE_PARAMETERS

_INAPPLICABLE_TUNABLES = frozenset({"dtype", "tensor_parallel_size"})

_INAPPLICABLE_FIELDS = frozenset(
    {
        "dtype",
        "quantization",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "seed",
    }
)
"""The :class:`LLMConfig` fields this backend has no ``llama-server`` target for.

``dtype`` is carried by the GGUF weights and ``quantization`` by the file's
metadata, tensor parallelism and the VRAM fraction are vLLM-only knobs, and the
sampling seed is not forwarded because the host client measures with greedy
decoding.
"""


def test_applicable_parameters_are_the_registry_minus_llama_cpp_inapplicable_ones() -> None:
    """The backend declares exactly the tunables llama.cpp has a flag for."""
    assert set(PARAMETER_TARGETS) == set(TUNABLE_PARAMETERS) - _INAPPLICABLE_TUNABLES


def test_every_llm_config_field_is_classified_as_mapped_or_inapplicable() -> None:
    """Every exposed parameter either reaches the engine or is deliberately ignored.

    A field added to :class:`LLMConfig` fails this test until it is given a
    mapping or added to the documented inapplicable set, so a parameter can
    never be exposed and then silently dropped before the engine sees it.
    """
    exposed = {field.name for field in fields(LLMConfig)}
    assert exposed == set(PARAMETER_TARGETS) | _INAPPLICABLE_FIELDS


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (LLMConfig(max_model_len=4096), ["--ctx-size", "4096"]),
        (LLMConfig(max_num_seqs=8), ["--parallel", "8"]),
        (
            LLMConfig(max_num_batched_tokens=8192),
            ["--batch-size", "8192", "--ubatch-size", "8192"],
        ),
        (
            LLMConfig(kv_cache_dtype=KVCacheDType.FLOAT16),
            ["--cache-type-k", "f16", "--cache-type-v", "f16"],
        ),
        (LLMConfig(attention_backend=AttentionBackend.FLASH_ATTENTION), ["--flash-attn", "on"]),
    ],
)
def test_each_tunable_maps_to_its_server_flags(config: LLMConfig, expected: list[str]) -> None:
    assert serve_arguments(config) == expected


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (KVCacheDType.FLOAT16, "f16"),
        (KVCacheDType.BFLOAT16, "bf16"),
    ],
)
def test_every_supported_kv_cache_dtype_translates(value: KVCacheDType, spelling: str) -> None:
    assert serve_arguments(LLMConfig(kv_cache_dtype=value)) == [
        "--cache-type-k",
        spelling,
        "--cache-type-v",
        spelling,
    ]


def test_the_supported_attention_backend_translates() -> None:
    assert serve_arguments(LLMConfig(attention_backend=AttentionBackend.FLASH_ATTENTION)) == [
        "--flash-attn",
        "on",
    ]


@pytest.mark.parametrize(
    ("config", "field"),
    [
        (LLMConfig(kv_cache_dtype=KVCacheDType.AUTO), "kv_cache_dtype"),
        (LLMConfig(kv_cache_dtype=KVCacheDType.FP8_E4M3), "kv_cache_dtype"),
        (LLMConfig(kv_cache_dtype=KVCacheDType.FP8_E5M2), "kv_cache_dtype"),
        (LLMConfig(attention_backend=AttentionBackend.FLASH_INFER), "attention_backend"),
        (LLMConfig(attention_backend=AttentionBackend.XFORMERS), "attention_backend"),
    ],
)
def test_unsupported_enum_member_is_rejected_rather_than_dropped(
    config: LLMConfig, field: str
) -> None:
    with pytest.raises(ExperimentError, match=field):
        serve_arguments(config)


def test_every_supported_value_has_a_map_entry_and_no_unsupported_one_does() -> None:
    """The curated maps cover the portable members and reject the rest."""
    assert VALUE_MAPS["kv_cache_dtype"] == {
        KVCacheDType.FLOAT16: "f16",
        KVCacheDType.BFLOAT16: "bf16",
    }
    assert VALUE_MAPS["attention_backend"] == {AttentionBackend.FLASH_ATTENTION: "on"}


def test_all_tunables_assemble_in_the_declared_order() -> None:
    config = LLMConfig(
        kv_cache_dtype=KVCacheDType.BFLOAT16,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        attention_backend=AttentionBackend.FLASH_ATTENTION,
    )

    assert serve_arguments(config) == [
        "--cache-type-k",
        "bf16",
        "--cache-type-v",
        "bf16",
        "--ctx-size",
        "2048",
        "--parallel",
        "4",
        "--batch-size",
        "4096",
        "--ubatch-size",
        "4096",
        "--flash-attn",
        "on",
    ]


def test_unset_fields_are_omitted_so_the_engine_keeps_its_defaults() -> None:
    assert serve_arguments(LLMConfig()) == []
    assert serve_arguments(LLMConfig(max_num_seqs=None, kv_cache_dtype=None)) == []


def test_fields_llama_cpp_does_not_model_are_not_forwarded() -> None:
    config = LLMConfig(
        dtype=DType.FLOAT16,
        quantization=QuantizationMethod.AWQ,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        seed=7,
    )

    assert serve_arguments(config) == []


def test_extra_args_are_appended_verbatim_after_the_mapped_parameters() -> None:
    config = LLMConfig(max_model_len=4096)

    assert serve_arguments(config, extra_args=("--n-gpu-layers", "99", "--no-mmap")) == [
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "99",
        "--no-mmap",
    ]
