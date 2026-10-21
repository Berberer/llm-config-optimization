"""Unit tests for the vLLM parameter mapping and argument assembly.

The backend's observable contract here is the argument list it assembles from an
engine-neutral config: which flags appear, in what order, how each value is
encoded, and which enum members are rejected. Serving parameters
(:class:`LLMConfig` and the backend's ``extra_args``) build the ``vllm serve``
command through :func:`serve_arguments`. The typed config is used where it
models a value, and the generic encoding primitive is exercised directly for
value shapes it does not yet model (booleans, nested JSON), which is exactly the
seam the backend's own assembly builds on.
"""

from dataclasses import fields
from enum import Enum, StrEnum
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DType,
    ExperimentError,
    KVCacheDType,
    LLMConfig,
    QuantizationMethod,
)
from llm_config_optimization.inference._vllm import (
    PARAMETER_TARGETS,
    VALUE_MAPS,
    FlatFlag,
    JsonGroup,
    encode_arguments,
    serve_arguments,
)


def test_every_llm_config_field_has_a_mapping_entry() -> None:
    """Every exposed parameter actually reaches the engine, and none are extra."""
    assert {field.name for field in fields(LLMConfig)} == set(PARAMETER_TARGETS)


def test_flat_flags_are_emitted_as_kebab_case_value_pairs() -> None:
    config = LLMConfig(
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        seed=1234,
    )

    assert serve_arguments(config) == [
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.9",
        "--tensor-parallel-size",
        "2",
        "--seed",
        "1234",
    ]


def test_categorical_fields_are_translated_to_vllm_spellings() -> None:
    config = LLMConfig(
        dtype=DType.BFLOAT16,
        quantization=QuantizationMethod.AWQ,
        kv_cache_dtype=KVCacheDType.FP8_E4M3,
    )

    assert serve_arguments(config) == [
        "--dtype",
        "bfloat16",
        "--quantization",
        "awq",
        "--kv-cache-dtype",
        "fp8_e4m3",
    ]


def test_attention_backend_is_forwarded_through_its_json_group() -> None:
    """The one non-identity mapping is sent as ``--attention-config`` JSON."""
    config = LLMConfig(attention_backend=AttentionBackend.FLASH_ATTENTION)

    assert serve_arguments(config) == ["--attention-config", '{"backend": "FLASH_ATTN"}']


def test_every_curated_value_has_a_vllm_spelling() -> None:
    """The maps pin every member vLLM can serve, so a spelling cannot be dropped."""
    assert VALUE_MAPS["dtype"] == {
        DType.AUTO: "auto",
        DType.FLOAT16: "float16",
        DType.BFLOAT16: "bfloat16",
        DType.FLOAT32: "float32",
    }
    assert VALUE_MAPS["quantization"] == {
        QuantizationMethod.AWQ: "awq",
        QuantizationMethod.GPTQ: "gptq",
        QuantizationMethod.FP8: "fp8",
    }
    assert VALUE_MAPS["kv_cache_dtype"] == {
        KVCacheDType.AUTO: "auto",
        KVCacheDType.FLOAT16: "float16",
        KVCacheDType.BFLOAT16: "bfloat16",
        KVCacheDType.FP8_E4M3: "fp8_e4m3",
        KVCacheDType.FP8_E5M2: "fp8_e5m2",
    }
    assert VALUE_MAPS["attention_backend"] == {
        AttentionBackend.FLASH_ATTENTION: "FLASH_ATTN",
        AttentionBackend.FLASH_INFER: "FLASHINFER",
    }


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("dtype", DType.AUTO, ["--dtype", "auto"]),
        ("dtype", DType.FLOAT16, ["--dtype", "float16"]),
        ("dtype", DType.BFLOAT16, ["--dtype", "bfloat16"]),
        ("dtype", DType.FLOAT32, ["--dtype", "float32"]),
        ("quantization", QuantizationMethod.AWQ, ["--quantization", "awq"]),
        ("quantization", QuantizationMethod.GPTQ, ["--quantization", "gptq"]),
        ("quantization", QuantizationMethod.FP8, ["--quantization", "fp8"]),
        ("kv_cache_dtype", KVCacheDType.AUTO, ["--kv-cache-dtype", "auto"]),
        ("kv_cache_dtype", KVCacheDType.FLOAT16, ["--kv-cache-dtype", "float16"]),
        ("kv_cache_dtype", KVCacheDType.BFLOAT16, ["--kv-cache-dtype", "bfloat16"]),
        ("kv_cache_dtype", KVCacheDType.FP8_E4M3, ["--kv-cache-dtype", "fp8_e4m3"]),
        ("kv_cache_dtype", KVCacheDType.FP8_E5M2, ["--kv-cache-dtype", "fp8_e5m2"]),
        (
            "attention_backend",
            AttentionBackend.FLASH_ATTENTION,
            ["--attention-config", '{"backend": "FLASH_ATTN"}'],
        ),
        (
            "attention_backend",
            AttentionBackend.FLASH_INFER,
            ["--attention-config", '{"backend": "FLASHINFER"}'],
        ),
    ],
)
def test_every_curated_value_reaches_the_engine_under_its_spelling(
    field: str, value: object, expected: list[str]
) -> None:
    """Every member the maps accept is translated end to end, not merely tabulated."""
    assert serve_arguments(LLMConfig(**{field: value})) == expected


def test_all_fields_assemble_in_the_declared_order() -> None:
    config = LLMConfig(
        dtype=DType.FLOAT16,
        quantization=QuantizationMethod.FP8,
        kv_cache_dtype=KVCacheDType.AUTO,
        max_model_len=2048,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        seed=7,
        attention_backend=AttentionBackend.FLASH_INFER,
    )

    assert serve_arguments(config) == [
        "--dtype",
        "float16",
        "--quantization",
        "fp8",
        "--kv-cache-dtype",
        "auto",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
        "--tensor-parallel-size",
        "1",
        "--seed",
        "7",
        "--attention-config",
        '{"backend": "FLASHINFER"}',
    ]


def test_unset_fields_are_omitted_so_the_engine_keeps_its_defaults() -> None:
    assert serve_arguments(LLMConfig()) == []
    assert serve_arguments(LLMConfig(dtype=None, max_num_seqs=None)) == []


def test_extra_args_are_appended_verbatim_after_the_mapped_parameters() -> None:
    config = LLMConfig(dtype=DType.FLOAT16)

    assert serve_arguments(config, extra_args=("--enforce-eager", "--block-size", "32")) == [
        "--dtype",
        "float16",
        "--enforce-eager",
        "--block-size",
        "32",
    ]


class _ExtendedDType(StrEnum):
    """Stands in for a member added to the engine-neutral enum after a release."""

    INT8 = "int8"


def test_unsupported_enum_member_is_rejected_rather_than_dropped() -> None:
    config = LLMConfig(dtype=_ExtendedDType.INT8)

    with pytest.raises(ExperimentError, match="dtype"):
        serve_arguments(config)


def test_an_attention_backend_vllm_no_longer_ships_is_rejected() -> None:
    """XFORMERS has no vLLM spelling in current releases, so it is refused."""
    config = LLMConfig(attention_backend=AttentionBackend.XFORMERS)

    with pytest.raises(ExperimentError, match="attention_backend"):
        serve_arguments(config)


def test_booleans_encode_as_flag_presence_or_negation() -> None:
    targets = {"trust_remote_code": FlatFlag()}

    assert encode_arguments({"trust_remote_code": True}, targets) == ["--trust-remote-code"]
    assert encode_arguments({"trust_remote_code": False}, targets) == ["--no-trust-remote-code"]


def test_nested_values_are_json_encoded_for_flat_flags() -> None:
    targets = {"kernel_config": FlatFlag(), "block_size": FlatFlag()}

    assert encode_arguments({"kernel_config": {"enable_flashinfer": True}}, targets) == [
        "--kernel-config",
        '{"enable_flashinfer": true}',
    ]
    assert encode_arguments({"block_size": [16, 32]}, targets) == ["--block-size", "[16, 32]"]


def test_json_group_members_are_merged_into_a_single_payload() -> None:
    targets = {
        "backend": JsonGroup(flag="--attention-config", key="backend"),
        "flash_attn_version": JsonGroup(flag="--attention-config", key="flash_attn_version"),
    }

    assert encode_arguments({"backend": "FLASH_ATTN", "flash_attn_version": 2}, targets) == [
        "--attention-config",
        '{"backend": "FLASH_ATTN", "flash_attn_version": 2}',
    ]


def test_json_group_omits_unset_members_and_is_emitted_once() -> None:
    targets = {
        "backend": JsonGroup(flag="--attention-config", key="backend"),
        "flash_attn_version": JsonGroup(flag="--attention-config", key="flash_attn_version"),
    }

    assert encode_arguments({"flash_attn_version": 3}, targets) == [
        "--attention-config",
        '{"flash_attn_version": 3}',
    ]


def test_nested_enums_paths_and_sets_serialise_inside_a_json_group() -> None:
    targets = {"config": FlatFlag()}

    assert encode_arguments(
        {"config": {"dtype": DType.FLOAT16, "download_dir": Path("/m"), "sizes": {2, 1}}},
        targets,
    ) == ["--config", '{"dtype": "float16", "download_dir": "/m", "sizes": [1, 2]}']


def test_sets_and_paths_normalise_deterministically() -> None:
    targets = {"cudagraph_capture_sizes": FlatFlag(), "download_dir": FlatFlag()}

    assert encode_arguments(
        {"cudagraph_capture_sizes": {256, 1, 64}, "download_dir": Path("/models")}, targets
    ) == [
        "--cudagraph-capture-sizes",
        "[1, 256, 64]",
        "--download-dir",
        "/models",
    ]


class _Plain(Enum):
    """A non-string enum, whose ``str`` is ``ClassName.MEMBER``, not its value."""

    OPTION = "option"


def test_plain_enums_normalise_to_their_value() -> None:
    assert encode_arguments({"mode": _Plain.OPTION}, {"mode": FlatFlag()}) == ["--mode", "option"]


def test_parameters_without_a_target_are_not_forwarded() -> None:
    assert encode_arguments({"unmodelled": 1}, {"dtype": FlatFlag()}) == []
