"""Unit tests for the vLLM parameter mapping and argument assembly.

The backend's observable contract here is the argument list it assembles from an
engine-neutral config: which flags appear, in what order, how each value is
encoded, which enum members are rejected, and — after the assembly was split by
destination — which command each flag reaches. Serving parameters
(:class:`LLMConfig` and the backend's ``extra_args``) build the ``vllm serve``
command through :func:`serve_arguments`; workload parameters
(:class:`BenchmarkConfig`) build the ``vllm bench serve`` command through
:func:`bench_arguments`. The typed config is used where it models a value, and
the generic encoding primitive is exercised directly for value shapes it does
not yet model (booleans, nested JSON), which is exactly the seam the backend's
own assembly builds on.
"""

from dataclasses import fields
from enum import Enum, StrEnum
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    BenchmarkConfig,
    DType,
    ExperimentError,
    KVCacheDType,
    LLMConfig,
    QuantizationMethod,
)
from llm_config_optimization.inference._vllm import (
    PARAMETER_TARGETS,
    FlatFlag,
    JsonGroup,
    bench_arguments,
    encode_arguments,
    serve_arguments,
)

HUB_MODEL = "meta-llama/Llama-3.1-8B"


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


def test_extra_args_reach_the_server_command_not_the_client() -> None:
    """``extra_args`` are serving flags, so they never leak into the workload."""
    backend_extra = ("--enforce-eager",)

    serve = serve_arguments(LLMConfig(), extra_args=backend_extra)
    bench = bench_arguments(BenchmarkConfig(model=HUB_MODEL), HUB_MODEL)

    assert "--enforce-eager" in serve
    assert "--enforce-eager" not in bench


def test_workload_fields_assemble_for_the_client_command() -> None:
    benchmark = BenchmarkConfig(
        model=HUB_MODEL,
        dataset_name="sharegpt",
        dataset_path="/data/prompts.jsonl",
        num_prompts=100,
        input_len=128,
        output_len=64,
    )

    assert bench_arguments(benchmark, HUB_MODEL) == [
        "--model",
        HUB_MODEL,
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        "/data/prompts.jsonl",
        "--num-prompts",
        "100",
        "--input-len",
        "128",
        "--output-len",
        "64",
    ]


def test_random_dataset_routes_lengths_to_the_random_flags() -> None:
    """vLLM's random dataset reads ``--random-*-len`` and ignores the generic flags."""
    benchmark = BenchmarkConfig(
        model=HUB_MODEL,
        dataset_name="random",
        num_prompts=8,
        input_len=32,
        output_len=8,
    )

    assert bench_arguments(benchmark, HUB_MODEL) == [
        "--model",
        HUB_MODEL,
        "--dataset-name",
        "random",
        "--num-prompts",
        "8",
        "--random-input-len",
        "32",
        "--random-output-len",
        "8",
    ]


def test_omitted_dataset_name_routes_lengths_to_the_random_flags() -> None:
    """vLLM defaults to the random dataset when no dataset name is given."""
    benchmark = BenchmarkConfig(model=HUB_MODEL, input_len=32, output_len=8)

    arguments = bench_arguments(benchmark, HUB_MODEL)

    assert "--random-input-len" in arguments
    assert "--random-output-len" in arguments
    assert "--input-len" not in arguments
    assert "--output-len" not in arguments


def test_unset_workload_fields_are_omitted() -> None:
    assert bench_arguments(BenchmarkConfig(model=HUB_MODEL), HUB_MODEL) == [
        "--model",
        HUB_MODEL,
    ]


def test_serving_and_workload_flags_do_not_cross_commands() -> None:
    """A serving flag reaches the server, a workload flag reaches the client."""
    serve = serve_arguments(LLMConfig(dtype=DType.FLOAT16, max_num_seqs=256))
    bench = bench_arguments(BenchmarkConfig(model=HUB_MODEL, num_prompts=32), HUB_MODEL)

    assert "--dtype" in serve
    assert "--max-num-seqs" in serve
    assert "--dtype" not in bench
    assert "--max-num-seqs" not in bench

    assert "--model" in bench
    assert "--num-prompts" in bench
    assert "--model" not in serve
    assert "--num-prompts" not in serve


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
