"""Unit tests for the engine-neutral tunable-parameter registry.

The registry is the single allow-list of what an experiment sweep may vary, so
these tests pin its contents and the kind each tunable resolves to from its
``LLMConfig`` field. They need no daemon: the registry is pure data and the
kind is read from the field's type annotation.
"""

from dataclasses import fields

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DType,
    KVCacheDType,
    LLMConfig,
    VLLMBackend,
)
from llm_config_optimization.search_space.registry import TUNABLE_PARAMETERS, expected_kind

TUNABLES = (
    "dtype",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "tensor_parallel_size",
    "attention_backend",
)


def test_registry_covers_the_initial_tunable_set_in_canonical_order() -> None:
    assert tuple(TUNABLE_PARAMETERS) == TUNABLES


def test_registry_maps_every_tunable_to_an_llm_config_field() -> None:
    config_fields = {field.name for field in fields(LLMConfig)}

    assert set(TUNABLE_PARAMETERS.values()) <= config_fields
    assert "seed" not in TUNABLE_PARAMETERS
    assert "gpu_memory_utilization" not in TUNABLE_PARAMETERS
    assert "quantization" not in TUNABLE_PARAMETERS


@pytest.mark.parametrize(
    ("parameter", "kind"),
    [
        ("dtype", DType),
        ("kv_cache_dtype", KVCacheDType),
        ("max_model_len", int),
        ("max_num_seqs", int),
        ("max_num_batched_tokens", int),
        ("tensor_parallel_size", int),
        ("attention_backend", AttentionBackend),
    ],
)
def test_expected_kind_is_read_from_the_config_annotation(parameter: str, kind: type) -> None:
    assert expected_kind(TUNABLE_PARAMETERS[parameter]) is kind


def test_the_vllm_backend_declares_every_registered_tunable_as_applicable() -> None:
    """vLLM is the reference backend: its applicable set is the whole registry."""
    assert VLLMBackend().applicable_parameters == frozenset(TUNABLE_PARAMETERS)
