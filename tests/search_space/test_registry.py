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
    GpuLayersMode,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    SpeculativeMethod,
    SpeculativeMode,
    VLLMBackend,
)
from llm_config_optimization.search_space.registry import (
    SPECULATIVE_PARAMETERS,
    TUNABLE_PARAMETERS,
    UnionKind,
    expected_kind,
)

TUNABLES = (
    "dtype",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_active_seqs",
    "max_num_scheduled_tokens",
    "tensor_parallel_size",
    "attention_backend",
    "speculative_mode",
    "num_speculative_tokens",
    "prefix_caching",
    "n_gpu_layers",
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
        ("max_num_active_seqs", int),
        ("max_num_scheduled_tokens", int),
        ("tensor_parallel_size", int),
        ("attention_backend", AttentionBackend),
        ("speculative_mode", SpeculativeMode),
        ("num_speculative_tokens", int),
        ("prefix_caching", bool),
    ],
)
def test_expected_kind_is_read_from_the_config_annotation(parameter: str, kind: type) -> None:
    assert expected_kind(TUNABLE_PARAMETERS[parameter]) is kind


def test_expected_kind_resolves_the_gpu_layers_union() -> None:
    """The real field resolves to both branches, integer first."""
    assert expected_kind("n_gpu_layers") == UnionKind((int, GpuLayersMode))


def test_expected_kind_resolves_a_boolean_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``bool | None`` annotation resolves to :class:`bool`, not :class:`int`."""

    def annotations(_: type) -> dict[str, object]:
        return {"prefix_caching": bool | None}

    monkeypatch.setattr("llm_config_optimization.search_space.registry.get_type_hints", annotations)

    assert expected_kind("prefix_caching") is bool


def test_expected_kind_resolves_a_union_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``int | enum | None`` annotation yields both branches in order."""

    def annotations(_: type) -> dict[str, object]:
        return {"n_gpu_layers": int | AttentionBackend | None}

    monkeypatch.setattr("llm_config_optimization.search_space.registry.get_type_hints", annotations)

    assert expected_kind("n_gpu_layers") == UnionKind((int, AttentionBackend))


def test_a_union_kind_keeps_its_branches_distinct() -> None:
    assert UnionKind((int, AttentionBackend)) != UnionKind((AttentionBackend, int))
    assert isinstance(UnionKind((int,)), UnionKind)


def test_prefix_caching_is_applicable_on_both_engines() -> None:
    """The boolean switch is unconditional: no setup gates it on either engine."""
    assert "prefix_caching" in VLLMBackend().applicable_parameters(LLMConfig())
    assert "prefix_caching" in LlamaCppBackend().applicable_parameters(LLMConfig())


def test_the_vllm_backend_declares_every_ungated_tunable_it_forwards_as_applicable() -> None:
    """Without a configured setup, the gated and llama.cpp-only dimensions are withheld."""
    assert VLLMBackend().applicable_parameters(LLMConfig()) == (
        frozenset(TUNABLE_PARAMETERS) - SPECULATIVE_PARAMETERS - {"n_gpu_layers"}
    )


@pytest.mark.parametrize(
    "llm",
    [
        LLMConfig(speculative_method=SpeculativeMethod.MTP),
        LLMConfig(draft_model="org/draft"),
    ],
    ids=["method", "draft-model"],
)
def test_a_configured_speculative_setup_makes_the_gated_tunables_applicable(
    llm: LLMConfig,
) -> None:
    assert VLLMBackend().applicable_parameters(llm) == (
        frozenset(TUNABLE_PARAMETERS) - {"n_gpu_layers"}
    )
