"""Unit tests for the engine-neutral tunable-parameter registry.

The registry is the single allow-list of what an experiment sweep may vary, so
these tests pin its contents and the kind each tunable resolves to from its
``LLMConfig`` field. They need no daemon: the registry is pure data and the
kind is read from the field's type annotation.
"""

from dataclasses import fields

import pytest

from llm_config_optimization.inference import (
    All2AllBackend,
    AttentionBackend,
    CpuMoeMode,
    DType,
    GpuLayersMode,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    LoadMode,
    SpecDraftSampling,
    SpeculativeMethod,
    SpeculativeMode,
    VLLMBackend,
)
from llm_config_optimization.search_space.registry import (
    DRAFT_MODEL_PARAMETERS,
    SPECULATIVE_PARAMETERS,
    TUNABLE_PARAMETERS,
    UnionKind,
    draft_configured,
    expected_kind,
)

TUNABLES = (
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_active_seqs",
    "max_num_scheduled_tokens",
    "ubatch_size",
    "dtype",
    "kv_cache_dtype",
    "attention_backend",
    "n_gpu_layers",
    "n_cpu_moe",
    "n_cpu_ffn",
    "kv_offload",
    "op_offload",
    "host_buffer",
    "repack",
    "swa_full",
    "load_mode",
    "prefix_caching",
    "cache_ram",
    "ctx_checkpoints",
    "checkpoint_min_step",
    "kv_unified",
    "continuous_batching",
    "cache_idle_slots",
    "tensor_parallel_size",
    "expert_parallel",
    "all2all_backend",
    "speculative_mode",
    "num_speculative_tokens",
    "spec_draft_n_min",
    "spec_draft_p_split",
    "spec_draft_sampling",
    "n_gpu_layers_draft",
    "n_cpu_moe_draft",
    "draft_kv_cache_dtype",
    "threads",
    "threads_batch",
    "poll",
    "prio",
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
        ("ubatch_size", int),
        ("n_cpu_ffn", int),
        ("kv_offload", bool),
        ("op_offload", bool),
        ("host_buffer", bool),
        ("repack", bool),
        ("swa_full", bool),
        ("load_mode", LoadMode),
        ("cache_ram", int),
        ("ctx_checkpoints", int),
        ("checkpoint_min_step", int),
        ("kv_unified", bool),
        ("continuous_batching", bool),
        ("cache_idle_slots", bool),
        ("tensor_parallel_size", int),
        ("attention_backend", AttentionBackend),
        ("speculative_mode", SpeculativeMode),
        ("num_speculative_tokens", int),
        ("spec_draft_n_min", int),
        ("spec_draft_p_split", float),
        ("spec_draft_sampling", SpecDraftSampling),
        ("draft_kv_cache_dtype", KVCacheDType),
        ("threads", int),
        ("threads_batch", int),
        ("poll", int),
        ("prio", int),
        ("prefix_caching", bool),
        ("expert_parallel", bool),
        ("all2all_backend", All2AllBackend),
    ],
)
def test_expected_kind_is_read_from_the_config_annotation(parameter: str, kind: type) -> None:
    assert expected_kind(TUNABLE_PARAMETERS[parameter]) is kind


@pytest.mark.parametrize(
    ("parameter", "kind"),
    [
        ("n_gpu_layers", UnionKind((int, GpuLayersMode))),
        ("n_cpu_moe", UnionKind((int, CpuMoeMode))),
        ("n_gpu_layers_draft", UnionKind((int, GpuLayersMode))),
        ("n_cpu_moe_draft", UnionKind((int, CpuMoeMode))),
    ],
    ids=["gpu-layers", "cpu-moe", "gpu-layers-draft", "cpu-moe-draft"],
)
def test_expected_kind_resolves_an_int_or_enum_union(parameter: str, kind: UnionKind) -> None:
    """Both union fields resolve to both branches, integer first."""
    assert expected_kind(parameter) == kind


def test_expected_kind_resolves_a_boolean_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``bool | None`` annotation resolves to :class:`bool`, not :class:`int`."""

    def annotations(_: type) -> dict[str, object]:
        return {"prefix_caching": bool | None}

    monkeypatch.setattr("llm_config_optimization.search_space.registry.get_type_hints", annotations)

    assert expected_kind("prefix_caching") is bool


def test_expected_kind_resolves_a_float_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``float | None`` annotation resolves to :class:`float`."""

    def annotations(_: type) -> dict[str, object]:
        return {"n_cpu_moe": float | None}

    monkeypatch.setattr("llm_config_optimization.search_space.registry.get_type_hints", annotations)

    assert expected_kind("n_cpu_moe") is float


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


def test_draft_configured_needs_a_draft_model_not_merely_a_speculative_method() -> None:
    assert not draft_configured(LLMConfig())
    assert not draft_configured(LLMConfig(speculative_method=SpeculativeMethod.MTP))
    assert draft_configured(LLMConfig(draft_model="org/draft"))


@pytest.mark.parametrize(
    "llm",
    [LLMConfig(), LLMConfig(draft_model="org/draft")],
    ids=["no-draft", "draft"],
)
def test_the_vllm_backend_never_requires_a_draft_only_tunable(llm: LLMConfig) -> None:
    """A llama.cpp-only draft knob stays out of vLLM's applicable set."""
    assert DRAFT_MODEL_PARAMETERS.isdisjoint(VLLMBackend().applicable_parameters(llm))


def test_the_vllm_backend_declares_every_ungated_tunable_it_forwards_as_applicable() -> None:
    """Without a configured setup, the gated and llama.cpp-only dimensions are withheld."""
    assert VLLMBackend().applicable_parameters(LLMConfig()) == (
        frozenset(TUNABLE_PARAMETERS)
        - SPECULATIVE_PARAMETERS
        - DRAFT_MODEL_PARAMETERS
        - {
            "n_gpu_layers",
            "n_cpu_moe",
            "n_cpu_ffn",
            "kv_offload",
            "op_offload",
            "host_buffer",
            "repack",
            "swa_full",
            "load_mode",
            "ubatch_size",
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
        frozenset(TUNABLE_PARAMETERS)
        - DRAFT_MODEL_PARAMETERS
        - {
            "n_gpu_layers",
            "n_cpu_moe",
            "n_cpu_ffn",
            "kv_offload",
            "op_offload",
            "host_buffer",
            "repack",
            "swa_full",
            "load_mode",
            "ubatch_size",
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
