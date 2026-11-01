"""Unit tests for llama.cpp's parameter mapping and argument assembly.

The observable contract is the argument list the translation assembles from an
engine-neutral config: which flags appear, which engine spelling each value
becomes, and which enum members are rejected. The typed config is used
throughout, because the translation forwards the fields it maps and ignores the
ones llama.cpp does not model.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    CpuMoeMode,
    DType,
    ExperimentError,
    FitMode,
    GpuLayersMode,
    KVCacheDType,
    LLMConfig,
    LoadMode,
    QuantizationMethod,
    SpecDraftSampling,
    SpeculativeMethod,
    SpeculativeMode,
)
from llm_config_optimization.inference._llamacpp import (
    PARAMETER_TARGETS,
    VALUE_MAPS,
    serve_arguments,
)
from llm_config_optimization.search_space.registry import TUNABLE_PARAMETERS

_INAPPLICABLE_TUNABLES = frozenset(
    {
        "dtype",
        "tensor_parallel_size",
        "max_num_active_seqs",
        "max_num_scheduled_tokens",
        "expert_parallel",
        "all2all_backend",
    }
)
"""The registry's tunables llama.cpp has no ``llama-server`` flag for."""

_INAPPLICABLE_FIELDS = frozenset(
    {
        "dtype",
        "quantization",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "max_num_active_seqs",
        "max_num_scheduled_tokens",
        "seed",
        "resolved_draft_path",
        "expert_parallel",
        "all2all_backend",
    }
)
"""The :class:`LLMConfig` fields this backend has no ``llama-server`` target for.

``dtype`` is carried by the GGUF weights and ``quantization`` by the file's
metadata, tensor parallelism, the VRAM fraction, and the scheduling caps are
vLLM-only knobs, the MoE expert-parallelism fields are vLLM's MoE counterpart,
and the sampling seed is not forwarded because the host client measures with
greedy decoding. ``resolved_draft_path`` is a host-side resolution artifact that
an engine never receives; the speculative settings reach the engine through
their ``--spec-*`` flags.
"""


def test_applicable_parameters_are_the_registry_minus_llama_cpp_inapplicable_ones() -> None:
    """The forwarding table covers exactly the tunables llama.cpp can serve."""
    forwarded_tunables = {name for name in TUNABLE_PARAMETERS if name in PARAMETER_TARGETS}
    assert forwarded_tunables == set(TUNABLE_PARAMETERS) - _INAPPLICABLE_TUNABLES
    fixed_only = {"speculative_method", "draft_model", "fit", "fit_target", "fit_ctx"}
    assert set(PARAMETER_TARGETS) == forwarded_tunables | fixed_only


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
            ["--batch-size", "8192"],
        ),
        (LLMConfig(ubatch_size=2048), ["--ubatch-size", "2048"]),
        (
            LLMConfig(kv_cache_dtype=KVCacheDType.FLOAT16),
            ["--cache-type-k", "f16", "--cache-type-v", "f16"],
        ),
        (LLMConfig(attention_backend=AttentionBackend.FLASH_ATTENTION), ["--flash-attn", "on"]),
        (LLMConfig(attention_backend=AttentionBackend.AUTO), ["--flash-attn", "auto"]),
        (LLMConfig(attention_backend=AttentionBackend.OFF), ["--flash-attn", "off"]),
        (LLMConfig(n_gpu_layers=GpuLayersMode.AUTO), ["--n-gpu-layers", "auto"]),
        (LLMConfig(n_gpu_layers=GpuLayersMode.ALL), ["--n-gpu-layers", "all"]),
        (LLMConfig(n_gpu_layers=0), ["--n-gpu-layers", "0"]),
        (LLMConfig(n_gpu_layers=40), ["--n-gpu-layers", "40"]),
        (LLMConfig(n_cpu_moe=0), ["--n-cpu-moe", "0"]),
        (LLMConfig(n_cpu_moe=8), ["--n-cpu-moe", "8"]),
        (LLMConfig(n_cpu_ffn=0), ["--n-cpu-ffn", "0"]),
        (LLMConfig(n_cpu_ffn=12), ["--n-cpu-ffn", "12"]),
        (LLMConfig(cache_ram=-1), ["--cache-ram", "-1"]),
        (LLMConfig(cache_ram=0), ["--cache-ram", "0"]),
        (LLMConfig(cache_ram=2048), ["--cache-ram", "2048"]),
        (LLMConfig(ctx_checkpoints=4), ["--ctx-checkpoints", "4"]),
        (LLMConfig(checkpoint_min_step=0), ["--checkpoint-min-step", "0"]),
        (LLMConfig(load_mode=LoadMode.MMAP), ["--load-mode", "mmap"]),
        (LLMConfig(spec_draft_n_min=2), ["--spec-draft-n-min", "2"]),
        (LLMConfig(spec_draft_p_split=0.75), ["--spec-draft-p-split", "0.75"]),
        (LLMConfig(n_gpu_layers_draft=8), ["--n-gpu-layers-draft", "8"]),
        (LLMConfig(n_cpu_moe_draft=8), ["--n-cpu-moe-draft", "8"]),
    ],
)
def test_each_tunable_maps_to_its_server_flags(config: LLMConfig, expected: list[str]) -> None:
    assert serve_arguments(config) == expected


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (KVCacheDType.FLOAT16, "f16"),
        (KVCacheDType.BFLOAT16, "bf16"),
        (KVCacheDType.FLOAT32, "f32"),
        (KVCacheDType.Q8_0, "q8_0"),
        (KVCacheDType.Q4_0, "q4_0"),
        (KVCacheDType.Q4_1, "q4_1"),
        (KVCacheDType.IQ4_NL, "iq4_nl"),
        (KVCacheDType.Q5_0, "q5_0"),
        (KVCacheDType.Q5_1, "q5_1"),
    ],
)
def test_every_supported_kv_cache_dtype_translates(value: KVCacheDType, spelling: str) -> None:
    assert serve_arguments(LLMConfig(kv_cache_dtype=value)) == [
        "--cache-type-k",
        spelling,
        "--cache-type-v",
        spelling,
    ]


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (AttentionBackend.AUTO, "auto"),
        (AttentionBackend.FLASH_ATTENTION, "on"),
        (AttentionBackend.OFF, "off"),
    ],
)
def test_every_supported_attention_backend_translates(
    value: AttentionBackend, spelling: str
) -> None:
    assert serve_arguments(LLMConfig(attention_backend=value)) == ["--flash-attn", spelling]


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (GpuLayersMode.AUTO, "auto"),
        (GpuLayersMode.ALL, "all"),
    ],
)
def test_every_supported_gpu_layers_mode_translates(value: GpuLayersMode, spelling: str) -> None:
    assert serve_arguments(LLMConfig(n_gpu_layers=value)) == ["--n-gpu-layers", spelling]


def test_an_unset_n_gpu_layers_emits_no_flag() -> None:
    assert serve_arguments(LLMConfig(n_gpu_layers=None)) == []


def test_the_all_cpu_moe_mode_emits_the_value_less_flag() -> None:
    """``all`` is a flag of its own, never ``--n-cpu-moe all``."""
    assert serve_arguments(LLMConfig(n_cpu_moe=CpuMoeMode.ALL)) == ["--cpu-moe"]


def test_an_unset_n_cpu_moe_emits_no_flag() -> None:
    assert serve_arguments(LLMConfig(n_cpu_moe=None)) == []


def test_the_all_draft_cpu_moe_mode_emits_the_value_less_flag() -> None:
    """``all`` is ``--cpu-moe-draft``, never ``--n-cpu-moe-draft all``."""
    assert serve_arguments(LLMConfig(n_cpu_moe_draft=CpuMoeMode.ALL)) == ["--cpu-moe-draft"]


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (GpuLayersMode.AUTO, "auto"),
        (GpuLayersMode.ALL, "all"),
    ],
)
def test_every_draft_gpu_layers_mode_translates(value: GpuLayersMode, spelling: str) -> None:
    assert serve_arguments(LLMConfig(n_gpu_layers_draft=value)) == [
        "--n-gpu-layers-draft",
        spelling,
    ]


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (SpecDraftSampling.GREEDY, "greedy"),
        (SpecDraftSampling.PROBABILISTIC, "probabilistic"),
    ],
)
def test_every_draft_sampling_strategy_translates(value: SpecDraftSampling, spelling: str) -> None:
    assert serve_arguments(LLMConfig(spec_draft_sampling=value)) == [
        "--spec-draft-sampling",
        spelling,
    ]


def test_the_draft_cache_dtype_targets_both_cache_flags() -> None:
    assert serve_arguments(LLMConfig(draft_kv_cache_dtype=KVCacheDType.Q8_0)) == [
        "--cache-type-k-draft",
        "q8_0",
        "--cache-type-v-draft",
        "q8_0",
    ]


@pytest.mark.parametrize(
    "field",
    [
        "spec_draft_n_min",
        "spec_draft_p_split",
        "spec_draft_sampling",
        "n_gpu_layers_draft",
        "n_cpu_moe_draft",
        "draft_kv_cache_dtype",
    ],
)
def test_an_unset_draft_field_emits_no_flag(field: str) -> None:
    assert serve_arguments(LLMConfig(**{field: None})) == []


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (LoadMode.AUTO, "auto"),
        (LoadMode.NONE, "none"),
        (LoadMode.MMAP, "mmap"),
        (LoadMode.MLOCK, "mlock"),
        (LoadMode.MMAP_MLOCK, "mmap+mlock"),
        (LoadMode.DIO, "dio"),
    ],
)
def test_every_load_mode_translates(value: LoadMode, spelling: str) -> None:
    """The compound mode becomes llama.cpp's ``mmap+mlock``, not its enum name."""
    assert serve_arguments(LLMConfig(load_mode=value)) == ["--load-mode", spelling]


@pytest.mark.parametrize(("value", "spelling"), [(FitMode.ON, "on"), (FitMode.OFF, "off")])
def test_every_fit_mode_translates(value: FitMode, spelling: str) -> None:
    assert serve_arguments(LLMConfig(fit=value)) == ["--fit", spelling]


def test_fit_settings_are_forwarded_only_when_fit_is_enabled() -> None:
    """llama.cpp discards a target or context that arrives without ``--fit on``."""
    assert serve_arguments(LLMConfig(fit=FitMode.ON, fit_target=1024, fit_ctx=8192)) == [
        "--fit",
        "on",
        "--fit-target",
        "1024",
        "--fit-ctx",
        "8192",
    ]
    assert serve_arguments(LLMConfig(fit=FitMode.OFF, fit_target=1024, fit_ctx=8192)) == [
        "--fit",
        "off",
    ]
    assert serve_arguments(LLMConfig(fit_target=1024, fit_ctx=8192)) == []


@pytest.mark.parametrize(
    ("field", "enabled", "disabled"),
    [
        ("kv_offload", "--kv-offload", "--no-kv-offload"),
        ("op_offload", "--op-offload", "--no-op-offload"),
        ("repack", "--repack", "--no-repack"),
        ("kv_unified", "--kv-unified", "--no-kv-unified"),
        ("continuous_batching", "--cont-batching", "--no-cont-batching"),
        ("cache_idle_slots", "--cache-idle-slots", "--no-cache-idle-slots"),
    ],
)
def test_each_two_sided_toggle_emits_both_flags(field: str, enabled: str, disabled: str) -> None:
    """Both sides are explicit, so ``False`` is a real setting."""
    assert serve_arguments(LLMConfig(**{field: True})) == [enabled]
    assert serve_arguments(LLMConfig(**{field: False})) == [disabled]


def test_the_cpu_and_process_fields_forward_their_flags() -> None:
    assert serve_arguments(LLMConfig(threads=8)) == ["--threads", "8"]
    assert serve_arguments(LLMConfig(threads_batch=4)) == ["--threads-batch", "4"]
    assert serve_arguments(LLMConfig(poll=50)) == ["--poll", "50"]
    assert serve_arguments(LLMConfig(prio=2)) == ["--prio", "2"]


@pytest.mark.parametrize("field", ["threads", "threads_batch", "poll", "prio"])
def test_an_unset_cpu_or_process_field_emits_no_flag(field: str) -> None:
    assert serve_arguments(LLMConfig(**{field: None})) == []


def test_the_one_sided_host_buffer_toggle_only_has_a_disabling_flag() -> None:
    """llama.cpp has ``--no-host`` but no enabling counterpart."""
    assert serve_arguments(LLMConfig(host_buffer=True)) == []
    assert serve_arguments(LLMConfig(host_buffer=False)) == ["--no-host"]


def test_the_one_sided_swa_full_toggle_only_has_an_enabling_flag() -> None:
    """llama.cpp has ``--swa-full`` but no disabling counterpart."""
    assert serve_arguments(LLMConfig(swa_full=True)) == ["--swa-full"]
    assert serve_arguments(LLMConfig(swa_full=False)) == []


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
        KVCacheDType.FLOAT32: "f32",
        KVCacheDType.Q8_0: "q8_0",
        KVCacheDType.Q4_0: "q4_0",
        KVCacheDType.Q4_1: "q4_1",
        KVCacheDType.IQ4_NL: "iq4_nl",
        KVCacheDType.Q5_0: "q5_0",
        KVCacheDType.Q5_1: "q5_1",
    }
    assert VALUE_MAPS["attention_backend"] == {
        AttentionBackend.AUTO: "auto",
        AttentionBackend.FLASH_ATTENTION: "on",
        AttentionBackend.OFF: "off",
    }
    assert VALUE_MAPS["speculative_method"] == {
        SpeculativeMethod.DRAFT_MODEL: "draft-simple",
        SpeculativeMethod.EAGLE3: "draft-eagle3",
        SpeculativeMethod.MTP: "draft-mtp",
        SpeculativeMethod.DFLASH: "draft-dflash",
        SpeculativeMethod.DSPARK: "draft-dspark",
        SpeculativeMethod.NGRAM: "ngram-simple",
    }
    assert VALUE_MAPS["speculative_mode"] == {
        SpeculativeMode.ENABLED: "enabled",
        SpeculativeMode.DISABLED: "disabled",
    }
    assert VALUE_MAPS["n_cpu_moe"] == {CpuMoeMode.ALL: "all"}
    assert VALUE_MAPS["load_mode"] == {
        LoadMode.AUTO: "auto",
        LoadMode.NONE: "none",
        LoadMode.MMAP: "mmap",
        LoadMode.MLOCK: "mlock",
        LoadMode.MMAP_MLOCK: "mmap+mlock",
        LoadMode.DIO: "dio",
    }


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (SpeculativeMethod.DRAFT_MODEL, "draft-simple"),
        (SpeculativeMethod.EAGLE3, "draft-eagle3"),
        (SpeculativeMethod.MTP, "draft-mtp"),
        (SpeculativeMethod.DFLASH, "draft-dflash"),
        (SpeculativeMethod.DSPARK, "draft-dspark"),
        (SpeculativeMethod.NGRAM, "ngram-simple"),
    ],
)
def test_every_speculative_method_translates(value: SpeculativeMethod, spelling: str) -> None:
    assert serve_arguments(LLMConfig(speculative_method=value)) == ["--spec-type", spelling]


def test_prefix_caching_emits_the_enabling_or_disabling_flag() -> None:
    """Both sides of the switch are explicit, so ``False`` is a real setting."""
    assert serve_arguments(LLMConfig(prefix_caching=True)) == ["--cache-prompt"]
    assert serve_arguments(LLMConfig(prefix_caching=False)) == ["--no-cache-prompt"]


def test_an_unset_prefix_caching_emits_neither_flag() -> None:
    assert serve_arguments(LLMConfig(prefix_caching=None)) == []


def test_speculative_settings_reach_their_own_flags() -> None:
    config = LLMConfig(
        speculative_method=SpeculativeMethod.EAGLE3,
        draft_model="org/eagle3-draft",
        speculative_mode=SpeculativeMode.ENABLED,
        num_speculative_tokens=5,
    )

    assert serve_arguments(config) == [
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-model",
        "org/eagle3-draft",
        "--spec-draft-n-max",
        "5",
    ]


def test_an_unset_mode_emits_the_configured_setup() -> None:
    """An unset mode means the setup's default, which is enabled."""
    config = LLMConfig(speculative_method=SpeculativeMethod.MTP)

    assert serve_arguments(config) == ["--spec-type", "draft-mtp"]


def test_an_unset_depth_omits_only_the_depth_flag() -> None:
    config = LLMConfig(
        speculative_method=SpeculativeMethod.DRAFT_MODEL,
        draft_model="org/draft.gguf",
    )

    assert serve_arguments(config) == [
        "--spec-type",
        "draft-simple",
        "--spec-draft-model",
        "org/draft.gguf",
    ]


def test_the_mode_is_never_forwarded_itself() -> None:
    assert serve_arguments(LLMConfig(speculative_mode=SpeculativeMode.ENABLED)) == []


def test_a_disabled_mode_suppresses_every_speculative_flag() -> None:
    """No speculative flag reaches the engine, so the draft's memory is freed."""
    config = LLMConfig(
        max_model_len=4096,
        speculative_method=SpeculativeMethod.EAGLE3,
        draft_model="org/eagle3-draft",
        speculative_mode=SpeculativeMode.DISABLED,
        num_speculative_tokens=5,
        spec_draft_n_min=2,
        spec_draft_p_split=0.5,
        spec_draft_sampling=SpecDraftSampling.PROBABILISTIC,
        n_gpu_layers_draft=8,
        n_cpu_moe_draft=4,
        draft_kv_cache_dtype=KVCacheDType.Q8_0,
    )

    assert serve_arguments(config) == ["--ctx-size", "4096"]


def test_a_resolved_draft_path_is_never_forwarded(tmp_path: Path) -> None:
    """The resolved path is host-side only; the configured draft string is forwarded."""
    config = LLMConfig(
        speculative_method=SpeculativeMethod.DRAFT_MODEL,
        draft_model="org/draft.gguf",
        resolved_draft_path=tmp_path / "draft.gguf",
    )

    assert serve_arguments(config) == [
        "--spec-type",
        "draft-simple",
        "--spec-draft-model",
        "org/draft.gguf",
    ]


def test_all_tunables_assemble_in_the_declared_order() -> None:
    config = LLMConfig(
        kv_cache_dtype=KVCacheDType.BFLOAT16,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        ubatch_size=2048,
        attention_backend=AttentionBackend.FLASH_ATTENTION,
        speculative_method=SpeculativeMethod.DRAFT_MODEL,
        draft_model="org/draft.gguf",
        speculative_mode=SpeculativeMode.ENABLED,
        num_speculative_tokens=3,
        spec_draft_n_min=2,
        spec_draft_p_split=0.5,
        spec_draft_sampling=SpecDraftSampling.PROBABILISTIC,
        n_gpu_layers_draft=8,
        n_cpu_moe_draft=4,
        draft_kv_cache_dtype=KVCacheDType.BFLOAT16,
        prefix_caching=True,
        n_cpu_moe=16,
        n_cpu_ffn=4,
        kv_offload=True,
        op_offload=False,
        host_buffer=False,
        repack=True,
        swa_full=True,
        load_mode=LoadMode.MMAP_MLOCK,
        threads=8,
        threads_batch=4,
        poll=50,
        prio=2,
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
        "2048",
        "--flash-attn",
        "on",
        "--spec-type",
        "draft-simple",
        "--spec-draft-model",
        "org/draft.gguf",
        "--spec-draft-n-max",
        "3",
        "--spec-draft-n-min",
        "2",
        "--spec-draft-p-split",
        "0.5",
        "--spec-draft-sampling",
        "probabilistic",
        "--n-gpu-layers-draft",
        "8",
        "--n-cpu-moe-draft",
        "4",
        "--cache-type-k-draft",
        "bf16",
        "--cache-type-v-draft",
        "bf16",
        "--cache-prompt",
        "--n-cpu-moe",
        "16",
        "--n-cpu-ffn",
        "4",
        "--kv-offload",
        "--no-op-offload",
        "--no-host",
        "--repack",
        "--swa-full",
        "--load-mode",
        "mmap+mlock",
        "--threads",
        "8",
        "--threads-batch",
        "4",
        "--poll",
        "50",
        "--prio",
        "2",
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
