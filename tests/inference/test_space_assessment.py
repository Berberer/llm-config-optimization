"""Unit tests for model-aware search-space assessment.

:meth:`VLLMBackend.assess_space` and :meth:`LlamaCppBackend.assess_space` turn a
declared space into the effective one for a resolved checkpoint. Their
observable contract is which dimensions the effective space keeps, what it notes
about each removal, and what it leaves untouched when the checkpoint's metadata
cannot be read. The fixtures are hand-written ``config.json`` and GGUF payloads
written into ``tmp_path``, so both backends are exercised without an engine, a
GPU, or a download. The tests assert on the parameter path and the kind of
reason rather than on exact wording, so notes can be reworded freely.
"""

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    All2AllBackend,
    AttentionBackend,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    CpuMoeMode,
    DetectedHardware,
    DType,
    ExperimentConfig,
    GpuLayersMode,
    HardwareTarget,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    VLLMBackend,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    SpaceAssessment,
    Spec,
)

MODEL = "Qwen/Qwen3.5-8B"
"""A Hub id, which only stands in as the experiment's model field here."""

_CUDA = DetectedHardware(target=HardwareTarget.CUDA, device_count=2)
_CPU = DetectedHardware(target=HardwareTarget.CPU, device_count=1)

VLLM_SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Fixed(DType.BFLOAT16)),
        Dimension("kv_cache_dtype", Default()),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(128)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("max_num_active_seqs", Fixed(16)),
        Dimension("max_num_scheduled_tokens", Fixed(2048)),
        Dimension("tensor_parallel_size", Categorical((1, 2))),
        Dimension("attention_backend", Default()),
        Dimension("prefix_caching", Fixed(True)),
        Dimension("expert_parallel", Fixed(True)),
        Dimension(
            "all2all_backend",
            Categorical((All2AllBackend.ALLGATHER_REDUCESCATTER, All2AllBackend.NAIVE)),
        ),
    )
)
"""A canonical-order vLLM space, including both MoE dimensions."""

VLLM_MOE = ("expert_parallel", "all2all_backend")
"""The vLLM dimensions that only affect a Mixture-of-Experts checkpoint."""

LLAMACPP_SPACE = ExperimentSpace(
    dimensions=(
        Dimension("kv_cache_dtype", Default()),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(4)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION)),
        Dimension("prefix_caching", Fixed(True)),
        Dimension("n_gpu_layers", Fixed(GpuLayersMode.AUTO)),
        Dimension("n_cpu_moe", Categorical((0, 8, CpuMoeMode.ALL))),
    )
)
"""A canonical-order llama.cpp space, including the CPU expert offload."""

EXPERTS_FACT = "experts"
"""The checkpoint fact every pruning note must name, rather than its wording."""

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_STRING = 8
_UINT32 = 4


def _experiment() -> ExperimentConfig:
    """Build the run's fixed configuration, which no MoE rule depends on."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=MODEL),
        llm=LLMConfig(),
    )


def _write_checkpoint(directory: Path, payload: object) -> Path:
    """Write a vLLM ``config.json`` with the given JSON-serialisable payload."""
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _encode_string(text: str) -> bytes:
    """Encode a length-prefixed UTF-8 GGUF string."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(value: object) -> bytes:
    """Encode one GGUF metadata value with the type the parser expects."""
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _encode_string(value)
    return struct.pack("<II", _UINT32, value)


def _write_gguf(path: Path, metadata: Mapping[str, object]) -> Path:
    """Write a minimal GGUF container holding only the given metadata entries."""
    entries = b"".join(
        _encode_string(key) + _encode_value(value) for key, value in metadata.items()
    )
    path.write_bytes(_GGUF_MAGIC + struct.pack("<IQQ", _GGUF_VERSION, 0, len(metadata)) + entries)
    return path


def _assess_vllm(definition: ExperimentSpace, model: Path) -> SpaceAssessment:
    """Assess ``definition`` against the checkpoint at ``model`` for vLLM."""
    return VLLMBackend().assess_space(definition, _experiment(), _CUDA, model)


def _assess_llamacpp(
    definition: ExperimentSpace,
    model: Path,
    *,
    hardware: DetectedHardware = _CUDA,
) -> SpaceAssessment:
    """Assess ``definition`` against the checkpoint at ``model`` for llama.cpp.

    The default target is a GPU host so a test about the checkpoint's own
    facts does not also exercise the CPU-target rules.
    """
    return LlamaCppBackend().assess_space(definition, _experiment(), hardware, model)


def _retained(definition: ExperimentSpace, dropped: tuple[str, ...]) -> tuple[Dimension, ...]:
    """The dimensions ``definition`` declares apart from ``dropped``, in order."""
    return tuple(dimension for dimension in definition.dimensions if dimension.name not in dropped)


def test_a_dense_checkpoint_drops_both_moe_dimensions(tmp_path: Path) -> None:
    assessment = _assess_vllm(VLLM_SPACE, _write_checkpoint(tmp_path, {"num_experts": 0}))

    assert assessment.space.dimensions == _retained(VLLM_SPACE, VLLM_MOE)
    assert assessment.problems == ()


def test_a_dense_checkpoint_notes_every_dropped_dimension(tmp_path: Path) -> None:
    assessment = _assess_vllm(VLLM_SPACE, _write_checkpoint(tmp_path, {}))

    assert len(assessment.notes) == len(VLLM_MOE)
    for name, note in zip(VLLM_MOE, assessment.notes, strict=True):
        assert f"space.{name}" in note
        assert EXPERTS_FACT in note


@pytest.mark.parametrize(
    "payload",
    [
        {"num_experts": 128},
        {"n_routed_experts": 64},
        {"num_local_experts": 8},
        {"text_config": {"num_experts": 128}},
    ],
    ids=["top-level", "routed", "local", "nested"],
)
def test_an_moe_checkpoint_keeps_both_moe_dimensions(payload: object, tmp_path: Path) -> None:
    assessment = _assess_vllm(VLLM_SPACE, _write_checkpoint(tmp_path, payload))

    assert assessment == SpaceAssessment(space=VLLM_SPACE)


def test_a_missing_configuration_is_unknown_rather_than_dense(tmp_path: Path) -> None:
    """Unreadable metadata fails open, so the declared dimensions are kept."""
    assert _assess_vllm(VLLM_SPACE, tmp_path) == SpaceAssessment(space=VLLM_SPACE)


def test_a_malformed_configuration_is_unknown_rather_than_dense(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not valid json", encoding="utf-8")

    assert _assess_vllm(VLLM_SPACE, tmp_path) == SpaceAssessment(space=VLLM_SPACE)


def test_a_non_object_configuration_is_unknown_rather_than_dense(tmp_path: Path) -> None:
    assert _assess_vllm(VLLM_SPACE, _write_checkpoint(tmp_path, [1, 2, 3])) == SpaceAssessment(
        space=VLLM_SPACE
    )


def test_a_dense_checkpoint_drops_the_cpu_expert_offload(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf", {"general.architecture": "llama", "llama.expert_count": 0}
    )

    assessment = _assess_llamacpp(LLAMACPP_SPACE, model)

    assert assessment.space.dimensions == _retained(LLAMACPP_SPACE, ("n_cpu_moe",))
    assert assessment.problems == ()
    assert len(assessment.notes) == 1
    assert "space.n_cpu_moe" in assessment.notes[0]
    assert EXPERTS_FACT in assessment.notes[0]


def test_a_header_without_an_expert_count_is_confidently_dense(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})

    assessment = _assess_llamacpp(LLAMACPP_SPACE, model)

    assert assessment.space.dimensions == _retained(LLAMACPP_SPACE, ("n_cpu_moe",))


def test_an_moe_checkpoint_keeps_the_cpu_expert_offload(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf", {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 128}
    )

    assert _assess_llamacpp(LLAMACPP_SPACE, model) == SpaceAssessment(space=LLAMACPP_SPACE)


def test_an_unparseable_header_is_unknown_rather_than_dense(tmp_path: Path) -> None:
    """A GGUF file whose header cannot be parsed fails open, keeping the dimension."""
    model = tmp_path / "model.gguf"
    model.write_bytes(_GGUF_MAGIC)

    assert _assess_llamacpp(LLAMACPP_SPACE, model) == SpaceAssessment(space=LLAMACPP_SPACE)


def test_a_non_gguf_checkpoint_is_unknown_rather_than_dense(tmp_path: Path) -> None:
    """``validate_model`` reports a non-GGUF file, so the assessment never fails twice."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a GGUF file")

    assert _assess_llamacpp(LLAMACPP_SPACE, model) == SpaceAssessment(space=LLAMACPP_SPACE)


def test_an_empty_effective_space_is_representable(tmp_path: Path) -> None:
    """Pruning every dimension is not an error here; the caller decides what it means."""
    definition = ExperimentSpace(
        dimensions=(
            Dimension("expert_parallel", Fixed(True)),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {}))

    assert assessment.space.dimensions == ()


MOE_CHECKPOINT: dict[str, object] = {
    "general.architecture": "qwen3moe",
    "qwen3moe.expert_count": 8,
    "qwen3moe.block_count": 32,
}
"""An llama.cpp checkpoint that declares experts and a block count."""


def _moe_space() -> ExperimentSpace:
    """A resolvable llama.cpp space that isolates the MoE dimensions."""
    return ExperimentSpace(
        dimensions=(
            Dimension("n_gpu_layers", Default()),
            Dimension("n_cpu_moe", Categorical((0, 8, CpuMoeMode.ALL))),
        )
    )


def _layers_space(spec: Spec) -> ExperimentSpace:
    """A llama.cpp space declaring only ``n_gpu_layers`` with ``spec``."""
    return ExperimentSpace(dimensions=(Dimension("n_gpu_layers", spec),))


@pytest.mark.parametrize(
    "spec",
    [
        Fixed(1),
        Categorical((1,)),
        Interval(1, 1),
    ],
    ids=["fixed", "categorical", "interval"],
)
def test_a_space_that_never_splits_drops_both_moe_dimensions(spec: Spec, tmp_path: Path) -> None:
    definition = ExperimentSpace(
        dimensions=(
            Dimension("tensor_parallel_size", spec),
            Dimension("expert_parallel", Fixed(True)),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {"num_experts": 8}))

    assert assessment.space.dimensions == (Dimension("tensor_parallel_size", spec),)
    assert len(assessment.notes) == 2
    assert {note.split(":")[0] for note in assessment.notes} == {
        "dropped space.expert_parallel",
        "dropped space.all2all_backend",
    }
    for note in assessment.notes:
        assert "tensor-parallel" in note


@pytest.mark.parametrize(
    "spec",
    [Fixed(2), Categorical((1, 2)), Interval(1, 4)],
    ids=["fixed", "categorical", "interval"],
)
def test_a_split_capable_space_keeps_both_moe_dimensions(spec: Spec, tmp_path: Path) -> None:
    definition = ExperimentSpace(
        dimensions=(
            Dimension("tensor_parallel_size", spec),
            Dimension("expert_parallel", Categorical((False, True))),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {"num_experts": 8}))

    assert assessment == SpaceAssessment(space=definition)


def test_a_deferred_tensor_parallel_size_resolves_through_the_device_count(tmp_path: Path) -> None:
    """``Default`` becomes the detected device count, exactly as planning does."""
    definition = ExperimentSpace(
        dimensions=(
            Dimension("tensor_parallel_size", Default()),
            Dimension("expert_parallel", Fixed(True)),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )
    model = _write_checkpoint(tmp_path, {"num_experts": 8})

    single = VLLMBackend().assess_space(
        definition,
        _experiment(),
        DetectedHardware(target=HardwareTarget.CUDA, device_count=1),
        model,
    )
    split = VLLMBackend().assess_space(
        definition,
        _experiment(),
        DetectedHardware(target=HardwareTarget.CUDA, device_count=2),
        model,
    )

    assert single.space.dimensions == (Dimension("tensor_parallel_size", Default()),)
    assert split == SpaceAssessment(space=definition)


@pytest.mark.parametrize(
    "spec",
    [Fixed(False), Categorical((False,)), Default()],
    ids=["fixed", "categorical", "deferred"],
)
def test_a_space_that_never_enables_expert_parallel_drops_only_the_transport(
    spec: Spec, tmp_path: Path
) -> None:
    definition = ExperimentSpace(
        dimensions=(
            Dimension("tensor_parallel_size", Categorical((1, 2))),
            Dimension("expert_parallel", spec),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {"num_experts": 8}))

    assert assessment.space.dimensions == _retained(definition, ("all2all_backend",))
    assert len(assessment.notes) == 1
    assert "space.all2all_backend" in assessment.notes[0]
    assert "expert parallelism" in assessment.notes[0]


def test_a_missing_tensor_parallel_dimension_leaves_the_dependency_rules_inactive(
    tmp_path: Path,
) -> None:
    """A partial space establishes no ceiling, so neither rule may fire."""
    definition = ExperimentSpace(
        dimensions=(
            Dimension("expert_parallel", Fixed(True)),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assert _assess_vllm(definition, _write_checkpoint(tmp_path, {"num_experts": 8})) == (
        SpaceAssessment(space=definition)
    )


def test_a_dense_checkpoint_notes_survive_the_dependency_rules(tmp_path: Path) -> None:
    """The dependency pass must not add a second note for an already dropped dimension."""
    definition = ExperimentSpace(
        dimensions=(
            Dimension("tensor_parallel_size", Categorical((1, 2))),
            Dimension("expert_parallel", Fixed(False)),
            Dimension("all2all_backend", Categorical((All2AllBackend.NAIVE,))),
        )
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {}))

    assert assessment.space.dimensions == _retained(definition, VLLM_MOE)
    assert len(assessment.notes) == len(VLLM_MOE)


def _dtype_space(dtype: Spec, cache: Spec) -> ExperimentSpace:
    """A partial vLLM space that isolates the dtype and KV-cache dimensions."""
    return ExperimentSpace(
        dimensions=(
            Dimension("dtype", dtype),
            Dimension("kv_cache_dtype", cache),
        )
    )


def _linear_attention_checkpoint(directory: Path, dtype: str = "bfloat16") -> Path:
    """Write a vLLM checkpoint whose layers mix linear and full attention."""
    return _write_checkpoint(
        directory,
        {
            "torch_dtype": dtype,
            "text_config": {
                "dtype": dtype,
                "layer_types": ["linear_attention", "full_attention"],
            },
        },
    )


def test_a_linear_attention_checkpoint_narrows_float32_out_of_the_dtype(tmp_path: Path) -> None:
    definition = _dtype_space(
        Categorical((DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16)),
        Default(),
    )

    assessment = _assess_vllm(definition, _linear_attention_checkpoint(tmp_path))

    assert assessment.space.dimensions == (
        Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16))),
        Dimension("kv_cache_dtype", Default()),
    )
    assert assessment.problems == ()
    assert len(assessment.notes) == 1
    assert "space.dtype.categorical[0]" in assessment.notes[0]
    assert "float32" in assessment.notes[0]
    assert "linear-attention" in assessment.notes[0]


def test_narrowing_a_dtype_keeps_the_declared_value_order(tmp_path: Path) -> None:
    definition = _dtype_space(
        Categorical((DType.BFLOAT16, DType.FLOAT32, DType.FLOAT16)),
        Default(),
    )

    assessment = _assess_vllm(definition, _linear_attention_checkpoint(tmp_path))

    assert assessment.space.dimensions[0] == Dimension(
        "dtype", Categorical((DType.BFLOAT16, DType.FLOAT16))
    )
    assert "space.dtype.categorical[1]" in assessment.notes[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"torch_dtype": "bfloat16", "layer_types": ["full_attention"]},
        {"torch_dtype": "bfloat16"},
        {},
    ],
    ids=["full-attention", "no-layer-types", "empty-config"],
)
def test_a_checkpoint_without_linear_attention_keeps_float32(
    payload: object, tmp_path: Path
) -> None:
    definition = _dtype_space(Categorical((DType.FLOAT32, DType.BFLOAT16)), Default())

    assert _assess_vllm(definition, _write_checkpoint(tmp_path, payload)) == SpaceAssessment(
        space=definition
    )


def test_an_unreadable_checkpoint_keeps_float32(tmp_path: Path) -> None:
    definition = _dtype_space(Categorical((DType.FLOAT32, DType.BFLOAT16)), Default())

    assert _assess_vllm(definition, tmp_path) == SpaceAssessment(space=definition)


def test_a_float32_only_dtype_on_a_linear_attention_checkpoint_is_a_problem(
    tmp_path: Path,
) -> None:
    definition = _dtype_space(Fixed(DType.FLOAT32), Default())

    assessment = _assess_vllm(definition, _linear_attention_checkpoint(tmp_path))

    assert assessment.space == definition
    assert assessment.notes == ()
    assert len(assessment.problems) == 1
    assert "space.dtype" in assessment.problems[0]
    assert "float32" in assessment.problems[0]


def test_a_matching_pinned_dtype_keeps_the_float_cache(tmp_path: Path) -> None:
    definition = _dtype_space(
        Fixed(DType.FLOAT16),
        Categorical((KVCacheDType.AUTO, KVCacheDType.FLOAT16, KVCacheDType.FP8_E4M3)),
    )

    assert _assess_vllm(definition, _write_checkpoint(tmp_path, {"torch_dtype": "float16"})) == (
        SpaceAssessment(space=definition)
    )


@pytest.mark.parametrize("dtype", [Fixed(DType.AUTO), Default()], ids=["auto", "deferred"])
def test_an_auto_dtype_resolved_from_the_checkpoint_keeps_the_matching_cache(
    dtype: Spec, tmp_path: Path
) -> None:
    definition = _dtype_space(dtype, Categorical((KVCacheDType.AUTO, KVCacheDType.BFLOAT16)))

    assert _assess_vllm(definition, _write_checkpoint(tmp_path, {"torch_dtype": "bfloat16"})) == (
        SpaceAssessment(space=definition)
    )


def test_a_float_cache_the_dtype_set_cannot_all_match_is_narrowed(tmp_path: Path) -> None:
    definition = _dtype_space(
        Categorical((DType.FLOAT16, DType.BFLOAT16)),
        Categorical(
            (
                KVCacheDType.FLOAT16,
                KVCacheDType.BFLOAT16,
                KVCacheDType.AUTO,
                KVCacheDType.FP8_E4M3,
            )
        ),
    )

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {"torch_dtype": "bfloat16"}))

    assert assessment.space.dimensions == (
        Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16))),
        Dimension("kv_cache_dtype", Categorical((KVCacheDType.AUTO, KVCacheDType.FP8_E4M3))),
    )
    assert assessment.problems == ()
    assert len(assessment.notes) == 2
    assert "space.kv_cache_dtype.categorical[0]" in assessment.notes[0]
    assert "float16" in assessment.notes[0]
    assert "space.kv_cache_dtype.categorical[1]" in assessment.notes[1]
    assert "bfloat16" in assessment.notes[1]


@pytest.mark.parametrize(
    "cache",
    [KVCacheDType.AUTO, KVCacheDType.FP8_E4M3, KVCacheDType.FP8_E5M2],
    ids=["auto", "fp8-e4m3", "fp8-e5m2"],
)
def test_auto_and_fp8_cache_formats_are_always_retained(
    cache: KVCacheDType, tmp_path: Path
) -> None:
    definition = _dtype_space(Fixed(DType.BFLOAT16), Fixed(cache))

    assert _assess_vllm(definition, _write_checkpoint(tmp_path, {"torch_dtype": "float16"})) == (
        SpaceAssessment(space=definition)
    )


def test_a_float_cache_with_no_matching_dtype_is_a_problem(tmp_path: Path) -> None:
    definition = _dtype_space(Fixed(DType.AUTO), Fixed(KVCacheDType.FLOAT16))

    assessment = _assess_vllm(definition, _write_checkpoint(tmp_path, {"torch_dtype": "bfloat16"}))

    assert assessment.space == definition
    assert assessment.notes == ()
    assert len(assessment.problems) == 1
    assert "space.kv_cache_dtype" in assessment.problems[0]
    assert "float" in assessment.problems[0]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"torch_dtype": "auto"},
        {"torch_dtype": "float8_e4m3fn"},
        {"torch_dtype": 16},
    ],
    ids=["unreadable", "empty-config", "auto", "unmodelled", "non-string"],
)
def test_an_unknown_model_dtype_leaves_the_float_cache_alone(
    payload: object, tmp_path: Path
) -> None:
    """Unknown metadata is not evidence a cache value is wrong: the engine decides."""
    if payload is not None:
        _write_checkpoint(tmp_path, payload)
    definition = _dtype_space(Fixed(DType.AUTO), Fixed(KVCacheDType.FLOAT16))

    assert _assess_vllm(definition, tmp_path) == SpaceAssessment(space=definition)


def test_the_cache_is_reconciled_after_the_dtype_is_narrowed(tmp_path: Path) -> None:
    definition = _dtype_space(
        Categorical((DType.FLOAT32, DType.BFLOAT16)),
        Categorical((KVCacheDType.BFLOAT16, KVCacheDType.FLOAT16)),
    )

    assessment = _assess_vllm(definition, _linear_attention_checkpoint(tmp_path))

    assert assessment.space.dimensions == (
        Dimension("dtype", Categorical((DType.BFLOAT16,))),
        Dimension("kv_cache_dtype", Categorical((KVCacheDType.BFLOAT16,))),
    )
    assert assessment.problems == ()
    assert len(assessment.notes) == 2
    assert "space.dtype.categorical[0]" in assessment.notes[0]
    assert "space.kv_cache_dtype.categorical[1]" in assessment.notes[1]


def test_a_cpu_target_drops_the_cpu_expert_offload(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)

    assessment = _assess_llamacpp(_moe_space(), model, hardware=_CPU)

    assert assessment.space.dimensions == _retained(_moe_space(), ("n_cpu_moe",))
    assert len(assessment.notes) == 1
    assert "space.n_cpu_moe" in assessment.notes[0]
    assert "CPU" in assessment.notes[0]


def test_a_gpu_target_keeps_the_cpu_expert_offload(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)

    assert _assess_llamacpp(_moe_space(), model) == SpaceAssessment(space=_moe_space())


@pytest.mark.parametrize(
    "spec",
    [
        Fixed(GpuLayersMode.AUTO),
        Fixed(0),
        Categorical((GpuLayersMode.AUTO, 0)),
        Interval(0, 0),
    ],
    ids=["auto", "zero", "categorical", "interval"],
)
def test_a_cpu_target_drops_a_gpu_layer_dimension_that_cannot_offload(
    spec: Spec, tmp_path: Path
) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)

    assessment = _assess_llamacpp(_layers_space(spec), model, hardware=_CPU)

    assert assessment.space.dimensions == ()
    assert len(assessment.notes) == 1
    assert "space.n_gpu_layers" in assessment.notes[0]
    assert "CPU" in assessment.notes[0]


def test_a_cpu_target_keeps_a_deferred_gpu_layer_dimension(tmp_path: Path) -> None:
    """A deferral declares no value, so it is not confidently CPU-valid."""
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)
    definition = _layers_space(Default())

    assert _assess_llamacpp(definition, model, hardware=_CPU) == SpaceAssessment(space=definition)


def test_an_exact_cpu_expert_count_above_the_block_count_is_a_problem(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)
    definition = ExperimentSpace(
        dimensions=(Dimension("n_cpu_moe", Categorical((0, 64, CpuMoeMode.ALL))),)
    )

    assessment = _assess_llamacpp(definition, model)

    assert assessment.space == definition
    assert len(assessment.problems) == 1
    assert "space.n_cpu_moe.categorical[1]" in assessment.problems[0]
    assert "64" in assessment.problems[0]
    assert "32" in assessment.problems[0]
    assert "n_cpu_moe=all" in assessment.problems[0]


def test_an_exact_gpu_layer_count_above_the_block_count_is_a_problem(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)
    definition = _layers_space(Fixed(64))

    assessment = _assess_llamacpp(definition, model)

    assert len(assessment.problems) == 1
    assert "space.n_gpu_layers.fixed" in assessment.problems[0]
    assert "n_gpu_layers=all" in assessment.problems[0]


@pytest.mark.parametrize("name", ["n_gpu_layers", "n_cpu_moe"])
def test_an_interval_above_the_block_count_is_a_problem(name: str, tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", MOE_CHECKPOINT)
    definition = ExperimentSpace(dimensions=(Dimension(name, Interval(0, 64)),))

    assessment = _assess_llamacpp(definition, model)

    assert len(assessment.problems) == 1
    assert f"space.{name}.interval" in assessment.problems[0]
    assert "64" in assessment.problems[0]


@pytest.mark.parametrize(
    "spec",
    [Fixed(GpuLayersMode.ALL), Fixed(GpuLayersMode.AUTO), Default()],
    ids=["all", "auto", "deferred"],
)
def test_a_gpu_layer_mode_is_never_bounded(spec: Spec, tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama", "llama.block_count": 2},
    )

    assert _assess_llamacpp(_layers_space(spec), model).problems == ()


def test_the_all_cpu_expert_mode_is_never_bounded(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama", "llama.block_count": 2},
    )
    definition = ExperimentSpace(dimensions=(Dimension("n_cpu_moe", Fixed(CpuMoeMode.ALL)),))

    assert _assess_llamacpp(definition, model).problems == ()


def test_a_checkpoint_without_a_block_count_is_not_bounded(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})
    definition = _layers_space(Fixed(64))

    assert _assess_llamacpp(definition, model).problems == ()


def test_an_unparseable_header_is_not_bounded(tmp_path: Path) -> None:
    """Unreadable block-count metadata fails open, so no count is rejected."""
    model = tmp_path / "model.gguf"
    model.write_bytes(_GGUF_MAGIC)
    definition = _layers_space(Fixed(64))

    assessment = _assess_llamacpp(definition, model)

    assert assessment == SpaceAssessment(space=definition)
