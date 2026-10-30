"""Unit tests for the shipped example search spaces.

A shipped example is executable documentation, so it must stay a valid space for
the engine it names: it parses with that backend's applicable parameter set
required and passes the same engine-support validation a real run performs. The
examples are read from disk, so a space that drifts from the loader or a backend
is caught here rather than by a user's sweep.
"""

import json
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    All2AllBackend,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    CpuMoeMode,
    DetectedHardware,
    ExperimentConfig,
    GpuLayersMode,
    HardwareTarget,
    LlamaCppBackend,
    LLMConfig,
    SpeculativeMethod,
    VLLMBackend,
)
from llm_config_optimization.search_space import (
    Categorical,
    ExperimentSpace,
    Fixed,
    load_experiment_space,
)

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
"""The example tree, resolved from the test file rather than the process cwd."""

_EXAMPLE_PATHS = sorted((*_EXAMPLES.glob("vllm/*.yaml"), *_EXAMPLES.glob("llamacpp/*.yaml")))
"""Every shipped example space, across both engines."""

_GPU = DetectedHardware(HardwareTarget.CUDA, 1)
"""A representative accelerator target the CUDA-only example values can run on."""

_MULTI_GPU = DetectedHardware(HardwareTarget.CUDA, 2)
"""A two-device target for the vLLM MoE example, which fixes tensor parallelism at two."""

_EXAMPLE_HARDWARE = {
    "vllm/space-moe.yaml": _MULTI_GPU,
}
"""The examples whose defining configuration needs more than one device."""

_PORTABLE_VLLM_CHECKPOINTS = {
    "dense": {"torch_dtype": "bfloat16"},
    "full-attention": {"torch_dtype": "bfloat16", "layer_types": ["full_attention"]},
    "linear-attention": {
        "torch_dtype": "bfloat16",
        "layer_types": ["linear_attention", "full_attention"],
    },
}
"""The checkpoint facts every shipped vLLM example must serve without a problem."""


def _load_example(path: Path) -> tuple[VLLMBackend | LlamaCppBackend, ExperimentSpace, LLMConfig]:
    """Load a shipped example against its engine's applicable parameter set."""
    backend = VLLMBackend() if path.parent.name == "vllm" else LlamaCppBackend()
    llm = (
        LLMConfig(speculative_method=SpeculativeMethod.MTP)
        if "speculative" in path.stem
        else LLMConfig()
    )
    return backend, load_experiment_space(path, required=backend.applicable_parameters(llm)), llm


@pytest.mark.parametrize(
    "path", _EXAMPLE_PATHS, ids=[f"{path.parent.name}/{path.name}" for path in _EXAMPLE_PATHS]
)
def test_every_example_space_parses_and_passes_backend_validation(path: Path) -> None:
    backend, space, llm = _load_example(path)
    hardware = _EXAMPLE_HARDWARE.get(f"{path.parent.name}/{path.name}", _GPU)

    assert list(backend.validate(space, llm, hardware)) == []


def test_vllm_moe_example_isolates_the_all2all_transport() -> None:
    path = _EXAMPLES / "vllm" / "space-moe.yaml"
    backend, space, llm = _load_example(path)
    dimensions = {dimension.name: dimension for dimension in space.dimensions}

    assert dimensions["tensor_parallel_size"].spec == Fixed(2)
    assert dimensions["expert_parallel"].spec == Fixed(True)
    assert dimensions["all2all_backend"].spec == Categorical(
        (All2AllBackend.ALLGATHER_REDUCESCATTER, All2AllBackend.NAIVE)
    )
    assert list(backend.validate(space, llm, _MULTI_GPU)) == []


def test_llamacpp_moe_example_sweeps_offload_counts_and_full_offload() -> None:
    path = _EXAMPLES / "llamacpp" / "space-moe.yaml"
    backend, space, llm = _load_example(path)
    dimensions = {dimension.name: dimension for dimension in space.dimensions}
    spec = dimensions["n_cpu_moe"].spec

    assert isinstance(spec, Categorical)
    assert 0 in spec.values
    assert CpuMoeMode.ALL in spec.values
    assert any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in spec.values
    )
    assert dimensions["n_gpu_layers"].spec == Fixed(GpuLayersMode.ALL)
    assert list(backend.validate(space, llm, _GPU)) == []


@pytest.mark.parametrize("path", sorted(_EXAMPLES.glob("vllm/*.yaml")), ids=lambda path: path.name)
@pytest.mark.parametrize("checkpoint", _PORTABLE_VLLM_CHECKPOINTS)
def test_vllm_examples_stay_portable_across_checkpoint_dtypes(
    path: Path, checkpoint: str, tmp_path: Path
) -> None:
    """An example must assess cleanly on dense, full-attention, and hybrid checkpoints."""
    backend, space, llm = _load_example(path)
    model = tmp_path / checkpoint
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_PORTABLE_VLLM_CHECKPOINTS[checkpoint]))
    hardware = _EXAMPLE_HARDWARE.get(f"{path.parent.name}/{path.name}", _GPU)
    experiment = ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model="Qwen/Qwen3.5-8B"),
        llm=llm,
    )

    assessment = backend.assess_space(space, experiment, hardware, model)

    assert assessment.problems == ()
