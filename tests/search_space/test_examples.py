"""Unit tests for the shipped example search spaces.

A shipped example is executable documentation, so it must stay a valid space for
the engine it names: it parses with that backend's applicable parameter set
required and passes the same engine-support validation a real run performs. The
examples are read from disk, so a space that drifts from the loader or a backend
is caught here rather than by a user's sweep.
"""

from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    DetectedHardware,
    HardwareTarget,
    LlamaCppBackend,
    LLMConfig,
    SpeculativeMethod,
    VLLMBackend,
)
from llm_config_optimization.search_space import load_experiment_space

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
"""The example tree, resolved from the test file rather than the process cwd."""

_EXAMPLE_PATHS = sorted((*_EXAMPLES.glob("vllm/*.yaml"), *_EXAMPLES.glob("llamacpp/*.yaml")))
"""Every shipped example space, across both engines."""

_GPU = DetectedHardware(HardwareTarget.CUDA, 1)
"""A representative accelerator target the CUDA-only example values can run on."""


@pytest.mark.parametrize(
    "path", _EXAMPLE_PATHS, ids=[f"{path.parent.name}/{path.name}" for path in _EXAMPLE_PATHS]
)
def test_every_example_space_parses_and_passes_backend_validation(path: Path) -> None:
    backend = VLLMBackend() if path.parent.name == "vllm" else LlamaCppBackend()
    llm = (
        LLMConfig(speculative_method=SpeculativeMethod.MTP)
        if "speculative" in path.stem
        else LLMConfig()
    )

    space = load_experiment_space(path, required=backend.applicable_parameters(llm))

    assert list(backend.validate(space, llm, _GPU)) == []
