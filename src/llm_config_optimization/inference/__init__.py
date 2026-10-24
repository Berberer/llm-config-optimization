"""Engine-agnostic LLM inference experiment layer.

This subpackage is the public import path for running inference benchmarks::

    from llm_config_optimization.inference import run_experiment

The shared types are engine-neutral. Engine-specific backends (vLLM and
llama.cpp) live alongside them and are re-exported here so callers need not
know the module layout.
"""

from ._benchmark import BenchmarkClient, validate_dataset
from ._llamacpp import LlamaCppBackend
from ._orchestrate import measure_dataset_request_tokens, run_experiment
from ._vllm import VLLMBackend
from .hardware import detect_hardware
from .types import (
    AttentionBackend,
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DetectedHardware,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    HubCheckpoint,
    InferenceBackend,
    KVCacheDType,
    LLMConfig,
    LocalCheckpoint,
    ModelRequest,
    QuantizationMethod,
    RunMetrics,
    ServerSpec,
    SpeculativeMethod,
    SpeculativeMode,
)

__all__ = [
    "AttentionBackend",
    "BenchmarkClient",
    "BenchmarkConfig",
    "BenchmarkOutcome",
    "ContainerRuntimeConfig",
    "DType",
    "DetectedHardware",
    "ExperimentConfig",
    "ExperimentError",
    "ExperimentResult",
    "HardwareTarget",
    "HubCheckpoint",
    "InferenceBackend",
    "KVCacheDType",
    "LLMConfig",
    "LlamaCppBackend",
    "LocalCheckpoint",
    "ModelRequest",
    "QuantizationMethod",
    "RunMetrics",
    "ServerSpec",
    "SpeculativeMethod",
    "SpeculativeMode",
    "VLLMBackend",
    "detect_hardware",
    "measure_dataset_request_tokens",
    "run_experiment",
    "validate_dataset",
]
