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
    All2AllBackend,
    AttentionBackend,
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    CpuMoeMode,
    DetectedHardware,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    FitMode,
    GpuLayersMode,
    HardwareTarget,
    HubCheckpoint,
    InferenceBackend,
    KVCacheDType,
    LLMConfig,
    LoadMode,
    LocalCheckpoint,
    ModelRequest,
    QuantizationMethod,
    RunMetrics,
    ServerSpec,
    SpecDraftSampling,
    SpeculativeMethod,
    SpeculativeMode,
)

__all__ = [
    "All2AllBackend",
    "AttentionBackend",
    "BenchmarkClient",
    "BenchmarkConfig",
    "BenchmarkOutcome",
    "ContainerRuntimeConfig",
    "CpuMoeMode",
    "DType",
    "DetectedHardware",
    "ExperimentConfig",
    "ExperimentError",
    "ExperimentResult",
    "FitMode",
    "GpuLayersMode",
    "HardwareTarget",
    "HubCheckpoint",
    "InferenceBackend",
    "KVCacheDType",
    "LLMConfig",
    "LlamaCppBackend",
    "LoadMode",
    "LocalCheckpoint",
    "ModelRequest",
    "QuantizationMethod",
    "RunMetrics",
    "ServerSpec",
    "SpecDraftSampling",
    "SpeculativeMethod",
    "SpeculativeMode",
    "VLLMBackend",
    "detect_hardware",
    "measure_dataset_request_tokens",
    "run_experiment",
    "validate_dataset",
]
