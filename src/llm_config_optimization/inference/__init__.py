"""Engine-agnostic LLM inference experiment layer.

This subpackage is the public import path for running inference benchmarks::

    from llm_config_optimization.inference import run_experiment

The shared types are engine-neutral. Engine-specific backends (starting with
vLLM) live alongside them and are re-exported here so callers need not know the
module layout.
"""

from ._benchmark import BenchmarkClient
from ._orchestrate import run_experiment
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
    "LocalCheckpoint",
    "ModelRequest",
    "QuantizationMethod",
    "RunMetrics",
    "ServerSpec",
    "VLLMBackend",
    "detect_hardware",
    "run_experiment",
]
