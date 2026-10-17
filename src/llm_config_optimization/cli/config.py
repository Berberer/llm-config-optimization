"""Base experiment construction for the optimization runner.

This module is the bridge between the parsed command line and the inference
layer: it turns the fixed settings the user passed as flags into the
:class:`~llm_config_optimization.inference.types.ExperimentConfig` that the optimizer
varies a search space around. The config's three domains come straight from the
arguments — the container settings, the benchmark workload, and the non-tunable
LLM values — while every tunable serving parameter is left at its unset default
for the space to fill in.

Two container values are deliberately left unresolved: ``target`` and ``image``
stay ``None`` when the corresponding flag is omitted, so the backend's own
hardware detection and default-image lookup apply at planning time, exactly as
they do for a hand-written config. The preparation stage resolves and reports
them separately rather than baking a guess into the base config.

The module also owns the single test for whether the model argument names a
local checkpoint: :func:`is_local_model` reports whether the argument is an
existing host path, and anything else is treated as a Hugging Face id that the
preparation stage must download. Existence is the only test, matching how the
vLLM backend decides whether to bind-mount a checkpoint or pass the identifier
through to the engine.
"""

import argparse
from pathlib import Path

from ..inference.types import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    QuantizationMethod,
)

__all__ = ["build_base_config", "is_local_model"]


def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the fixed experiment the search space is varied around.

    The container domain takes the target, image, shared-memory size, memory
    limit, and pull policy; the benchmark domain takes the model and the
    workload shape; and the LLM domain takes only the non-tunable serving
    values, the quantization method, engine seed, and GPU memory utilization.
    Every tunable serving parameter is left ``None`` so the space supplies it,
    and an omitted ``--target`` or ``--image`` stays ``None`` so the backend
    resolves it.
    """
    container = ContainerRuntimeConfig(
        target=HardwareTarget(args.target) if args.target is not None else None,
        image=args.image,
        mem_limit=args.mem_limit,
        shm_size=args.shm_size,
        pull_policy=args.pull_policy,
    )
    benchmark = BenchmarkConfig(
        model=args.model,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        timeout=args.run_timeout,
    )
    llm = LLMConfig(
        quantization=(
            QuantizationMethod(args.quantization) if args.quantization is not None else None
        ),
        seed=args.llm_seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return ExperimentConfig(container=container, benchmark=benchmark, llm=llm)


def is_local_model(model: str) -> bool:
    """Return whether ``model`` names an existing host checkpoint.

    An existing path is used directly and needs no download; anything else is
    treated as a Hugging Face id and fetched by the preparation stage. The test
    is existence only, matching the vLLM backend, so a Hugging Face identifier
    that happens to resolve to a local file is treated as a checkpoint.
    """
    return Path(model).expanduser().exists()
