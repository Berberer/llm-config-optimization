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

The base config carries the user's model string unchanged. The backend decides
what that string refers to through its model-resolution hook, and the
preparation stage records the local path it resolved on the benchmark config.
"""

import argparse

from ..inference.types import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    QuantizationMethod,
)

__all__ = ["build_base_config"]


def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the fixed experiment the search space is varied around.

    The container domain takes the target, image, shared-memory size, memory
    limit, and pull policy; the benchmark domain takes the model, the workload
    shape, and the measurement controls; and the LLM domain takes only the
    non-tunable serving values, the quantization method, engine seed, and GPU
    memory utilization. Every tunable serving parameter is left ``None`` so the
    space supplies it, and an omitted ``--target`` or ``--image`` stays ``None``
    so the backend resolves it.
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
        repetitions=args.repetitions,
        warmup=args.warmup_runs,
        startup_timeout=args.startup_timeout,
        timeout=args.timeout,
    )
    llm = LLMConfig(
        quantization=(
            QuantizationMethod(args.quantization) if args.quantization is not None else None
        ),
        seed=args.llm_seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return ExperimentConfig(container=container, benchmark=benchmark, llm=llm)
