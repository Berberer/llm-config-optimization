"""Base experiment and objective construction for the optimization runner.

This module is the bridge between the parsed command line and the inference and
optimization layers: it turns the fixed settings the user passed as flags into
the :class:`~llm_config_optimization.inference.types.ExperimentConfig` that the
optimizer varies a search space around, and resolves the chosen objective mode
and its token-mix weights from the same flags. The config's three domains come
straight from the arguments — the container settings, the benchmark workload,
and the non-tunable LLM values — while every tunable serving parameter is left
at its unset default for the space to fill in.

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
from ..optimization import ObjectiveMode, ObjectiveWeights

__all__ = ["build_base_config", "resolve_objective"]


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


def resolve_objective(
    args: argparse.Namespace,
    benchmark: BenchmarkConfig,
) -> tuple[ObjectiveMode, ObjectiveWeights | None]:
    """Resolve the objective mode and its token-mix weights from the arguments.

    The mode is the ``--objective`` choice. The weights are a token mix and
    come from ``--prefill-weight`` and ``--decode-weight`` when given,
    defaulting to the benchmark's ``input_len`` and ``output_len`` for the
    phase those flags leave unset, so ``--objective weighted`` works without
    arithmetic on the standard random workload. Both are resolved and
    validated here rather than during the sweep, so a typo is a usage error
    before the space is loaded or a container is pulled.

    The multi and weighted modes both need a token mix — the weighted one
    scores it, and the multi one uses it to pick the recommended front point —
    so when either side cannot be resolved the run is rejected rather than
    scored under an invented default. The combined mode needs no weights and
    tolerates their absence; a pair that can be resolved is still validated,
    because a value outside the model describes no token mix in any mode.

    Returns the engine-neutral mode and the weights, or ``None`` for the
    combined mode when no pair can be built. Raises ``ValueError`` for a
    missing mix a phase-scoring mode requires and for an invalid pair; the
    caller maps that to a usage error.
    """
    mode = ObjectiveMode(args.objective)
    prefill = args.prefill_weight if args.prefill_weight is not None else benchmark.input_len
    decode = args.decode_weight if args.decode_weight is not None else benchmark.output_len
    if prefill is None or decode is None:
        if mode is ObjectiveMode.COMBINED:
            return mode, None
        raise ValueError(
            f"the {mode.value} objective needs a token mix, but the benchmark defines no "
            f"input/output length: pass both --prefill-weight and --decode-weight"
        )
    return mode, ObjectiveWeights(prefill=float(prefill), decode=float(decode))
