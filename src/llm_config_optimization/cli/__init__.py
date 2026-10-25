"""Command-line interface for the optimization runner.

This subpackage is the composition root that turns the console script into a
usable tool: it parses the command line, builds the fixed
:class:`~llm_config_optimization.inference.types.ExperimentConfig` the search space is
varied around, reports progress, and maps outcomes onto exit codes. The
modules here own argument parsing, base-config construction, the model download
progress line, the log and TUI reporters, and the final summary and
configuration writer.

The package is driven through a single :func:`main` entry point that returns an
integer exit code, so both the console script and ``python -m llm_config_optimization``
can share one implementation. The inference engine is chosen by the required
``ENGINE`` positional and resolved through a small registry of backend factories,
so the rest of the composition root never imports an engine class. That registry
also records which command-line options each engine rejects, so an option listed
for rejection is a usage error rather than a value the engine would silently
ignore.
``main`` is a pure composition root: it loads the search space, resolves the
hardware target,
validates the space against that target, builds the base configuration, resolves
the model to a local directory, validates the model against that configuration,
pulls the container image, and then calls the optimization adapter with the
injected backend, client, and reporter. Nothing about the study itself is
duplicated here.

Exit codes are distinct so scripts can react to each outcome: ``0`` success,
``2`` a usage error (including an explicit TUI request without a terminal),
``3`` an invalid or unreadable search-space definition, ``4`` a configuration
the backend does not support (the search space or the resolved model), ``5`` a
sweep that finished with no completed trial, ``130`` an interrupt, and ``1`` any
unexpected error. Progress and the summary go to stdout; warnings and errors go
to stderr.
"""

import argparse
import sys
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import optuna

from ..inference import (
    BenchmarkClient,
    LlamaCppBackend,
    LocalCheckpoint,
    VLLMBackend,
    detect_hardware,
    measure_dataset_request_tokens,
    validate_dataset,
)
from ..inference.types import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    HardwareTarget,
    InferenceBackend,
    LLMConfig,
    ServerSpec,
)
from ..optimization import (
    ObjectiveMode,
    ObjectiveWeights,
    OptimizationError,
    OptimizationInterrupted,
    OptimizationResult,
    optimize,
)
from ..output import OutputFormat, RenderContext, writer_for
from ..runtime import ContainerRunError, pull_image
from ..search_space import (
    ExperimentDefinitionError,
    ExperimentSpace,
    load_experiment_space,
    validate_space,
)
from .arguments import parse_args
from .config import build_base_config, resolve_objective
from .download import download_model
from .log import LogReporter
from .summary import render_summary
from .tui import TuiReporter

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = ["main"]

_EXIT_SUCCESS = 0
_EXIT_ERROR = 1
_EXIT_USAGE = 2
_EXIT_DEFINITION = 3
_EXIT_UNSUPPORTED = 4
_EXIT_EMPTY_SWEEP = 5
_EXIT_INTERRUPTED = 130


@contextmanager
def _vllm_backend() -> Generator[InferenceBackend]:
    """Build the vLLM backend with a per-run compile cache.

    The cache directory lives for the whole sweep, so the trials of one run
    reuse the kernels the first one built, and is removed when the run ends.
    """
    with tempfile.TemporaryDirectory(
        prefix="llm-config-optimization-cache-", ignore_cleanup_errors=True
    ) as cache_directory:
        yield VLLMBackend(cache_dir=Path(cache_directory))


def _vllm_deployment_backend() -> InferenceBackend:
    """Build vLLM without the trial-scoped compile cache for an artifact plan.

    A generated quadlet or Compose file describes the server to deploy, not the
    scratch resources a benchmark run used. Planning through a cache-less
    backend keeps the ephemeral compile-cache mount and its redirect environment
    out of the artifact, so the rendered file is deterministic and every host
    path in it outlives the run.
    """
    return VLLMBackend()


@contextmanager
def _llamacpp_backend() -> Generator[InferenceBackend]:
    """Build the llama.cpp backend for the duration of a sweep.

    llama.cpp needs no per-run host resources, so the factory only yields the
    backend; it follows the registry's uniform context-manager contract rather
    than leaking the engine's construction into the rest of the CLI.
    """
    yield LlamaCppBackend()


def _llamacpp_deployment_backend() -> InferenceBackend:
    """Build llama.cpp for an artifact plan.

    llama.cpp owns no trial-scoped host resources, so its deployment plan is its
    regular plan; the factory exists to keep the registry contract uniform.
    """
    return LlamaCppBackend()


@dataclass(frozen=True, slots=True)
class _EngineEntry:
    """One registered engine: its sweep and deployment factories and rejected options.

    ``factory`` builds the backend for the duration of a sweep and is a context
    manager, so an engine can own the resources it needs (vLLM's per-run compile
    cache) without leaking that detail into the CLI. ``deployment_factory``
    builds a backend used only to plan a written artifact: it must omit
    trial-scoped resources, because a deployed server must not depend on a cache
    the runner removes. ``rejected_options`` names the command-line options that
    are meaningful only for other engines; passing one of them selects a usage
    error instead of a value the backend would silently ignore.
    """

    factory: Callable[[], AbstractContextManager[InferenceBackend]]
    deployment_factory: Callable[[], InferenceBackend]
    rejected_options: tuple[str, ...] = ()


_ENGINES: Mapping[str, _EngineEntry] = {
    "vllm": _EngineEntry(
        factory=_vllm_backend,
        deployment_factory=_vllm_deployment_backend,
    ),
    "llamacpp": _EngineEntry(
        factory=_llamacpp_backend,
        deployment_factory=_llamacpp_deployment_backend,
        rejected_options=("quantization", "gpu-memory-utilization"),
    ),
}
"""Maps the CLI's ``ENGINE`` token to its registry entry.

The registry is the only place that knows which engines exist and which
engine-scoped options they reject; the rest of the composition root selects a
backend by token and never imports an engine class. A factory is a context
manager so it can own the resources a backend needs for the whole sweep
(vLLM's compile cache) without leaking that detail into the CLI. The separate
deployment factory plans artifacts without those trial-only resources.
"""


class _CliFailure(Exception):
    """An expected failure carrying the exit code it maps to.

    Preparation steps raise this with a human-readable message and the exit code
    the plan assigns to that class of failure, so :func:`_run` can print the
    message on stderr and return the code without threading returns through
    every helper.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: InferenceBackend | None = None,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the command line and return the process exit code.

    ``argv`` defaults to ``sys.argv[1:]``; ``backend``, ``client``, and
    ``benchmark_client`` are the composition seams the optimizer, runtime, and
    host benchmark client are driven with, and default to the backend selected by
    the ``ENGINE`` positional, a client discovered from the environment, and the
    default benchmark client. ``stdout`` and ``stderr`` default to the process
    streams and are injectable so the entry point can be exercised as a string
    producer.

    Optuna's own logging is silenced first, so the only output is the chosen
    reporter's. Argparse's own exits are folded into return values, so
    ``-h``/``--help`` and ``--version`` return ``0`` and a usage error returns
    ``2``. A Ctrl-C that escapes the sweep (for example during preparation) is
    reported and returns the conventional interrupt code; anything unexpected is
    reported on stderr and returns ``1``.
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    try:
        with redirect_stdout(out), redirect_stderr(err):
            args = parse_args(argv)
    except SystemExit as exit:
        return int(exit.code or 0)

    try:
        return _run(
            args,
            backend=backend,
            client=client,
            benchmark_client=benchmark_client,
            out=out,
            err=err,
        )
    except KeyboardInterrupt:
        print("interrupted", file=err)
        return _EXIT_INTERRUPTED
    except Exception as error:
        print(f"error: {error}", file=err)
        return _EXIT_ERROR


def _run(
    args: argparse.Namespace,
    *,
    backend: InferenceBackend | None,
    client: PodmanClient | None,
    benchmark_client: BenchmarkClient | None,
    out: TextIO,
    err: TextIO,
) -> int:
    """Resolve the output mode, the engine, and the base config, then run one invocation.

    The output mode is resolved first, because an explicit TUI request without a
    terminal is a usage error that must fail before any expensive work, and the
    engine next, so an unwired engine or a vLLM-only option paired with
    llama.cpp fails as a usage error before the header is printed or any
    preparation runs. The base configuration and the objective are resolved
    after that, so a measurement control outside its valid range or an invalid
    objective weight is reported as a usage error before the space is loaded,
    the hardware is probed, or a container is pulled. A JSONL ``--dataset-path``
    is read and validated at the same point, so a missing or malformed dataset
    is a usage error before the first trial rather than a failure repeated by
    every trial. A caller-supplied backend is used as-is for both the sweep and
    any artifact plan; otherwise the entry's factory builds the selected backend
    for the duration of the sweep, so resources such as vLLM's compile cache live
    exactly as long as the run, while its deployment factory plans written
    artifacts without those trial-only resources.
    """
    mode = _output_mode(args, out)
    if mode is None:
        print("error: --output tui requires a terminal", file=err)
        return _EXIT_USAGE

    engine = _ENGINES.get(args.engine)
    if engine is None:
        print(f"error: engine {args.engine!r} is not available", file=err)
        return _EXIT_USAGE
    rejected = _rejected_option(args, engine)
    if rejected is not None:
        print(f"error: {rejected} is not supported by the {args.engine} engine", file=err)
        return _EXIT_USAGE

    try:
        base = build_base_config(args)
        objective_mode, objective_weights = resolve_objective(args, base.benchmark)
        validate_dataset(base.benchmark)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return _EXIT_USAGE
    except ExperimentError as error:
        print(f"error: {error}", file=err)
        return _EXIT_USAGE

    if backend is None:
        _print_header(args, base, out)
        with engine.factory() as resolved_backend:
            return _sweep(
                args,
                base,
                resolved_backend,
                mode=mode,
                objective_mode=objective_mode,
                objective_weights=objective_weights,
                output_planning_factory=engine.deployment_factory,
                client=client,
                benchmark_client=benchmark_client,
                out=out,
                err=err,
            )

    _print_header(args, base, out)
    return _sweep(
        args,
        base,
        backend,
        mode=mode,
        objective_mode=objective_mode,
        objective_weights=objective_weights,
        output_planning_factory=lambda: backend,
        client=client,
        benchmark_client=benchmark_client,
        out=out,
        err=err,
    )


def _sweep(
    args: argparse.Namespace,
    base: ExperimentConfig,
    resolved_backend: InferenceBackend,
    *,
    mode: str,
    objective_mode: ObjectiveMode,
    objective_weights: ObjectiveWeights | None,
    output_planning_factory: Callable[[], InferenceBackend],
    client: PodmanClient | None,
    benchmark_client: BenchmarkClient | None,
    out: TextIO,
    err: TextIO,
) -> int:
    """Run the preparation sequence and the sweep, mapping outcomes to exit codes.

    The base configuration and the objective were already resolved and
    validated by :func:`_run`; the preparation steps run in order and announce
    themselves, and the sweep is driven through the optimizer with the chosen
    reporter and objective. Every outcome the plan assigns an exit code to is
    mapped here; anything else propagates to :func:`main` as an unexpected
    error.
    """
    try:
        space = _load_space(args.space, resolved_backend, base.llm, out)
        hardware = _detect_hardware(args, out)
        _check_supported(space, resolved_backend, base.llm, hardware, out)
        model = _prefetch_model(args.model, resolved_backend, out)
        base = replace(base, benchmark=replace(base.benchmark, resolved_path=model))
        if base.llm.draft_model is not None:
            draft = _prefetch_draft(base.llm.draft_model, resolved_backend, out)
            base = replace(base, llm=replace(base.llm, resolved_draft_path=draft))
        _check_model(base, model, resolved_backend, out)
        _warn_speculation_without_dataset(base, err)
        image = _pull_image(args, hardware.target, resolved_backend, client, out)
        base = _with_dataset_request_tokens(
            base,
            resolved_backend,
            hardware,
            client=client,
            benchmark_client=benchmark_client,
            err=err,
        )
        reporter = _make_reporter(mode, args, out, objective_mode=objective_mode)
        print("preparing study", file=out)
        reporter.start()
        result = optimize(
            space,
            base,
            n_trials=args.n_trials,
            timeout=args.sweep_timeout,
            seed=args.sampler_seed,
            mode=objective_mode,
            weights=objective_weights,
            storage=args.storage,
            study_name=args.study_name or _default_study_name(args.model, objective_mode),
            backend=resolved_backend,
            client=client,
            benchmark_client=benchmark_client,
            n_jobs=args.n_jobs,
            callback=reporter,
            on_trial_start=reporter.trial_started if isinstance(reporter, TuiReporter) else None,
        )
    except _CliFailure as failure:
        print(failure, file=err)
        return failure.exit_code
    except OptimizationInterrupted as interrupted:
        return _finish(
            interrupted.result,
            args,
            target=hardware.target,
            image=image,
            hardware=hardware,
            output_planning_factory=output_planning_factory,
            out=out,
            err=err,
            exit_code=_EXIT_INTERRUPTED,
        )
    except OptimizationError as error:
        print(f"error: {error}", file=err)
        return _EXIT_EMPTY_SWEEP

    return _finish(
        result,
        args,
        target=hardware.target,
        image=image,
        hardware=hardware,
        output_planning_factory=output_planning_factory,
        out=out,
        err=err,
        exit_code=_EXIT_SUCCESS,
    )


def _rejected_option(args: argparse.Namespace, engine: _EngineEntry) -> str | None:
    """Name the first engine-scoped option this engine rejects, or ``None``.

    Argparse parses every engine's options, because it cannot know the selected
    engine at flag-definition time, and stores each option under the attribute
    name derived from its spelling. An option that was not passed is ``None``,
    so a non-``None`` value the entry rejects is a usage error raised before any
    preparation rather than a value the backend would silently ignore.
    """
    for option in engine.rejected_options:
        if getattr(args, option.replace("-", "_")) is not None:
            return f"--{option}"
    return None


def _output_mode(args: argparse.Namespace, out: TextIO) -> str | None:
    """Resolve the reporter mode, or ``None`` for an explicit TUI without a terminal.

    An explicit ``--output`` always wins: ``tui`` needs a terminal and is
    rejected otherwise, and ``log`` is accepted anywhere. With no flag the mode
    follows the stream, so an interactive run plots and a redirected run logs.
    """
    if args.output is not None:
        if args.output == "tui" and not out.isatty():
            return None
        return args.output
    return "tui" if out.isatty() else "log"


def _print_header(args: argparse.Namespace, base: ExperimentConfig, out: TextIO) -> None:
    """Print the startup header naming the run's inputs and budget.

    The target and image are the requested values — ``auto`` and ``default``
    when the flags were omitted — while the preparation steps that follow report
    what they actually resolved to. A configured speculative setup is named with
    its method and, when it has one, its draft source.
    """
    print(f"model: {args.model}", file=out)
    print(f"space: {args.space}", file=out)
    print(f"engine: {args.engine}", file=out)
    print(f"target: {args.target or 'auto'}", file=out)
    print(f"image: {args.image or 'default'}", file=out)
    if base.llm.speculative_method is not None:
        setup = base.llm.speculative_method.value
        if base.llm.draft_model is not None:
            setup += f" ({base.llm.draft_model})"
        print(f"speculation: {setup}", file=out)
    print(f"budget: {_budget_text(args)}", file=out)


def _budget_text(args: argparse.Namespace) -> str:
    """Render the sweep's budget as trials, seconds, both, or ``unbounded``."""
    bounds: list[str] = []
    if args.n_trials is not None:
        bounds.append(f"{args.n_trials} trials")
    if args.sweep_timeout is not None:
        bounds.append(f"{args.sweep_timeout:g}s")
    return ", ".join(bounds) if bounds else "unbounded"


def _load_space(
    path: Path, backend: InferenceBackend, llm: LLMConfig, out: TextIO
) -> ExperimentSpace:
    """Load and parse the search space, mapping a bad file to the definition code.

    The selected backend's :meth:`applicable_parameters` is the required set for
    the run's fixed configuration, so the definition is checked to be total over
    exactly the parameters the engine forwards: a parameter it does not support
    may be omitted, and one it does must be declared. A missing or unreadable
    file is reported as a definition failure rather than a usage error, because
    the path was syntactically valid but the definition behind it could not be
    read.
    """
    print(f"loading search space: {path}", file=out)
    try:
        return load_experiment_space(path, required=backend.applicable_parameters(llm))
    except (ExperimentDefinitionError, OSError) as error:
        raise _CliFailure(str(error), _EXIT_DEFINITION) from error


def _check_supported(
    space: ExperimentSpace,
    backend: InferenceBackend,
    llm: LLMConfig,
    hardware: DetectedHardware,
    out: TextIO,
) -> None:
    """Ask whether the backend supports the space on ``hardware``, mapping problems to code ``4``.

    The host is resolved first so a value the engine supports in general but the
    host cannot run — a CUDA-only attention backend on ROCm, or a tensor-parallel
    size larger than the device count — is reported here instead of failing
    every trial.
    """
    print("validating search space", file=out)
    try:
        validate_space(space, backend, llm, hardware)
    except ExperimentDefinitionError as error:
        raise _CliFailure(str(error), _EXIT_UNSUPPORTED) from error


def _check_model(
    experiment: ExperimentConfig,
    model: Path,
    backend: InferenceBackend,
    out: TextIO,
) -> None:
    """Ask whether the resolved checkpoint supports the base configuration.

    The model-aware half of preparation: the backend inspects the checkpoint
    the run will mount and reports every way it cannot serve the configured
    experiment. The problems are aggregated under an "unsupported configuration"
    headline and mapped to the unsupported exit code, mirroring how
    :func:`_check_supported` reports the space. It runs after the model is
    resolved and before the image is pulled, so a doomed configuration does not
    wait on a multi-gigabyte download. A checkpoint the backend cannot read
    reports no problems, leaving the engine as the final authority at load time.
    """
    print("validating model compatibility", file=out)
    problems = tuple(backend.validate_model(experiment, model))
    if not problems:
        return
    detail = "\n".join(f"  - {problem}" for problem in problems)
    raise _CliFailure(f"unsupported configuration:\n{detail}", _EXIT_UNSUPPORTED)


def _detect_hardware(args: argparse.Namespace, out: TextIO) -> DetectedHardware:
    """Resolve the host's accelerator target and device count, honouring ``--target``."""
    print("detecting hardware target", file=out)
    override = HardwareTarget(args.target) if args.target is not None else None
    hardware = detect_hardware(override)
    print(f"resolved target: {hardware.target.value}", file=out)
    return hardware


def _prefetch_model(model: str, backend: InferenceBackend, out: TextIO) -> Path:
    """Resolve the model to a local path, downloading a Hub repository if needed.

    The backend's model-resolution hook decides what the model argument refers
    to — a local checkpoint, or a Hub repository possibly narrowed to a single
    file — and the shared download helper fetches anything not already on the
    host. The resolved path is returned so a later model-aware check inspects
    the same checkpoint the run mounts; the caller records it on the benchmark
    config so the backend plans the run from the checkpoint it actually has. An
    argument the backend refuses to resolve is an unsupported-model failure (exit
    code 4) reported before any download, and a download failure fails the run
    before any trial.
    """
    return _prefetch_checkpoint("model", model, backend, out)


def _prefetch_draft(draft_model: str, backend: InferenceBackend, out: TextIO) -> Path:
    """Resolve the draft model to a local path, downloading it if needed.

    A draft follows the same model-argument rules as the main checkpoint, so
    the backend's resolution hook decides what the draft source refers to and
    the shared download helper fetches it with the same single progress line.
    The resolved path is recorded on the base configuration, which is what the
    backend later mounts into the container and inspects for compatibility. A
    draft the backend refuses to resolve is an unsupported-model failure (exit
    code 4) before any download, just like an unresolvable main model.
    """
    return _prefetch_checkpoint("draft model", draft_model, backend, out)


def _prefetch_checkpoint(role: str, source: str, backend: InferenceBackend, out: TextIO) -> Path:
    """Resolve one checkpoint source and download it when the backend says to.

    ``role`` names the checkpoint in the progress line and the failure
    messages — ``model`` or ``draft model`` — so the main model and its draft
    share one resolution path rather than duplicating the download and error
    handling. The backend's hook decides between a local checkpoint and a Hub
    repository; anything it rejects is the unsupported exit code, and a download
    failure is an ordinary run failure.
    """
    print(f"resolving {role}: {source}", file=out)
    try:
        request = backend.model_request(source)
    except ExperimentError as error:
        raise _CliFailure(f"unsupported {role}: {error}", _EXIT_UNSUPPORTED) from error
    if isinstance(request, LocalCheckpoint):
        print(f"using local {role}: {request.path}", file=out)
        return request.path
    try:
        return download_model(request.repo_id, out, filename=request.filename)
    except Exception as error:
        raise _CliFailure(f"failed to download {role} {source!r}: {error}", _EXIT_ERROR) from error


def _with_dataset_request_tokens(
    base: ExperimentConfig,
    backend: InferenceBackend,
    hardware: DetectedHardware,
    *,
    client: PodmanClient | None,
    benchmark_client: BenchmarkClient | None,
    err: TextIO,
) -> ExperimentConfig:
    """Record a JSONL dataset's measured request length on the base config.

    The engine tokenizes the dataset once before the sweep, so every trial's
    backend can cap its slot count against the real workload instead of the
    text length. A backend without a tokenizer endpoint returns ``None`` and
    leaves the config alone. A failed preflight is a warning rather than a
    fatal error: the sweep can still run with the backend's conservative
    fallback, and aborting would hide an otherwise viable configuration behind
    a diagnostic-only failure.
    """
    if base.benchmark.dataset_path is None:
        return base
    try:
        request_tokens = measure_dataset_request_tokens(
            base,
            backend,
            hardware,
            client=client,
            benchmark_client=benchmark_client,
        )
    except ExperimentError as error:
        print(f"warning: could not measure the dataset's token lengths: {error}", file=err)
        return base
    if request_tokens is None:
        return base
    return replace(base, benchmark=replace(base.benchmark, max_request_tokens=request_tokens))


def _warn_speculation_without_dataset(base: ExperimentConfig, err: TextIO) -> None:
    """Warn that the synthetic workload cannot show speculation's benefit.

    Speculative decoding earns its overhead from accepted draft tokens, and the
    default random-token workload proposes drafts the target almost always
    rejects. The warning is advisory rather than an error because measuring the
    worst case can be deliberate; a caller who supplies a dataset is already
    measuring with real prompts, so it stays silent.
    """
    if base.llm.speculative_method is None or base.benchmark.dataset_path is not None:
        return
    print(
        "warning: speculative decoding rarely helps on the synthetic random workload; "
        "pass `--dataset-path` with real prompts",
        file=err,
    )


def _pull_image(
    args: argparse.Namespace,
    target: HardwareTarget,
    backend: InferenceBackend,
    client: PodmanClient | None,
    out: TextIO,
) -> str:
    """Resolve the container image for ``target`` and pull it, returning its name.

    The image is the explicit ``--image`` or the backend's image for the target,
    and is pulled through the runtime's public helper unless the policy is
    ``never``.
    """
    image = args.image or backend.image_for(target)
    if args.pull_policy == "never":
        print(f"using container image: {image} (pull policy never)", file=out)
        return image
    print(f"pulling container image: {image}", file=out)
    try:
        pull_image(image, args.pull_policy, client=client)
    except ContainerRunError as error:
        raise _CliFailure(f"failed to pull image {image!r}: {error}", _EXIT_ERROR) from error
    return image


def _make_reporter(
    mode: str,
    args: argparse.Namespace,
    out: TextIO,
    *,
    objective_mode: ObjectiveMode,
) -> LogReporter | TuiReporter:
    """Build the reporter for the resolved output mode, writing to ``out``.

    The TUI is told the objective mode because it must choose its view — the
    prefill-versus-decode scatter or the historical best-so-far line — before
    the first frame, when its argument-less trial-start hook has no study to
    read. The log reporter infers the mode from the study on every line.
    """
    if mode == "tui":
        return TuiReporter(n_trials=args.n_trials, mode=objective_mode, stream=out)
    return LogReporter(n_trials=args.n_trials, stream=out)


def _default_study_name(model: str, mode: ObjectiveMode) -> str:
    """Derive a study name from the model and the objective mode.

    The mode is part of the name because a persisted study records the
    objective that produced its scores and refuses a resume under a different
    one, so the zero-config path must not collide with a study of the same
    model scored under another mode. An explicit ``--study-name`` is the
    user's own choice and is never suffixed.
    """
    return f"{model.replace('/', '_')}_{mode.value}"


def _finish(
    result: OptimizationResult,
    args: argparse.Namespace,
    *,
    target: HardwareTarget,
    image: str,
    hardware: DetectedHardware,
    output_planning_factory: Callable[[], InferenceBackend],
    out: TextIO,
    err: TextIO,
    exit_code: int,
) -> int:
    """Print the recommended-configuration summary and optionally write it to disk.

    The summary and the file share the resolved target and image, so both name
    the hardware and image that actually ran rather than the ``None`` the
    optimizer leaves in the reconstructed config. Both describe the result's
    recommended point, the one point of the frontier a caller should re-run.

    The finish path runs on both a completed and an interrupted sweep, so it
    returns the exit code to use: the caller's ``exit_code`` on success, or the
    output failure's code when the requested artifact could not be produced.
    """
    print(render_summary(result, target=target, image=image), file=out)
    if args.output_config is not None:
        try:
            _write_output(
                result,
                args,
                target=target,
                image=image,
                hardware=hardware,
                output_planning_factory=output_planning_factory,
            )
        except _CliFailure as failure:
            print(failure, file=err)
            return failure.exit_code
    return exit_code


def _write_output(
    result: OptimizationResult,
    args: argparse.Namespace,
    *,
    target: HardwareTarget,
    image: str,
    hardware: DetectedHardware,
    output_planning_factory: Callable[[], InferenceBackend],
) -> None:
    """Render the winning configuration in the selected format and write it.

    The writer is selected from the format's registry entry, so this path never
    imports a concrete renderer. The YAML description is built from the
    recommended config plus the resolved target and image alone and declares
    that it needs no plan, so it keeps its historical side effects and never
    invokes backend planning. An executable format needs the server plan a trial
    ran, so the recommended configuration is re-planned through the deployment
    backend before its writer renders it. That backend omits trial-only scratch
    resources, so the artifact does not reference a cache directory the run
    deletes. Planning and writing failures are reported as :class:`_CliFailure`
    so both the normal and interrupted finish paths exit non-zero.
    """
    output_format = OutputFormat(args.output_format)
    config = result.recommended.config
    writer = writer_for(output_format)
    spec: ServerSpec | None = None
    try:
        if writer.requires_plan:
            spec = output_planning_factory().plan(config, hardware)
        text = writer.render(RenderContext(config=config, target=target, image=image, spec=spec))
    except Exception as error:
        raise _CliFailure(
            f"failed to render the winning configuration as {output_format.value}: {error}",
            _EXIT_ERROR,
        ) from error
    try:
        args.output_config.write_text(text, encoding="utf-8")
    except OSError as error:
        raise _CliFailure(f"failed to write {args.output_config}: {error}", _EXIT_ERROR) from error
