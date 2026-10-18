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
so the rest of the composition root never imports an engine class. ``main`` is a
pure composition root: it loads the search space, resolves the hardware target,
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
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import optuna

from ..inference import BenchmarkClient, LocalCheckpoint, VLLMBackend, detect_hardware
from ..inference.types import DetectedHardware, ExperimentConfig, HardwareTarget, InferenceBackend
from ..optimization import (
    OptimizationError,
    OptimizationInterrupted,
    OptimizationResult,
    optimize,
)
from ..runtime import ContainerRunError, pull_image
from ..search_space import (
    ExperimentDefinitionError,
    ExperimentSpace,
    load_experiment_space,
    validate_space,
)
from .arguments import parse_args
from .config import build_base_config
from .download import download_model
from .log import LogReporter
from .summary import render_summary, write_config
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


_ENGINE_FACTORIES: dict[str, Callable[[], AbstractContextManager[InferenceBackend]]] = {
    "vllm": _vllm_backend,
}
"""Maps the CLI's ``ENGINE`` token to a backend factory.

The registry is the only place that knows which engines exist; the rest of the
composition root selects a backend by token and never imports an engine class.
A factory is a context manager so it can own the resources a backend needs for
the whole sweep (vLLM's compile cache) without leaking that detail into the CLI.
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
    """Resolve the output mode and the base config, then run one invocation.

    The output mode is resolved first, because an explicit TUI request without a
    terminal is a usage error that must fail before any expensive work. The base
    configuration is built next, so a measurement control outside its valid
    range is reported as a usage error before the space is loaded, the hardware
    is probed, or a container is pulled. The engine is then resolved through the
    registry, so an engine that is not wired fails before the header is printed
    or any preparation runs. A caller-supplied backend is used as-is; otherwise
    the factory for ``--engine`` builds one for the duration of the sweep, so
    resources such as vLLM's compile cache live exactly as long as the run.
    """
    mode = _output_mode(args, out)
    if mode is None:
        print("error: --output tui requires a terminal", file=err)
        return _EXIT_USAGE

    try:
        base = build_base_config(args)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return _EXIT_USAGE

    if backend is None:
        factory = _ENGINE_FACTORIES.get(args.engine)
        if factory is None:
            print(f"error: engine {args.engine!r} is not available", file=err)
            return _EXIT_USAGE
        _print_header(args, out)
        with factory() as resolved_backend:
            return _sweep(
                args,
                base,
                resolved_backend,
                mode=mode,
                client=client,
                benchmark_client=benchmark_client,
                out=out,
                err=err,
            )

    _print_header(args, out)
    return _sweep(
        args,
        base,
        backend,
        mode=mode,
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
    client: PodmanClient | None,
    benchmark_client: BenchmarkClient | None,
    out: TextIO,
    err: TextIO,
) -> int:
    """Run the preparation sequence and the sweep, mapping outcomes to exit codes.

    The base configuration was already built and validated by :func:`_run`; the
    preparation steps run in order and announce themselves, and the sweep is
    driven through the optimizer with the chosen reporter. Every outcome the
    plan assigns an exit code to is mapped here; anything else propagates to
    :func:`main` as an unexpected error.
    """
    try:
        space = _load_space(args.space, resolved_backend, out)
        hardware = _detect_hardware(args, out)
        _check_supported(space, resolved_backend, hardware, out)
        model = _prefetch_model(args.model, resolved_backend, out)
        base = replace(base, benchmark=replace(base.benchmark, resolved_path=model))
        _check_model(base, model, resolved_backend, out)
        image = _pull_image(args, hardware.target, resolved_backend, client, out)
        reporter = _make_reporter(mode, args, out)
        print("preparing study", file=out)
        reporter.start()
        result = optimize(
            space,
            base,
            n_trials=args.n_trials,
            timeout=args.sweep_timeout,
            seed=args.sampler_seed,
            storage=args.storage,
            study_name=args.study_name or _default_study_name(args.model),
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
        _finish(interrupted.result, args, target=hardware.target, image=image, out=out)
        return _EXIT_INTERRUPTED
    except OptimizationError as error:
        print(f"error: {error}", file=err)
        return _EXIT_EMPTY_SWEEP

    _finish(result, args, target=hardware.target, image=image, out=out)
    return _EXIT_SUCCESS


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


def _print_header(args: argparse.Namespace, out: TextIO) -> None:
    """Print the startup header naming the run's inputs and budget.

    The target and image are the requested values — ``auto`` and ``default``
    when the flags were omitted — while the preparation steps that follow report
    what they actually resolved to.
    """
    print(f"model: {args.model}", file=out)
    print(f"space: {args.space}", file=out)
    print(f"engine: {args.engine}", file=out)
    print(f"target: {args.target or 'auto'}", file=out)
    print(f"image: {args.image or 'default'}", file=out)
    print(f"budget: {_budget_text(args)}", file=out)


def _budget_text(args: argparse.Namespace) -> str:
    """Render the sweep's budget as trials, seconds, both, or ``unbounded``."""
    bounds: list[str] = []
    if args.n_trials is not None:
        bounds.append(f"{args.n_trials} trials")
    if args.sweep_timeout is not None:
        bounds.append(f"{args.sweep_timeout:g}s")
    return ", ".join(bounds) if bounds else "unbounded"


def _load_space(path: Path, backend: InferenceBackend, out: TextIO) -> ExperimentSpace:
    """Load and parse the search space, mapping a bad file to the definition code.

    The selected backend's :attr:`applicable_parameters` is the required set, so
    the definition is checked to be total over exactly the parameters the engine
    forwards: a parameter it does not support may be omitted, and one it does
    must be declared. A missing or unreadable file is reported as a definition
    failure rather than a usage error, because the path was syntactically valid
    but the definition behind it could not be read.
    """
    print(f"loading search space: {path}", file=out)
    try:
        return load_experiment_space(path, required=backend.applicable_parameters)
    except (ExperimentDefinitionError, OSError) as error:
        raise _CliFailure(str(error), _EXIT_DEFINITION) from error


def _check_supported(
    space: ExperimentSpace,
    backend: InferenceBackend,
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
        validate_space(space, backend, hardware)
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
    config so the backend plans the run from the checkpoint it actually has. A
    download failure fails the run before any trial.
    """
    print(f"resolving model: {model}", file=out)
    request = backend.model_request(model)
    if isinstance(request, LocalCheckpoint):
        print(f"using local model: {request.path}", file=out)
        return request.path
    try:
        return download_model(request.repo_id, out, filename=request.filename)
    except Exception as error:
        raise _CliFailure(f"failed to download model {model!r}: {error}", _EXIT_ERROR) from error


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


def _make_reporter(mode: str, args: argparse.Namespace, out: TextIO) -> LogReporter | TuiReporter:
    """Build the reporter for the resolved mode, writing to ``out``."""
    if mode == "tui":
        return TuiReporter(n_trials=args.n_trials, stream=out)
    return LogReporter(n_trials=args.n_trials, stream=out)


def _default_study_name(model: str) -> str:
    """Derive a study name from the model, so resuming needs no extra bookkeeping."""
    return model.replace("/", "_")


def _finish(
    result: OptimizationResult,
    args: argparse.Namespace,
    *,
    target: HardwareTarget,
    image: str,
    out: TextIO,
) -> None:
    """Print the winning-configuration summary and optionally write it to disk.

    The summary and the file share the resolved target and image, so both name
    the hardware and image that actually ran rather than the ``None`` the
    optimizer leaves in the reconstructed config.
    """
    print(render_summary(result, target=target, image=image), file=out)
    if args.output_config is not None:
        write_config(args.output_config, result.best_config, target=target, image=image)
