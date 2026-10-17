"""Command-line interface for the optimization runner.

This subpackage is the composition root that turns the console script into a
usable tool: it parses the command line, builds the fixed
:class:`~llm_config_optimization.inference.types.ExperimentConfig` the search space is
varied around, reports progress, and maps outcomes onto exit codes. The
modules here own argument parsing, base-config construction, the log and TUI
reporters, and the final summary and configuration writer.

The package is driven through a single :func:`main` entry point that returns an
integer exit code, so both the console script and ``python -m llm_config_optimization``
can share one implementation. ``main`` is a pure composition root: it loads the search space, resolves the
hardware target, validates the space against that target, prefetches the model
and image, and then calls the optimization adapter with the injected backend,
client, and reporter. Nothing about the study itself is duplicated here.

Exit codes are distinct so scripts can react to each outcome: ``0`` success,
``2`` a usage error (including an explicit TUI request without a terminal),
``3`` an invalid or unreadable search-space definition, ``4`` a space the
backend does not support, ``5`` a sweep that finished with no completed trial,
``130`` an interrupt, and ``1`` any unexpected error. Progress and the summary
go to stdout; warnings and errors go to stderr.
"""

import argparse
import sys
import tempfile
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import optuna
from huggingface_hub import snapshot_download

from ..inference import VLLMBackend, detect_hardware
from ..inference.types import DetectedHardware, HardwareTarget, InferenceBackend
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
from .config import build_base_config, is_local_model
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
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the command line and return the process exit code.

    ``argv`` defaults to ``sys.argv[1:]``; ``backend`` and ``client`` are the
    composition seams the optimizer and runtime are driven with, and default to
    the vLLM backend and a client discovered from the environment. ``stdout``
    and ``stderr`` default to the process streams and are injectable so the
    entry point can be exercised as a string producer.

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
        return _run(args, backend=backend, client=client, out=out, err=err)
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
    out: TextIO,
    err: TextIO,
) -> int:
    """Resolve the output mode and cache, then run one invocation.

    The output mode is resolved first, because an explicit TUI request without a
    terminal is a usage error that must fail before any expensive work. A
    caller-supplied backend is used as-is; the default vLLM backend is given a
    temporary compile cache for the run, so the trials of one sweep reuse the
    kernels the first one built while the directory is removed when the run
    ends.
    """
    mode = _output_mode(args, out)
    if mode is None:
        print("error: --output tui requires a terminal", file=err)
        return _EXIT_USAGE

    _print_header(args, out)

    if backend is not None:
        return _sweep(args, backend, mode=mode, client=client, out=out, err=err)

    with tempfile.TemporaryDirectory(
        prefix="llm-config-optimization-cache-", ignore_cleanup_errors=True
    ) as cache_directory:
        resolved_backend = VLLMBackend(cache_dir=Path(cache_directory))
        return _sweep(args, resolved_backend, mode=mode, client=client, out=out, err=err)


def _sweep(
    args: argparse.Namespace,
    resolved_backend: InferenceBackend,
    *,
    mode: str,
    client: PodmanClient | None,
    out: TextIO,
    err: TextIO,
) -> int:
    """Run the preparation sequence and the sweep, mapping outcomes to exit codes.

    The preparation steps run in order and announce themselves, and the sweep is
    driven through the optimizer with the chosen reporter. Every outcome the
    plan assigns an exit code to is mapped here; anything else propagates to
    :func:`main` as an unexpected error.
    """
    try:
        space = _load_space(args.space, out)
        hardware = _detect_hardware(args, out)
        _check_supported(space, resolved_backend, hardware, out)
        image = _prefetch(args, hardware.target, resolved_backend, client, out)
        base = build_base_config(args)
        reporter = _make_reporter(mode, args, out)
        print("preparing study", file=out)
        reporter.start()
        result = optimize(
            space,
            base,
            n_trials=args.n_trials,
            timeout=args.timeout,
            seed=args.sampler_seed,
            storage=args.storage,
            study_name=args.study_name or _default_study_name(args.model),
            backend=resolved_backend,
            client=client,
            n_jobs=args.n_jobs,
            callback=reporter,
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
    print(f"target: {args.target or 'auto'}", file=out)
    print(f"image: {args.image or 'default'}", file=out)
    print(f"budget: {_budget_text(args)}", file=out)


def _budget_text(args: argparse.Namespace) -> str:
    """Render the sweep's budget as trials, seconds, both, or ``unbounded``."""
    bounds: list[str] = []
    if args.n_trials is not None:
        bounds.append(f"{args.n_trials} trials")
    if args.timeout is not None:
        bounds.append(f"{args.timeout:g}s")
    return ", ".join(bounds) if bounds else "unbounded"


def _load_space(path: Path, out: TextIO) -> ExperimentSpace:
    """Load and parse the search space, mapping a bad file to the definition code.

    A missing or unreadable file is reported as a definition failure rather than
    a usage error, because the path was syntactically valid but the definition
    behind it could not be read.
    """
    print(f"loading search space: {path}", file=out)
    try:
        return load_experiment_space(path)
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


def _detect_hardware(args: argparse.Namespace, out: TextIO) -> DetectedHardware:
    """Resolve the host's accelerator target and device count, honouring ``--target``."""
    print("detecting hardware target", file=out)
    override = HardwareTarget(args.target) if args.target is not None else None
    hardware = detect_hardware(override)
    print(f"resolved target: {hardware.target.value}", file=out)
    return hardware


def _prefetch(
    args: argparse.Namespace,
    target: HardwareTarget,
    backend: InferenceBackend,
    client: PodmanClient | None,
    out: TextIO,
) -> str:
    """Resolve and prefetch the model and the container image, returning the image.

    The model is downloaded on the host only when it is a Hugging Face id, so a
    local checkpoint is used directly and the first trial reads the cache the
    backend bind-mounts rather than a download. The image is the explicit
    ``--image`` or the backend's image for the target, and is pulled through the
    runtime's public helper unless the policy is ``never``.
    """
    _prefetch_model(args.model, out)
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


def _prefetch_model(model: str, out: TextIO) -> None:
    """Download a Hugging Face model to the host cache, or use a local path directly.

    The snapshot download is called without an explicit token, so
    ``huggingface_hub``'s own credential resolution applies and a gated model is
    fetched from ``HF_TOKEN`` or a stored login. A download failure fails the
    run before any trial.
    """
    print(f"resolving model: {model}", file=out)
    if is_local_model(model):
        print(f"using local model: {model}", file=out)
        return
    print(f"downloading model: {model}", file=out)
    try:
        snapshot_download(model)
    except Exception as error:
        raise _CliFailure(f"failed to download model {model!r}: {error}", _EXIT_ERROR) from error


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
