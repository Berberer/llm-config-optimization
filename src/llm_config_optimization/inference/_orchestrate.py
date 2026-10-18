"""Orchestration for the engine-agnostic inference experiment layer.

This module is the entry point that ties the layers together: it resolves the
host hardware, asks the configured backend to plan a container run, executes
that plan through the runtime layer, and hands the captured outcome back to the
backend for parsing. It knows nothing about any inference engine — every
engine-specific decision lives behind the injected backend.
"""

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..runtime import run_container
from ..runtime.types import ContainerResult, ContainerRunError
from ._vllm import VLLMBackend
from .hardware import detect_hardware
from .types import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    InferenceBackend,
)

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = ["run_experiment"]


def run_experiment(
    config: ExperimentConfig,
    backend: InferenceBackend | None = None,
    *,
    client: PodmanClient | None = None,
    output_dir: Path | None = None,
) -> ExperimentResult:
    """Run one benchmark experiment and return its typed result.

    The host hardware is detected once — honouring the container config's
    ``target`` override — and passed to the backend's ``plan``, so the backend
    never probes the host itself. The plan is then executed through
    :func:`~llm_config_optimization.runtime.run_container` and the captured outcome is
    handed back to the backend's ``parse_result``.

    ``backend`` defaults to :class:`~llm_config_optimization.inference.VLLMBackend`.
    ``client`` is forwarded to ``run_container`` so a caller (or a test) can
    supply a pre-configured Podman client; when it is omitted the runtime
    discovers one from the environment.

    When ``output_dir`` is supplied it is used as-is and left in place. When it
    is omitted a unique temporary directory is created for the run and removed
    once the result has been parsed, so concurrent runs never share a result
    file; the parsed JSON remains available on the result as the per-run
    ``raw_json`` payloads.

    Raises:
        ExperimentError: If the container run fails — the underlying
            :class:`~llm_config_optimization.runtime.ContainerRunError` is preserved as
            ``__cause__`` and its partial result attached — or if the backend
            cannot parse the run's output.
    """
    resolved_backend = backend if backend is not None else VLLMBackend()
    hardware = detect_hardware(config.container.target)
    if output_dir is not None:
        return _run(config, resolved_backend, hardware, output_dir, client)
    with tempfile.TemporaryDirectory(prefix="llm-config-optimization-") as directory:
        return _run(config, resolved_backend, hardware, Path(directory), client)


_FAILURE_OUTPUT_LIMIT = 2000
"""How many characters of a failed run's output to attach to the error.

The engine's own diagnostics — a traceback, a missing-library message — live on
its stderr, so the error message is nearly useless without them. The engine
prints the root cause before the traceback that follows it, so the excerpt keeps
both: the root-cause line is rendered first and the trailing window follows. The
whole excerpt is bounded so a chatty engine cannot flood a log line.
"""

_EXCEPTION_LINE = re.compile(r"\b\w*(?:Error|Exception):")
"""Matches a line that names an exception, e.g. ``AssertionError: ...``."""


def _root_cause(text: str) -> str | None:
    """Return the last line of ``text`` that names an exception, if any.

    An engine prints the cause of a failure — for example ``AssertionError:
    ...`` — before the traceback that follows it, so a trailing window alone can
    discard the most useful line. The line is selected by its shape rather than
    its position because the traceback length varies with the engine and the
    failure.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if _EXCEPTION_LINE.search(stripped):
            return stripped
    return None


def _failure_output(result: ContainerResult | None) -> str:
    """Return a bounded excerpt of a failed run's output, or an empty string.

    stderr is preferred because that is where an engine reports a failure, with
    stdout as the fallback for engines that log there. The excerpt keeps the
    trailing window and, when the root cause falls outside it, the root-cause
    line rendered first, so the most specific line survives at the front. The
    result stays within :data:`_FAILURE_OUTPUT_LIMIT` characters, so a chatty
    engine cannot flood the log.
    """
    if result is None:
        return ""
    text = result.stderr_text.strip() or result.stdout_text.strip()
    if not text:
        return ""
    if len(text) <= _FAILURE_OUTPUT_LIMIT:
        return text
    tail = text[-(_FAILURE_OUTPUT_LIMIT - 1) :]
    root_cause = _root_cause(text)
    if root_cause is None or root_cause in tail:
        return "…" + tail
    prefix = f"{root_cause}\n…"
    room = _FAILURE_OUTPUT_LIMIT - len(prefix)
    if room <= 0:
        return root_cause[:_FAILURE_OUTPUT_LIMIT]
    return prefix + text[-room:]


def _run(
    config: ExperimentConfig,
    backend: InferenceBackend,
    hardware: DetectedHardware,
    output_dir: Path,
    client: PodmanClient | None,
) -> ExperimentResult:
    """Plan, execute, and parse one run against an already-resolved directory."""
    plan = backend.plan(config, hardware, output_dir)
    try:
        result = run_container(plan, client=client)
    except ContainerRunError as exc:
        message = f"benchmark container for model {config.benchmark.model!r} failed: {exc}"
        detail = _failure_output(exc.result)
        if detail:
            message = f"{message}\n{detail}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    return backend.parse_result(result, plan, output_dir)
