"""Orchestration for the engine-agnostic inference experiment layer.

This module is the entry point that ties the layers together: it resolves the
host hardware, asks the configured backend to plan a server run, starts that
server container, drives it with the host benchmark client, stops it, and hands
the measured outcome back to the backend for parsing. It knows nothing about
any inference engine — every engine-specific decision lives behind the injected
backend and the engine-neutral :class:`ServerSpec` it returns.
"""

import re
from typing import TYPE_CHECKING

from ..runtime import start_container
from ..runtime.types import ContainerResult, ContainerRunError, RunningContainer
from ._benchmark import BenchmarkClient
from ._vllm import VLLMBackend
from .hardware import detect_hardware
from .types import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    InferenceBackend,
    ServerSpec,
)

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = ["run_experiment"]


def run_experiment(
    config: ExperimentConfig,
    backend: InferenceBackend | None = None,
    *,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
) -> ExperimentResult:
    """Run one host-driven benchmark experiment and return its typed result.

    The host hardware is detected once — honouring the container config's
    ``target`` override — and passed to the backend's ``plan``, so the backend
    never probes the host itself. The planned server container is started
    through :func:`~llm_config_optimization.runtime.start_container`; the host
    benchmark client waits for its health, takes the warmup and measured runs,
    and returns its measured outcome; the container is stopped and the outcome
    is handed to the backend's ``parse_result``.

    ``backend`` defaults to :class:`~llm_config_optimization.inference.VLLMBackend`.
    ``client`` is forwarded to ``start_container`` so a caller (or a test) can
    supply a pre-configured Podman client; when it is omitted the runtime
    discovers one from the environment. ``benchmark_client`` is the host
    measurement client, injectable so a test needs neither a daemon nor a GPU;
    ``None`` uses the default client.

    Raises:
        ExperimentError: If the server container cannot be started or stopped,
            if the client fails to measure the server, or if the backend cannot
            assemble the outcome into a result. The server container's logs are
            attached to the error when they could be captured.
    """
    resolved_backend = backend if backend is not None else VLLMBackend()
    resolved_client = benchmark_client if benchmark_client is not None else BenchmarkClient()
    hardware = detect_hardware(config.container.target)
    return _run(config, resolved_backend, hardware, client, resolved_client)


def _base_url(spec: ServerSpec, handle: RunningContainer) -> str:
    """Return the host URL the client reaches ``spec``'s server on.

    The spec declares the container port it publishes and the started handle
    reports the host port the runtime assigned it, so the first published port
    resolves to ``http://127.0.0.1:<host port>``. A spec with no published port
    — or one the runtime did not resolve — cannot be measured and raises.
    """
    for container_port in spec.container.ports:
        host_port = handle.ports.get(container_port)
        if host_port is not None:
            return f"http://127.0.0.1:{host_port}"
    raise ExperimentError("the server container published no host port to measure")


def _stop_with_logs(handle: RunningContainer) -> ContainerResult | None:
    """Stop ``handle`` and return its result, or ``None`` when stopping fails.

    Used on the failure path, where the original error matters more than a
    teardown failure: a stop that fails is followed by a forced removal so the
    container does not leak, and the caller re-raises its own error.
    """
    try:
        return handle.stop()
    except ContainerRunError:
        handle.remove()
        return None


def _run(
    config: ExperimentConfig,
    backend: InferenceBackend,
    hardware: DetectedHardware,
    client: PodmanClient | None,
    benchmark_client: BenchmarkClient,
) -> ExperimentResult:
    """Plan, start, measure, stop, and parse one run."""
    spec = backend.plan(config, hardware)
    try:
        handle = start_container(spec.container, client=client)
    except ContainerRunError as exc:
        message = f"benchmark server for model {config.benchmark.model!r} failed to start: {exc}"
        detail = _failure_output(exc.result)
        if detail:
            message = f"{message}\n{detail}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    try:
        base_url = _base_url(spec, handle)
        outcome = benchmark_client.measure_sync(
            base_url, config.benchmark.model, spec, config.benchmark
        )
    except ExperimentError as exc:
        result = _stop_with_logs(handle)
        detail = _failure_output(result)
        message = str(exc) if not detail else f"{exc}\n{detail}"
        raise ExperimentError(message, raw_result=result) from exc
    except BaseException:
        handle.remove()
        raise
    try:
        result = handle.stop()
    except ContainerRunError as exc:
        message = f"failed to stop the benchmark server for {config.benchmark.model!r}: {exc}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    return backend.parse_result(outcome, spec, result)


_FAILURE_OUTPUT_LIMIT = 2000
"""How many characters of a failed run's output to attach to the error.

The engine's own diagnostics — a traceback, a missing-library message — live on
its stderr, so the error message is nearly useless without them. A failure can
print several tracebacks, with the actual cause in the first and only a wrapper
in the last, so the excerpt keeps both: the earliest root-cause line is rendered
first and the trailing window follows. The whole excerpt is bounded so a chatty
engine cannot flood a log line.
"""

_EXCEPTION_LINE = re.compile(r"\b\w*(?:Error|Exception):")
"""Matches a line that names an exception, e.g. ``AssertionError: ...``."""


def _root_cause(text: str) -> str | None:
    """Return the earliest line of ``text`` that names an exception, if any.

    An engine can print several tracebacks in one log: a vLLM run prints the
    engine core's failure first and the frontend's wrapper traceback last, and a
    chained exception prints the original cause before the re-raise. The root
    cause is therefore the *earliest* such line — the later ones restate it
    ("Engine core initialization failed. See root cause above"). Selecting the
    earliest line surfaces the actual cause, while the trailing window keeps the
    surrounding context. The line is selected by its shape rather than its
    position because the traceback length varies with the engine and the
    failure.
    """
    for line in text.splitlines():
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
