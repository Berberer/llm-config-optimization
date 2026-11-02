"""Orchestration for the engine-agnostic inference experiment layer.

This module is the entry point that ties the layers together: it resolves the
host hardware, asks the configured backend to plan a server run, starts that
server container, drives it with the host benchmark client, stops it, and hands
the measured outcome back to the backend for parsing. It knows nothing about
any inference engine — every engine-specific decision lives behind the injected
backend and the engine-neutral :class:`ServerSpec` it returns.

A trial's memory footprint is measured alongside its throughput. The host
sampler is built for the started container, samples from the moment it is up —
so the engine's model-load allocation is observed — and stops just before the
container is torn down, so nothing polls the host after a trial ends. The whole
container's peak rides on the parsed result. Sampling is always best-effort:
an absent vendor probe, an unreadable container stat, or a sampler that raises
leaves the numbers absent rather than failing the trial.
"""

import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

from ..runtime import start_container
from ..runtime.types import ContainerResult, ContainerRunError, RunningContainer
from ._benchmark import BenchmarkClient
from ._resources import ThreadedResourceSampler
from ._vllm import VLLMBackend
from .hardware import MemoryProbe, detect_hardware, memory_probe_for
from .types import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    InferenceBackend,
    ResourceSampler,
    ResourceUsage,
    ServerSpec,
)

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = ["SamplerFactory", "measure_dataset_request_tokens", "run_experiment"]

type SamplerFactory = Callable[[RunningContainer, int | None, MemoryProbe | None], ResourceSampler]
"""Builds the memory sampler for a container the orchestrator just started.

It receives the running container — the only place a container's RAM can be
read — the idle VRAM baseline taken before that container started, and the
vendor memory probe selected for the detected target (``None`` on a CPU
target). It is injectable so a caller can substitute a sampler, and so a test
never shells out to a vendor tool or touches a real host.
"""


def measure_dataset_request_tokens(
    config: ExperimentConfig,
    backend: InferenceBackend,
    hardware: DetectedHardware,
    *,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
) -> int | None:
    """Measure a JSONL dataset's longest request with the engine's tokenizer.

    A dataset's prompts are text, so only the engine knows their token counts.
    This starts the backend's server once, before the sweep, with a minimal plan
    (one slot and no speculative setup), asks the spec's tokenizer endpoint for
    every prompt's length, and returns the largest ``prompt_tokens +
    output_len``. The result is recorded on the base configuration so every
    trial can cap its slot count against the real workload instead of a guess.

    A benchmark without a dataset path, or a backend whose spec declares no
    tokenizer endpoint, needs no preflight and returns ``None`` without
    starting a container. The one server start is the cost of an accurate cap;
    a failure is raised as :class:`ExperimentError` with the container output
    attached, and the CLI may choose to warn and fall back to a conservative
    slot count.
    """
    if config.benchmark.dataset_path is None:
        return None
    resolved_client = benchmark_client if benchmark_client is not None else BenchmarkClient()
    preflight = replace(
        config,
        llm=replace(
            config.llm,
            max_num_seqs=1,
            speculative_method=None,
            draft_model=None,
            speculative_mode=None,
            num_speculative_tokens=None,
            resolved_draft_path=None,
        ),
    )
    spec = backend.plan(preflight, hardware)
    if spec.tokenize_path is None:
        return None
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
        tokens = resolved_client.dataset_request_tokens_sync(
            base_url,
            config.benchmark.model,
            spec,
            config.benchmark,
            is_alive=handle.is_running,
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
        handle.stop()
    except ContainerRunError as exc:
        message = f"failed to stop the benchmark server for {config.benchmark.model!r}: {exc}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    return tokens


def run_experiment(
    config: ExperimentConfig,
    backend: InferenceBackend | None = None,
    *,
    client: PodmanClient | None = None,
    benchmark_client: BenchmarkClient | None = None,
    sampler_factory: SamplerFactory | None = None,
) -> ExperimentResult:
    """Run one host-driven benchmark experiment and return its typed result.

    The host hardware is detected once — honouring the container config's
    ``target`` override — and passed to the backend's ``plan``, so the backend
    never probes the host itself. The planned server container is started
    through :func:`~llm_config_optimization.runtime.start_container`; the host
    benchmark client waits for its health — aborting as soon as the started
    container exits during that wait — takes the warmup and measured runs, and
    returns its measured outcome; the container is stopped and the outcome is
    handed to the backend's ``parse_result``.

    ``backend`` defaults to :class:`~llm_config_optimization.inference.VLLMBackend`.
    ``client`` is forwarded to ``start_container`` so a caller (or a test) can
    supply a pre-configured Podman client; when it is omitted the runtime
    discovers one from the environment. ``benchmark_client`` is the host
    measurement client, injectable so a test needs neither a daemon nor a GPU;
    ``None`` uses the default client.

    Memory is measured for every trial, silently and never as an objective: the
    container's RAM comes through its runtime handle and VRAM from the detected
    target's vendor probe, the idle device baseline is read before the container
    starts, and the container's lifetime peak is attached to the parsed result.
    ``sampler_factory`` builds that sampler for the started container;
    ``None`` uses the host sampler. Nothing about the measurement can fail a
    trial.

    Raises:
        ExperimentError: If the server container cannot be started or stopped,
            if the client fails to measure the server, or if the backend cannot
            assemble the outcome into a result. The server container's logs are
            attached to the error when they could be captured.
    """
    resolved_backend = backend if backend is not None else VLLMBackend()
    resolved_client = benchmark_client if benchmark_client is not None else BenchmarkClient()
    resolved_factory = sampler_factory if sampler_factory is not None else _default_sampler
    hardware = detect_hardware(config.container.target)
    return _run(config, resolved_backend, hardware, client, resolved_client, resolved_factory)


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
    sampler_factory: SamplerFactory,
) -> ExperimentResult:
    """Plan, start, measure, stop, and parse one run."""
    spec = backend.plan(config, hardware)
    probe = memory_probe_for(hardware.target)
    baseline_vram_bytes = _idle_vram_bytes(probe)
    try:
        handle = start_container(spec.container, client=client)
    except ContainerRunError as exc:
        message = f"benchmark server for model {config.benchmark.model!r} failed to start: {exc}"
        detail = _failure_output(exc.result)
        if detail:
            message = f"{message}\n{detail}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    sampler: ResourceSampler | None = None
    try:
        sampler = sampler_factory(handle, baseline_vram_bytes, probe)
        _start_sampler(sampler)
        base_url = _base_url(spec, handle)
        outcome = benchmark_client.measure_sync(
            base_url,
            config.benchmark.model,
            spec,
            config.benchmark,
            is_alive=handle.is_running,
            sampler=sampler,
        )
    except ExperimentError as exc:
        _stop_sampler(sampler)
        result = _stop_with_logs(handle)
        detail = _failure_output(result)
        message = str(exc) if not detail else f"{exc}\n{detail}"
        raise ExperimentError(message, raw_result=result) from exc
    except BaseException:
        _stop_sampler(sampler)
        handle.remove()
        raise
    _stop_sampler(sampler)
    try:
        result = handle.stop()
    except ContainerRunError as exc:
        message = f"failed to stop the benchmark server for {config.benchmark.model!r}: {exc}"
        raise ExperimentError(message, raw_result=exc.result) from exc
    parsed = backend.parse_result(outcome, spec, result)
    return replace(parsed, lifetime_resources=_lifetime_peak(sampler))


def _default_sampler(
    handle: RunningContainer,
    baseline_vram_bytes: int | None,
    probe: MemoryProbe | None,
) -> ResourceSampler:
    """Build the host sampler for a container the orchestrator started.

    Container RAM is read through the runtime handle, so the runtime layer keeps
    owning how a container's memory is obtained, and VRAM comes from the probe
    selected for the detected target — absent on a CPU target. The baseline was
    captured before the container started, so a device-level VRAM figure is the
    trial's own growth over an idle device.
    """
    return ThreadedResourceSampler(
        ram_reader=handle.memory_usage,
        container_id=handle.container_id,
        baseline_vram_bytes=baseline_vram_bytes,
        probe=probe,
    )


def _idle_vram_bytes(probe: MemoryProbe | None) -> int | None:
    """Read the device's used memory before a trial's container starts.

    The device-level VRAM fallback subtracts this baseline, so it must be
    observed while nothing of this trial is running. A CPU target has no probe
    and a failed query yields no baseline; neither fails the trial.
    """
    if probe is None:
        return None
    try:
        return probe().used_bytes
    except Exception:
        return None


def _start_sampler(sampler: ResourceSampler) -> None:
    """Begin sampling, treating a sampler failure as "nothing measured"."""
    with suppress(Exception):
        sampler.start()


def _stop_sampler(sampler: ResourceSampler | None) -> None:
    """End sampling, treating an absent or failing sampler as "nothing measured"."""
    if sampler is None:
        return
    with suppress(Exception):
        sampler.stop()


def _lifetime_peak(sampler: ResourceSampler | None) -> ResourceUsage:
    """Return the whole-container peak, or an unmeasured usage when unreadable."""
    if sampler is None:
        return ResourceUsage()
    try:
        return sampler.lifetime_peak()
    except Exception:
        return ResourceUsage()


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
