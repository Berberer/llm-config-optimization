"""Host-side benchmark client for the engine-agnostic experiment layer.

A trial no longer measures from inside the engine's image: it starts the
engine's server in a container and drives it from the host over the server's
OpenAI-compatible HTTP API. This module owns that measurement, so both engines
are exercised by identical logic and no engine image needs to carry a benchmark
client.

The client is asynchronous and uses ``httpx2`` for the requests. Its
transport and clock are injectable, so a unit test can drive it with a fake
HTTP transport and a fake clock and needs neither a daemon nor a GPU.

The measured workload is random prompts by default — ``num_prompts`` requests
of ``input_len`` random token ids asking for ``output_len`` tokens — and a
JSONL dataset when the benchmark names one. Every run is fired concurrently and
its elapsed wall-clock, completed requests, and token usage are aggregated into
a :class:`~llm_config_optimization.inference.types.RunMetrics`; the successful
runs and the failed-run count become a
:class:`~llm_config_optimization.inference.types.BenchmarkOutcome`. A trial is
rejected unless a strict majority of its measured runs completed at least one
request, mirroring the engine client this replaces.
"""

import asyncio
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx2

from .types import (
    BenchmarkConfig,
    BenchmarkOutcome,
    ExperimentError,
    RunMetrics,
    ServerSpec,
)

__all__ = ["DEFAULT_VOCAB_SIZE", "BenchmarkClient"]

DEFAULT_VOCAB_SIZE = 32000
"""The token-id ceiling the random workload samples from by default.

The host client has no tokenizer, so it cannot know a model's real vocabulary
size. The default is a value most served checkpoints include, and a caller or
backend that knows better can widen or narrow it through
:attr:`BenchmarkClient.vocab_size`.
"""

_COMPLETIONS_PATH = "/v1/completions"
_RANDOM_DATASETS = frozenset({"random", "random-mm", "random-rerank"})
"""Dataset names that select the random workload rather than a file.

The CLI defaults to ``random``; the multimodal and rerank variants also draw
their prompts rather than reading a dataset, so pairing any of them with a
``dataset_path`` is allowed and the path simply wins.
"""


@dataclass(frozen=True, slots=True)
class _Request:
    """One prompt to send and the number of tokens to generate for it."""

    prompt: str | tuple[int, ...]
    output_len: int


def _positive(value: int | None, name: str) -> int:
    """Return ``value`` when it is at least one, else raise a configuration error.

    The random workload cannot be measured without a request count and a
    positive prompt and generation length, and :class:`BenchmarkConfig` leaves
    each unset until a caller supplies it, so a missing value is a configuration
    error rather than something to default away.
    """
    if value is None or value < 1:
        raise ValueError(f"the random workload requires {name} to be at least 1, got {value!r}")
    return value


def _random_requests(benchmark: BenchmarkConfig, seed: int, vocab_size: int) -> list[_Request]:
    """Build ``num_prompts`` random token-id prompts for one run.

    Each prompt is ``input_len`` token ids drawn from ``range(vocab_size)`` with
    a generator seeded by ``seed``, so a run draws a fresh, reproducible prompt
    set and the model receives the exact prompt length the benchmark configured.
    """
    num_prompts = _positive(benchmark.num_prompts, "num_prompts")
    input_len = _positive(benchmark.input_len, "input_len")
    output_len = _positive(benchmark.output_len, "output_len")
    generator = random.Random(seed)
    return [
        _Request(
            prompt=tuple(generator.randrange(vocab_size) for _ in range(input_len)),
            output_len=output_len,
        )
        for _ in range(num_prompts)
    ]


def _dataset_requests(benchmark: BenchmarkConfig) -> list[_Request]:
    """Read a JSONL dataset into the prompts one run sends.

    Each non-empty line is a JSON object with a required string ``prompt`` and
    an optional positive integer ``output_len`` that falls back to the
    benchmark's ``output_len``. ``num_prompts`` caps how many lines are used;
    ``input_len`` is ignored because the dataset carries the prompts. A missing
    or malformed file, a non-object line, a missing prompt, and an invalid
    output length are all reported as :class:`ExperimentError` before any
    request is sent.
    """
    path = Path(cast("str", benchmark.dataset_path)).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(f"could not read the dataset at {path}: {exc}") from exc
    requests: list[_Request] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"dataset line {number} at {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ExperimentError(f"dataset line {number} at {path} is not a JSON object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ExperimentError(f"dataset line {number} at {path} has no string 'prompt'")
        output_len = payload.get("output_len", benchmark.output_len)
        if not isinstance(output_len, int) or isinstance(output_len, bool) or output_len < 1:
            raise ExperimentError(
                f"dataset line {number} at {path} has no positive integer 'output_len' "
                f"and no benchmark output_len to fall back to"
            )
        requests.append(_Request(prompt=prompt, output_len=output_len))
        if benchmark.num_prompts is not None and len(requests) >= benchmark.num_prompts:
            break
    if not requests:
        raise ExperimentError(f"the dataset at {path} produced no prompts")
    return requests


def _build_workload(benchmark: BenchmarkConfig, vocab_size: int) -> Callable[[int], list[_Request]]:
    """Return a callable that builds one run's requests for a given seed.

    A JSONL dataset is fixed, so every run replays it; the random workload
    rebuilds its prompt set from the run's seed, so each repetition draws a
    fresh set. Combining a non-random dataset name with a ``dataset_path`` is a
    usage error, because the named dataset would silently be ignored.
    """
    if benchmark.dataset_path is not None:
        if benchmark.dataset_name is not None and benchmark.dataset_name not in _RANDOM_DATASETS:
            raise ValueError(
                f"dataset_name={benchmark.dataset_name!r} cannot be combined with dataset_path; "
                f"a custom dataset uses the JSONL contract"
            )
        requests = _dataset_requests(benchmark)
        return lambda _seed: requests
    return lambda seed: _random_requests(benchmark, seed, vocab_size)


def _usage_tokens(payload: Mapping[str, object]) -> tuple[int, int] | None:
    """Return ``(prompt_tokens, completion_tokens)`` from a completion payload.

    ``None`` when the payload carries no usable ``usage`` object, so the caller
    can treat the response as a failed request rather than counting zero tokens
    for a request that did generate output.
    """
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or isinstance(prompt, bool):
        return None
    if not isinstance(completion, int) or isinstance(completion, bool):
        return None
    return prompt, completion


@dataclass(frozen=True, slots=True)
class BenchmarkClient:
    """Drive an engine's OpenAI-compatible server and aggregate its runs.

    ``transport`` is the injectable HTTP transport; ``None`` uses the default
    one, while a test passes an :class:`httpx2.MockTransport`. ``clock`` is the
    monotonic source used for elapsed times and health deadlines, so a test can
    make the measured duration deterministic. ``vocab_size`` bounds the token
    ids the random workload samples. ``request_timeout`` bounds a single request
    and is ``None`` by default, leaving the overall :attr:`BenchmarkConfig.timeout`
    as the only time limit. ``poll_interval`` is how long to wait between health
    probes.
    """

    transport: httpx2.AsyncBaseTransport | None = None
    clock: Callable[[], float] = time.monotonic
    vocab_size: int = DEFAULT_VOCAB_SIZE
    request_timeout: float | None = None
    poll_interval: float = 1.0

    async def measure(
        self,
        base_url: str,
        model: str,
        spec: ServerSpec,
        benchmark: BenchmarkConfig,
        *,
        concurrency: int | None = None,
        is_alive: Callable[[], bool] | None = None,
    ) -> BenchmarkOutcome:
        """Measure ``spec``'s server at ``base_url`` and return its runs.

        The client waits for the server's health endpoint, takes ``warmup``
        discarded runs, then ``repetitions`` measured runs. Before every run it
        resets the spec's cache paths when it declares any, and it seeds each
        run distinctly so every repetition draws a fresh prompt set. Requests
        are fired concurrently, bounded by ``concurrency`` when it is given and
        unbounded otherwise.

        ``is_alive``, when given, is polled during the health wait so a server
        container that has already exited — a model-load crash, an out-of-memory
        kill — ends the run immediately with that reason instead of waiting out
        the whole startup timeout. The orchestrator passes the started
        container's liveness check.

        A run whose requests all fail is counted in ``failed_runs`` and does not
        contribute a metric. Raises :class:`ExperimentError` when the server
        never becomes healthy, when the server container exits during the wait,
        when the workload cannot be built, when the overall timeout is exceeded,
        or when fewer than a strict majority of the measured runs succeeded.
        """
        workload = _build_workload(benchmark, self.vocab_size)
        if benchmark.timeout is not None:
            try:
                async with asyncio.timeout(benchmark.timeout):
                    return await self._measure(
                        base_url, model, spec, benchmark, workload, concurrency, is_alive
                    )
            except TimeoutError as exc:
                raise ExperimentError(
                    f"the benchmark for {benchmark.model!r} exceeded its "
                    f"{benchmark.timeout}s overall timeout"
                ) from exc
        return await self._measure(
            base_url, model, spec, benchmark, workload, concurrency, is_alive
        )

    def measure_sync(
        self,
        base_url: str,
        model: str,
        spec: ServerSpec,
        benchmark: BenchmarkConfig,
        *,
        concurrency: int | None = None,
        is_alive: Callable[[], bool] | None = None,
    ) -> BenchmarkOutcome:
        """Synchronous entry point for a caller that is not already in a loop.

        The orchestrator is synchronous, so it drives the asynchronous
        :meth:`measure` through :func:`asyncio.run` rather than owning an event
        loop of its own.
        """
        return asyncio.run(
            self.measure(
                base_url, model, spec, benchmark, concurrency=concurrency, is_alive=is_alive
            )
        )

    async def _measure(
        self,
        base_url: str,
        model: str,
        spec: ServerSpec,
        benchmark: BenchmarkConfig,
        workload: Callable[[int], list[_Request]],
        concurrency: int | None,
        is_alive: Callable[[], bool] | None,
    ) -> BenchmarkOutcome:
        """Run the health wait, warmup, and measured runs against one server."""
        limits = httpx2.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        )
        async with httpx2.AsyncClient(
            base_url=base_url,
            transport=self.transport,
            limits=limits,
            timeout=self.request_timeout,
        ) as client:
            await self._wait_for_health(client, spec, is_alive)
            measured: list[RunMetrics] = []
            failed_runs = 0
            for warmup_index in range(benchmark.warmup):
                await self._reset_caches(client, spec)
                # Warmup seeds are offset past the measured range so no measured
                # run ever replays the prompt set a warmup run drew.
                await self._take_run(
                    client,
                    model,
                    workload(benchmark.repetitions + warmup_index),
                    concurrency,
                )
            for run_index in range(benchmark.repetitions):
                await self._reset_caches(client, spec)
                run = await self._take_run(client, model, workload(run_index), concurrency)
                if run is None:
                    failed_runs += 1
                else:
                    measured.append(run)
        if len(measured) * 2 <= benchmark.repetitions:
            raise ExperimentError(
                f"only {len(measured)}/{benchmark.repetitions} measured runs succeeded; "
                f"a strict majority is required"
            )
        return BenchmarkOutcome(runs=tuple(measured), failed_runs=failed_runs)

    async def _wait_for_health(
        self,
        client: httpx2.AsyncClient,
        spec: ServerSpec,
        is_alive: Callable[[], bool] | None,
    ) -> None:
        """Poll the spec's health path until it succeeds, the container dies, or the deadline passes.

        A failed probe is followed by ``is_alive`` — when the caller supplied
        one — so a container that crashed on startup is reported as such at
        once, with the engine's logs attached by the orchestrator, instead of
        polling a dead endpoint until the startup timeout elapses.
        """
        deadline = self.clock() + spec.health_timeout
        while True:
            try:
                response = await client.get(spec.health_path)
            except httpx2.HTTPError:
                response = None
            if response is not None and response.is_success:
                return
            if is_alive is not None and not is_alive():
                raise ExperimentError(
                    f"the server container exited before becoming healthy at {spec.health_path}"
                )
            if self.clock() >= deadline:
                raise ExperimentError(
                    f"the server did not become healthy at {spec.health_path} within "
                    f"{spec.health_timeout}s"
                )
            await asyncio.sleep(self.poll_interval)

    async def _reset_caches(self, client: httpx2.AsyncClient, spec: ServerSpec) -> None:
        """Call the spec's cache-reset paths, ignoring any failure.

        A reset is best-effort exactly as the engine client treated it: an
        engine that does not expose an endpoint, or answers with an error, must
        not fail the run.
        """
        for path in spec.reset_paths:
            try:
                await client.post(path)
            except httpx2.HTTPError:
                continue

    async def _take_run(
        self,
        client: httpx2.AsyncClient,
        model: str,
        requests: list[_Request],
        concurrency: int | None,
    ) -> RunMetrics | None:
        """Fire one run's requests and aggregate them, or ``None`` if none completed."""
        semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None

        async def send(request: _Request) -> Mapping[str, object]:
            if semaphore is None:
                return await self._post(client, model, request)
            async with semaphore:
                return await self._post(client, model, request)

        start = self.clock()
        results = await asyncio.gather(
            *(send(request) for request in requests), return_exceptions=True
        )
        elapsed = self.clock() - start
        responses: list[Mapping[str, object]] = []
        prompt_tokens = 0
        completion_tokens = 0
        for result in results:
            if isinstance(result, BaseException):
                continue
            usage = _usage_tokens(result)
            if usage is None:
                continue
            responses.append(result)
            prompt_tokens += usage[0]
            completion_tokens += usage[1]
        completed = len(responses)
        if completed == 0:
            return None
        total_tokens = prompt_tokens + completion_tokens
        return RunMetrics(
            tokens_per_second=total_tokens / elapsed,
            requests_per_second=completed / elapsed,
            elapsed_time=elapsed,
            num_prompts=completed,
            raw_json={
                "completed": completed,
                "failed": len(requests) - completed,
                "elapsed_time": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "responses": responses,
            },
        )

    async def _post(
        self, client: httpx2.AsyncClient, model: str, request: _Request
    ) -> Mapping[str, object]:
        """Send one completion request and return its JSON object.

        ``min_tokens`` and ``ignore_eos`` ask the engine to generate the exact
        configured output length, matching the engine client this replaces, so
        a run measures the workload it was configured with.
        """
        prompt = list(request.prompt) if isinstance(request.prompt, tuple) else request.prompt
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": request.output_len,
            "min_tokens": request.output_len,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": False,
        }
        response = await client.post(_COMPLETIONS_PATH, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise ExperimentError("the server returned a non-object completion payload")
        return cast("Mapping[str, object]", body)
