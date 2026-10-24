"""Unit tests for the host-side benchmark client.

The client's observable contract is what it sends to the engine's
OpenAI-compatible server and the metrics it derives from the responses: the
random and JSONL workloads, the per-run seed, the warmup runs, the cache-reset
calls, the concurrency bound, the strict-majority rule, the aggregation of
elapsed time and token usage, and the best-effort snapshots of a spec-declared
metrics endpoint. These tests drive it through an injected
:class:`httpx2.MockTransport` and an injected clock, so no daemon, engine, or
GPU is required and every elapsed time is deterministic.
"""

import asyncio
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx2
import pytest

from llm_config_optimization.inference import (
    BenchmarkClient,
    BenchmarkConfig,
    ExperimentError,
    ServerSpec,
    validate_dataset,
)
from llm_config_optimization.runtime import ContainerConfig

MODEL = "org/model"


class FakeClock:
    """A monotonic-looking clock whose every reading advances by ``step``.

    Two readings taken around a measured run therefore differ by exactly
    ``step``, so the run's elapsed time is deterministic regardless of how many
    other readings the client took.
    """

    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._now = start
        self._step = step

    def __call__(self) -> float:
        now = self._now
        self._now += self._step
        return now


class FakeEngine:
    """A fake OpenAI-compatible engine that records what the client sends.

    ``health_failures`` makes the first N health probes fail before the server
    reports healthy. ``fail_completions`` makes the first N completion requests
    fail, so a test can fail whole runs by covering their request count.
    ``metrics``, when given, is called with the 1-based scrape number for each
    ``GET /metrics`` and returns that scrape's body, or ``None`` to answer with
    an error; leaving it unset makes the endpoint 404, modelling an engine
    without scrapeable metrics. The handler yields once so an unbounded run can
    have every request in flight at the same time, which lets a test observe
    the concurrency bound.
    """

    def __init__(
        self,
        *,
        health_failures: int = 0,
        fail_completions: int = 0,
        metrics: Callable[[int], str | None] | None = None,
    ) -> None:
        self.health_failures = health_failures
        self.fail_completions = fail_completions
        self.metrics = metrics
        self.health_attempts = 0
        self.metrics_attempts = 0
        self.completions: list[dict[str, object]] = []
        self.tokenizations: list[str] = []
        self.resets: list[str] = []
        self.events: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handle)

    async def _handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/health":
            self.health_attempts += 1
            if self.health_attempts <= self.health_failures:
                return httpx2.Response(503, json={"status": "loading"})
            return httpx2.Response(200, json={"status": "ok"})
        if request.method == "POST" and path.startswith("/reset"):
            self.resets.append(path)
            self.events.append(f"reset:{path}")
            return httpx2.Response(200, json={})
        if path == "/metrics":
            self.metrics_attempts += 1
            self.events.append("metrics")
            if self.metrics is None:
                return httpx2.Response(404, json={"error": "not found"})
            text = self.metrics(self.metrics_attempts)
            if text is None:
                return httpx2.Response(500, json={"error": "boom"})
            return httpx2.Response(200, text=text)
        if path == "/tokenize":
            body = json.loads(request.content)
            content = body["content"]
            self.tokenizations.append(content)
            return httpx2.Response(200, json={"tokens": [0] * len(content.split())})
        if path == "/v1/completions":
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0)
            self.in_flight -= 1
            body = json.loads(request.content)
            self.completions.append(body)
            self.events.append("completion")
            if len(self.completions) <= self.fail_completions:
                return httpx2.Response(500, json={"error": "boom"})
            prompt = body["prompt"]
            prompt_len = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
            output_len = body["max_tokens"]
            return httpx2.Response(
                200,
                json={
                    "id": "cmpl",
                    "choices": [{"text": "x", "index": 0, "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": output_len,
                        "total_tokens": prompt_len + output_len,
                    },
                },
            )
        return httpx2.Response(404, json={"error": "not found"})


def _spec(
    *,
    health_timeout: float = 60.0,
    reset_paths: tuple[str, ...] = (),
    metrics_path: str | None = None,
    tokenize_path: str | None = None,
) -> ServerSpec:
    return ServerSpec(
        container=ContainerConfig(image="fake/server:latest"),
        health_path="/health",
        health_timeout=health_timeout,
        reset_paths=reset_paths,
        metrics_path=metrics_path,
        tokenize_path=tokenize_path,
        tokenize_prompt_field="content",
    )


def _client(engine: FakeEngine, **kwargs: object) -> BenchmarkClient:
    return BenchmarkClient(transport=engine.transport(), clock=FakeClock(), **kwargs)


def _random_benchmark(**overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "model": MODEL,
        "dataset_name": "random",
        "num_prompts": 2,
        "input_len": 4,
        "output_len": 3,
        "repetitions": 1,
        "warmup": 0,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)  # type: ignore[arg-type]


def _prompts(engine: FakeEngine) -> list[object]:
    return [body["prompt"] for body in engine.completions]


def test_random_workload_sends_the_configured_requests() -> None:
    engine = FakeEngine()
    client = _client(engine, vocab_size=100)

    outcome = client.measure_sync("http://engine", "served-model", _spec(), _random_benchmark())

    assert len(engine.completions) == 2
    for body in engine.completions:
        assert body["model"] == "served-model"
        assert isinstance(body["prompt"], list)
        assert len(body["prompt"]) == 4
        assert all(0 <= token < 100 for token in body["prompt"])
        assert body["max_tokens"] == 3
        assert body["min_tokens"] == 3
        assert body["ignore_eos"] is True
    assert len(outcome.runs) == 1
    assert outcome.failed_runs == 0


def test_run_metrics_aggregate_tokens_and_elapsed_time() -> None:
    engine = FakeEngine()
    client = BenchmarkClient(transport=engine.transport(), clock=FakeClock(step=1.0))

    outcome = client.measure_sync("http://engine", MODEL, _spec(), _random_benchmark())

    run = outcome.runs[0]
    assert run.elapsed_time == 1.0
    assert run.num_prompts == 2
    # (4 prompt + 3 completion) tokens per request, two requests, one second.
    assert run.tokens_per_second == 14.0
    assert run.requests_per_second == 2.0
    assert run.raw_json["completed"] == 2
    assert run.raw_json["total_tokens"] == 14
    assert len(run.raw_json["responses"]) == 2


def test_warmup_runs_are_discarded_and_each_run_draws_a_fresh_seed() -> None:
    engine = FakeEngine()
    client = _client(engine, vocab_size=50)
    benchmark = _random_benchmark(repetitions=2, warmup=1)

    outcome = client.measure_sync("http://engine", MODEL, _spec(), benchmark)

    def expected(seed: int) -> set[tuple[int, ...]]:
        generator = random.Random(seed)
        return {tuple(generator.randrange(50) for _ in range(4)) for _ in range(2)}

    sent = [tuple(body["prompt"]) for body in engine.completions]  # type: ignore[arg-type]
    assert len(sent) == 6
    assert set(sent[0:2]) == expected(2)  # warmup seed is offset past the measured range
    assert set(sent[2:4]) == expected(0)
    assert set(sent[4:6]) == expected(1)
    assert len(outcome.runs) == 2


def test_jsonl_dataset_uses_prompts_and_output_len_fallback(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        '{"prompt": "hello"}\n'
        '{"prompt": "world", "output_len": 7}\n'
        '{"prompt": "ignored", "output_len": 2}\n',
        encoding="utf-8",
    )
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(
        dataset_name=None,
        dataset_path=str(dataset),
        num_prompts=2,
        input_len=999,
        output_len=5,
    )

    client.measure_sync("http://engine", MODEL, _spec(), benchmark)

    sent = {body["prompt"]: body["max_tokens"] for body in engine.completions}
    assert sent == {"hello": 5, "world": 7}


def test_non_random_dataset_with_a_path_is_a_usage_error(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(dataset_name="sharegpt", dataset_path=str(dataset))

    with pytest.raises(ValueError, match="dataset_path"):
        client.measure_sync("http://engine", MODEL, _spec(), benchmark)


def test_a_malformed_dataset_line_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "hello"}\n{"output_len": 3}\n', encoding="utf-8")
    engine = FakeEngine()
    client = _client(engine)

    with pytest.raises(ExperimentError, match="prompt"):
        client.measure_sync(
            "http://engine",
            MODEL,
            _spec(),
            _random_benchmark(dataset_name=None, dataset_path=str(dataset)),
        )


def test_validate_dataset_accepts_a_valid_file(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    validate_dataset(_random_benchmark(dataset_name=None, dataset_path=str(dataset)))


def test_validate_dataset_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(ExperimentError, match="could not read the dataset"):
        validate_dataset(_random_benchmark(dataset_name=None, dataset_path=str(missing)))


def test_validate_dataset_reports_a_malformed_file(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"output_len": 3}\n', encoding="utf-8")

    with pytest.raises(ExperimentError, match="prompt"):
        validate_dataset(_random_benchmark(dataset_name=None, dataset_path=str(dataset)))


def test_validate_dataset_rejects_a_named_dataset_with_a_path(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_path"):
        validate_dataset(_random_benchmark(dataset_name="sharegpt", dataset_path=str(dataset)))


def test_validate_dataset_leaves_the_random_workload_alone() -> None:
    validate_dataset(_random_benchmark())


def test_dataset_request_tokens_measures_the_longest_request(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        '{"prompt": "one two three", "output_len": 5}\n{"prompt": "a b", "output_len": 10}\n',
        encoding="utf-8",
    )
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(
        dataset_name=None,
        dataset_path=str(dataset),
        num_prompts=None,
        output_len=1,
    )

    tokens = client.dataset_request_tokens_sync(
        "http://engine", MODEL, _spec(tokenize_path="/tokenize"), benchmark
    )

    assert tokens == 12  # "a b" is 2 tokens plus its 10 requested output tokens
    assert engine.tokenizations == ["one two three", "a b"]


def test_dataset_request_tokens_is_skipped_without_a_tokenizer(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(dataset_name=None, dataset_path=str(dataset))

    tokens = client.dataset_request_tokens_sync("http://engine", MODEL, _spec(), benchmark)

    assert tokens is None
    assert engine.tokenizations == []


def test_a_run_whose_requests_all_fail_is_counted_as_failed() -> None:
    engine = FakeEngine(fail_completions=2)
    client = _client(engine)
    benchmark = _random_benchmark(repetitions=3)

    outcome = client.measure_sync("http://engine", MODEL, _spec(), benchmark)

    assert len(outcome.runs) == 2
    assert outcome.failed_runs == 1


def test_a_tie_between_successful_and_failed_runs_is_rejected() -> None:
    engine = FakeEngine(fail_completions=2)
    client = _client(engine)
    benchmark = _random_benchmark(repetitions=2)

    with pytest.raises(ExperimentError, match="strict majority"):
        client.measure_sync("http://engine", MODEL, _spec(), benchmark)


def test_a_partial_request_failure_still_measures_completed_requests() -> None:
    engine = FakeEngine(fail_completions=1)
    client = _client(engine)

    outcome = client.measure_sync("http://engine", MODEL, _spec(), _random_benchmark())

    run = outcome.runs[0]
    assert run.num_prompts == 1
    assert run.tokens_per_second == 7.0
    assert run.raw_json["completed"] == 1
    assert run.raw_json["failed"] == 1


def test_reset_paths_are_called_before_every_run() -> None:
    engine = FakeEngine()
    client = _client(engine)
    spec = _spec(reset_paths=("/reset_prefix_cache", "/reset_mm_cache"))
    benchmark = _random_benchmark(repetitions=2, warmup=1)

    client.measure_sync("http://engine", MODEL, spec, benchmark)

    assert engine.resets == ["/reset_prefix_cache", "/reset_mm_cache"] * 3
    assert engine.events[0] == "reset:/reset_prefix_cache"


def test_absent_reset_paths_skip_resetting() -> None:
    engine = FakeEngine()
    client = _client(engine)

    client.measure_sync("http://engine", MODEL, _spec(), _random_benchmark())

    assert engine.resets == []


def _exposition(scrape: int) -> str:
    """Build a realistic exposition whose counters advance with the scrape."""
    return (
        "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n"
        "# TYPE vllm:prompt_tokens_total counter\n"
        f'vllm:prompt_tokens_total{{model_name="{MODEL}"}} {scrape * 10}.0\n'
        f'vllm:request_decode_time_seconds_sum{{model_name="{MODEL}"}} {scrape}.5\n'
    )


def test_declared_metrics_are_snapshotted_before_and_after_the_run() -> None:
    engine = FakeEngine(metrics=_exposition)
    client = _client(engine)

    outcome = client.measure_sync(
        "http://engine", MODEL, _spec(metrics_path="/metrics"), _random_benchmark()
    )

    assert outcome.runs[0].raw_json["metrics"] == {
        "before": {
            "vllm:prompt_tokens_total": 10.0,
            "vllm:request_decode_time_seconds_sum": 1.5,
        },
        "after": {
            "vllm:prompt_tokens_total": 20.0,
            "vllm:request_decode_time_seconds_sum": 2.5,
        },
    }
    assert engine.metrics_attempts == 2
    assert engine.events == ["metrics", "completion", "completion", "metrics"]


def test_every_measured_run_is_bracketed_and_warmup_is_not_scraped() -> None:
    engine = FakeEngine(metrics=lambda scrape: f"vllm:prompt_tokens_total {scrape}.0\n")
    client = _client(engine)
    benchmark = _random_benchmark(repetitions=2, warmup=1)

    outcome = client.measure_sync("http://engine", MODEL, _spec(metrics_path="/metrics"), benchmark)

    snapshots = [run.raw_json["metrics"] for run in outcome.runs]
    assert [snapshot["before"]["vllm:prompt_tokens_total"] for snapshot in snapshots] == [1.0, 3.0]
    assert [snapshot["after"]["vllm:prompt_tokens_total"] for snapshot in snapshots] == [2.0, 4.0]
    assert engine.metrics_attempts == 4


def test_a_spec_without_a_metrics_path_is_never_scraped() -> None:
    engine = FakeEngine(metrics=_exposition)
    client = _client(engine)

    outcome = client.measure_sync("http://engine", MODEL, _spec(), _random_benchmark())

    assert "metrics" not in outcome.runs[0].raw_json
    assert engine.metrics_attempts == 0


def test_a_missing_metrics_endpoint_leaves_the_run_intact() -> None:
    engine = FakeEngine()  # the endpoint 404s
    client = _client(engine)

    outcome = client.measure_sync(
        "http://engine", MODEL, _spec(metrics_path="/metrics"), _random_benchmark()
    )

    assert len(outcome.runs) == 1
    assert "metrics" not in outcome.runs[0].raw_json
    assert engine.metrics_attempts == 2


def test_an_unparseable_metrics_payload_is_ignored() -> None:
    engine = FakeEngine(metrics=lambda _scrape: '{"error": "this is not an exposition"}')
    client = _client(engine)

    outcome = client.measure_sync(
        "http://engine", MODEL, _spec(metrics_path="/metrics"), _random_benchmark()
    )

    assert len(outcome.runs) == 1
    assert "metrics" not in outcome.runs[0].raw_json
    assert engine.metrics_attempts == 2


def test_a_failed_half_of_the_snapshot_pair_leaves_no_metrics_key() -> None:
    engine = FakeEngine(
        metrics=lambda scrape: None if scrape == 1 else "vllm:gpu_cache_usage 0.5\n"
    )
    client = _client(engine)

    outcome = client.measure_sync(
        "http://engine", MODEL, _spec(metrics_path="/metrics"), _random_benchmark()
    )

    assert engine.metrics_attempts == 2
    assert "metrics" not in outcome.runs[0].raw_json


def test_a_server_that_never_becomes_healthy_fails() -> None:
    engine = FakeEngine(health_failures=100)
    client = BenchmarkClient(
        transport=engine.transport(), clock=FakeClock(step=1000.0), poll_interval=0.0
    )

    with pytest.raises(ExperimentError, match="healthy"):
        client.measure_sync("http://engine", MODEL, _spec(health_timeout=10.0), _random_benchmark())

    assert engine.completions == []


def test_a_dead_server_container_ends_the_health_wait_immediately() -> None:
    engine = FakeEngine(health_failures=100)
    client = BenchmarkClient(
        transport=engine.transport(), clock=FakeClock(step=1.0), poll_interval=0.0
    )

    with pytest.raises(ExperimentError, match="exited before becoming healthy"):
        client.measure_sync(
            "http://engine",
            MODEL,
            _spec(health_timeout=600.0),
            _random_benchmark(),
            is_alive=lambda: False,
        )

    assert engine.health_attempts == 1
    assert engine.completions == []


def test_a_live_container_does_not_shorten_the_health_wait() -> None:
    engine = FakeEngine(health_failures=2)
    client = BenchmarkClient(
        transport=engine.transport(), clock=FakeClock(step=1.0), poll_interval=0.0
    )

    outcome = client.measure_sync(
        "http://engine",
        MODEL,
        _spec(health_timeout=100.0),
        _random_benchmark(),
        is_alive=lambda: True,
    )

    assert engine.health_attempts == 3
    assert len(outcome.runs) == 1


def test_the_client_waits_for_health_before_measuring() -> None:
    engine = FakeEngine(health_failures=2)
    client = BenchmarkClient(
        transport=engine.transport(), clock=FakeClock(step=1.0), poll_interval=0.0
    )

    outcome = client.measure_sync(
        "http://engine", MODEL, _spec(health_timeout=100.0), _random_benchmark()
    )

    assert engine.health_attempts == 3
    assert len(outcome.runs) == 1


def test_concurrency_bounds_the_requests_in_flight() -> None:
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(num_prompts=5)

    client.measure_sync("http://engine", MODEL, _spec(), benchmark, concurrency=1)

    assert engine.max_in_flight == 1


def test_requests_are_unbounded_by_default() -> None:
    engine = FakeEngine()
    client = _client(engine)
    benchmark = _random_benchmark(num_prompts=5)

    client.measure_sync("http://engine", MODEL, _spec(), benchmark)

    assert engine.max_in_flight == 5


def test_the_overall_timeout_is_reported_as_an_experiment_error() -> None:
    engine = FakeEngine(health_failures=100)
    client = BenchmarkClient(
        transport=engine.transport(), clock=FakeClock(step=0.1), poll_interval=0.0
    )
    benchmark = _random_benchmark(timeout=0.01)

    with pytest.raises(ExperimentError, match="timeout"):
        client.measure_sync("http://engine", MODEL, _spec(health_timeout=1000.0), benchmark)


def test_the_async_entry_point_matches_the_synchronous_one() -> None:
    engine = FakeEngine()
    client = _client(engine)

    outcome = asyncio.run(client.measure("http://engine", MODEL, _spec(), _random_benchmark()))

    assert len(outcome.runs) == 1
    assert isinstance(outcome.raw_payloads, tuple)
    assert isinstance(outcome.raw_payloads[0], Mapping)
