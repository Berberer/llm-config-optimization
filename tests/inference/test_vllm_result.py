"""Unit tests for vLLM result parsing.

``VLLMBackend.parse_result``'s observable contract is the typed
:class:`~llm_config_optimization.inference.types.ExperimentResult` it assembles
from the host client's measured :class:`BenchmarkOutcome` and the planned
:class:`ServerSpec`: the successful runs and the failure count come straight
from the client, while the target, image, and device mapping are recovered from
the plan so the result records what actually ran, and each run's prefill and
decode rates are derived from the metric snapshots the client stored. These
tests drive the backend with an in-memory outcome and a planned spec — no
Podman daemon, no inference engine, and no result files are involved.
"""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    HardwareTarget,
    LLMConfig,
    RunMetrics,
    ServerSpec,
    VLLMBackend,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult

HUB_MODEL = "meta-llama/Llama-3.1-8B"

_PREFILL_TOKENS = "vllm:prompt_tokens_total"
"""vLLM's processed-prompt-token counter, as the exposition names it."""

_DECODE_TOKENS = "vllm:generation_tokens_total"
"""vLLM's generated-token counter."""

_PREFILL_SECONDS = "vllm:request_prefill_time_seconds_sum"
"""The sum series of vLLM's prefill-time histogram."""

_DECODE_SECONDS = "vllm:request_decode_time_seconds_sum"
"""The sum series of vLLM's decode-time histogram."""


def _experiment() -> ExperimentConfig:
    """Build a minimal experiment; parsing does not depend on its tuning."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _plan(target: HardwareTarget) -> ServerSpec:
    """Plan a run for ``target`` so parsing can recover what ran."""
    return VLLMBackend().plan(_experiment(), DetectedHardware(target, 1))


def _container_result() -> ContainerResult:
    """Build a successful server container outcome with some captured output."""
    return ContainerResult(
        exit_code=0,
        stdout=b"benchmark log\n",
        stderr=b"",
        container_id="c0ffee",
    )


def _run(**overrides: object) -> RunMetrics:
    """Build one measured run, overridable field by field."""
    fields: dict[str, object] = {
        "tokens_per_second": 1234.5,
        "requests_per_second": 12.25,
        "elapsed_time": 8.5,
        "num_prompts": 100,
        "raw_json": {"completed": 100, "model_id": HUB_MODEL},
    }
    fields.update(overrides)
    return RunMetrics(**fields)  # type: ignore[arg-type]


def _outcome(runs: tuple[RunMetrics, ...] | None = None, failed_runs: int = 0) -> BenchmarkOutcome:
    """Build a measured outcome, defaulting to one successful run."""
    return BenchmarkOutcome(
        runs=(_run(),) if runs is None else runs,
        failed_runs=failed_runs,
    )


def _metrics_run(before: Mapping[str, float], after: Mapping[str, float]) -> RunMetrics:
    """Build a run carrying one before/after metric snapshot pair."""
    return _run(raw_json={"completed": 100, "metrics": {"before": before, "after": after}})


def _phase_run(prefill_rate: float, decode_rate: float) -> RunMetrics:
    """Build a run whose one-second snapshot deltas imply the given phase rates."""
    return _metrics_run(
        before={
            _PREFILL_TOKENS: 0.0,
            _PREFILL_SECONDS: 0.0,
            _DECODE_TOKENS: 0.0,
            _DECODE_SECONDS: 0.0,
        },
        after={
            _PREFILL_TOKENS: prefill_rate,
            _PREFILL_SECONDS: 1.0,
            _DECODE_TOKENS: decode_rate,
            _DECODE_SECONDS: 1.0,
        },
    )


def test_the_measured_runs_become_the_typed_result() -> None:
    spec = _plan(HardwareTarget.CUDA)

    result = VLLMBackend().parse_result(_outcome(), spec, _container_result())

    assert result.tokens_per_second == 1234.5
    assert result.requests_per_second == 12.25
    assert result.elapsed_time == 8.5
    assert result.num_prompts == 100


def test_result_records_the_plan_that_ran_not_the_request() -> None:
    spec = _plan(HardwareTarget.ROCM)

    result = VLLMBackend().parse_result(_outcome(), spec, _container_result())

    assert result.target == HardwareTarget.ROCM
    assert result.image == "docker.io/vllm/vllm-openai-rocm:latest"
    assert result.devices == ("/dev/kfd", "/dev/dri")


def test_raw_json_and_container_result_are_preserved() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _run(raw_json={"completed": 100, "model_id": HUB_MODEL, "failed": 0})
    container_result = _container_result()

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, container_result)

    assert len(result.runs) == 1
    assert dict(result.raw_json[0]) == {"completed": 100, "model_id": HUB_MODEL, "failed": 0}
    assert result.raw_result is container_result


def test_successful_runs_are_aggregated_and_the_failure_count_is_recorded() -> None:
    spec = _plan(HardwareTarget.CUDA)
    first = _run(tokens_per_second=10.0, requests_per_second=1.0, elapsed_time=2.0, num_prompts=4)
    second = _run(tokens_per_second=20.0, requests_per_second=3.0, elapsed_time=4.0, num_prompts=6)

    result = VLLMBackend().parse_result(
        _outcome(runs=(first, second), failed_runs=1), spec, _container_result()
    )

    assert result.tokens_per_second == 15.0
    assert result.requests_per_second == 2.0
    assert result.elapsed_time == 3.0
    assert result.num_prompts == 10
    assert result.failed_runs == 1
    assert result.runs == (first, second)


def test_an_outcome_with_no_successful_run_raises_with_the_raw_result() -> None:
    spec = _plan(HardwareTarget.CUDA)
    container_result = _container_result()

    with pytest.raises(ExperimentError, match="no successful run") as excinfo:
        VLLMBackend().parse_result(BenchmarkOutcome(runs=(), failed_runs=2), spec, container_result)

    assert excinfo.value.raw_result is container_result


def test_devices_the_backend_never_plans_raise_rather_than_guess() -> None:
    spec = replace(
        _plan(HardwareTarget.CUDA),
        container=ContainerConfig(image="custom/vllm:dev", devices=("/dev/nvidia0",)),
    )
    container_result = _container_result()

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(_outcome(), spec, container_result)

    assert "accelerator class" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_phase_rates_are_derived_from_the_counter_and_histogram_deltas() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _metrics_run(
        before={
            _PREFILL_TOKENS: 1000.0,
            _PREFILL_SECONDS: 2.0,
            _DECODE_TOKENS: 5000.0,
            _DECODE_SECONDS: 10.0,
        },
        after={
            _PREFILL_TOKENS: 5000.0,
            _PREFILL_SECONDS: 6.0,
            _DECODE_TOKENS: 8000.0,
            _DECODE_SECONDS: 12.0,
        },
    )

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 1000.0  # 4000 tokens in 4 prefill seconds
    assert result.decode_tokens_per_second == 1500.0  # 3000 tokens in 2 decode seconds


def test_phase_rates_average_over_the_successful_runs() -> None:
    spec = _plan(HardwareTarget.CUDA)
    first = _phase_run(prefill_rate=1000.0, decode_rate=500.0)
    second = _phase_run(prefill_rate=3000.0, decode_rate=1500.0)

    result = VLLMBackend().parse_result(_outcome(runs=(first, second)), spec, _container_result())

    assert result.prefill_tokens_per_second == 2000.0
    assert result.decode_tokens_per_second == 1000.0


def test_a_run_without_metric_snapshots_reports_no_phase_rates() -> None:
    spec = _plan(HardwareTarget.CUDA)

    result = VLLMBackend().parse_result(_outcome(), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second is None
    assert result.tokens_per_second == 1234.5


def test_a_half_snapshot_pair_reports_no_phase_rates() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _run(raw_json={"completed": 100, "metrics": {"before": {_PREFILL_TOKENS: 1.0}}})

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second is None


def test_a_missing_phase_series_leaves_only_that_phase_absent() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _metrics_run(
        before={_PREFILL_TOKENS: 0.0, _PREFILL_SECONDS: 0.0, _DECODE_TOKENS: 0.0},
        after={_PREFILL_TOKENS: 1000.0, _PREFILL_SECONDS: 1.0, _DECODE_TOKENS: 500.0},
    )

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 1000.0
    assert result.decode_tokens_per_second is None


def test_a_zero_time_delta_yields_no_rate_for_that_phase() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _metrics_run(
        before={
            _PREFILL_TOKENS: 0.0,
            _PREFILL_SECONDS: 1.0,
            _DECODE_TOKENS: 0.0,
            _DECODE_SECONDS: 0.0,
        },
        after={
            _PREFILL_TOKENS: 1000.0,
            _PREFILL_SECONDS: 1.0,
            _DECODE_TOKENS: 2000.0,
            _DECODE_SECONDS: 4.0,
        },
    )

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second == 500.0


def test_a_zero_counter_delta_yields_no_rate_for_that_phase() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _metrics_run(
        before={
            _PREFILL_TOKENS: 0.0,
            _PREFILL_SECONDS: 0.0,
            _DECODE_TOKENS: 100.0,
            _DECODE_SECONDS: 0.0,
        },
        after={
            _PREFILL_TOKENS: 2000.0,
            _PREFILL_SECONDS: 4.0,
            _DECODE_TOKENS: 100.0,
            _DECODE_SECONDS: 2.0,
        },
    )

    result = VLLMBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 500.0
    assert result.decode_tokens_per_second is None
