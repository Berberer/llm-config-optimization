"""Unit tests for llama.cpp result parsing.

``LlamaCppBackend.parse_result``'s observable contract is the typed
:class:`~llm_config_optimization.inference.types.ExperimentResult` it assembles from the
host client's measured :class:`BenchmarkOutcome` and the planned
:class:`ServerSpec`: the successful runs and the failure count come straight
from the client, while the target, image, and device mapping are recovered from
the plan so the result records what actually ran, and each run's prefill and
decode rates are derived from the response timings the client stored. These
tests drive the backend with an in-memory outcome and a planned spec — no
Podman daemon, no inference engine, and no checkpoint files are involved.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    HardwareTarget,
    LlamaCppBackend,
    LLMConfig,
    RunMetrics,
    ServerSpec,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult

HUB_MODEL = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
"""A model string whose checkpoint a direct caller resolved itself."""

_RESOLVED = Path("/models/model.gguf")
"""A stand-in for the file the composition root resolved; planning never opens it."""


def _experiment() -> ExperimentConfig:
    """Build a minimal experiment; parsing does not depend on its tuning."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=HUB_MODEL, resolved_path=_RESOLVED),
        llm=LLMConfig(),
    )


def _plan(target: HardwareTarget) -> ServerSpec:
    """Plan a run for ``target`` so parsing can recover what ran."""
    return LlamaCppBackend().plan(_experiment(), DetectedHardware(target, 1))


def _container_result() -> ContainerResult:
    """Build a successful server container outcome with some captured output."""
    return ContainerResult(
        exit_code=0,
        stdout=b"server log\n",
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


def _response(
    prompt_n: int,
    prompt_ms: float,
    predicted_n: int,
    predicted_ms: float,
    cache_n: int = 0,
) -> Mapping[str, object]:
    """Build one completion payload carrying llama.cpp's timings object."""
    return {
        "timings": {
            "cache_n": cache_n,
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
        }
    }


def _timings_run(*responses: Mapping[str, object]) -> RunMetrics:
    """Build a run whose raw JSON stores the given completion payloads."""
    return _run(raw_json={"completed": len(responses), "responses": list(responses)})


def _phase_run(prefill_rate: float, decode_rate: float) -> RunMetrics:
    """Build a run whose one-second timings imply the given phase rates."""
    return _timings_run(
        _response(
            prompt_n=int(prefill_rate),
            prompt_ms=1000.0,
            predicted_n=int(decode_rate),
            predicted_ms=1000.0,
        )
    )


def test_the_measured_runs_become_the_typed_result() -> None:
    spec = _plan(HardwareTarget.CUDA)

    result = LlamaCppBackend().parse_result(_outcome(), spec, _container_result())

    assert result.tokens_per_second == 1234.5
    assert result.requests_per_second == 12.25
    assert result.elapsed_time == 8.5
    assert result.num_prompts == 100


def test_result_records_the_plan_that_ran_not_the_request() -> None:
    spec = _plan(HardwareTarget.ROCM)

    result = LlamaCppBackend().parse_result(_outcome(), spec, _container_result())

    assert result.target == HardwareTarget.ROCM
    assert result.image == "ghcr.io/ggml-org/llama.cpp:server-rocm"
    assert result.devices == ("/dev/kfd", "/dev/dri")


def test_raw_json_and_container_result_are_preserved() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _run(raw_json={"completed": 100, "model_id": HUB_MODEL, "failed": 0})
    container_result = _container_result()

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, container_result)

    assert len(result.runs) == 1
    assert dict(result.raw_json[0]) == {"completed": 100, "model_id": HUB_MODEL, "failed": 0}
    assert result.raw_result is container_result


def test_successful_runs_are_aggregated_and_the_failure_count_is_recorded() -> None:
    spec = _plan(HardwareTarget.CUDA)
    first = _run(tokens_per_second=10.0, requests_per_second=1.0, elapsed_time=2.0, num_prompts=4)
    second = _run(tokens_per_second=20.0, requests_per_second=3.0, elapsed_time=4.0, num_prompts=6)

    result = LlamaCppBackend().parse_result(
        _outcome(runs=(first, second), failed_runs=1), spec, _container_result()
    )

    assert result.tokens_per_second == 15.0
    assert result.requests_per_second == 2.0
    assert result.elapsed_time == 3.0
    assert result.num_prompts == 10
    assert result.failed_runs == 1
    assert result.runs == (first, second)


def test_an_outcome_with_no_successful_run_raises_with_the_raw_result() -> None:
    spec = _plan(HardwareTarget.CPU)
    container_result = _container_result()

    with pytest.raises(ExperimentError, match="no successful run") as excinfo:
        LlamaCppBackend().parse_result(
            BenchmarkOutcome(runs=(), failed_runs=2), spec, container_result
        )

    assert excinfo.value.raw_result is container_result


def test_devices_the_backend_never_plans_raise_rather_than_guess() -> None:
    spec = replace(
        _plan(HardwareTarget.CPU),
        container=ContainerConfig(image="custom/llama.cpp:dev", devices=("/dev/nvidia0",)),
    )
    container_result = _container_result()

    with pytest.raises(ExperimentError) as excinfo:
        LlamaCppBackend().parse_result(_outcome(), spec, container_result)

    assert "accelerator class" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_phase_rates_are_derived_from_the_response_timings() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(
        _response(prompt_n=4000, prompt_ms=4000.0, predicted_n=3000, predicted_ms=2000.0)
    )

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 1000.0  # 4000 tokens in 4 prefill seconds
    assert result.decode_tokens_per_second == 1500.0  # 3000 tokens in 2 decode seconds


def test_response_timings_are_summed_across_the_run() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(
        _response(prompt_n=1000, prompt_ms=1000.0, predicted_n=100, predicted_ms=500.0),
        _response(prompt_n=3000, prompt_ms=3000.0, predicted_n=500, predicted_ms=1500.0),
    )

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 1000.0  # 4000 tokens in 4 seconds
    assert result.decode_tokens_per_second == 300.0  # 600 tokens in 2 seconds


def test_cached_prompt_tokens_are_excluded_from_the_prefill_rate() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(
        _response(prompt_n=1000, prompt_ms=500.0, predicted_n=200, predicted_ms=400.0, cache_n=8000)
    )

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 2000.0  # only the processed prompt tokens
    assert result.decode_tokens_per_second == 500.0


def test_phase_rates_average_over_the_successful_runs() -> None:
    spec = _plan(HardwareTarget.CUDA)
    first = _phase_run(prefill_rate=1000.0, decode_rate=500.0)
    second = _phase_run(prefill_rate=3000.0, decode_rate=1500.0)

    result = LlamaCppBackend().parse_result(
        _outcome(runs=(first, second)), spec, _container_result()
    )

    assert result.prefill_tokens_per_second == 2000.0
    assert result.decode_tokens_per_second == 1000.0


def test_a_run_without_response_timings_reports_no_phase_rates() -> None:
    spec = _plan(HardwareTarget.CUDA)

    result = LlamaCppBackend().parse_result(_outcome(), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second is None
    assert result.tokens_per_second == 1234.5


def test_a_response_without_timings_reports_no_phase_rates() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(
        _response(prompt_n=1000, prompt_ms=500.0, predicted_n=200, predicted_ms=400.0),
        {"id": "cmpl-untimed"},
    )

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second is None


def test_a_missing_phase_timing_leaves_only_that_phase_absent() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run({"timings": {"prompt_n": 1000, "prompt_ms": 500.0, "predicted_n": 200}})

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second == 2000.0
    assert result.decode_tokens_per_second is None


def test_a_non_numeric_timing_is_treated_as_absent() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(
        {
            "timings": {
                "prompt_n": 1000,
                "prompt_ms": "slow",
                "predicted_n": 200,
                "predicted_ms": 400.0,
            }
        }
    )

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second == 500.0


def test_a_zero_time_denominator_yields_no_rate_for_that_phase() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(_response(prompt_n=1000, prompt_ms=0.0, predicted_n=200, predicted_ms=400.0))

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second == 500.0


def test_a_zero_token_count_yields_no_rate_for_that_phase() -> None:
    spec = _plan(HardwareTarget.CUDA)
    run = _timings_run(_response(prompt_n=0, prompt_ms=500.0, predicted_n=200, predicted_ms=400.0))

    result = LlamaCppBackend().parse_result(_outcome(runs=(run,)), spec, _container_result())

    assert result.prefill_tokens_per_second is None
    assert result.decode_tokens_per_second == 500.0
