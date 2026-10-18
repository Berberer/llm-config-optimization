"""Unit tests for vLLM result parsing.

``VLLMBackend.parse_result``'s observable contract is the typed
:class:`~llm_config_optimization.inference.types.ExperimentResult` it assembles
from the host client's measured :class:`BenchmarkOutcome` and the planned
:class:`ServerSpec`: the successful runs and the failure count come straight
from the client, while the target, image, and device mapping are recovered from
the plan so the result records what actually ran. These tests drive the backend
with an in-memory outcome and a planned spec — no Podman daemon, no inference
engine, and no result files are involved.
"""

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
