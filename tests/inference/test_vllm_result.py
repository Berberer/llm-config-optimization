"""Unit tests for vLLM result parsing and the failures around it.

``VLLMBackend.parse_result``'s observable contract is the typed
:class:`~llm_config_optimization.inference.types.ExperimentResult` it builds from the
engine's result JSON, and the :class:`~llm_config_optimization.inference.types.ExperimentError`
it raises when that JSON is absent, unreadable, or malformed. These tests drive
the backend with a planned container config and a captured container result —
no Podman daemon and no inference engine is involved — and assert both the typed
result and the error payloads, including the preserved cause.
"""

import json
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    HardwareTarget,
    LLMConfig,
    VLLMBackend,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult

HUB_MODEL = "meta-llama/Llama-3.1-8B"

_RESULT_FILENAME = "result.json"
_WELL_FORMED = {
    "tokens_per_second": 1234.5,
    "requests_per_second": 12.25,
    "elapsed_time": 8.5,
    "num_requests": 100,
}


def _experiment() -> ExperimentConfig:
    """Build a minimal experiment; parsing does not depend on its tuning."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _plan(target: HardwareTarget, output_dir: Path) -> ContainerConfig:
    """Plan a run for ``target`` so parsing can recover what ran."""
    return VLLMBackend().plan(_experiment(), DetectedHardware(target, 1), output_dir=output_dir)


def _container_result() -> ContainerResult:
    """Build a successful container outcome with some captured output."""
    return ContainerResult(
        exit_code=0,
        stdout=b"benchmark log\n",
        stderr=b"",
        container_id="c0ffee",
    )


def _write_result(output_dir: Path, payload: object) -> None:
    """Write ``payload`` as the result file the backend reads back."""
    (output_dir / _RESULT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_well_formed_result_is_parsed_into_typed_fields(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    _write_result(tmp_path, _WELL_FORMED)

    result = VLLMBackend().parse_result(_container_result(), plan, tmp_path)

    assert result.tokens_per_second == 1234.5
    assert result.requests_per_second == 12.25
    assert result.elapsed_time == 8.5
    assert result.num_prompts == 100


def test_result_records_the_plan_that_ran_not_the_request(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.ROCM, tmp_path)
    _write_result(tmp_path, _WELL_FORMED)

    result = VLLMBackend().parse_result(_container_result(), plan, tmp_path)

    assert result.target == HardwareTarget.ROCM
    assert result.image == "docker.io/vllm/vllm-openai-rocm:latest"
    assert result.devices == ("/dev/kfd", "/dev/dri")


def test_raw_json_and_container_result_are_preserved(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    payload = {**_WELL_FORMED, "model_id": HUB_MODEL, "completed": 100}
    _write_result(tmp_path, payload)
    container_result = _container_result()

    result = VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert dict(result.raw_json) == payload
    assert result.raw_result is container_result


def test_parse_result_falls_back_to_the_planned_default_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no directory is given, parsing reads where the plan mounted."""
    monkeypatch.setattr("llm_config_optimization.inference._vllm.DEFAULT_OUTPUT_DIR", tmp_path)
    plan = _plan(HardwareTarget.CPU, tmp_path)
    _write_result(tmp_path, _WELL_FORMED)

    result = VLLMBackend().parse_result(_container_result(), plan)

    assert result.num_prompts == 100


def test_missing_result_file_raises_with_the_raw_result_attached(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "could not read" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_unparseable_result_file_raises_and_preserves_the_decode_error(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    (tmp_path / _RESULT_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not valid JSON" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_result_that_is_not_a_json_object_is_rejected(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_result(tmp_path, [1, 2, 3])

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not a JSON object" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    "metric",
    ["tokens_per_second", "requests_per_second", "elapsed_time", "num_requests"],
)
def test_result_missing_a_required_metric_is_rejected(tmp_path: Path, metric: str) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    payload = {key: value for key, value in _WELL_FORMED.items() if key != metric}
    _write_result(tmp_path, payload)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_result_with_a_non_numeric_metric_is_rejected(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_result(tmp_path, {**_WELL_FORMED, "tokens_per_second": "fast"})

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_devices_the_backend_never_plans_raise_rather_than_guess(tmp_path: Path) -> None:
    plan = ContainerConfig(image="custom/vllm:dev", devices=("/dev/nvidia0",))
    container_result = _container_result()
    _write_result(tmp_path, _WELL_FORMED)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "accelerator class" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
