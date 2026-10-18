"""Unit tests for vLLM result parsing and the failures around it.

``VLLMBackend.parse_result``'s observable contract is the typed
:class:`~llm_config_optimization.inference.types.ExperimentResult` it builds from the
runner's manifest and the per-run online benchmark JSON files it names, and the
:class:`~llm_config_optimization.inference.types.ExperimentError` it raises when
any of those files is absent, unreadable, malformed, or too few to form a strict
majority. These tests drive the backend with a planned container config and a
captured container result — no Podman daemon and no inference engine is involved
— and assert both the typed result and the error payloads, including the
preserved cause.
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

_MANIFEST_FILENAME = "manifest.json"
_WELL_FORMED = {
    "total_token_throughput": 1234.5,
    "request_throughput": 12.25,
    "duration": 8.5,
    "completed": 100,
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


def _write_run(output_dir: Path, filename: str, payload: object) -> None:
    """Write one run's result file exactly as the runner would."""
    (output_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest_payload(output_dir: Path, payload: object) -> None:
    """Write an arbitrary manifest payload, for malformed-manifest cases."""
    (output_dir / _MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    *,
    runs: list[str],
    failed_runs: int = 0,
    repetitions: int | None = None,
    warmup: int = 0,
) -> None:
    """Write a well-formed manifest naming ``runs`` as the successful runs."""
    if repetitions is None:
        repetitions = len(runs) + failed_runs
    _write_manifest_payload(
        output_dir,
        {
            "repetitions": repetitions,
            "warmup": warmup,
            "failed_runs": failed_runs,
            "runs": runs,
        },
    )


def _write_result(output_dir: Path, payload: object) -> None:
    """Write a single successful run plus the manifest that names it."""
    _write_run(output_dir, "run-0.json", payload)
    _write_manifest(output_dir, runs=["run-0.json"])


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
    payload = {**_WELL_FORMED, "model_id": HUB_MODEL, "failed": 0}
    _write_result(tmp_path, payload)
    container_result = _container_result()

    result = VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert len(result.runs) == 1
    assert dict(result.raw_json[0]) == payload
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


def test_missing_manifest_raises_with_the_raw_result_attached(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "could not read" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_unparseable_manifest_raises_and_preserves_the_decode_error(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    (tmp_path / _MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not valid JSON" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_manifest_that_is_not_a_json_object_is_rejected(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_manifest_payload(tmp_path, [1, 2, 3])

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not a JSON object" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize("key", ["repetitions", "failed_runs", "runs"])
def test_manifest_missing_a_required_key_is_rejected(tmp_path: Path, key: str) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    payload = {
        "repetitions": 1,
        "warmup": 0,
        "failed_runs": 0,
        "runs": ["run-0.json"],
    }
    del payload[key]
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_manifest_runs_must_be_a_list_of_filenames(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_manifest_payload(
        tmp_path,
        {"repetitions": 1, "warmup": 0, "failed_runs": 0, "runs": "run-0.json"},
    )

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


@pytest.mark.parametrize(
    "name",
    ["../run-0.json", "/etc/passwd", "nested/run-0.json", ""],
    ids=["parent", "absolute", "subdirectory", "empty"],
)
def test_manifest_run_names_must_be_bare_filenames(tmp_path: Path, name: str) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_manifest_payload(
        tmp_path,
        {"repetitions": 1, "warmup": 0, "failed_runs": 0, "runs": [name]},
    )

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_manifest_failure_count_must_agree_with_the_runs_it_names(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_run(tmp_path, "run-0.json", _WELL_FORMED)
    _write_run(tmp_path, "run-1.json", _WELL_FORMED)
    # Two of three runs succeeded, so the majority holds; only the count is wrong.
    _write_manifest(tmp_path, runs=["run-0.json", "run-1.json"], failed_runs=0, repetitions=3)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "inconsistent" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_run_referenced_by_the_manifest_must_exist(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_manifest(tmp_path, runs=["run-0.json"])

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "could not read" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_unparseable_run_file_raises_and_preserves_the_decode_error(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    (tmp_path / "run-0.json").write_text("{not json", encoding="utf-8")
    _write_manifest(tmp_path, runs=["run-0.json"])

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not valid JSON" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_run_file_that_is_not_a_json_object_is_rejected(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_run(tmp_path, "run-0.json", [1, 2, 3])
    _write_manifest(tmp_path, runs=["run-0.json"])

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "not a JSON object" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


@pytest.mark.parametrize(
    "metric",
    ["total_token_throughput", "request_throughput", "duration", "completed"],
)
def test_run_missing_a_required_metric_is_rejected(tmp_path: Path, metric: str) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    payload = {key: value for key, value in _WELL_FORMED.items() if key != metric}
    _write_result(tmp_path, payload)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_run_with_a_non_numeric_metric_is_rejected(tmp_path: Path) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    _write_result(tmp_path, {**_WELL_FORMED, "total_token_throughput": "fast"})

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "missing or malformed" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_successful_runs_are_aggregated_and_the_failure_count_is_recorded(
    tmp_path: Path,
) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    first = {
        "total_token_throughput": 10.0,
        "request_throughput": 1.0,
        "duration": 2.0,
        "completed": 4,
    }
    second = {
        "total_token_throughput": 20.0,
        "request_throughput": 3.0,
        "duration": 4.0,
        "completed": 6,
    }
    _write_run(tmp_path, "run-0.json", first)
    _write_run(tmp_path, "run-1.json", second)
    _write_manifest(tmp_path, runs=["run-0.json", "run-1.json"], failed_runs=1, repetitions=3)

    result = VLLMBackend().parse_result(_container_result(), plan, tmp_path)

    assert result.tokens_per_second == 15.0
    assert result.requests_per_second == 2.0
    assert result.elapsed_time == 3.0
    assert result.num_prompts == 10
    assert result.failed_runs == 1
    assert [dict(run.raw_json) for run in result.runs] == [first, second]


@pytest.mark.parametrize(
    ("successes", "repetitions"),
    [
        (2, 3),
        (3, 4),
        (4, 4),
    ],
)
def test_a_strict_majority_of_successful_runs_passes(
    tmp_path: Path, successes: int, repetitions: int
) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    filenames = [f"run-{index}.json" for index in range(successes)]
    for filename in filenames:
        _write_run(tmp_path, filename, _WELL_FORMED)
    _write_manifest(
        tmp_path,
        runs=filenames,
        failed_runs=repetitions - successes,
        repetitions=repetitions,
    )

    result = VLLMBackend().parse_result(_container_result(), plan, tmp_path)

    assert len(result.runs) == successes
    assert result.failed_runs == repetitions - successes


@pytest.mark.parametrize(
    ("successes", "repetitions"),
    [
        (1, 3),
        (2, 4),
        (0, 1),
    ],
)
def test_too_few_successful_runs_raise_even_when_the_container_exited_zero(
    tmp_path: Path, successes: int, repetitions: int
) -> None:
    plan = _plan(HardwareTarget.CUDA, tmp_path)
    container_result = _container_result()
    filenames = [f"run-{index}.json" for index in range(successes)]
    for filename in filenames:
        _write_run(tmp_path, filename, _WELL_FORMED)
    _write_manifest(
        tmp_path,
        runs=filenames,
        failed_runs=repetitions - successes,
        repetitions=repetitions,
    )

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "majority" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result


def test_devices_the_backend_never_plans_raise_rather_than_guess(tmp_path: Path) -> None:
    plan = ContainerConfig(image="custom/vllm:dev", devices=("/dev/nvidia0",))
    container_result = _container_result()
    _write_result(tmp_path, _WELL_FORMED)

    with pytest.raises(ExperimentError) as excinfo:
        VLLMBackend().parse_result(container_result, plan, tmp_path)

    assert "accelerator class" in str(excinfo.value)
    assert excinfo.value.raw_result is container_result
