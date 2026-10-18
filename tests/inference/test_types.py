"""Unit tests for the inference layer's immutability and hashability contract.

The experiment types promise two properties their callers rely on: an
experiment cannot be mutated after it is described, and it can be hashed so it
can be logged, compared, and used as a cache key. These tests assert that
contract directly — every type rejects attribute assignment with
``FrozenInstanceError`` and produces a hash — and that equal values hash equally
regardless of mapping insertion order, so a reordered ``env`` or ``raw_json``
does not change a config's identity. They mirror the analogous runtime-layer
test, ``tests/runtime/test_runtime.py::test_config_env_is_immutable_and_config_is_hashable``.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    ExperimentResult,
    HardwareTarget,
    LLMConfig,
    RunMetrics,
    ServerSpec,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult

HUB_MODEL = "meta-llama/Llama-3.1-8B"


def _experiment() -> ExperimentConfig:
    """Build a complete experiment whose sub-configs are all hashable."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _run(**overrides: object) -> RunMetrics:
    """Build a measured run whose ``raw_json`` is a mapping, so hashing is exercised."""
    fields: dict[str, object] = {
        "tokens_per_second": 1234.5,
        "requests_per_second": 12.25,
        "elapsed_time": 8.5,
        "num_prompts": 100,
        "raw_json": {"tokens_per_second": 1234.5, "num_requests": 100},
    }
    fields.update(overrides)
    return RunMetrics(**fields)  # type: ignore[arg-type]


def _result() -> ExperimentResult:
    """Build a result carrying one measured run and a container outcome."""
    return ExperimentResult(
        runs=(_run(),),
        failed_runs=0,
        target=HardwareTarget.CUDA,
        image="docker.io/vllm/vllm-openai:latest",
        devices=("nvidia.com/gpu=all",),
        raw_result=ContainerResult(exit_code=0, stdout=b"benchmark log\n", stderr=b""),
    )


def _server_spec() -> ServerSpec:
    """Build a server spec whose container and reset paths are hashable."""
    return ServerSpec(
        container=ContainerConfig(
            image="docker.io/vllm/vllm-openai:latest",
            command=("serve",),
            ports={"8000": None},
        ),
        health_path="/health",
        health_timeout=600.0,
        reset_paths=("/reset_prefix_cache",),
    )


@pytest.mark.parametrize(
    ("config", "attribute", "value"),
    [
        (ContainerRuntimeConfig(), "shm_size", "1g"),
        (BenchmarkConfig(model=HUB_MODEL), "model", "other/model"),
        (LLMConfig(), "dtype", DType.FLOAT16),
        (_experiment(), "llm", LLMConfig(dtype=DType.FLOAT16)),
        (_result(), "image", "other/image"),
        (_run(), "tokens_per_second", 1.0),
        (_server_spec(), "health_path", "/ready"),
    ],
    ids=["container", "benchmark", "llm", "experiment", "result", "run", "server"],
)
def test_config_types_are_frozen_and_hashable(
    config: object, attribute: str, value: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(config, attribute, value)
    assert isinstance(hash(config), int)


def test_container_config_env_is_immutable_and_hashes_regardless_of_order() -> None:
    config = ContainerRuntimeConfig(env={"FOO": "bar", "BAZ": "qux"})

    assert config.env == {"FOO": "bar", "BAZ": "qux"}
    with pytest.raises(TypeError):
        config.env["FOO"] = "changed"
    assert config == ContainerRuntimeConfig(env={"FOO": "bar", "BAZ": "qux"})
    assert hash(config) == hash(ContainerRuntimeConfig(env={"BAZ": "qux", "FOO": "bar"}))


def test_result_hashes_regardless_of_raw_json_order() -> None:
    first = replace(_result(), runs=(_run(raw_json={"a": 1, "b": 2}),))
    second = replace(_result(), runs=(_run(raw_json={"b": 2, "a": 1}),))

    assert first == second
    assert hash(first) == hash(second)


def test_result_hashes_the_nested_payload_the_engine_writes() -> None:
    """``vllm bench serve`` writes list-valued percentile fields, not only scalars."""
    run = _run(
        raw_json={
            "total_token_throughput": 1234.5,
            "completed": 100,
            "percentiles_ttft_ms": [[50, 12.3], [99, 45.6]],
        }
    )

    assert isinstance(hash(replace(_result(), runs=(run,))), int)


def test_run_hashes_regardless_of_nested_mapping_order() -> None:
    first = _run(raw_json={"stats": {"a": 1, "b": [1, 2]}})
    second = _run(raw_json={"stats": {"b": [1, 2], "a": 1}})

    assert first == second
    assert hash(first) == hash(second)


def test_result_derives_its_metrics_from_the_successful_runs() -> None:
    runs = (
        _run(tokens_per_second=10.0, requests_per_second=1.0, elapsed_time=2.0, num_prompts=4),
        _run(tokens_per_second=20.0, requests_per_second=3.0, elapsed_time=4.0, num_prompts=6),
    )

    result = replace(_result(), runs=runs, failed_runs=1)

    assert result.tokens_per_second == 15.0
    assert result.requests_per_second == 2.0
    assert result.elapsed_time == 3.0
    assert result.num_prompts == 10
    assert result.raw_json == tuple(run.raw_json for run in runs)
    assert result.failed_runs == 1


def test_result_rejects_an_empty_run_set() -> None:
    with pytest.raises(ValueError):
        ExperimentResult(
            runs=(),
            failed_runs=0,
            target=HardwareTarget.CPU,
            image="fake/engine:latest",
            devices=(),
            raw_result=ContainerResult(exit_code=0, stdout=b"", stderr=b""),
        )


def test_replace_yields_a_distinct_hash() -> None:
    experiment = _experiment()

    varied = replace(experiment, llm=replace(experiment.llm, dtype=DType.FLOAT16))

    assert varied != experiment
    assert hash(varied) != hash(experiment)


def test_server_spec_defaults_to_no_reset_paths() -> None:
    spec = ServerSpec(
        container=ContainerConfig(image="img"),
        health_path="/health",
        health_timeout=1.0,
    )

    assert spec.reset_paths == ()
    assert spec.container.image == "img"
    assert spec.health_path == "/health"
    assert spec.health_timeout == 1.0


def test_server_spec_hashes_regardless_of_container_env_order() -> None:
    first = ServerSpec(
        container=ContainerConfig(image="img", env={"A": "1", "B": "2"}),
        health_path="/health",
        health_timeout=1.0,
    )
    second = ServerSpec(
        container=ContainerConfig(image="img", env={"B": "2", "A": "1"}),
        health_path="/health",
        health_timeout=1.0,
    )

    assert first == second
    assert hash(first) == hash(second)
