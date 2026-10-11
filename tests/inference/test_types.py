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

from llm_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    ExperimentResult,
    HardwareTarget,
    LLMConfig,
)
from llm_optimization.runtime import ContainerResult

HUB_MODEL = "meta-llama/Llama-3.1-8B"


def _experiment() -> ExperimentConfig:
    """Build a complete experiment whose sub-configs are all hashable."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=HUB_MODEL),
        llm=LLMConfig(),
    )


def _result() -> ExperimentResult:
    """Build a result carrying a mapping for ``raw_json`` and a container outcome."""
    return ExperimentResult(
        tokens_per_second=1234.5,
        requests_per_second=12.25,
        elapsed_time=8.5,
        num_prompts=100,
        target=HardwareTarget.CUDA,
        image="docker.io/vllm/vllm-openai:latest",
        devices=("nvidia.com/gpu=all",),
        raw_json={"tokens_per_second": 1234.5, "num_requests": 100},
        raw_result=ContainerResult(exit_code=0, stdout=b"benchmark log\n", stderr=b""),
    )


@pytest.mark.parametrize(
    ("config", "attribute", "value"),
    [
        (ContainerRuntimeConfig(), "shm_size", "1g"),
        (BenchmarkConfig(model=HUB_MODEL), "model", "other/model"),
        (LLMConfig(), "dtype", DType.FLOAT16),
        (_experiment(), "llm", LLMConfig(dtype=DType.FLOAT16)),
        (_result(), "image", "other/image"),
    ],
    ids=["container", "benchmark", "llm", "experiment", "result"],
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
    first = replace(_result(), raw_json={"a": 1, "b": 2})
    second = replace(_result(), raw_json={"b": 2, "a": 1})

    assert first == second
    assert hash(first) == hash(second)


def test_replace_yields_a_distinct_hash() -> None:
    experiment = _experiment()

    varied = replace(experiment, llm=replace(experiment.llm, dtype=DType.FLOAT16))

    assert varied != experiment
    assert hash(varied) != hash(experiment)
