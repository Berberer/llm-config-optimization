"""Unit tests for the benchmark workload's measurement controls.

A :class:`~llm_config_optimization.inference.types.BenchmarkConfig` carries three
non-tunable controls that decide how many measurements a trial takes:
``repetitions``, ``warmup``, and ``startup_timeout``. These tests pin their
documented defaults and the constructor's rejection of values that could not
produce a usable trial — a repetition count below one, a negative warmup, and a
non-positive startup timeout — which the layer turns into a usage error before
any container runs. They are pure unit tests: no engine and no daemon.
"""

import pytest

from llm_config_optimization.inference import BenchmarkConfig

HUB_MODEL = "meta-llama/Llama-3.1-8B"


def test_measurement_controls_have_the_documented_defaults() -> None:
    benchmark = BenchmarkConfig(model=HUB_MODEL)

    assert benchmark.repetitions == 3
    assert benchmark.warmup == 1
    assert benchmark.startup_timeout == 600.0
    assert benchmark.timeout is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 0},
        {"repetitions": -1},
        {"warmup": -1},
        {"startup_timeout": 0},
        {"startup_timeout": -1.0},
    ],
    ids=[
        "zero-repetitions",
        "negative-repetitions",
        "negative-warmup",
        "zero-timeout",
        "negative-timeout",
    ],
)
def test_out_of_range_measurement_controls_are_rejected(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(model=HUB_MODEL, **kwargs)


def test_boundary_values_are_accepted() -> None:
    benchmark = BenchmarkConfig(model=HUB_MODEL, repetitions=1, warmup=0, startup_timeout=0.001)

    assert benchmark.repetitions == 1
    assert benchmark.warmup == 0
    assert benchmark.startup_timeout == 0.001
