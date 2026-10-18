"""Unit tests for the space-to-trial mapping and config reconstruction.

The mapping's observable contract is twofold. Given an engine-neutral
:class:`~llm_config_optimization.search_space.ExperimentSpace`, a trial must offer
exactly the choices each spec kind declares — a one-choice categorical for
``Fixed`` so its value is recorded on the trial, the declared values in declared
order for ``Categorical``, an inclusive integer range for ``Interval``, and no
suggestion at all for ``Default`` so deferral to the engine survives the sweep.
Given a sampled parameter mapping, reconstruction must turn it back into an
:class:`~llm_config_optimization.inference.LLMConfig`: sampled strings become the
field's engine-neutral enum members, a ``Default`` field is unset, and the
non-tunable fields are inherited from the base config.

These tests drive the mapping functions directly with a real in-memory Optuna
study, so no daemon, GPU, or inference engine is involved. They mirror the
search-space loader tests' coverage of every spec kind and the inference-layer
tests' use of engine-neutral enums.
"""

from enum import StrEnum

import optuna
import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    HardwareTarget,
    KVCacheDType,
    LLMConfig,
    QuantizationMethod,
)
from llm_config_optimization.optimization._optuna import (
    reconstruct_config,
    reconstruct_llm,
    suggest_parameters,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
)

FULL_SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Categorical((DType.FLOAT16, DType.BFLOAT16))),
        Dimension("kv_cache_dtype", Categorical((KVCacheDType.AUTO, KVCacheDType.FP8_E4M3))),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(256)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("tensor_parallel_size", Categorical((1, 2, 4))),
        Dimension(
            "attention_backend",
            Categorical((AttentionBackend.XFORMERS, AttentionBackend.FLASH_ATTENTION)),
        ),
    )
)
"""A complete space over every tunable, with all four spec kinds represented."""

PARAMETERS: dict[str, object] = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "fp8_e4m3",
    "max_model_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 4096,
    "tensor_parallel_size": 2,
    "attention_backend": "flash_attention",
}
"""A parameter mapping as a persisted study would return it: plain JSON values."""

BASE_LLM = LLMConfig(quantization=QuantizationMethod.AWQ, seed=7, gpu_memory_utilization=0.85)
BASE_EXPERIMENT = ExperimentConfig(
    container=ContainerRuntimeConfig(image="example/engine:latest", target=HardwareTarget.CUDA),
    benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B", num_prompts=32),
    llm=BASE_LLM,
)


def _sample(space: ExperimentSpace, count: int = 1, *, seed: int = 0) -> list[optuna.Trial]:
    """Sample ``count`` trials of ``space`` through a real in-memory study."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    trials: list[optuna.Trial] = []
    for _ in range(count):
        trial = study.ask()
        suggest_parameters(trial, space)
        study.tell(trial, 1.0)
        trials.append(trial)
    return trials


def test_fixed_dimension_is_a_one_choice_categorical_recorded_on_the_trial() -> None:
    """A fixed value is offered once and recorded, so the trial stays explainable."""
    space = ExperimentSpace(dimensions=(Dimension("dtype", Fixed(DType.FLOAT16)),))

    (trial,) = _sample(space)

    assert trial.distributions["dtype"].choices == ("float16",)
    assert trial.params["dtype"] == "float16"


def test_categorical_dimension_offers_its_declared_values_in_order() -> None:
    space = ExperimentSpace(
        dimensions=(
            Dimension(
                "attention_backend",
                Categorical((AttentionBackend.XFORMERS, AttentionBackend.FLASH_ATTENTION)),
            ),
        )
    )

    (trial,) = _sample(space)

    assert trial.distributions["attention_backend"].choices == ("xformers", "flash_attention")
    assert trial.params["attention_backend"] in {"xformers", "flash_attention"}


def test_categorical_dimension_samples_only_declared_values() -> None:
    space = ExperimentSpace(
        dimensions=(
            Dimension(
                "attention_backend",
                Categorical((AttentionBackend.XFORMERS, AttentionBackend.FLASH_ATTENTION)),
            ),
        )
    )

    sampled = {trial.params["attention_backend"] for trial in _sample(space, 20, seed=1)}

    assert sampled <= {"xformers", "flash_attention"}
    assert sampled


def test_numeric_categorical_samples_only_its_declared_integers() -> None:
    """A numeric ``Categorical`` offers its declared integers and nothing between them."""
    space = ExperimentSpace(dimensions=(Dimension("max_num_seqs", Categorical((1, 20, 99))),))

    trials = _sample(space, 30, seed=3)
    distribution = trials[0].distributions["max_num_seqs"]
    sampled = {trial.params["max_num_seqs"] for trial in trials}

    assert distribution.choices == (1, 20, 99)
    assert sampled <= {1, 20, 99}
    assert sampled


def test_interval_dimension_samples_integers_within_its_inclusive_bounds() -> None:
    space = ExperimentSpace(dimensions=(Dimension("max_model_len", Interval(1024, 8192)),))

    trials = _sample(space, 20, seed=2)
    distribution = trials[0].distributions["max_model_len"]

    assert (distribution.low, distribution.high, distribution.step) == (1024, 8192, 1)
    assert all(1024 <= trial.params["max_model_len"] <= 8192 for trial in trials)


def test_default_dimension_produces_no_suggestion() -> None:
    space = ExperimentSpace(dimensions=(Dimension("attention_backend", Default()),))

    (trial,) = _sample(space)

    assert trial.params == {}
    assert trial.distributions == {}


def test_default_dimension_leaves_its_field_unset() -> None:
    """A ``Default`` dimension genuinely defers, even when the base set a value."""
    space = ExperimentSpace(dimensions=(Dimension("attention_backend", Default()),))
    base = LLMConfig(attention_backend=AttentionBackend.XFORMERS, seed=7)

    result = reconstruct_llm(space, base, {})

    assert result.attention_backend is None
    assert result.seed == 7


@pytest.mark.parametrize(
    ("field", "sampled", "expected"),
    [
        ("dtype", "bfloat16", DType.BFLOAT16),
        ("kv_cache_dtype", "fp8_e4m3", KVCacheDType.FP8_E4M3),
        ("attention_backend", "flash_attention", AttentionBackend.FLASH_ATTENTION),
    ],
    ids=["dtype", "kv-cache-dtype", "attention-backend"],
)
def test_sampled_strings_round_trip_to_enum_members(
    field: str, sampled: str, expected: StrEnum
) -> None:
    """A persisted plain string resolves back to the field's engine-neutral member.

    The attention backend is included because its engine spelling differs from
    its enum value (vLLM, for instance, spells ``flash_attention``
    ``FLASH_ATTN``); the neutral member is what lets the backend's value map
    apply.
    """
    space = ExperimentSpace(dimensions=(Dimension(field, Fixed(expected)),))

    result = reconstruct_llm(space, LLMConfig(), {field: sampled})

    assert getattr(result, field) is expected


def test_reconstruct_llm_takes_tunables_from_the_mapping_and_inherits_the_rest() -> None:
    result = reconstruct_llm(FULL_SPACE, BASE_LLM, PARAMETERS)

    assert result == LLMConfig(
        dtype=DType.BFLOAT16,
        quantization=QuantizationMethod.AWQ,
        kv_cache_dtype=KVCacheDType.FP8_E4M3,
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=2,
        seed=7,
        attention_backend=AttentionBackend.FLASH_ATTENTION,
    )


def test_reconstruct_config_reuses_the_base_container_and_benchmark() -> None:
    result = reconstruct_config(FULL_SPACE, BASE_EXPERIMENT, PARAMETERS)

    assert result.container is BASE_EXPERIMENT.container
    assert result.benchmark is BASE_EXPERIMENT.benchmark
    assert result.llm == reconstruct_llm(FULL_SPACE, BASE_EXPERIMENT.llm, PARAMETERS)
