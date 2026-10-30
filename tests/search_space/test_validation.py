"""Unit tests for engine-support validation of a search-space definition.

Engine-support validation is the second, explicit half of checking a
definition: once a file has been parsed engine-agnostically, a backend is asked
whether it can forward every declared parameter and value.
:meth:`VLLMBackend.validate` reports the problems it sees as strings, and
:func:`validate_space` aggregates them into a single
:class:`ExperimentDefinitionError`. These tests craft ``ExperimentSpace`` values
directly and drive both surfaces, so no inference engine and no Podman daemon is
involved. They assert on the parameter path and the kind of problem rather than
on exact wording, so messages can be reworded freely.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    DetectedHardware,
    DType,
    HardwareTarget,
    KVCacheDType,
    LLMConfig,
    SpeculativeMethod,
    SpeculativeMode,
    VLLMBackend,
)
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentDefinitionError,
    ExperimentSpace,
    Fixed,
    Interval,
    validate_space,
)
from llm_config_optimization.search_space.registry import SPECULATIVE_PARAMETERS

SUPPORTED_SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("kv_cache_dtype", Default()),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(256)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("max_num_active_seqs", Fixed(16)),
        Dimension("max_num_scheduled_tokens", Fixed(2048)),
        Dimension("tensor_parallel_size", Categorical((1, 2, 4))),
        Dimension("attention_backend", Default()),
        Dimension("prefix_caching", Fixed(True)),
    )
)
"""A space whose every parameter and value vLLM supports."""


@dataclass(frozen=True, slots=True)
class StubBackend:
    """A backend whose support check returns a canned set of problems.

    ``validate_space`` is a thin aggregator over the backend hook, so a stub
    that returns fixed problems isolates the aggregation contract from any
    engine's actual support rules.
    """

    problems: tuple[str, ...] = ()

    def validate(
        self,
        space: ExperimentSpace,
        llm: LLMConfig,
        hardware: DetectedHardware | None = None,
    ) -> Sequence[str]:
        """Ignore the space and report the canned problems."""
        return list(self.problems)


def space(*dimensions: Dimension) -> ExperimentSpace:
    """Assemble a space from explicit dimensions, bypassing the file parser."""
    return ExperimentSpace(dimensions=dimensions)


def validate(
    definition: ExperimentSpace, hardware: DetectedHardware | None = None
) -> Sequence[str]:
    """Validate ``definition`` against vLLM with a configuration that has no setup."""
    return VLLMBackend().validate(definition, LLMConfig(), hardware)


def test_a_fully_supported_space_has_no_problems() -> None:
    assert list(validate(SUPPORTED_SPACE)) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("speculative_mode", Fixed("enabled")),
        Dimension("num_speculative_tokens", Fixed(4)),
    ],
    ids=["mode", "depth"],
)
def test_a_gated_tunable_without_a_configured_setup_is_reported(
    dimension: Dimension,
) -> None:
    problems = validate(space(dimension))

    assert any(
        f"space.{dimension.name}" in problem and "does not forward" in problem
        for problem in problems
    ), problems


def test_a_gated_tunable_is_checked_against_a_configured_setup() -> None:
    """With a setup the parameter is applicable, forwardable, and value-checked."""
    llm = LLMConfig(speculative_method=SpeculativeMethod.MTP)
    definition = space(
        Dimension("speculative_mode", Fixed(SpeculativeMode.ENABLED)),
        Dimension("num_speculative_tokens", Fixed(4)),
    )

    applicable = VLLMBackend().applicable_parameters(llm)

    assert SPECULATIVE_PARAMETERS.issubset(applicable)
    assert list(VLLMBackend().validate(definition, llm)) == []


@pytest.mark.parametrize(
    "dimension",
    [
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("attention_backend", Categorical((AttentionBackend.FLASH_ATTENTION,))),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(256)),
        Dimension("kv_cache_dtype", Default()),
        Dimension("tensor_parallel_size", Default()),
    ],
    ids=[
        "fixed-enum",
        "categorical-enum",
        "interval",
        "fixed-integer",
        "default-enum",
        "default-int",
    ],
)
def test_each_spec_kind_is_accepted_for_a_supported_parameter(dimension: Dimension) -> None:
    assert list(validate(space(dimension))) == []


def test_a_parameter_the_backend_does_not_forward_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_config_optimization.inference._vllm.PARAMETER_TARGETS", MappingProxyType({})
    )

    problems = validate(space(Dimension("dtype", Fixed(DType.FLOAT16))))

    assert any("space.dtype" in problem and "does not forward" in problem for problem in problems)


def test_llama_cpp_gpu_layers_are_reported_as_not_forwarded_by_vllm() -> None:
    problems = validate(space(Dimension("n_gpu_layers", Fixed(0))))

    assert any(
        "space.n_gpu_layers" in problem and "does not forward" in problem for problem in problems
    )


def test_a_default_dimension_still_requires_a_forwarded_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferral skips the value check, not the parameter check."""
    monkeypatch.setattr(
        "llm_config_optimization.inference._vllm.PARAMETER_TARGETS", MappingProxyType({})
    )

    problems = validate(space(Dimension("attention_backend", Default())))

    assert any("space.attention_backend" in problem for problem in problems)


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        (Fixed(DType.FLOAT16), "space.dtype.fixed"),
        (Categorical((DType.AUTO, DType.FLOAT16)), "space.dtype.categorical[1]"),
    ],
    ids=["fixed", "categorical"],
)
def test_an_unsupported_enum_value_is_reported(
    monkeypatch: pytest.MonkeyPatch, spec: Fixed | Categorical, path: str
) -> None:
    monkeypatch.setattr(
        "llm_config_optimization.inference._vllm.VALUE_MAPS",
        MappingProxyType({"dtype": {DType.AUTO: "auto"}}),
    )

    problems = validate(space(Dimension("dtype", spec)))

    assert any(problem.startswith(path) for problem in problems), problems


def test_a_space_that_can_violate_an_engine_bound_is_reported() -> None:
    violating = space(
        Dimension("max_num_batched_tokens", Interval(1, 512)),
        Dimension("max_num_seqs", Interval(1024, 2048)),
    )

    problems = validate(violating)

    assert any(
        "space.max_num_batched_tokens" in problem and "max_num_seqs" in problem
        for problem in problems
    )


def test_a_space_that_satisfies_the_engine_bound_has_no_problems() -> None:
    satisfying = space(
        Dimension("max_num_batched_tokens", Interval(2048, 4096)),
        Dimension("max_num_seqs", Interval(1, 1024)),
    )

    assert list(validate(satisfying)) == []


def test_a_matching_pinned_float_cache_dtype_is_accepted() -> None:
    accepted = space(
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("kv_cache_dtype", Categorical((KVCacheDType.AUTO, KVCacheDType.FLOAT16))),
    )

    assert list(validate(accepted)) == []


@pytest.mark.parametrize(
    ("dtype_spec", "cache_spec"),
    [
        (Fixed(DType.FLOAT16), Fixed(KVCacheDType.BFLOAT16)),
        (Fixed(DType.BFLOAT16), Fixed(KVCacheDType.FLOAT16)),
    ],
    ids=["float-to-bfloat", "bfloat-to-float"],
)
def test_an_explicit_float_cache_pinned_to_the_wrong_dtype_is_reported(
    dtype_spec: Fixed,
    cache_spec: Fixed,
) -> None:
    problems = validate(
        space(
            Dimension("dtype", dtype_spec),
            Dimension("kv_cache_dtype", cache_spec),
        )
    )

    assert any(
        problem.startswith("space.kv_cache_dtype") and "explicit float KV cache" in problem
        for problem in problems
    ), problems


@pytest.mark.parametrize(
    ("dtype_spec", "cache_spec"),
    [
        (Categorical((DType.FLOAT16, DType.BFLOAT16)), Fixed(KVCacheDType.FLOAT16)),
        (Fixed(DType.AUTO), Fixed(KVCacheDType.FLOAT16)),
        (Default(), Fixed(KVCacheDType.BFLOAT16)),
    ],
    ids=["mixed-dtype", "auto-dtype", "default-dtype"],
)
def test_an_explicit_float_cache_left_to_the_model_aware_assessment_is_not_reported(
    dtype_spec: Fixed | Categorical | Default,
    cache_spec: Fixed,
) -> None:
    """These pairs depend on the resolved model dtype, so pruning decides them."""
    accepted = space(
        Dimension("dtype", dtype_spec),
        Dimension("kv_cache_dtype", cache_spec),
    )

    assert list(validate(accepted)) == []


def test_an_fp8_cache_is_exempt_from_the_model_dtype_match() -> None:
    accepted = space(
        Dimension("dtype", Categorical((DType.AUTO, DType.BFLOAT16))),
        Dimension("kv_cache_dtype", Categorical((KVCacheDType.AUTO, KVCacheDType.FP8_E4M3))),
    )

    assert list(validate(accepted)) == []


def test_a_space_that_can_violate_the_active_sequence_cap_is_reported() -> None:
    violating = space(
        Dimension("max_num_seqs", Interval(1, 16)),
        Dimension("max_num_active_seqs", Interval(1, 32)),
    )

    problems = validate(violating)

    assert any(
        "space.max_num_seqs" in problem and "max_num_active_seqs" in problem for problem in problems
    )


def test_a_space_that_can_violate_the_scheduled_token_cap_is_reported() -> None:
    violating = space(
        Dimension("max_num_batched_tokens", Fixed(4096)),
        Dimension("max_num_scheduled_tokens", Interval(2048, 8192)),
    )

    problems = validate(violating)

    assert any(
        "space.max_num_batched_tokens" in problem and "max_num_scheduled_tokens" in problem
        for problem in problems
    )


@pytest.mark.parametrize(
    ("upper", "lower"),
    [
        ("max_num_seqs", "max_num_active_seqs"),
        ("max_num_batched_tokens", "max_num_scheduled_tokens"),
    ],
)
def test_a_deferred_scheduling_companion_skips_the_bound_check(upper: str, lower: str) -> None:
    """An engine default is unknown, so the bound is left to the engine at run time."""
    deferred = space(Dimension(upper, Fixed(8)), Dimension(lower, Default()))
    absent = space(Dimension(upper, Fixed(8)))

    assert not any("requires" in problem for problem in validate(deferred))
    assert not any("requires" in problem for problem in validate(absent))


def test_the_permissive_corners_of_the_scheduling_caps_are_accepted() -> None:
    """The broad example ranges satisfy every companion bound at every corner."""
    accepted = space(
        Dimension("max_num_seqs", Interval(16, 256)),
        Dimension("max_num_active_seqs", Interval(1, 16)),
        Dimension("max_num_batched_tokens", Interval(4096, 8192)),
        Dimension("max_num_scheduled_tokens", Interval(2048, 4096)),
    )

    assert list(validate(accepted)) == []


def test_a_categorical_engine_bound_is_checked_by_its_extremes() -> None:
    """A categorical bound can violate the rule too, so its extremes are compared."""
    violating = space(
        Dimension("max_num_batched_tokens", Categorical((512, 4096))),
        Dimension("max_num_seqs", Categorical((1024, 2048))),
    )

    problems = validate(violating)

    assert any("space.max_num_batched_tokens" in problem for problem in problems)


def test_a_non_numeric_engine_bound_value_skips_the_bound_check() -> None:
    """A spec the bound cannot be compared against is skipped, not crashed on."""
    incomparable = space(
        Dimension("max_num_batched_tokens", Fixed(512)),
        Dimension("max_num_seqs", Categorical((1024, "wide"))),
    )

    assert list(validate(incomparable)) == []


def test_an_enum_engine_bound_value_skips_the_bound_check() -> None:
    """An enum member is incomparable with an integer bound, like a non-numeric value."""
    incomparable = space(
        Dimension("max_num_batched_tokens", Fixed(512)),
        Dimension("max_num_seqs", Categorical((DType.FLOAT16,))),
    )

    problems = validate(incomparable)

    assert not any("requires" in problem for problem in problems)
    assert any("space.max_num_seqs.categorical[0]" in problem for problem in problems)


def test_an_attention_backend_vllm_no_longer_ships_is_reported() -> None:
    problems = validate(space(Dimension("attention_backend", Fixed(AttentionBackend.XFORMERS))))

    assert any("space.attention_backend.fixed" in problem for problem in problems)


def test_a_non_enum_attention_backend_value_skips_the_target_check() -> None:
    """The target check classifies enum members; the kind check owns every other value."""
    problems = validate(
        space(Dimension("attention_backend", Fixed(7))),
        DetectedHardware(HardwareTarget.CUDA, 1),
    )

    assert list(problems) == []


def test_a_parameter_absent_from_the_value_maps_rejects_every_enum_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field with no curated spelling cannot translate any member."""
    monkeypatch.setattr("llm_config_optimization.inference._vllm.VALUE_MAPS", MappingProxyType({}))

    problems = validate(space(Dimension("dtype", Fixed(DType.FLOAT16))))

    assert any(problem.startswith("space.dtype.fixed") for problem in problems)


@pytest.mark.parametrize("target", [HardwareTarget.ROCM, HardwareTarget.CPU])
def test_a_cuda_only_attention_backend_is_reported_on_a_non_cuda_target(
    target: HardwareTarget,
) -> None:
    problems = validate(
        space(Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION))),
        DetectedHardware(target, 1),
    )

    assert any(
        "space.attention_backend.fixed" in problem and target.value in problem
        for problem in problems
    )


def test_a_cuda_attention_backend_is_accepted_on_cuda() -> None:
    problems = validate(
        space(Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION))),
        DetectedHardware(HardwareTarget.CUDA, 1),
    )

    assert list(problems) == []


def test_the_hardware_check_is_skipped_when_no_hardware_is_given() -> None:
    problems = validate(
        space(Dimension("attention_backend", Fixed(AttentionBackend.FLASH_ATTENTION)))
    )

    assert list(problems) == []


def test_a_tensor_parallel_size_larger_than_the_device_count_is_reported() -> None:
    problems = validate(
        space(Dimension("tensor_parallel_size", Categorical((1, 2, 4)))),
        DetectedHardware(HardwareTarget.CUDA, 1),
    )

    assert any("space.tensor_parallel_size.categorical[1]" in problem for problem in problems)
    assert any("space.tensor_parallel_size.categorical[2]" in problem for problem in problems)


def test_a_tensor_parallel_size_interval_above_the_device_count_is_reported() -> None:
    problems = validate(
        space(Dimension("tensor_parallel_size", Interval(1, 8))),
        DetectedHardware(HardwareTarget.CUDA, 4),
    )

    assert any(
        "space.tensor_parallel_size.interval" in problem and "8" in problem for problem in problems
    )


def test_a_tensor_parallel_size_that_fits_the_device_count_is_accepted() -> None:
    problems = validate(
        space(Dimension("tensor_parallel_size", Categorical((1, 2)))),
        DetectedHardware(HardwareTarget.CUDA, 2),
    )

    assert list(problems) == []


def test_a_default_tensor_parallel_size_skips_the_device_count_check() -> None:
    problems = validate(
        space(Dimension("tensor_parallel_size", Default())),
        DetectedHardware(HardwareTarget.CUDA, 1),
    )

    assert list(problems) == []


class RecordingBackend:
    """A backend that records the hardware and configuration it is asked to validate against."""

    def __init__(self) -> None:
        self.hardware: list[DetectedHardware | None] = []
        self.llm: list[LLMConfig] = []

    def validate(
        self,
        space: ExperimentSpace,
        llm: LLMConfig,
        hardware: DetectedHardware | None = None,
    ) -> Sequence[str]:
        self.llm.append(llm)
        self.hardware.append(hardware)
        return []


def test_validate_space_forwards_the_hardware_to_the_backend() -> None:
    backend = RecordingBackend()
    hardware = DetectedHardware(HardwareTarget.ROCM, 2)

    validate_space(SUPPORTED_SPACE, backend, LLMConfig(), hardware)

    assert backend.hardware == [hardware]


def test_validate_space_forwards_the_run_configuration_to_the_backend() -> None:
    backend = RecordingBackend()
    llm = LLMConfig(max_model_len=4096)

    validate_space(SUPPORTED_SPACE, backend, llm)

    assert backend.llm == [llm]


def test_a_default_parameter_skips_the_engine_bound() -> None:
    deferred = space(
        Dimension("max_num_batched_tokens", Default()),
        Dimension("max_num_seqs", Interval(1024, 2048)),
    )

    assert list(validate(deferred)) == []


def test_validate_space_returns_none_when_the_backend_supports_the_space() -> None:
    assert validate_space(SUPPORTED_SPACE, VLLMBackend(), LLMConfig()) is None


def test_validate_space_raises_every_backend_problem() -> None:
    backend = StubBackend(("space.dtype: first", "space.max_model_len: second"))

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        validate_space(SUPPORTED_SPACE, backend, LLMConfig())

    assert excinfo.value.problems == ("space.dtype: first", "space.max_model_len: second")
