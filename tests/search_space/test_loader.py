"""Unit tests for loading and engine-agnostically validating a search space.

A definition's observable contract is the
:class:`~llm_config_optimization.search_space.types.ExperimentSpace` it parses into and
the :class:`~llm_config_optimization.search_space.types.ExperimentDefinitionError` it
raises, whose ``.problems`` tuple holds every problem found. These tests drive
the public ``parse_experiment_space`` / ``load_experiment_space`` surface with
in-memory mappings and temporary files, so no backend, no engine, and no Podman
daemon is involved. They assert on the parameter path and the kind of problem
rather than on exact wording, so messages can be reworded freely.
"""

from pathlib import Path

import pytest
import yaml

from llm_config_optimization.inference import (
    All2AllBackend,
    AttentionBackend,
    CpuMoeMode,
    DType,
    GpuLayersMode,
    LoadMode,
    SpecDraftSampling,
    SpeculativeMode,
)
from llm_config_optimization.search_space import (
    TUNABLE_PARAMETERS,
    Categorical,
    Default,
    Dimension,
    ExperimentDefinitionError,
    Fixed,
    Interval,
    UnionKind,
    load_experiment_space,
    parse_experiment_space,
)

VALID_SPACE: dict[str, object] = {
    "max_model_len": {"interval": {"min": 1024, "max": 8192}},
    "max_num_seqs": {"fixed": 256},
    "max_num_batched_tokens": {"interval": {"min": 2048, "max": 8192}},
    "max_num_active_seqs": {"fixed": 16},
    "max_num_scheduled_tokens": {"fixed": 4096},
    "ubatch_size": {"interval": {"min": 256, "max": 2048}},
    "dtype": {"fixed": "float16"},
    "kv_cache_dtype": {"default": True},
    "attention_backend": {"default": True},
    "n_gpu_layers": {"default": True},
    "n_cpu_moe": {"categorical": [0, 8, "all"]},
    "n_cpu_ffn": {"interval": {"min": 0, "max": 16}},
    "kv_offload": {"fixed": True},
    "op_offload": {"fixed": False},
    "host_buffer": {"categorical": [True, False]},
    "repack": {"default": True},
    "swa_full": {"default": True},
    "load_mode": {"categorical": ["mmap", "mmap_mlock"]},
    "prefix_caching": {"fixed": True},
    "cache_ram": {"categorical": [-1, 0, 1024]},
    "ctx_checkpoints": {"interval": {"min": 0, "max": 8}},
    "checkpoint_min_step": {"fixed": 0},
    "kv_unified": {"fixed": True},
    "continuous_batching": {"categorical": [True, False]},
    "cache_idle_slots": {"default": True},
    "tensor_parallel_size": {"categorical": [1, 2, 4]},
    "expert_parallel": {"fixed": False},
    "all2all_backend": {"categorical": ["allgather_reducescatter", "naive"]},
    "speculative_mode": {"fixed": "enabled"},
    "num_speculative_tokens": {"interval": {"min": 1, "max": 5}},
    "spec_draft_n_min": {"interval": {"min": 0, "max": 5}},
    "spec_draft_p_split": {"interval": {"min": 0.0, "max": 1.0}},
    "spec_draft_sampling": {"categorical": ["greedy", "probabilistic"]},
    "n_gpu_layers_draft": {"categorical": [0, 8, "all"]},
    "n_cpu_moe_draft": {"categorical": [0, 8, "all"]},
    "draft_kv_cache_dtype": {"default": True},
    "threads": {"interval": {"min": 1, "max": 32}},
    "threads_batch": {"fixed": 8},
    "poll": {"interval": {"min": 0, "max": 100}},
    "prio": {"categorical": [-1, 0, 1, 2, 3]},
}
"""A complete, valid definition covering every tunable and every spec kind."""


def definition(**overrides: object) -> dict[str, object]:
    """Wrap a complete space, replacing or adding the given dimensions."""
    return {"space": {**VALID_SPACE, **overrides}}


def only_space(name: str, spec: object) -> dict[str, object]:
    """A complete definition where every tunable defers except ``name``."""
    space: dict[str, object] = {parameter: {"default": True} for parameter in TUNABLE_PARAMETERS}
    space[name] = spec
    return {"space": space}


def spec_of(space: object, name: str) -> object:
    """Return the single spec of the named dimension, failing if it is absent."""
    matches = [dimension.spec for dimension in space.dimensions if dimension.name == name]
    assert len(matches) == 1, f"expected one dimension named {name!r}, found {matches!r}"
    return matches[0]


BOOLEAN_PARAMETER = "prefix_caching"
"""The registry's boolean tunable, whose strict parsing the tests below pin."""


UNION_PARAMETER = "n_gpu_layers"
"""The registry's int-or-enum tunable, whose two branches the tests below pin."""


def parse_only(parameter: str, spec: object) -> object:
    """Parse a document declaring only one tunable, with it required."""
    return parse_experiment_space({"space": {parameter: spec}}, required={parameter})


def test_a_complete_definition_parses_into_the_expected_dimensions() -> None:
    space = parse_experiment_space(definition())

    assert space.dimensions == (
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(256)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("max_num_active_seqs", Fixed(16)),
        Dimension("max_num_scheduled_tokens", Fixed(4096)),
        Dimension("ubatch_size", Interval(256, 2048)),
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("kv_cache_dtype", Default()),
        Dimension("attention_backend", Default()),
        Dimension("n_gpu_layers", Default()),
        Dimension("n_cpu_moe", Categorical((0, 8, CpuMoeMode.ALL))),
        Dimension("n_cpu_ffn", Interval(0, 16)),
        Dimension("kv_offload", Fixed(True)),
        Dimension("op_offload", Fixed(False)),
        Dimension("host_buffer", Categorical((True, False))),
        Dimension("repack", Default()),
        Dimension("swa_full", Default()),
        Dimension("load_mode", Categorical((LoadMode.MMAP, LoadMode.MMAP_MLOCK))),
        Dimension("prefix_caching", Fixed(True)),
        Dimension("cache_ram", Categorical((-1, 0, 1024))),
        Dimension("ctx_checkpoints", Interval(0, 8)),
        Dimension("checkpoint_min_step", Fixed(0)),
        Dimension("kv_unified", Fixed(True)),
        Dimension("continuous_batching", Categorical((True, False))),
        Dimension("cache_idle_slots", Default()),
        Dimension("tensor_parallel_size", Categorical((1, 2, 4))),
        Dimension("expert_parallel", Fixed(False)),
        Dimension(
            "all2all_backend",
            Categorical((All2AllBackend.ALLGATHER_REDUCESCATTER, All2AllBackend.NAIVE)),
        ),
        Dimension("speculative_mode", Fixed(SpeculativeMode.ENABLED)),
        Dimension("num_speculative_tokens", Interval(1, 5)),
        Dimension("spec_draft_n_min", Interval(0, 5)),
        Dimension("spec_draft_p_split", Interval(0.0, 1.0)),
        Dimension(
            "spec_draft_sampling",
            Categorical((SpecDraftSampling.GREEDY, SpecDraftSampling.PROBABILISTIC)),
        ),
        Dimension("n_gpu_layers_draft", Categorical((0, 8, GpuLayersMode.ALL))),
        Dimension("n_cpu_moe_draft", Categorical((0, 8, CpuMoeMode.ALL))),
        Dimension("draft_kv_cache_dtype", Default()),
        Dimension("threads", Interval(1, 32)),
        Dimension("threads_batch", Fixed(8)),
        Dimension("poll", Interval(0, 100)),
        Dimension("prio", Categorical((-1, 0, 1, 2, 3))),
    )


def test_dimensions_follow_the_registry_order_not_the_file_order() -> None:
    reversed_space = dict(reversed(list(VALID_SPACE.items())))

    space = parse_experiment_space({"space": reversed_space})

    assert tuple(dimension.name for dimension in space.dimensions) == tuple(TUNABLE_PARAMETERS)


@pytest.mark.parametrize(
    ("name", "spec", "expected"),
    [
        ("dtype", {"fixed": "bfloat16"}, Fixed(DType.BFLOAT16)),
        (
            "attention_backend",
            {"categorical": ["xformers", "flash_attention"]},
            Categorical((AttentionBackend.XFORMERS, AttentionBackend.FLASH_ATTENTION)),
        ),
        ("max_model_len", {"interval": {"min": 1024, "max": 8192}}, Interval(1024, 8192)),
        ("max_num_seqs", {"categorical": [1, 20, 99]}, Categorical((1, 20, 99))),
        (
            "load_mode",
            {"categorical": ["mmap", "mmap_mlock"]},
            Categorical((LoadMode.MMAP, LoadMode.MMAP_MLOCK)),
        ),
        ("kv_cache_dtype", {"default": True}, Default()),
    ],
    ids=[
        "fixed",
        "categorical",
        "interval",
        "categorical-integers",
        "categorical-load-mode",
        "default",
    ],
)
def test_each_spec_kind_parses_to_its_engine_neutral_type(
    name: str, spec: object, expected: object
) -> None:
    space = parse_experiment_space(only_space(name, spec))

    assert spec_of(space, name) == expected


def test_the_all2all_backend_coerces_its_string_values_to_the_enum() -> None:
    space = parse_only("all2all_backend", {"categorical": ["allgather_reducescatter", "naive"]})

    assert spec_of(space, "all2all_backend") == Categorical(
        (All2AllBackend.ALLGATHER_REDUCESCATTER, All2AllBackend.NAIVE)
    )


def test_the_all2all_backend_rejects_an_engine_spelling() -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only("all2all_backend", {"fixed": "deepep"})

    assert any(
        "space.all2all_backend.fixed: 'deepep' is not a valid All2AllBackend" in problem
        for problem in excinfo.value.problems
    ), excinfo.value.problems


def test_the_n_cpu_moe_dimension_accepts_zero_and_the_all_mode() -> None:
    space = parse_only("n_cpu_moe", {"categorical": [0, 8, "all"]})

    assert spec_of(space, "n_cpu_moe") == Categorical((0, 8, CpuMoeMode.ALL))


def test_the_n_cpu_moe_dimension_rejects_a_negative_count() -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only("n_cpu_moe", {"fixed": -1})

    assert any(
        "space.n_cpu_moe.fixed: value must be at least 0, got -1" in problem
        for problem in excinfo.value.problems
    ), excinfo.value.problems


def test_the_n_cpu_ffn_dimension_accepts_zero_and_rejects_a_negative_count() -> None:
    """``0`` disables the dense FFN offload, so the floor is zero rather than one."""
    space = parse_only("n_cpu_ffn", {"categorical": [0, 8]})

    assert spec_of(space, "n_cpu_ffn") == Categorical((0, 8))
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only("n_cpu_ffn", {"fixed": -1})

    assert any(
        "space.n_cpu_ffn.fixed: value must be at least 0, got -1" in problem
        for problem in excinfo.value.problems
    ), excinfo.value.problems


def test_every_missing_tunable_is_named_with_its_accepted_spec_kinds() -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space({"space": {"dtype": {"fixed": "float16"}}})

    problems = excinfo.value.problems
    for parameter in TUNABLE_PARAMETERS:
        if parameter == "dtype":
            continue
        expected = f"space.{parameter}: missing required tunable parameter"
        assert any(problem.startswith(expected) for problem in problems), problems
    assert all("fixed, categorical, interval, default" in problem for problem in problems)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"fixed": True}, Fixed(True)),
        ({"fixed": False}, Fixed(False)),
        ({"categorical": [True, False]}, Categorical((True, False))),
        ({"default": True}, Default()),
    ],
    ids=["fixed-true", "fixed-false", "categorical", "default"],
)
def test_a_boolean_dimension_parses_to_its_engine_neutral_type(
    spec: dict[str, object], expected: object
) -> None:
    space = parse_only(BOOLEAN_PARAMETER, spec)

    assert spec_of(space, BOOLEAN_PARAMETER) == expected


def test_a_boolean_value_is_kept_distinct_from_an_integer() -> None:
    """``True == 1``, so assert on identity rather than equality alone."""
    fixed = spec_of(parse_only(BOOLEAN_PARAMETER, {"fixed": True}), BOOLEAN_PARAMETER)

    assert isinstance(fixed, Fixed)
    assert fixed.value is True


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"fixed": 1}, "space.prefix_caching.fixed: expected a boolean"),
        ({"fixed": 0}, "space.prefix_caching.fixed: expected a boolean"),
        ({"fixed": "true"}, "space.prefix_caching.fixed: expected a boolean"),
        (
            {"categorical": [True, 1]},
            "space.prefix_caching.categorical[1]: expected a boolean",
        ),
        (
            {"interval": {"min": 0, "max": 1}},
            "space.prefix_caching.interval: not valid for a boolean parameter",
        ),
    ],
    ids=["integer-one", "integer-zero", "quoted-string", "categorical-integer", "interval"],
)
def test_boolean_values_are_validated_strictly(spec: dict[str, object], problem: str) -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only(BOOLEAN_PARAMETER, spec)

    assert any(problem in found for found in excinfo.value.problems), excinfo.value.problems


def test_a_boolean_dimension_still_rejects_duplicate_categorical_values() -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only(BOOLEAN_PARAMETER, {"categorical": [True, True]})

    assert any("values must be unique" in found for found in excinfo.value.problems)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"fixed": "auto"}, Fixed(GpuLayersMode.AUTO)),
        ({"fixed": 8}, Fixed(8)),
        (
            {"categorical": ["auto", 0, "all", 16]},
            Categorical((GpuLayersMode.AUTO, 0, GpuLayersMode.ALL, 16)),
        ),
        ({"interval": {"min": 0, "max": 40}}, Interval(0, 40)),
        ({"default": True}, Default()),
    ],
    ids=["fixed-enum", "fixed-integer", "mixed-categorical", "interval", "default"],
)
def test_an_int_or_enum_dimension_accepts_either_branch(
    spec: dict[str, object], expected: object
) -> None:
    space = parse_only(UNION_PARAMETER, spec)

    assert spec_of(space, UNION_PARAMETER) == expected


def test_a_union_integer_is_kept_distinct_from_the_enum_branch() -> None:
    """``0`` is an exact count, not an enum member; assert on its exact type."""
    fixed = spec_of(parse_only(UNION_PARAMETER, {"fixed": 0}), UNION_PARAMETER)

    assert isinstance(fixed, Fixed)
    assert fixed.value == 0
    assert type(fixed.value) is int


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"fixed": True}, "space.n_gpu_layers.fixed: expected an integer or one of"),
        ({"fixed": "bogus"}, "space.n_gpu_layers.fixed: expected an integer or one of"),
        (
            {"categorical": ["auto", "bogus"]},
            "space.n_gpu_layers.categorical[1]: expected an integer or one of",
        ),
        (
            {"interval": {"min": "auto", "max": 4}},
            "space.n_gpu_layers.interval.min: expected an integer",
        ),
    ],
    ids=["boolean", "unknown-enum", "categorical-unknown-enum", "interval-enum-bound"],
)
def test_a_union_dimension_rejects_values_outside_both_branches(
    spec: dict[str, object], problem: str
) -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only(UNION_PARAMETER, spec)

    assert any(problem in found for found in excinfo.value.problems), excinfo.value.problems


@pytest.mark.parametrize(
    "spec",
    [
        {"fixed": 0},
        {"categorical": ["auto", 0, "all"]},
        {"interval": {"min": 0, "max": 40}},
    ],
    ids=["fixed-zero", "categorical-zero", "interval-zero"],
)
def test_a_non_negative_dimension_accepts_zero(
    spec: dict[str, object],
) -> None:
    space = parse_only(UNION_PARAMETER, spec)

    assert spec_of(space, UNION_PARAMETER) is not None


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"fixed": -1}, "space.n_gpu_layers.fixed: value must be at least 0, got -1"),
        (
            {"categorical": ["auto", -1]},
            "space.n_gpu_layers.categorical[1]: value must be at least 0, got -1",
        ),
        (
            {"interval": {"min": -1, "max": 40}},
            "space.n_gpu_layers.interval.min: value must be at least 0, got -1",
        ),
    ],
    ids=["fixed-negative", "categorical-negative", "interval-negative"],
)
def test_a_non_negative_dimension_rejects_a_negative_count(
    spec: dict[str, object], problem: str
) -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only(UNION_PARAMETER, spec)

    assert any(problem in found for found in excinfo.value.problems), excinfo.value.problems


def test_a_union_without_an_integer_branch_rejects_an_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_config_optimization.search_space.loader.expected_kind",
        lambda _: UnionKind((AttentionBackend,)),
    )

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only(UNION_PARAMETER, {"interval": {"min": 0, "max": 4}})

    assert any(
        "space.n_gpu_layers.interval: not valid for a categorical parameter" in found
        for found in excinfo.value.problems
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"fixed": 0.5}, Fixed(0.5)),
        ({"fixed": 1}, Fixed(1.0)),
        ({"categorical": [0.0, 0.5, 1.0]}, Categorical((0.0, 0.5, 1.0))),
        ({"categorical": [0, 0.5, 1]}, Categorical((0.0, 0.5, 1.0))),
        ({"interval": {"min": 0.0, "max": 1.0}}, Interval(0.0, 1.0)),
        ({"interval": {"min": 0, "max": 1}}, Interval(0.0, 1.0)),
        ({"default": True}, Default()),
    ],
    ids=[
        "fixed-float",
        "fixed-integer",
        "categorical-floats",
        "categorical-mixed",
        "interval-floats",
        "interval-integers",
        "default",
    ],
)
def test_a_float_dimension_parses_to_floats(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, object], expected: object
) -> None:
    monkeypatch.setattr(
        "llm_config_optimization.search_space.loader.expected_kind",
        lambda _: float,
    )

    space = parse_only("n_cpu_moe", spec)

    assert spec_of(space, "n_cpu_moe") == expected


@pytest.mark.parametrize(
    "spec",
    [{"fixed": True}, {"categorical": [False, 0.5]}, {"interval": {"min": True, "max": 1}}],
    ids=["fixed-boolean", "categorical-boolean", "interval-boolean"],
)
def test_a_float_dimension_keeps_booleans_out(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, object]
) -> None:
    monkeypatch.setattr(
        "llm_config_optimization.search_space.loader.expected_kind",
        lambda _: float,
    )

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_only("n_cpu_moe", spec)

    assert any("expected a number" in found for found in excinfo.value.problems)


def test_a_reduced_required_set_makes_a_declared_space_total_over_it() -> None:
    required = {"max_model_len", "max_num_seqs"}
    document = {"space": {"max_model_len": {"default": True}, "max_num_seqs": {"default": True}}}

    space = parse_experiment_space(document, required=required)

    assert tuple(dimension.name for dimension in space.dimensions) == (
        "max_model_len",
        "max_num_seqs",
    )


def test_a_reduced_required_set_reports_only_its_own_missing_members() -> None:
    required = {"max_model_len", "max_num_seqs", "attention_backend"}

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space({"space": {"max_model_len": {"default": True}}}, required=required)

    problems = excinfo.value.problems
    assert any("space.max_num_seqs: missing required tunable parameter" in p for p in problems)
    assert any("space.attention_backend: missing required tunable parameter" in p for p in problems)
    assert not any("space.dtype" in problem for problem in problems)
    assert len(problems) == 2


def test_a_declared_parameter_outside_the_required_set_is_not_a_loader_problem() -> None:
    """Whether the engine forwards it is the backend's separate ``validate`` check."""
    required = {"max_model_len", "max_num_seqs"}
    document = {
        "space": {
            "max_model_len": {"default": True},
            "max_num_seqs": {"default": True},
            "dtype": {"fixed": "float16"},
        }
    }

    space = parse_experiment_space(document, required=required)

    assert tuple(dimension.name for dimension in space.dimensions) == (
        "max_model_len",
        "max_num_seqs",
        "dtype",
    )


def test_the_file_loader_forwards_the_required_set(tmp_path: Path) -> None:
    path = tmp_path / "space.yaml"
    path.write_text(
        yaml.safe_dump(
            {"space": {"max_model_len": {"default": True}, "max_num_seqs": {"default": True}}}
        ),
        encoding="utf-8",
    )

    space = load_experiment_space(path, required={"max_model_len", "max_num_seqs"})

    assert tuple(dimension.name for dimension in space.dimensions) == (
        "max_model_len",
        "max_num_seqs",
    )


def test_an_invalid_declared_tunable_is_not_also_reported_missing() -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space(definition(dtype={"fixed": "int8"}))

    assert len(excinfo.value.problems) == 1
    assert "space.dtype.fixed" in excinfo.value.problems[0]


def test_independent_problems_are_collected_into_one_error() -> None:
    space = {
        **VALID_SPACE,
        "dtype": {"fixed": "int8"},
        "max_num_seqs": {"fixed": 0},
        "seed": {"fixed": 1},
    }
    del space["tensor_parallel_size"]

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space({"space": space})

    problems = excinfo.value.problems
    assert isinstance(problems, tuple)
    joined = "\n".join(problems)
    assert "space.dtype.fixed" in joined
    assert "space.max_num_seqs.fixed" in joined
    assert "space.seed" in joined
    assert "space.tensor_parallel_size: missing required tunable parameter" in joined
    assert len(problems) == 4


@pytest.mark.parametrize(
    ("document", "problem"),
    [
        ({"space": VALID_SPACE, "extra": 1}, "unknown top-level key"),
        ({"spaces": VALID_SPACE}, "missing required top-level key"),
        ({"space": []}, "'space' must be a mapping"),
        ([], "definition must be a mapping"),
        ("space", "definition must be a mapping"),
    ],
    ids=["unknown-key", "missing-space", "space-not-mapping", "list", "scalar"],
)
def test_top_level_shape_problems_are_reported(document: object, problem: str) -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space(document)

    assert any(problem in found for found in excinfo.value.problems)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"unknown": {"fixed": 1}}, "space.unknown: unknown parameter"),
        ({"seed": {"fixed": 1}}, "space.seed: 'seed' is a known parameter but is not tunable"),
        (
            {"quantization": {"fixed": "awq"}},
            "space.quantization: 'quantization' is a known parameter but is not tunable",
        ),
        ({"dtype": 5}, "space.dtype: spec must be a mapping"),
        ({"dtype": {"float": 1}}, "space.dtype: unknown spec key(s): 'float'"),
        ({"dtype": {}}, "space.dtype: spec must contain exactly one of"),
        (
            {"dtype": {"fixed": "float16", "default": True}},
            "space.dtype: spec must contain exactly one of",
        ),
        ({"dtype": {"fixed": "int8"}}, "space.dtype.fixed: 'int8' is not a valid DType"),
        ({"dtype": {"fixed": 5}}, "space.dtype.fixed: expected one of"),
        (
            {"dtype": {"interval": {"min": 1, "max": 2}}},
            "space.dtype.interval: not valid for a categorical parameter",
        ),
        ({"max_model_len": {"fixed": "4096"}}, "space.max_model_len.fixed: expected an integer"),
        ({"max_model_len": {"fixed": True}}, "space.max_model_len.fixed: expected an integer"),
        ({"max_model_len": {"fixed": 1.5}}, "space.max_model_len.fixed: expected an integer"),
        (
            {"dtype": {"categorical": "float16"}},
            "space.dtype.categorical: expected a non-empty list",
        ),
        ({"dtype": {"categorical": []}}, "space.dtype.categorical: must not be empty"),
        (
            {"attention_backend": {"categorical": ["xformers", "xformers"]}},
            "space.attention_backend.categorical: values must be unique",
        ),
        (
            {"max_model_len": {"interval": {"min": 8192, "max": 1024}}},
            "space.max_model_len.interval: 'min'",
        ),
        (
            {"max_model_len": {"interval": {"min": 1024}}},
            "space.max_model_len.interval: missing required key(s)",
        ),
        (
            {"max_model_len": {"interval": {"min": 1, "max": 2, "step": 1}}},
            "space.max_model_len.interval: unknown key(s)",
        ),
        (
            {"max_model_len": {"interval": {"min": 1.5, "max": 2}}},
            "space.max_model_len.interval.min: expected an integer",
        ),
        ({"max_num_seqs": {"fixed": 0}}, "space.max_num_seqs.fixed: value must be at least 1"),
        (
            {"max_num_active_seqs": {"fixed": 0}},
            "space.max_num_active_seqs.fixed: value must be at least 1",
        ),
        (
            {"max_num_scheduled_tokens": {"interval": {"min": 0, "max": 10}}},
            "space.max_num_scheduled_tokens.interval.min: value must be at least 1",
        ),
        (
            {"ubatch_size": {"fixed": 0}},
            "space.ubatch_size.fixed: value must be at least 1",
        ),
        (
            {"ctx_checkpoints": {"fixed": -1}},
            "space.ctx_checkpoints.fixed: value must be at least 0",
        ),
        (
            {"checkpoint_min_step": {"interval": {"min": -1, "max": 10}}},
            "space.checkpoint_min_step.interval.min: value must be at least 0",
        ),
        (
            {"num_speculative_tokens": {"fixed": 0}},
            "space.num_speculative_tokens.fixed: value must be at least 1",
        ),
        (
            {"spec_draft_n_min": {"fixed": -1}},
            "space.spec_draft_n_min.fixed: value must be at least 0",
        ),
        (
            {"n_gpu_layers_draft": {"interval": {"min": -1, "max": 8}}},
            "space.n_gpu_layers_draft.interval.min: value must be at least 0",
        ),
        (
            {"n_cpu_moe_draft": {"fixed": -1}},
            "space.n_cpu_moe_draft.fixed: value must be at least 0",
        ),
        (
            {"threads": {"fixed": 0}},
            "space.threads.fixed: value must be at least 1",
        ),
        (
            {"threads_batch": {"interval": {"min": 0, "max": 32}}},
            "space.threads_batch.interval.min: value must be at least 1",
        ),
        (
            {"poll": {"fixed": -1}},
            "space.poll.fixed: value must be at least 0",
        ),
        (
            {"max_model_len": {"interval": {"min": 0, "max": 10}}},
            "space.max_model_len.interval.min: value must be at least 1",
        ),
        ({"dtype": {"default": False}}, "space.dtype.default: must be exactly true"),
        ({"dtype": {"default": "yes"}}, "space.dtype.default: must be exactly true"),
    ],
)
def test_invalid_dimensions_report_the_parameter_path(
    overrides: dict[str, object], problem: str
) -> None:
    with pytest.raises(ExperimentDefinitionError) as excinfo:
        parse_experiment_space(definition(**overrides))

    assert any(problem in found for found in excinfo.value.problems), excinfo.value.problems


def test_the_loader_parses_a_valid_definition_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(definition()), encoding="utf-8")

    space = load_experiment_space(path)

    assert tuple(dimension.name for dimension in space.dimensions) == tuple(TUNABLE_PARAMETERS)


def test_malformed_yaml_is_reported_through_the_definition_error(tmp_path: Path) -> None:
    path = tmp_path / "space.yaml"
    path.write_text("space: [unclosed\n", encoding="utf-8")

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        load_experiment_space(path)

    assert any("malformed YAML" in problem for problem in excinfo.value.problems)


def test_a_non_mapping_document_is_reported_through_the_file_loader(tmp_path: Path) -> None:
    path = tmp_path / "space.yaml"
    path.write_text("- dtype\n- quantization\n", encoding="utf-8")

    with pytest.raises(ExperimentDefinitionError) as excinfo:
        load_experiment_space(path)

    assert any("definition must be a mapping" in problem for problem in excinfo.value.problems)
