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
    AttentionBackend,
    DType,
)
from llm_config_optimization.search_space import (
    TUNABLE_PARAMETERS,
    Categorical,
    Default,
    Dimension,
    ExperimentDefinitionError,
    Fixed,
    Interval,
    load_experiment_space,
    parse_experiment_space,
)

VALID_SPACE: dict[str, object] = {
    "dtype": {"fixed": "float16"},
    "kv_cache_dtype": {"default": True},
    "max_model_len": {"interval": {"min": 1024, "max": 8192}},
    "max_num_seqs": {"fixed": 256},
    "max_num_batched_tokens": {"interval": {"min": 2048, "max": 8192}},
    "tensor_parallel_size": {"categorical": [1, 2, 4]},
    "attention_backend": {"default": True},
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


def test_a_complete_definition_parses_into_the_expected_dimensions() -> None:
    space = parse_experiment_space(definition())

    assert space.dimensions == (
        Dimension("dtype", Fixed(DType.FLOAT16)),
        Dimension("kv_cache_dtype", Default()),
        Dimension("max_model_len", Interval(1024, 8192)),
        Dimension("max_num_seqs", Fixed(256)),
        Dimension("max_num_batched_tokens", Interval(2048, 8192)),
        Dimension("tensor_parallel_size", Categorical((1, 2, 4))),
        Dimension("attention_backend", Default()),
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
        ("kv_cache_dtype", {"default": True}, Default()),
    ],
    ids=["fixed", "categorical", "interval", "categorical-integers", "default"],
)
def test_each_spec_kind_parses_to_its_engine_neutral_type(
    name: str, spec: object, expected: object
) -> None:
    space = parse_experiment_space(only_space(name, spec))

    assert spec_of(space, name) == expected


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
