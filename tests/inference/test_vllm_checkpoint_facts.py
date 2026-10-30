"""Unit tests for reading a vLLM checkpoint's declared model facts.

``read_checkpoint_facts``'s observable contract is the
:class:`~llm_config_optimization.inference._vllm.CheckpointFacts` it returns for a
resolved checkpoint directory: whether its configuration confidently declares
Mixture-Of-Experts experts, which execution dtype it prefers, and whether its
layers use linear attention. The fixtures are hand-written ``config.json``
payloads written into ``tmp_path``, so detection is exercised without an engine,
a GPU, or a download. Unreadable metadata reports ``None`` rather than guessed
facts, so the space assessment can fail open.
"""

import json
from pathlib import Path

import pytest

from llm_config_optimization.inference import DType
from llm_config_optimization.inference._vllm import CheckpointFacts, read_checkpoint_facts

_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
"""Every ``config.json`` spelling that declares a checkpoint's expert count."""

_DTYPE_KEYS = ("dtype", "torch_dtype")
"""Every ``config.json`` spelling that declares a checkpoint's execution dtype."""


def _write_checkpoint(directory: Path, payload: object) -> Path:
    """Write ``config.json`` with the given JSON-serialisable payload."""
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _facts(
    *,
    moe: bool = False,
    model_dtype: DType | None = None,
    linear_attention: bool = False,
) -> CheckpointFacts:
    """Describe the facts a payload with the given properties must report."""
    return CheckpointFacts(moe=moe, model_dtype=model_dtype, linear_attention=linear_attention)


@pytest.mark.parametrize("key", _EXPERT_KEYS)
def test_a_top_level_expert_count_reports_an_moe_checkpoint(key: str, tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {key: 128})

    assert read_checkpoint_facts(model) == _facts(moe=True)


@pytest.mark.parametrize("key", _EXPERT_KEYS)
def test_a_nested_expert_count_reports_an_moe_checkpoint(key: str, tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {"text_config": {key: 8}})

    assert read_checkpoint_facts(model) == _facts(moe=True)


def test_a_nested_expert_count_wins_over_a_dense_top_level(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {"num_experts": 0, "text_config": {"num_experts": 64}})

    assert read_checkpoint_facts(model) == _facts(moe=True)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"num_experts": 0},
        {"n_routed_experts": 0, "num_local_experts": 0},
        {"text_config": {}},
        {"text_config": {"num_experts": 0}},
    ],
)
def test_a_readable_configuration_without_positive_experts_is_dense(
    payload: object, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, payload)

    assert read_checkpoint_facts(model) == _facts()


@pytest.mark.parametrize(
    "payload",
    [
        {"num_experts": "128"},
        {"num_experts": True},
        {"num_experts": 0.5},
        {"text_config": ["not", "an", "object"]},
    ],
)
def test_a_non_integer_expert_count_is_not_evidence_of_experts(
    payload: object, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, payload)

    assert read_checkpoint_facts(model) == _facts()


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("float16", DType.FLOAT16),
        ("bfloat16", DType.BFLOAT16),
        ("float32", DType.FLOAT32),
    ],
)
@pytest.mark.parametrize("key", _DTYPE_KEYS)
def test_a_top_level_dtype_key_reports_the_preferred_model_dtype(
    key: str, declared: str, expected: DType, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, {key: declared})

    assert read_checkpoint_facts(model) == _facts(model_dtype=expected)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("float16", DType.FLOAT16),
        ("bfloat16", DType.BFLOAT16),
        ("float32", DType.FLOAT32),
    ],
)
def test_a_nested_text_config_dtype_reports_the_model_dtype(
    declared: str, expected: DType, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, {"text_config": {"dtype": declared}})

    assert read_checkpoint_facts(model) == _facts(model_dtype=expected)


def test_a_nested_dtype_wins_over_the_top_level(tmp_path: Path) -> None:
    model = _write_checkpoint(
        tmp_path, {"torch_dtype": "float32", "text_config": {"dtype": "bfloat16"}}
    )

    assert read_checkpoint_facts(model) == _facts(model_dtype=DType.BFLOAT16)


def test_a_top_level_dtype_is_reported_when_text_config_is_not_an_object(
    tmp_path: Path,
) -> None:
    model = _write_checkpoint(tmp_path, {"torch_dtype": "float16", "text_config": "not an object"})

    assert read_checkpoint_facts(model) == _facts(model_dtype=DType.FLOAT16)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"dtype": "auto"},
        {"torch_dtype": "float8_e4m3fn"},
        {"dtype": "int8"},
        {"dtype": None},
        {"dtype": 16},
        {"dtype": True},
        {"dtype": ["bfloat16"]},
        {"text_config": {"dtype": "auto"}},
        {"text_config": {"dtype": None}},
    ],
)
def test_a_missing_or_unmodelled_dtype_reports_an_unknown_model_dtype(
    payload: object, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, payload)

    assert read_checkpoint_facts(model) == _facts()


def test_an_unmodelled_nested_dtype_does_not_fall_through_to_the_top_level(
    tmp_path: Path,
) -> None:
    model = _write_checkpoint(
        tmp_path, {"torch_dtype": "bfloat16", "text_config": {"dtype": "auto"}}
    )

    assert read_checkpoint_facts(model) == _facts()


@pytest.mark.parametrize(
    "layer_types", [["linear_attention", "full_attention"], ["linear_attention"]]
)
def test_a_nested_linear_attention_layer_type_is_reported(
    layer_types: list[str], tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, {"text_config": {"layer_types": layer_types}})

    assert read_checkpoint_facts(model) == _facts(linear_attention=True)


@pytest.mark.parametrize(
    "layer_types", [["linear_attention", "full_attention"], ["linear_attention"]]
)
def test_a_top_level_linear_attention_layer_type_is_reported(
    layer_types: list[str], tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, {"layer_types": layer_types})

    assert read_checkpoint_facts(model) == _facts(linear_attention=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"text_config": {"layer_types": ["full_attention", "full_attention"]}},
        {"layer_types": ["full_attention"]},
        {"layer_types": []},
        {"text_config": {}},
        {"text_config": []},
        {"layer_types": "linear_attention"},
        {"layer_types": ["Linear_Attention"]},
        {"layer_types": [None, 3]},
    ],
)
def test_a_configuration_without_a_linear_attention_layer_is_not_linear_attention(
    payload: object, tmp_path: Path
) -> None:
    model = _write_checkpoint(tmp_path, payload)

    assert read_checkpoint_facts(model) == _facts()


def test_a_multimodal_hybrid_checkpoint_reports_all_facts(tmp_path: Path) -> None:
    model = _write_checkpoint(
        tmp_path,
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "dtype": "bfloat16",
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                "num_experts": 512,
            },
        },
    )

    assert read_checkpoint_facts(model) == _facts(
        moe=True, model_dtype=DType.BFLOAT16, linear_attention=True
    )


def test_a_missing_configuration_is_unknown(tmp_path: Path) -> None:
    assert read_checkpoint_facts(tmp_path) is None


def test_a_malformed_configuration_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not valid json", encoding="utf-8")

    assert read_checkpoint_facts(tmp_path) is None


@pytest.mark.parametrize("payload", [["not", "an", "object"], "a string", 7, None])
def test_a_non_object_configuration_is_unknown(payload: object, tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, payload)

    assert read_checkpoint_facts(model) is None


def test_a_configuration_that_cannot_be_read_as_a_file_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "config.json").mkdir()

    assert read_checkpoint_facts(tmp_path) is None
