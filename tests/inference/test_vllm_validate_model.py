"""Unit tests for the vLLM checkpoint-quantization compatibility rule.

:meth:`VLLMBackend.validate_model`'s observable contract is the problems it
returns for a resolved checkpoint and a configured experiment: whether the
requested quantization can be served by the method the checkpoint declares.
These tests write hand-written ``config.json`` payloads into temporary
directories and drive the backend directly, so the whole compatibility matrix —
including the fail-open paths — is covered without an engine, a GPU, or a
daemon. They assert whether a problem is reported and that its wording names the
checkpoint's method, not the exact sentence, so messages can be reworded freely.
"""

import json
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    LLMConfig,
    QuantizationMethod,
    VLLMBackend,
)

MODEL = "Qwen/Qwen3.5-0.8B"
"""A Hub id, which only stands in as the experiment's model field here."""

_MISSING = object()
"""Marker for a checkpoint directory with no ``config.json``."""

_MALFORMED = object()
"""Marker for a checkpoint whose ``config.json`` is not valid JSON."""


def _experiment(quantization: QuantizationMethod | None) -> ExperimentConfig:
    """Build an experiment whose only model-relevant field is the quantization."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=MODEL),
        llm=LLMConfig(quantization=quantization),
    )


def _write_checkpoint(directory: Path, config: object) -> Path:
    """Write ``config.json`` with the given JSON-serialisable payload."""
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _checkpoint(directory: Path, payload: object) -> Path:
    """Materialise a checkpoint directory from a matrix payload."""
    if payload is _MISSING:
        return directory
    if payload is _MALFORMED:
        (directory / "config.json").write_text("{not valid json", encoding="utf-8")
        return directory
    return _write_checkpoint(directory, payload)


_CHECKPOINTS: dict[str, object] = {
    "missing-config": _MISSING,
    "malformed-config": _MALFORMED,
    "non-object-config": ["not", "a", "json", "object"],
    "no-quantization-config": {},
    "non-string-method": {"quantization_config": {"quant_method": 7}},
    "unquantized": {"quantization_config": {}},
    "awq": {"quantization_config": {"quant_method": "awq"}},
    "gptq": {"quantization_config": {"quant_method": "gptq"}},
    "fp8": {"quantization_config": {"quant_method": "fp8"}},
    "unmodeled": {"quantization_config": {"quant_method": "compressed-tensors"}},
}
"""Every checkpoint shape the rule distinguishes, keyed by a readable name."""

_ALL_QUANTIZATIONS = (None, QuantizationMethod.AWQ, QuantizationMethod.GPTQ, QuantizationMethod.FP8)

_ACCEPTED: dict[str, frozenset[QuantizationMethod | None]] = {
    "missing-config": frozenset(_ALL_QUANTIZATIONS),
    "malformed-config": frozenset(_ALL_QUANTIZATIONS),
    "non-object-config": frozenset(_ALL_QUANTIZATIONS),
    "no-quantization-config": frozenset({None, QuantizationMethod.FP8}),
    "non-string-method": frozenset({None, QuantizationMethod.FP8}),
    "unquantized": frozenset({None, QuantizationMethod.FP8}),
    "awq": frozenset({None, QuantizationMethod.AWQ}),
    "gptq": frozenset({None, QuantizationMethod.GPTQ}),
    "fp8": frozenset({None, QuantizationMethod.FP8}),
    "unmodeled": frozenset({None}),
}
"""The configured quantizations each checkpoint shape accepts."""


@pytest.mark.parametrize("checkpoint", sorted(_CHECKPOINTS))
@pytest.mark.parametrize("quantization", _ALL_QUANTIZATIONS)
def test_every_checkpoint_method_is_crossed_with_every_configured_value(
    checkpoint: str,
    quantization: QuantizationMethod | None,
    tmp_path: Path,
) -> None:
    model = _checkpoint(tmp_path, _CHECKPOINTS[checkpoint])

    problems = list(VLLMBackend().validate_model(_experiment(quantization), model))

    assert (problems == []) is (quantization in _ACCEPTED[checkpoint])


@pytest.mark.parametrize(
    ("declared", "quantization"),
    [
        ("awq", QuantizationMethod.GPTQ),
        ("gptq", QuantizationMethod.AWQ),
        ("fp8", QuantizationMethod.AWQ),
        ("compressed-tensors", QuantizationMethod.AWQ),
    ],
)
def test_a_mismatch_names_the_checkpoints_method(
    declared: str,
    quantization: QuantizationMethod,
    tmp_path: Path,
) -> None:
    model = _write_checkpoint(tmp_path, {"quantization_config": {"quant_method": declared}})

    problems = list(VLLMBackend().validate_model(_experiment(quantization), model))

    assert len(problems) == 1
    assert declared in problems[0]


def test_an_explicit_value_on_an_unquantized_checkpoint_is_rejected(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {})

    problems = list(VLLMBackend().validate_model(_experiment(QuantizationMethod.AWQ), model))

    assert len(problems) == 1
    assert "no quantization method" in problems[0]


@pytest.mark.parametrize("declared", ["awq", "auto_awq", "awq_marlin"])
def test_awq_checkpoint_spellings_accept_awq(declared: str, tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {"quantization_config": {"quant_method": declared}})

    assert list(VLLMBackend().validate_model(_experiment(QuantizationMethod.AWQ), model)) == []


@pytest.mark.parametrize("declared", ["gptq", "auto_gptq", "gptq_marlin"])
def test_gptq_checkpoint_spellings_accept_gptq(declared: str, tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {"quantization_config": {"quant_method": declared}})

    assert list(VLLMBackend().validate_model(_experiment(QuantizationMethod.GPTQ), model)) == []


def test_a_checkpoints_method_is_matched_case_insensitively(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path, {"quantization_config": {"quant_method": "AWQ"}})

    assert list(VLLMBackend().validate_model(_experiment(QuantizationMethod.AWQ), model)) == []


def test_an_omitted_quantization_accepts_even_a_mismatched_checkpoint(tmp_path: Path) -> None:
    """The engine auto-detects, so the check inspects nothing and reports nothing."""
    model = _write_checkpoint(tmp_path, {"quantization_config": {"quant_method": "awq"}})

    assert list(VLLMBackend().validate_model(_experiment(None), model)) == []
