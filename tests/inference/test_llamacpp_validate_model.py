"""Unit tests for llama.cpp checkpoint validation.

:meth:`LlamaCppBackend.validate_model`'s observable contract is the problems it
returns for a resolved checkpoint and a configured experiment: whether the file
is a GGUF checkpoint llama.cpp can serve and whether a vLLM-only quantization
value was requested for it. The fixtures are hand-written GGUF containers
written into ``tmp_path``, so the check is exercised without an engine, a GPU,
or a download. The tests assert that a problem is reported and that its wording
names the relevant fact — the file is not GGUF, or the quantization the
checkpoint declares — rather than the exact sentence, so messages can be
reworded freely.
"""

import struct
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    GpuLayersMode,
    LlamaCppBackend,
    LLMConfig,
    QuantizationMethod,
)

MODEL = "bartowski/Llama-3.2-1B-Instruct-GGUF"
"""A Hub id, which only stands in as the experiment's model field here."""

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_STRING = 8
_UINT32 = 4
_Q4_K_M = 15
"""llama.cpp's ``general.file_type`` value for a ``Q4_K_M`` checkpoint."""


def _encode_string(text: str) -> bytes:
    """Encode a length-prefixed UTF-8 GGUF string."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(
    path: Path,
    file_type: int | None,
    *,
    architecture: str | None = None,
    block_count: int | None = None,
) -> Path:
    """Write a minimal GGUF container with the requested metadata facts."""
    entries = b""
    count = 0
    if file_type is not None:
        entries += _encode_string("general.file_type") + struct.pack("<II", _UINT32, file_type)
        count += 1
    if architecture is not None:
        entries += (
            _encode_string("general.architecture")
            + struct.pack("<I", _STRING)
            + _encode_string(architecture)
        )
        count += 1
    if block_count is not None:
        entries += _encode_string(f"{architecture or 'llama'}.block_count") + struct.pack(
            "<II", _UINT32, block_count
        )
        count += 1
    path.write_bytes(_GGUF_MAGIC + struct.pack("<IQQ", _GGUF_VERSION, 0, count) + entries)
    return path


def _experiment(
    quantization: QuantizationMethod | None,
    *,
    draft: Path | None = None,
    n_gpu_layers: int | GpuLayersMode | None = None,
) -> ExperimentConfig:
    """Build an experiment whose model-relevant fields are quant and draft."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model=MODEL),
        llm=LLMConfig(
            quantization=quantization,
            resolved_draft_path=draft,
            n_gpu_layers=n_gpu_layers,
        ),
    )


def test_a_gguf_checkpoint_without_a_configured_quantization_is_accepted(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M)

    assert list(LlamaCppBackend().validate_model(_experiment(None), model)) == []


def test_a_gguf_checkpoint_without_a_declared_file_type_is_accepted(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", None)

    assert list(LlamaCppBackend().validate_model(_experiment(None), model)) == []


@pytest.mark.parametrize(
    "quantization",
    [QuantizationMethod.AWQ, QuantizationMethod.GPTQ, QuantizationMethod.FP8],
)
def test_a_configured_quantization_is_rejected_and_names_the_checkpoint_quant(
    quantization: QuantizationMethod, tmp_path: Path
) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M)

    problems = list(LlamaCppBackend().validate_model(_experiment(quantization), model))

    assert len(problems) == 1
    assert "Q4_K_M" in problems[0]
    assert "quantization" in problems[0]


def test_a_gguf_without_a_declared_file_type_still_rejects_a_configured_value(
    tmp_path: Path,
) -> None:
    model = _write_gguf(tmp_path / "model.gguf", None)

    problems = list(LlamaCppBackend().validate_model(_experiment(QuantizationMethod.FP8), model))

    assert len(problems) == 1
    assert "quantization" in problems[0]


def test_a_non_gguf_checkpoint_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"this is a safetensors file, honestly")

    problems = list(LlamaCppBackend().validate_model(_experiment(None), model))

    assert len(problems) == 1
    assert "not a GGUF file" in problems[0]


def test_a_directory_is_rejected_as_not_gguf(tmp_path: Path) -> None:
    problems = list(LlamaCppBackend().validate_model(_experiment(None), tmp_path))

    assert any("not a GGUF file" in problem for problem in problems)


def test_a_missing_checkpoint_is_rejected_as_not_gguf(tmp_path: Path) -> None:
    problems = list(LlamaCppBackend().validate_model(_experiment(None), tmp_path / "missing.gguf"))

    assert any("not a GGUF file" in problem for problem in problems)


def test_a_non_gguf_checkpoint_with_a_configured_quantization_reports_both(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a gguf file")

    problems = list(LlamaCppBackend().validate_model(_experiment(QuantizationMethod.FP8), model))

    assert len(problems) == 2
    assert any("quantization" in problem for problem in problems)
    assert any("not a GGUF file" in problem for problem in problems)


def test_a_truncated_gguf_header_fails_open(tmp_path: Path) -> None:
    """A header that cannot be read is not evidence of a mismatch."""
    model = tmp_path / "model.gguf"
    model.write_bytes(_GGUF_MAGIC + struct.pack("<I", _GGUF_VERSION))

    assert list(LlamaCppBackend().validate_model(_experiment(None), model)) == []


def test_a_broken_header_still_rejects_a_configured_quantization(tmp_path: Path) -> None:
    """A flag llama.cpp has no equivalent for is rejected regardless of the file."""
    model = tmp_path / "model.gguf"
    model.write_bytes(_GGUF_MAGIC + struct.pack("<I", _GGUF_VERSION))

    problems = list(LlamaCppBackend().validate_model(_experiment(QuantizationMethod.FP8), model))

    assert len(problems) == 1
    assert "quantization" in problems[0]


def test_a_non_gguf_resolved_draft_is_rejected(tmp_path: Path) -> None:
    """The draft is loaded by the same GGUF reader, so its format matters too."""
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"this is a safetensors draft, honestly")

    problems = list(LlamaCppBackend().validate_model(_experiment(None, draft=draft), model))

    assert len(problems) == 1
    assert "draft model" in problems[0]
    assert "not a GGUF file" in problems[0]


def test_a_draft_with_an_unreadable_header_fails_open(tmp_path: Path) -> None:
    """A header this library cannot parse is not evidence of a mismatch."""
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(_GGUF_MAGIC + struct.pack("<I", _GGUF_VERSION))

    assert list(LlamaCppBackend().validate_model(_experiment(None, draft=draft), model)) == []


def test_an_exact_gpu_layer_count_within_the_block_count_is_accepted(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M, block_count=16)

    assert list(LlamaCppBackend().validate_model(_experiment(None, n_gpu_layers=16), model)) == []


def test_an_exact_gpu_layer_count_above_the_block_count_is_rejected(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M, block_count=16)

    problems = list(LlamaCppBackend().validate_model(_experiment(None, n_gpu_layers=40), model))

    assert len(problems) == 1
    assert "16" in problems[0]
    assert "all" in problems[0]


def test_the_gpu_layer_mode_all_is_not_bounded_by_the_block_count(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M, block_count=16)

    problems = LlamaCppBackend().validate_model(
        _experiment(None, n_gpu_layers=GpuLayersMode.ALL), model
    )

    assert list(problems) == []


def test_the_gpu_layer_mode_auto_is_not_bounded_by_the_block_count(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M, block_count=16)

    problems = LlamaCppBackend().validate_model(
        _experiment(None, n_gpu_layers=GpuLayersMode.AUTO), model
    )

    assert list(problems) == []


def test_an_exact_gpu_layer_count_fails_open_without_a_block_count(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", _Q4_K_M)

    assert list(LlamaCppBackend().validate_model(_experiment(None, n_gpu_layers=40), model)) == []


def test_an_exact_gpu_layer_count_fails_open_on_an_unreadable_header(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(_GGUF_MAGIC + struct.pack("<I", _GGUF_VERSION))

    assert list(LlamaCppBackend().validate_model(_experiment(None, n_gpu_layers=40), model)) == []
