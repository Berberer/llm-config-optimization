"""Unit tests for reading a GGUF checkpoint's declared quantization and structure.

``read_gguf_metadata``'s observable contract is the
:class:`~llm_config_optimization.inference._llamacpp.GgufMetadata` it returns, the
:class:`~llm_config_optimization.inference.types.ExperimentError` it raises for a
path that is not a GGUF file, and the ``None`` it returns for a GGUF file whose
header cannot be parsed. The fixtures are hand-written GGUF containers written
into ``tmp_path``, so the parser is exercised without a real model, a GPU, or a
download.
"""

import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

from llm_config_optimization.inference import ExperimentError
from llm_config_optimization.inference._llamacpp import GgufMetadata, read_gguf_metadata

_GGUF_VERSION = 3
_GGUF_MAGIC = b"GGUF"

_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT32 = 4
_UINT64 = 10
_FLOAT32 = 6


def _encode_string(text: str) -> bytes:
    """Encode a length-prefixed UTF-8 GGUF string."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_element(value: object) -> bytes:
    """Encode one array element without its type id."""
    if isinstance(value, bool):
        return struct.pack("<?", value)
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, float):
        return struct.pack("<f", value)
    return struct.pack("<I", value)


def _encode_value(value: object) -> bytes:
    """Encode one metadata value with the type a GGUF file would use for it."""
    if isinstance(value, bool):
        return struct.pack("<I", _BOOL) + struct.pack("<?", value)
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _encode_string(value)
    if isinstance(value, list):
        element_type = _STRING if all(isinstance(item, str) for item in value) else _UINT32
        elements = b"".join(_encode_element(item) for item in value)
        return struct.pack("<IIQ", _ARRAY, element_type, len(value)) + elements
    if isinstance(value, float):
        return struct.pack("<If", _FLOAT32, value)
    return struct.pack("<II", _UINT32, value)


def _write_gguf(
    path: Path, metadata: Mapping[str, object], *, version: int = _GGUF_VERSION
) -> Path:
    """Write a GGUF container holding only the given metadata entries."""
    entries = b"".join(
        _encode_string(key) + _encode_value(value) for key, value in metadata.items()
    )
    path.write_bytes(_GGUF_MAGIC + struct.pack("<IQQ", version, 0, len(metadata)) + entries)
    return path


@pytest.mark.parametrize(
    ("file_type", "quantization"),
    [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (7, "Q8_0"),
        (15, "Q4_K_M"),
        (30, "IQ4_XS"),
        (32, "BF16"),
        (36, "TQ1_0"),
        (41, "Q2_0"),
    ],
)
def test_the_declared_file_type_is_reported_as_a_tag(
    file_type: int, quantization: str, tmp_path: Path
) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.file_type": file_type})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=quantization)


def test_a_checkpoint_without_a_file_type_declares_no_quantization(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_the_architecture_block_count_is_reported(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama", "llama.block_count": 16},
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, block_count=16)


def test_a_block_count_is_reported_without_a_declared_architecture(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"qwen2.block_count": 28})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, block_count=28)


def test_only_the_declared_architecture_block_count_is_reported(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {
            "general.architecture": "llama",
            "llama.block_count": 16,
            "qwen2.block_count": 28,
        },
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, block_count=16)


def test_an_ambiguous_block_count_is_not_reported(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"llama.block_count": 16, "qwen2.block_count": 28},
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_boolean_block_count_is_not_reported(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"llama.block_count": True})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_checkpoint_without_a_block_count_reports_none(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, block_count=None)


@pytest.mark.parametrize("architecture", ["llama", "qwen3moe", "deepseek2"])
def test_the_architecture_expert_count_is_reported(architecture: str, tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": architecture, f"{architecture}.expert_count": 128},
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, expert_count=128)


def test_an_expert_count_is_reported_without_a_declared_architecture(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"qwen3moe.expert_count": 128})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, expert_count=128)


def test_only_the_declared_architecture_expert_count_is_reported(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {
            "general.architecture": "llama",
            "llama.expert_count": 8,
            "qwen3moe.expert_count": 128,
        },
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, expert_count=8)


def test_an_ambiguous_expert_count_is_not_reported(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"llama.expert_count": 8, "qwen3moe.expert_count": 128},
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_zero_expert_count_is_reported(tmp_path: Path) -> None:
    """A readable checkpoint that declares no experts is confidently dense."""
    model = _write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "llama", "llama.expert_count": 0},
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, expert_count=0)


def test_a_boolean_expert_count_is_not_reported(tmp_path: Path) -> None:
    """A boolean is an ``int`` subclass, so it must not count as an expert count."""
    model = _write_gguf(tmp_path / "model.gguf", {"llama.expert_count": True})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_checkpoint_without_an_expert_count_reports_none(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None, expert_count=None)


def test_block_and_expert_counts_are_reported_together(tmp_path: Path) -> None:
    model = _write_gguf(
        tmp_path / "model.gguf",
        {
            "general.architecture": "qwen3moe",
            "qwen3moe.block_count": 48,
            "qwen3moe.expert_count": 128,
        },
    )

    assert read_gguf_metadata(model) == GgufMetadata(
        quantization=None, block_count=48, expert_count=128
    )


def test_an_unmodeled_file_type_declares_no_quantization(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.file_type": 250})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_boolean_file_type_declares_no_quantization(tmp_path: Path) -> None:
    """A boolean is an ``int`` subclass, so it must not match a numeric tag."""
    model = _write_gguf(tmp_path / "model.gguf", {"general.file_type": True})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_a_floating_point_file_type_declares_no_quantization(tmp_path: Path) -> None:
    """A numeric but non-integer value is not a llama.cpp file-type tag."""
    model = _write_gguf(tmp_path / "model.gguf", {"general.file_type": 15.5})

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)


def test_the_file_type_is_found_after_a_fixed_width_array(tmp_path: Path) -> None:
    """A numeric array is skipped in one seek, and the reader stays aligned."""
    model = _write_gguf(
        tmp_path / "model.gguf",
        {
            "tokenizer.ggml.token_type": [1, 1, 3, 4],
            "general.file_type": 15,
        },
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization="Q4_K_M")


def test_the_file_type_is_found_after_entries_of_every_scalar_type(tmp_path: Path) -> None:
    """Skipping strings, booleans, floats, integers, and arrays must not lose alignment."""
    model = _write_gguf(
        tmp_path / "model.gguf",
        {
            "general.name": "Llama 3.2 1B Instruct",
            "general.architecture": "llama",
            "general.is_instruct": True,
            "general.sampling.temperature": 0.7,
            "general.context_length": 131072,
            "general.tags": ["llama", "text-generation"],
            "general.file_type": 15,
        },
    )

    assert read_gguf_metadata(model) == GgufMetadata(quantization="Q4_K_M")


def test_a_file_without_the_gguf_magic_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"this is a safetensors file, honestly")

    with pytest.raises(ExperimentError, match="not a GGUF file"):
        read_gguf_metadata(model)


def test_a_directory_is_rejected_as_not_gguf(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="not a GGUF file"):
        read_gguf_metadata(tmp_path)


def test_a_missing_path_is_rejected_as_not_gguf(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="not a GGUF file"):
        read_gguf_metadata(tmp_path / "missing.gguf")


@pytest.mark.parametrize(
    "payload",
    [
        _GGUF_MAGIC,
        _GGUF_MAGIC + struct.pack("<I", _GGUF_VERSION),
        _GGUF_MAGIC + struct.pack("<IQQ", _GGUF_VERSION, 0, 1),
        _GGUF_MAGIC
        + struct.pack("<IQQ", _GGUF_VERSION, 0, 1)
        + _encode_string("general.file_type"),
        _GGUF_MAGIC
        + struct.pack("<IQQ", _GGUF_VERSION, 0, 1)
        + _encode_string("general.file_type")
        + struct.pack("<I", _UINT32)
        + b"\x0f\x00",
    ],
)
def test_a_truncated_gguf_fails_open(payload: bytes, tmp_path: Path) -> None:
    """A header that cannot be read is not evidence of a mismatch."""
    model = tmp_path / "model.gguf"
    model.write_bytes(payload)

    assert read_gguf_metadata(model) is None


def test_an_unsupported_container_version_fails_open(tmp_path: Path) -> None:
    model = _write_gguf(tmp_path / "model.gguf", {"general.file_type": 15}, version=1)

    assert read_gguf_metadata(model) is None


def test_an_entry_with_an_unknown_value_type_fails_open(tmp_path: Path) -> None:
    payload = (
        _GGUF_MAGIC
        + struct.pack("<IQQ", _GGUF_VERSION, 0, 1)
        + _encode_string("general.unknown")
        + struct.pack("<I", 99)
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(payload)

    assert read_gguf_metadata(model) is None


def test_a_file_type_only_counts_when_it_is_a_numeric_value(tmp_path: Path) -> None:
    """A string under the file-type key is skipped like any other entry."""
    payload = (
        _GGUF_MAGIC
        + struct.pack("<IQQ", _GGUF_VERSION, 0, 1)
        + _encode_string("general.file_type")
        + struct.pack("<I", _STRING)
        + _encode_string("Q4_K_M")
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(payload)

    assert read_gguf_metadata(model) == GgufMetadata(quantization=None)
