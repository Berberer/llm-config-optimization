"""Unit tests for the single-line Hugging Face download helper.

``download_model``'s observable contract is what it writes to the progress
stream and what it returns: one updating line that names the model, the Hub's
warnings intact on their own line, and the resolved snapshot path. These tests
inject a fake download that drives the ``tqdm_class`` the helper supplies, so
the real Hugging Face progress internals are exercised without the network, and
capture the stream in memory. No daemon, GPU, or download is involved.
"""

import io
import logging
from pathlib import Path

import pytest
from huggingface_hub.utils import tqdm

from llm_config_optimization.cli.download import download_model

MODEL = "meta-llama/Llama-3-8B"
"""The repository id every fake download is asked to fetch."""


def test_the_progress_line_names_the_model() -> None:
    """The generic Hub description is replaced by the model being fetched."""
    stream = io.StringIO()

    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        bar = tqdm_class(total=10, desc="Downloading bytes")
        bar.update(4)
        bar.close()
        return "/cache/model"

    download_model(MODEL, stream, download=fake_download)

    output = stream.getvalue()
    assert f"downloading {MODEL}" in output
    assert "Downloading bytes" not in output


def test_only_the_unnamed_file_bar_renders_on_one_updating_line() -> None:
    """The named transfer/reconstruction bars are disabled and the line uses ``\\r``."""
    stream = io.StringIO()

    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        file_bar = tqdm_class(total=10)
        file_bar.update(4)
        transfer_bar = tqdm_class(
            total=100, name="huggingface_hub.snapshot_download", desc="transfer bar"
        )
        transfer_bar.update(50)
        transfer_bar.close()
        file_bar.close()
        return "/cache/model"

    download_model(MODEL, stream, download=fake_download)

    output = stream.getvalue()
    assert f"downloading {MODEL}" in output
    assert "transfer bar" not in output
    # One newline means the updates redrew a single line instead of appending frames.
    assert output.count("\n") == 1


def test_the_progress_line_shows_no_transfer_rate() -> None:
    """The kept bar advances once per file, so a per-second rate would read as files/s."""
    stream = io.StringIO()

    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        bar = tqdm_class(total=10)
        bar.update(4)
        bar.close()
        return "/cache/model"

    download_model(MODEL, stream, download=fake_download)

    output = stream.getvalue()
    assert f"downloading {MODEL}:  40%" in output
    assert "4/10" in output
    assert "it/s" not in output


def test_a_hub_warning_is_written_intact_on_its_own_line() -> None:
    """A warning mid-download survives the next bar refresh without losing a character."""
    stream = io.StringIO()
    logger = logging.getLogger("huggingface_hub")
    warning = "Warning: You are sending unauthenticated requests to the HF Hub."

    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        bar = tqdm_class(total=10)
        bar.update(4)
        logger.warning(warning)
        bar.update(3)
        bar.close()
        return "/cache/model"

    download_model(MODEL, stream, download=fake_download)

    assert f"\n{warning}\n" in stream.getvalue()


def test_the_snapshot_directory_is_returned_as_a_path() -> None:
    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        return "/cache/models--meta-llama--Llama-3-8B/snapshots/abc"

    result = download_model(MODEL, io.StringIO(), download=fake_download)

    assert result == Path("/cache/models--meta-llama--Llama-3-8B/snapshots/abc")


def test_a_single_file_download_returns_the_file_path_and_names_it() -> None:
    """A ``filename`` fetches one file and returns its path, not a snapshot dir."""
    stream = io.StringIO()

    def fake_file_download(repo_id: str, filename: str, *, tqdm_class: type[tqdm]) -> str:
        assert repo_id == MODEL
        assert filename == "model-Q4_K_M.gguf"
        bar = tqdm_class(total=100, name="huggingface_hub.http_get", desc="Downloading bytes")
        bar.update(40)
        bar.close()
        return "/cache/model-Q4_K_M.gguf"

    result = download_model(
        MODEL, stream, filename="model-Q4_K_M.gguf", download_file=fake_file_download
    )

    assert result == Path("/cache/model-Q4_K_M.gguf")
    assert f"downloading {MODEL}/model-Q4_K_M.gguf" in stream.getvalue()


def test_a_single_file_download_keeps_only_the_transfer_bar() -> None:
    """The Xet reconstruction bar is disabled so the file transfer is one line."""
    stream = io.StringIO()

    def fake_file_download(repo_id: str, filename: str, *, tqdm_class: type[tqdm]) -> str:
        reconstruction = tqdm_class(
            total=100, name="huggingface_hub.xet_get", desc="reconstructing"
        )
        reconstruction.update(10)
        reconstruction.close()
        transfer = tqdm_class(
            total=100, name="huggingface_hub.xet_get.transfer", desc="transfer bar"
        )
        transfer.update(50)
        transfer.close()
        return "/cache/file.gguf"

    download_model(MODEL, stream, filename="file.gguf", download_file=fake_file_download)

    output = stream.getvalue()
    assert f"downloading {MODEL}/file.gguf" in output
    assert "reconstructing" not in output
    assert "transfer bar" not in output


def test_a_snapshot_download_is_used_when_no_filename_is_given() -> None:
    """The single-file callable is untouched unless a file was requested."""

    def fake_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        return "/cache/snapshot"

    def unexpected_file_download(*args: object, **kwargs: object) -> str:
        raise AssertionError("the single-file download must not be used for a snapshot")

    result = download_model(
        MODEL, io.StringIO(), download=fake_download, download_file=unexpected_file_download
    )

    assert result == Path("/cache/snapshot")


def test_the_hub_logger_is_restored_when_the_download_fails() -> None:
    """The temporary handler must not leak when the download raises."""
    logger = logging.getLogger("huggingface_hub")
    previous_handlers = list(logger.handlers)
    previous_propagate = logger.propagate

    def failing_download(model: str, *, tqdm_class: type[tqdm]) -> str:
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        download_model(MODEL, io.StringIO(), download=failing_download)

    assert logger.handlers == previous_handlers
    assert logger.propagate == previous_propagate
