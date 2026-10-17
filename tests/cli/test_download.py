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
