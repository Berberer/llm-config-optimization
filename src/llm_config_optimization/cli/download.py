"""Hugging Face download with a single, legible progress line.

The preparation stage resolves a model to a local path by calling
:func:`huggingface_hub.snapshot_download` for a whole repository or
:func:`huggingface_hub.hf_hub_download` for one file. Left to itself the Hub
draws several bars — a per-file transfer bar, a reconstruction bar, and the
file-count bar driven by its thread map — and writes them to stderr. The bars
interleave, the generic descriptions never name the model being fetched, and a
download that takes minutes looks the same as any other.

This module replaces that rendering with exactly one line. For a snapshot it
keeps the file-count bar created by the Hub's thread map, renamed to the model
and written to the CLI's progress stream; the file bar is kept because it has a
stable total from the first frame, so its percentage is honest, while the Hub's
byte-transfer bar has no total until each file begins. For a single file it
keeps the byte-transfer bar instead, because the file-count bar does not exist
then. In both cases the other bars are disabled so only the kept line renders.

Hugging Face's progress internals are touched only here, so a future change to
the Hub's bar layout is confined to this module. The same is true of the Hub's
warning handling: a temporary logging handler terminates the progress line
before a warning is written, so a warning printed mid-download — the missing
``HF_TOKEN`` advice, for example — is not overwritten by the next bar refresh.
"""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import tqdm

__all__ = ["download_model"]

_LOGGER_NAME = "huggingface_hub"
"""The Hub's library-root logger, whose children emit download warnings."""

_FILE_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
"""Rendering of the kept file-count bar, with no rate or ETA.

The bar advances once per finished file, so tqdm's default rate would read as
files per second — a poor proxy for transfer speed because the files differ in
size and download concurrently. The rate, and the remaining-time estimate
derived from it, are therefore omitted; the model name, percentage, bar, and
file count are what the line shows.
"""


class _ProgressLineWarningHandler(logging.Handler):
    """Write a Hub log record on its own line, below the progress bar.

    The single progress line is written without a trailing newline, so a record
    written straight to the stream would land on that line and have its first
    characters overwritten by the next bar refresh. This handler terminates the
    line first and flushes after the record, so the warning stays intact and the
    following refresh redraws on a fresh line.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        """Terminate the progress line, then write the record and a newline."""
        self._stream.write("\n")
        self._stream.write(self.format(record))
        self._stream.write("\n")
        self._stream.flush()


@contextmanager
def _hub_warning_handler(stream: TextIO) -> Generator[None]:
    """Route the Hub's log records through a progress-aware handler on ``stream``.

    For the duration of the block the ``huggingface_hub`` logger writes through
    a :class:`_ProgressLineWarningHandler` and no longer propagates to the root
    handlers, so a warning is rendered once, on the progress stream, with the
    bar terminated ahead of it. The logger's previous handlers and propagation
    flag are restored on exit — including on failure — so repeated downloads do
    not accumulate handlers or leave the process's logging reconfigured.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    previous_handlers = list(logger.handlers)
    previous_propagate = logger.propagate
    handler = _ProgressLineWarningHandler(stream)
    logger.handlers = [handler]
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers = previous_handlers
        logger.propagate = previous_propagate


def _single_bar_progress(
    desc: str, stream: TextIO, *, keep: Callable[[str | None], bool]
) -> type[tqdm]:
    """Build the ``tqdm`` subclass a download passes to the Hub.

    The Hub creates one bar per role and identifies each with a ``name``
    keyword: a snapshot's file-count bar is unnamed while its transfer and
    reconstruction bars are named, and a single-file transfer names its bar
    after the transfer. ``keep`` decides, from that name, which bar renders;
    every other bar is disabled so exactly one line reaches ``stream``. The
    kept bar is renamed to ``desc``, redirected to ``stream``, and given the
    rate-free :data:`_FILE_BAR_FORMAT` unless the Hub already chose a format.
    """

    class _SingleBarProgress(tqdm):
        """A Hub progress bar that renders only the bar ``keep`` selects."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raw_name = kwargs.pop("name", None)
            name = raw_name if isinstance(raw_name, str) else None
            if keep(name):
                kwargs["desc"] = desc
                kwargs["file"] = stream
                kwargs.setdefault("bar_format", _FILE_BAR_FORMAT)
            else:
                kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _SingleBarProgress


def _keep_snapshot_bar(name: str | None) -> bool:
    """Select the snapshot download's unnamed file-count bar."""
    return name is None


def _keep_file_bar(name: str | None) -> bool:
    """Select a single-file download's byte-transfer bar.

    The Hub names the plain HTTP transfer ``huggingface_hub.http_get`` and the
    Xet transfer ``<name>.transfer``; the Xet reconstruction bar shares the base
    name and is disabled so only the transfer line renders.
    """
    return name is not None and (name == "huggingface_hub.http_get" or name.endswith(".transfer"))


def download_model(
    repo_id: str,
    stream: TextIO,
    *,
    filename: str | None = None,
    download: Callable[..., str] = snapshot_download,
    download_file: Callable[..., str] = hf_hub_download,
) -> Path:
    """Download ``repo_id`` to the Hub cache, rendering one progress line.

    With no ``filename`` the whole repository snapshot is fetched, and the
    snapshot directory is returned. With a ``filename`` only that one file is
    fetched — a single GGUF quant, for example — and its path in the cache is
    returned, so a caller that needs one file never downloads its siblings.
    Either way the transfer renders as a single line on ``stream`` —
    ``downloading <repo>: 40%|…| 4/10`` for a snapshot, or
    ``downloading <repo>/<file>: 40%|…| 1.2G/3.0G`` for one file — so a long
    download names what it is fetching and does not flood the terminal. A
    temporary logging handler on the Hub's logger writes any warning it emits
    on its own line on ``stream``, terminated ahead of the progress bar, so a
    mid-download warning survives the next refresh.

    Args:
        repo_id: The Hugging Face repository id to download.
        stream: The CLI progress stream the single line is written to.
        filename: An optional single file to fetch from the repository; ``None``
            fetches the whole snapshot.
        download: The snapshot download callable, injectable so tests can drive
            the rendering without the network. Defaults to
            :func:`huggingface_hub.snapshot_download`.
        download_file: The single-file download callable, injectable for the
            same reason. Defaults to :func:`huggingface_hub.hf_hub_download`.

    Returns:
        The local snapshot directory, or the downloaded file's path when
        ``filename`` was given.

    Raises:
        Exception: Whatever the download callable raises; the caller maps it to
            the CLI's failure contract.
    """
    with _hub_warning_handler(stream):
        if filename is None:
            progress = _single_bar_progress(
                f"downloading {repo_id}", stream, keep=_keep_snapshot_bar
            )
            return Path(download(repo_id, tqdm_class=progress))
        progress = _single_bar_progress(
            f"downloading {repo_id}/{filename}", stream, keep=_keep_file_bar
        )
        return Path(download_file(repo_id, filename, tqdm_class=progress))
