"""Hugging Face snapshot download with a single, legible progress line.

The preparation stage resolves a Hugging Face id to a local snapshot by calling
:func:`huggingface_hub.snapshot_download`. Left to itself the Hub draws several
bars — a per-file transfer bar, a reconstruction bar, and the file-count bar
driven by its thread map — and writes them to stderr. The bars interleave, the
generic descriptions never name the model being fetched, and a download that
takes minutes looks the same as any other.

This module replaces that rendering with exactly one line: the file-count bar
created by the Hub's thread map, renamed to the model and written to the CLI's
progress stream. The file bar is kept because it has a stable total from the
first frame, so its percentage is honest; the Hub's byte-transfer bar has no
total until each file begins and is therefore disabled. The named transfer and
reconstruction bars are disabled so only the kept line renders.

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

from huggingface_hub import snapshot_download
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


def _single_line_progress(model: str, stream: TextIO) -> type[tqdm]:
    """Build the ``tqdm`` subclass :func:`download_model` passes to the Hub.

    ``snapshot_download`` creates its transfer and reconstruction bars with a
    ``name`` while the thread map's file-count bar is created without one, so a
    bar is identified by the presence of that keyword. A named bar is disabled
    so it never renders; the unnamed file-count bar keeps its default rendering
    but has its description rewritten to name the model, its output redirected
    to ``stream``, and its rate-free :data:`_FILE_BAR_FORMAT` applied, which is
    what makes it the only visible line.
    """

    class _SingleLineProgress(tqdm):
        """A Hub progress bar that renders only the unnamed file-count bar."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            name = kwargs.pop("name", None)
            if name is None:
                kwargs["desc"] = f"downloading {model}"
                kwargs["file"] = stream
                kwargs.setdefault("bar_format", _FILE_BAR_FORMAT)
            else:
                kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _SingleLineProgress


def download_model(
    model: str,
    stream: TextIO,
    *,
    download: Callable[..., str] = snapshot_download,
) -> Path:
    """Download ``model`` to the Hub cache, rendering one progress line.

    The snapshot is fetched with a custom :class:`huggingface_hub.utils.tqdm`
    subclass that shows a single line — ``downloading <model>: 40%|…| 4/10`` —
    on ``stream``, so a long download names what it is fetching and does not
    flood the terminal. A temporary logging handler on the Hub's logger writes
    any warning it emits on its own line on ``stream``, terminated ahead of the
    progress bar, so a mid-download warning survives the next refresh. The
    resolved snapshot directory is returned as a path, ready to be mounted or
    inspected.

    Args:
        model: The Hugging Face repository id to download.
        stream: The CLI progress stream the single line is written to.
        download: The download callable, injectable so tests can drive the
            rendering without the network. Defaults to
            :func:`huggingface_hub.snapshot_download`.

    Returns:
        The local snapshot directory the model was downloaded into.

    Raises:
        Exception: Whatever the download callable raises; the caller maps it to
            the CLI's failure contract.
    """
    progress = _single_line_progress(model, stream)
    with _hub_warning_handler(stream):
        return Path(download(model, tqdm_class=progress))
