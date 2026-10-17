"""Live TUI reporter for the optimization runner.

This module is the TUI output mode: a plotext graph of the sweep's progress,
redrawn after every finished trial so the search can be watched converge in a
terminal. It is the callback the optimizer invokes after each trial, so it
needs no study loop of its own and cannot diverge from the sweep's accounting.

Each frame draws two series against the trial number, which counts the trials
finished in this invocation:

- a line of the best tokens per second found so far, read from the study, so a
  resumed study's earlier best is reflected; and
- the raw per-trial points, so the noise around the incumbent is visible.

A failed trial contributes no raw point but still advances the x-axis, with the
best-so-far line carried forward. A status line under the graph repeats the
headline numbers — trial count against the budget, best value, and elapsed
time — so they can be read without parsing the plot.

Like the log reporter, this one is exercised in tests as a string producer: the
elapsed time comes from an injectable clock and the frame is written to an
injectable stream, while :meth:`TuiReporter.render` returns the frame without
printing it or advancing the reporter's own state. There is deliberately no
background timer and no thread; a preparation message is printed at start and a
frame is drawn once per completed trial, so a long trial simply leaves the
screen static.
"""

import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TextIO

import plotext

if TYPE_CHECKING:
    import optuna

__all__ = ["TuiReporter"]

_BEST_LABEL = "best"
"""Legend label of the best-so-far line."""

_TRIALS_LABEL = "trials"
"""Legend label of the raw per-trial points."""

_PREPARING_MESSAGE = (
    "preparing experiments: the first trial warms up the engine and can take several minutes"
)
"""Printed at start, before any trial, so the cold start is explained rather than blank."""


class TuiReporter:
    """Draw a live best-so-far graph, with a status line, once per trial.

    The reporter is a callable matching the optimizer's per-trial callback, so
    it is passed straight through as ``callback``. It keeps the two plotted
    series — the raw per-trial values and the best-so-far value after each
    trial — and derives the trial number from their length, so the index counts
    the trials finished in this invocation. The best-so-far value is read from
    the study on every frame rather than remembered here, so a resumed study
    reflects the best an earlier run found.

    ``n_trials`` is the sweep's trial budget and only shapes the status line's
    progress column; ``None`` means an unbounded sweep and prints the count
    alone. ``stream`` defaults to stdout, and ``clock`` to :func:`time.monotonic`
    for a wall-clock duration that is immune to system clock changes.

    The graph is drawn with plotext's process-wide figure, which is cleared
    before every frame; the reporter is single-threaded by design, so there is
    no second writer to race with.
    """

    def __init__(
        self,
        *,
        n_trials: int | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = n_trials
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock
        self._started_at = clock()
        self._values: list[float | None] = []
        self._best: list[float | None] = []

    def start(self) -> None:
        """Reset the series and clock and announce that preparation is underway.

        The clock starts when the reporter is built; calling this after the
        preparation steps keeps the elapsed column measuring the sweep rather
        than the setup that preceded it. Clearing the series makes the call
        idempotent, so a restart cannot replay an earlier sweep's points.

        No frame is drawn yet: the first trial loads the model and compiles
        kernels, which can take minutes, so an empty plot would sit on screen
        for the whole cold start. The preparation message explains the wait, and
        the first completed trial draws the graph.
        """
        self._started_at = self._clock()
        self._values.clear()
        self._best.clear()
        print(_PREPARING_MESSAGE, file=self._stream)

    def render(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> str:
        """Return the frame for ``trial`` without printing or recording it.

        This is the string producer the tests drive directly. The frame shows
        the trials already recorded plus ``trial`` as the next point, so the
        trial count and best value match the state after the trial completes;
        the reporter's own series are untouched.
        """
        values = [*self._values, trial.value]
        best = [*self._best, _best_value(study)]
        return self._frame(values, best)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print the trial's frame and record it as the next point."""
        print(self.render(study, trial), file=self._stream)
        self._values.append(trial.value)
        self._best.append(_best_value(study))

    def _frame(self, values: Sequence[float | None], best: Sequence[float | None]) -> str:
        """Build the graph for the two series, followed by the status line.

        A trial without an objective value failed, so it contributes no raw
        point but still occupies an x position; the best-so-far series carries
        the previous value across that gap. An all-failed (or empty) sweep has
        nothing to plot, so only the axes are drawn.
        """
        figure = plotext.figure
        figure.clear()
        x = list(range(1, len(values) + 1))
        best_x, best_y = _points(x, best)
        raw_x, raw_y = _points(x, values)
        if best_y:
            line = figure.signal(best_x, best_y, marker="braille")
            line.lines(True).label(_BEST_LABEL)
            figure.draw(line)
        if raw_y:
            points = figure.signal(raw_x, raw_y, marker="dot")
            points.lines(False).label(_TRIALS_LABEL)
            figure.draw(points)
        figure.title("tokens per second")
        figure.label("trial", axis=0)
        figure.label("tps", axis=1)
        if best_y or raw_y:
            figure.legend(True)
        graph = figure.build().string(colorless=True)
        return f"{graph}\n{self._status(len(values), best[-1] if best else None)}"

    def _status(self, completed: int, best: float | None) -> str:
        """Render the line under the graph: count, best value, and elapsed time."""
        progress = f"{completed}/{self._budget}" if self._budget is not None else str(completed)
        best_text = "n/a" if best is None else f"{best:.2f} tps"
        return f"trials {progress}  best {best_text}  ({self._elapsed():.1f}s)"

    def _elapsed(self) -> float:
        """Return the seconds elapsed since the reporter's clock started."""
        return self._clock() - self._started_at


def _points(x: Sequence[int], y: Sequence[float | None]) -> tuple[list[int], list[float]]:
    """Pair each x with its non-``None`` y, dropping the gaps.

    A failed trial's missing value is skipped rather than plotted as zero, so
    the raw series shows a gap while the best-so-far line simply spans it.
    """
    xs = [position for position, value in zip(x, y, strict=True) if value is not None]
    ys = [value for value in y if value is not None]
    return xs, ys


def _best_value(study: optuna.Study) -> float | None:
    """Return the study's best objective value, or ``None`` before any completes.

    Optuna raises :class:`ValueError` from ``best_value`` while no trial has
    completed, which is exactly the failed-first-trial case the status line must
    still render, so it becomes ``None`` for the ``n/a`` column.
    """
    try:
        return study.best_value
    except ValueError:
        return None
