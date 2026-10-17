"""Live per-trial log reporter for the optimization runner.

This module is the log output mode: one line per finished trial, printed to
stdout so a run can be watched in a terminal or redirected to a file. It is the
callback the optimizer invokes after every trial, so it needs no study loop of
its own and cannot diverge from the sweep's own accounting. Unlike the TUI
reporter, log mode has no running-trial marker: the format is deliberately one
line per finished trial, so a redirected log stays easy to parse.

The line is::

    trial <i>/<N>  <STATE>  <tps> tps  best <best>  (<elapsed>)

where ``<i>`` counts the trials finished in this invocation, ``<N>`` is the
trial budget and is omitted when the sweep is unbounded, ``<STATE>`` is the
trial's terminal state, ``<best>`` is the best objective value the study holds
so far (``n/a`` until a trial completes), and ``<elapsed>`` is the wall-clock
time since the reporter started. A failed trial prints the message the
optimizer recorded under the trial's ``error`` user attribute in place of the
throughput, so one bad configuration is visible without ending the sweep.

The elapsed time is read from an injectable clock and the line is written to an
injectable stream, so the reporter is exercised in tests as a string producer
without a terminal or real time. :meth:`LogReporter.render` returns the line
without printing it, and calling the reporter is what writes it and advances the
trial index.
"""

import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import optuna

__all__ = ["LogReporter"]

_ERROR_ATTRIBUTE = "error"
"""The trial user attribute the optimizer records a failed run's message under."""


class LogReporter:
    """Print one line per finished trial, with best-so-far and elapsed time.

    The reporter is a callable matching the optimizer's per-trial callback, so
    it is passed straight through as ``callback``. It tracks two things across
    calls: the number of trials finished in this invocation, which numbers the
    lines, and the start time used for the elapsed column. The best-so-far value
    is read from the study on every line rather than remembered here, so a
    resumed study reflects the best an earlier run found.

    ``n_trials`` is the sweep's trial budget and only shapes the ``<i>/<N>``
    progress column; ``None`` means an unbounded sweep and prints the index
    alone. ``stream`` defaults to stdout, and ``clock`` to :func:`time.monotonic`
    for a wall-clock duration that is immune to system clock changes.
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
        self._completed = 0

    def start(self) -> None:
        """Restart the elapsed-time clock, e.g. when the sweep actually begins.

        The clock starts when the reporter is built; calling this after the
        preparation steps keeps the elapsed column measuring the sweep rather
        than the setup that preceded it.
        """
        self._started_at = self._clock()

    def render(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> str:
        """Return the line for ``trial`` without printing or advancing the index.

        This is the string producer the tests drive directly; the index shown is
        the trial's position among those finished in this invocation, one past
        the lines already printed.
        """
        index = self._completed + 1
        progress = f"{index}/{self._budget}" if self._budget is not None else str(index)
        best = _best_value(study)
        best_text = "n/a" if best is None else f"{best:.2f}"
        return (
            f"trial {progress}  {trial.state.name}  {_outcome(trial)}"
            f"  best {best_text}  ({self._elapsed():.1f}s)"
        )

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print the trial's line and advance the index by one."""
        print(self.render(study, trial), file=self._stream)
        self._completed += 1

    def _elapsed(self) -> float:
        """Return the seconds elapsed since the reporter's clock started."""
        return self._clock() - self._started_at


def _best_value(study: optuna.Study) -> float | None:
    """Return the study's best objective value, or ``None`` before any completes.

    Optuna raises :class:`ValueError` from ``best_value`` while no trial has
    completed, which is exactly the failed-first-trial case the log line must
    still render, so it becomes ``None`` for the ``n/a`` column.
    """
    try:
        return study.best_value
    except ValueError:
        return None


def _outcome(trial: optuna.trial.FrozenTrial) -> str:
    """Render a trial's result column: its throughput, or its recorded error.

    A completed trial has an objective value, which is printed as tokens per
    second. A trial without one failed, so the message the optimizer recorded on
    it is used instead; a failure that recorded no message — a pruned trial, for
    instance — falls back to a plain placeholder rather than an empty column.
    The message is collapsed to a single line so the documented one-line-per-
    trial format survives a multi-line engine traceback.
    """
    if trial.value is not None:
        return f"{trial.value:.2f} tps"
    error = trial.user_attrs.get(_ERROR_ATTRIBUTE)
    if error is None:
        return "no result"
    return " ".join(str(error).split())
