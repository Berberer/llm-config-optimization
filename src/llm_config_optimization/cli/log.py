"""Live per-trial log reporter for the optimization runner.

This module is the log output mode: one line per finished trial, printed to
stdout so a run can be watched in a terminal or redirected to a file. It is the
callback the optimizer invokes after every trial, so it needs no study loop of
its own and cannot diverge from the sweep's own accounting. Unlike the TUI
reporter, log mode has no running-trial marker: the format is deliberately one
line per finished trial, so a redirected log stays easy to parse.

A scalar trial's line is::

    trial <i>/<N>  <STATE>  <tps> tps  [prefill <p> tps  decode <d> tps ] best <best>  (<elapsed>)

where ``<i>`` counts the trials finished in this invocation, ``<N>`` is the
trial budget and is omitted when the sweep is unbounded, ``<STATE>`` is the
trial's terminal state, ``<tps>`` is the objective value the trial submitted,
``<best>`` is the best objective value the study holds so far (``n/a`` until a
trial completes), and ``<elapsed>`` is the wall-clock time since the reporter
started. The phase rates are the engine-reported prefill and decode
tokens-per-second the optimizer recorded on the trial; they appear as a pair
when the engine reported both and are omitted — keeping the historical line —
when it did not.

A multi-objective study has no single objective value, so the line names the
trial's two rates instead and its best column reports the highest rate reached
in each phase::

    trial <i>/<N>  <STATE>  prefill <p> tps  decode <d> tps  best prefill <bp> tps  decode <bd> tps  (<elapsed>)

A failed trial prints the message the optimizer recorded under the trial's
``error`` user attribute in place of the value and phase columns, so one bad
configuration is visible without ending the sweep.

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

_RESULT_ATTRIBUTE = "result"
"""The trial user attribute the optimizer records a successful run's metrics under."""

_PREFILL_KEY = "prefill_tokens_per_second"
"""The result-projection key holding the aggregate prefill tokens per second."""

_DECODE_KEY = "decode_tokens_per_second"
"""The result-projection key holding the aggregate decode tokens per second."""


class LogReporter:
    """Print one line per finished trial, with best-so-far and elapsed time.

    The reporter is a callable matching the optimizer's per-trial callback, so
    it is passed straight through as ``callback``. It tracks two things across
    calls: the number of trials finished in this invocation, which numbers the
    lines, and the start time used for the elapsed column. The best-so-far value
    is read from the study on every line rather than remembered here — the best
    objective in a scalar study, the highest rate of each phase in a
    multi-objective one — so a resumed study reflects the best an earlier run
    found.

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
        return (
            f"trial {progress}  {trial.state.name}  {_outcome(trial)}"
            f"  best {_best_text(study)}  ({self._elapsed():.1f}s)"
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


def _best_text(study: optuna.Study) -> str:
    """Render the best column: one value, one phase-rate pair, or ``n/a``.

    A scalar study reports the best objective value. A multi-objective study
    has none, so its column reports the highest prefill and decode rate any
    trial has reached instead of inventing a single winner.
    """
    if len(study.directions) > 1:
        rates = _best_rates(study)
        return "n/a" if rates is None else _rates_text(*rates)
    best = _best_value(study)
    return "n/a" if best is None else f"{best:.2f}"


def _best_rates(study: optuna.Study) -> tuple[float, float] | None:
    """Return the highest prefill and decode rate reached in a multi study.

    Every completed trial of a multi study carries its two rates as its
    objective values, so the per-phase maxima are read straight off the study.
    The two maxima may come from different trials; each is the edge of the
    Pareto front the sweep has reached in that phase. ``None`` until a trial
    completes.
    """
    pairs = [values for trial in study.trials if (values := trial.values) is not None]
    if not pairs:
        return None
    return max(values[0] for values in pairs), max(values[1] for values in pairs)


def _outcome(trial: optuna.trial.FrozenTrial) -> str:
    """Render a trial's result column: its rates, throughput, or recorded error.

    A multi-objective trial's two objective values are exactly its prefill and
    decode rates, so they are printed as the pair. A completed scalar trial
    prints its objective as tokens per second and appends the phase-rate pair
    the optimizer recorded on it when the engine reported both, so the same
    line exposes the phase split without needing the study's mode. A trial
    without values failed, so the message the optimizer recorded on it is used
    instead; a failure that recorded no message — a pruned trial, for instance
    — falls back to a plain placeholder rather than an empty column. The
    message is collapsed to a single line so the documented one-line-per-trial
    format survives a multi-line engine traceback.
    """
    values = trial.values
    if values is not None:
        if len(values) > 1:
            return _rates_text(values[0], values[1])
        scalar = f"{values[0]:.2f} tps"
        rates = _recorded_rates(trial)
        return scalar if rates is None else f"{scalar}  {_rates_text(*rates)}"
    error = trial.user_attrs.get(_ERROR_ATTRIBUTE)
    if error is None:
        return "no result"
    return " ".join(str(error).split())


def _rates_text(prefill: float, decode: float) -> str:
    """Render a phase-rate pair as the label/value columns the log line uses."""
    return f"prefill {prefill:.2f} tps  decode {decode:.2f} tps"


def _recorded_rates(trial: optuna.trial.FrozenTrial) -> tuple[float, float] | None:
    """Return the phase-rate pair the optimizer recorded on a scalar trial.

    The optimizer stores a JSON-serializable projection of every successful
    run under the ``result`` user attribute, including the prefill and decode
    rates when the engine reported them. They are shown as a pair or not at
    all: a trial whose engine reported only one phase keeps the historical
    single-value line rather than a half-labelled one.
    """
    recorded = trial.user_attrs.get(_RESULT_ATTRIBUTE)
    if not isinstance(recorded, dict):
        return None
    prefill = recorded.get(_PREFILL_KEY)
    decode = recorded.get(_DECODE_KEY)
    if not isinstance(prefill, int | float) or not isinstance(decode, int | float):
        return None
    return float(prefill), float(decode)
