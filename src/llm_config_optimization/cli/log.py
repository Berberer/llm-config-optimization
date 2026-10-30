"""Live per-trial log reporter for the optimization runner.

This module is the log output mode: one line per finished trial, printed to
stdout so a run can be watched in a terminal or redirected to a file. It is the
callback the optimizer invokes after every trial, so it needs no study loop of
its own and cannot diverge from the sweep's own accounting. Unlike the TUI
reporter, log mode has no running-trial marker: the format is deliberately one
line per finished trial, so a redirected log stays easy to parse.

A scalar trial's line is::

    trial <i>/<N> | <STATE> | <tps> tps  [prefill <p> tps  decode <d> tps ] | failed <pct>% | best <best> | elapsed <elapsed> | eta <eta>

where ``<i>`` counts the trials finished in this invocation, ``<N>`` is the
trial budget and is omitted when the sweep is unbounded, ``<STATE>`` is the
trial's terminal state, ``<tps>`` is the objective value the trial submitted,
``<pct>`` is the share of this invocation's finished trials that failed,
``<best>`` is the best objective value the study holds so far (``n/a`` until a
trial completes), ``<elapsed>`` is the wall-clock time since the reporter
started, rendered by the shared duration format (``0s``, ``42s``, ``2m 10s``,
``1h 5m``), and ``<eta>`` is the budget's projection of the sweep's remaining
time in the same format. Top-level segments are separated by the shared live
separator ``" | "``; the phase rates are the engine-reported prefill and decode
tokens-per-second the optimizer recorded on the trial and form one segment when
the engine reported both, keeping their internal double-space separation. They
are omitted — keeping the historical outcome — when only one was reported.

The ``eta`` segment is absent when the budget cannot project one: an unbounded
sweep, a trial-count budget with no successful trial of this invocation to
average, or a parallel sweep without a timeout. A timeout-only budget shows it
from the first line, because the deadline is already known. The projection uses
the mean duration of this invocation's successful trials — the same basis the
TUI reports as ``avg trial`` — so the two modes cannot disagree.

The failure rate counts the trials whose terminal state is ``FAIL`` and only
those finished in this invocation, so a resumed study does not report an
earlier run's failures as the current run's health. The current trial's own
state is included on its line, so a first-trial failure immediately reads
``failed 100%``. The rate uses the shared whole-percent format, including its
``<1%`` floor for a small but real failure rate.

A multi-objective study has no single objective value, so the line names the
trial's two rates instead and its best column reports the highest rate reached
in each phase::

    trial <i>/<N> | <STATE> | prefill <p> tps  decode <d> tps | failed <pct>% | best prefill <bp> tps  decode <bd> tps | elapsed <elapsed> | eta <eta>

A failed trial prints the message the optimizer recorded under the trial's
``error`` user attribute in place of the value and phase columns, so one bad
configuration is visible without ending the sweep. Its time segments remain, so
the sweep's progress information survives an error line.

The elapsed time is read from an injectable clock and the line is written to an
injectable stream, so the reporter is exercised in tests as a string producer
without a terminal or real time. :meth:`LogReporter.render` returns the line
without printing it, and calling the reporter is what writes it and advances the
trial index. The estimate is computed in the render path from the trial's start
instant and the budget's projection, so rendering stays side-effect free.
"""

import sys
import time
from collections.abc import Callable
from typing import TextIO

import optuna

from llm_config_optimization.cli._budget import Budget, TrialTimings
from llm_config_optimization.cli._formatting import LIVE_SEPARATOR, duration_text, failure_text

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
    it is passed straight through as ``callback``. It tracks three things across
    calls: the number of trials finished in this invocation, which numbers the
    lines, how many of those failed, which feeds the ``failed <pct>%`` segment,
    and the successful trials' durations, which feed the budget's ``eta``
    projection. The start time used for the ``elapsed`` segment comes from its
    clock, and the durations are measured by a :class:`TrialTimings` over the
    same clock so the TUI's ``avg trial`` column uses the same basis. The
    best-so-far value is read from the study on every line rather than
    remembered here — the best objective in a scalar study, the highest rate of
    each phase in a multi-objective one — so a resumed study reflects the best
    an earlier run found.

    ``budget`` carries the sweep's stop budgets; its trial count shapes the
    ``<i>/<N>`` progress column and ``None`` means an unbounded sweep, which
    prints the index alone, and its projection supplies ``eta`` when one is
    possible. ``stream`` defaults to stdout, and ``clock`` to
    :func:`time.monotonic` for a wall-clock duration that is immune to system
    clock changes.
    """

    def __init__(
        self,
        *,
        budget: Budget | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = Budget() if budget is None else budget
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock
        self._started_at = clock()
        self._timings = TrialTimings(clock)
        self._completed = 0
        self._failed = 0

    def start(self) -> None:
        """Restart the elapsed-time clock, e.g. when the sweep actually begins.

        The clock starts when the reporter is built; calling this after the
        preparation steps keeps the elapsed column measuring the sweep rather
        than the setup that preceded it. The trial timings restart too, so a
        restart cannot replay an earlier sweep's mean or its in-flight start.
        """
        self._started_at = self._clock()
        self._timings.start()

    def trial_started(self) -> None:
        """Record when the in-flight trial started, to time its completion.

        The optimizer invokes this through its argument-less trial-start hook,
        so the reporter measures the same start-to-finish duration the TUI does.
        The hook adds no output and does not change the one-line-per-finished-
        trial shape.
        """
        self._timings.trial_started()

    def render(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> str:
        """Return the line for ``trial`` without printing or advancing the index.

        This is the string producer the tests drive directly; the index shown is
        the trial's position among those finished in this invocation, one past
        the lines already printed.
        """
        index = self._completed + 1
        failed = self._failed + (1 if _is_failed(trial) else 0)
        n_trials = self._budget.n_trials
        progress = f"{index}/{n_trials}" if n_trials is not None else str(index)
        rate = failure_text(failed, index) or "0%"
        segments = [
            f"trial {progress}",
            trial.state.name,
            _outcome(trial),
            f"failed {rate}",
            f"best {_best_text(study)}",
            f"elapsed {duration_text(self._elapsed())}",
        ]
        remaining = self._budget.remaining_seconds(
            index,
            mean=self._timings.mean(successful=trial.values is not None),
            elapsed=self._elapsed(),
        )
        if remaining is not None:
            segments.append(f"eta {duration_text(remaining)}")
        return LIVE_SEPARATOR.join(segments)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print the trial's line and advance the index, failure, and duration counts."""
        print(self.render(study, trial), file=self._stream)
        self._completed += 1
        if _is_failed(trial):
            self._failed += 1
        self._timings.record(successful=trial.values is not None)

    def _elapsed(self) -> float:
        """Return the seconds elapsed since the reporter's clock started."""
        return self._clock() - self._started_at


def _is_failed(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether the trial's terminal state is ``FAIL``.

    Only ``FAIL`` counts toward the failure rate. This project does not prune,
    and a successful trial's run-level ``failed_runs`` is a different signal, so
    neither is conflated with a failed trial.
    """
    return trial.state is optuna.trial.TrialState.FAIL


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
