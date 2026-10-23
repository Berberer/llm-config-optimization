"""Live TUI reporter for the optimization runner.

This module is the TUI output mode: a plotext graph of the sweep's progress,
redrawn after every finished trial so the search can be watched converge in a
terminal. It is the callback the optimizer invokes after each trial, so it
needs no study loop of its own and cannot diverge from the sweep's accounting.

Each frame follows the objective mode the reporter was built for. A scalar
weighted or combined sweep draws two series against the trial number, which
counts the trials finished in this invocation:

- a line of the best tokens per second found so far, read from the study, so a
  resumed study's earlier best is reflected; and
- the raw per-trial points, so the noise around the incumbent is visible.

A multi-objective sweep instead draws the trade-off: every rate pair measured in
this invocation as a prefill-versus-decode scatter point, with the study's
Pareto front highlighted as a connected line, read from the study for the same
reason as the scalar best line.

A failed trial contributes no raw point in either view but still counts toward
the trial budget, so the scalar best-so-far line spans the gap and the scatter
simply has one fewer point. A status line under the graph repeats the headline
numbers — trial count against the budget, the best value or the two best phase
rates, mean seconds per successful trial, and elapsed time — so they can be read
without parsing the plot. The average is omitted until a trial succeeds, and
failed trials do not contribute to it.

A frame is drawn at the start of each trial as well as when it finishes: the
optimizer's trial-start hook calls :meth:`TuiReporter.trial_started`, which
records the start time and draws a frame whose status line marks the in-flight
trial, so a trial that takes minutes is visibly running rather than frozen. The
running frame and the completion frame share one status-line format, so the
display does not change shape between them. The graph's y-axis is labelled with
thousands-separated plain numbers, never scientific notation, and its x-axis
with integer trial numbers, thinned to an even step once a sweep grows long.

Like the log reporter, this one is exercised in tests as a string producer: the
elapsed time comes from an injectable clock and the frame is written to an
injectable stream, while :meth:`TuiReporter.render` returns the frame without
printing it or advancing the reporter's own state. There is deliberately no
background timer and no thread of its own.

The reporter is not thread-safe: with ``--n-jobs`` greater than one the
trial-start hook and the completion callback fire from several worker threads,
so frames can interleave. Log mode is the safer choice when a clean record
matters.
"""

import math
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TextIO

import plotext

from ..optimization import ObjectiveMode

if TYPE_CHECKING:
    import optuna

__all__ = ["TuiReporter"]

_TITLE = "tokens per second"
"""Title of the scalar best-so-far view."""

_MULTI_TITLE = "prefill vs decode"
"""Title of the multi-objective prefill-versus-decode scatter."""

_BEST_LABEL = "best"
"""Legend label of the scalar best-so-far line."""

_FRONT_LABEL = "front"
"""Legend label of the multi-objective Pareto front line."""

_TRIALS_LABEL = "trials"
"""Legend label of the raw per-trial points."""

_PREPARING_MESSAGE = (
    "preparing experiments: the first trial warms up the engine and can take several minutes"
)
"""Printed at start, before any trial, so the cold start is explained rather than blank."""


class TuiReporter:
    """Draw a live best-so-far graph, with a status line, once per trial.

    The reporter is a callable matching the optimizer's per-trial callback, so
    it is passed straight through as ``callback``. It keeps the plotted series —
    the raw per-trial values and the best-so-far value in the scalar view, the
    raw prefill-versus-decode pairs in the multi view — and derives the trial
    number from their length, so the index counts the trials finished in this
    invocation. The scalar best-so-far value and the multi Pareto front are read
    from the study on every frame rather than remembered here, so a resumed
    study reflects what an earlier run found.

    ``mode`` selects the view, which is fixed for the life of a sweep: the
    multi mode draws the prefill-versus-decode scatter with the study's Pareto
    front highlighted, while the scalar weighted and combined modes keep the
    historical best-so-far line. It is passed in rather than inferred from the
    study because the trial-start frame is drawn from the reporter's own state,
    before any study is seen, so the view must be known from the first frame.

    ``n_trials`` is the sweep's trial budget and only shapes the status line's
    progress column; ``None`` means an unbounded sweep and prints the count
    alone. ``stream`` defaults to stdout, and ``clock`` to :func:`time.monotonic`
    for a wall-clock duration that is immune to system clock changes.

    The graph is drawn with plotext's process-wide figure, which is cleared
    before every frame, so a second concurrent writer would race with it. With
    ``--n-jobs`` greater than one the trial-start hook and the completion
    callback fire from worker threads, so the reporter is not thread-safe and
    frames can interleave.
    """

    def __init__(
        self,
        *,
        n_trials: int | None = None,
        mode: ObjectiveMode = ObjectiveMode.COMBINED,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = n_trials
        self._multi = mode is ObjectiveMode.MULTI
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock
        self._started_at = clock()
        self._trial_started_at: float | None = None
        self._values: list[float | None] = []
        self._best: list[float | None] = []
        self._pairs: list[tuple[float, float] | None] = []
        self._front: list[tuple[float, float]] = []
        self._durations: list[float] = []

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
        self._pairs.clear()
        self._front.clear()
        self._durations.clear()
        print(_PREPARING_MESSAGE, file=self._stream)

    def trial_started(self) -> None:
        """Record a trial's start and draw a frame marking it in flight.

        The optimizer invokes this through its argument-less trial-start hook,
        so the reporter knows a trial is running before its result arrives. The
        in-flight trial is numbered from the reporter's own finished count, so
        it matches the ``trials i/N`` column, and the frame is drawn from the
        series recorded so far — on the first trial that is an empty plot with
        only the running status line. The start time is kept so the completion
        callback can measure the trial's duration. In the multi view the front
        and best rates cached by the last completed trial are redrawn, since
        this hook receives no study to read them from.
        """
        self._trial_started_at = self._clock()
        if self._multi:
            print(
                self._frame(
                    pairs=self._pairs,
                    front=self._front,
                    rates=_front_rates(self._front),
                    durations=self._durations,
                    running=len(self._pairs) + 1,
                ),
                file=self._stream,
            )
            return
        print(
            self._frame(
                self._values,
                self._best,
                durations=self._durations,
                running=len(self._values) + 1,
            ),
            file=self._stream,
        )

    def render(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> str:
        """Return the frame for ``trial`` without printing or recording it.

        This is the string producer the tests drive directly. In the scalar
        view the frame shows the trials already recorded plus ``trial`` as the
        next point, so the trial count, best value, and average match the state
        after the trial completes. In the multi view the scatter series is the
        recorded pairs plus ``trial``'s pair, while the highlighted front and
        the best rates are read from the study, so a resumed sweep shows the
        front it already had. The reporter's own series are untouched.
        """
        durations = self._durations_for(trial)
        if self._multi:
            front = _front_points(study)
            return self._frame(
                pairs=[*self._pairs, _trial_pair(trial)],
                front=front,
                rates=_front_rates(front),
                durations=durations,
            )
        values = [*self._values, trial.value]
        best = [*self._best, _best_value(study)]
        return self._frame(values, best, durations=durations)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print the trial's frame and record it as the next point."""
        print(self.render(study, trial), file=self._stream)
        if self._multi:
            self._pairs.append(_trial_pair(trial))
            self._front = _front_points(study)
        else:
            self._values.append(trial.value)
            self._best.append(_best_value(study))
        if trial.values is not None:
            self._durations.append(self._trial_duration())

    def _durations_for(self, trial: optuna.trial.FrozenTrial) -> list[float]:
        """Return the successful durations as they stand once ``trial`` completes.

        A successful trial's own duration is measured from its start, so the
        frame that reports it already includes it; a failed trial contributes
        nothing and the average stays over the earlier successes only.
        """
        if trial.values is None:
            return list(self._durations)
        return [*self._durations, self._trial_duration()]

    def _trial_duration(self) -> float:
        """Return the in-flight trial's seconds so far.

        The start is the one recorded by :meth:`trial_started`; when the
        reporter is driven without it (as the string-producer tests do), the
        reporter's own start time stands in so the average stays defined.
        """
        started_at = self._started_at if self._trial_started_at is None else self._trial_started_at
        return self._clock() - started_at

    def _frame(
        self,
        values: Sequence[float | None] = (),
        best: Sequence[float | None] = (),
        *,
        pairs: Sequence[tuple[float, float] | None] = (),
        front: Sequence[tuple[float, float]] = (),
        rates: tuple[float, float] | None = None,
        durations: Sequence[float] = (),
        running: int | None = None,
    ) -> str:
        """Build the graph for the mode's series, followed by the status line.

        In the scalar view ``values`` is the raw per-trial series and ``best``
        the best-so-far series, drawn against the trial number: a trial without
        an objective value failed, so it contributes no raw point but still
        occupies an x position, while the best-so-far series carries the
        previous value across that gap. In the multi view ``pairs`` is the raw
        prefill-versus-decode series and ``front`` the study's Pareto front,
        connected as the trade-off line and drawn against the two rate axes.
        An all-failed (or empty) sweep has nothing to plot, so only the axes
        are drawn. ``durations`` are the successful trials' seconds, averaged
        into the status line, and ``running`` is the trial in flight for a
        running frame, or ``None`` once it has finished.
        """
        figure = plotext.figure
        figure.clear()
        y_values: list[float] = []
        x_values: list[float] = []
        if self._multi:
            raw_x, raw_y = _pair_points(pairs)
            front_x, front_y = _pair_points(front)
            if front_y:
                line = figure.signal(front_x, front_y, marker="braille")
                line.lines(True).label(_FRONT_LABEL)
                figure.draw(line)
            if raw_y:
                points = figure.signal(raw_x, raw_y, marker="dot")
                points.lines(False).label(_TRIALS_LABEL)
                figure.draw(points)
            figure.title(_MULTI_TITLE)
            figure.label("prefill tps", axis=0)
            figure.label("decode tps", axis=1)
            y_values = [*raw_y, *front_y]
            x_values = [*raw_x, *front_x]
            best_text = "n/a" if rates is None else _rates_text(*rates)
            completed = len(pairs)
        else:
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
            figure.title(_TITLE)
            figure.label("trial", axis=0)
            figure.label("tps", axis=1)
            y_values = [*best_y, *raw_y]
            last_best = best[-1] if best else None
            best_text = "n/a" if last_best is None else f"{last_best:.2f} tps"
            completed = len(values)
        if y_values:
            figure.legend(True)
            positions, labels = _axis_ticks(y_values)
            figure.ruler(1, 0).ticks(positions, labels)
            positions, labels = _axis_ticks(x_values) if self._multi else _x_ticks(len(values))
            figure.ruler(0, 0).ticks(positions, labels)
        graph = figure.build().string(colorless=True)
        average = sum(durations) / len(durations) if durations else None
        return (
            f"{graph}\n"
            f"{self._status(completed, best_text=best_text, running=running, average=average)}"
        )

    def _status(
        self,
        completed: int,
        *,
        best_text: str,
        running: int | None = None,
        average: float | None = None,
    ) -> str:
        """Render the line under the graph: count, best text, average, and elapsed time.

        ``best_text`` is already formatted for the mode — one value in the
        scalar view, both phase rates in the multi view — and ``running`` is the
        trial currently in flight, or ``None`` once it has finished;
        ``average`` is the mean seconds per successful trial, or ``None`` until
        one has completed. The running frame and the completion frame share this
        format and differ only by those fields.
        """
        progress = f"{completed}/{self._budget}" if self._budget is not None else str(completed)
        running_text = "" if running is None else f"  running trial {running}"
        average_text = "" if average is None else f"  avg trial {average:.1f}s"
        return f"trials {progress}{running_text}  best {best_text}{average_text}  ({self._elapsed():.1f}s)"

    def _elapsed(self) -> float:
        """Return the seconds elapsed since the reporter's clock started."""
        return self._clock() - self._started_at


_TICK_TARGET = 5
"""Number of ticks aimed for on either numeric axis; the nice-step rounding may add a few."""

_X_TICK_CAP = 10
"""Most x ticks drawn; a longer sweep is thinned to an integer step instead."""


def _axis_ticks(values: Sequence[float]) -> tuple[list[float], list[str]]:
    """Return tick positions and labels for a numeric axis plotting ``values``.

    The positions are round multiples of a *nice* step — 1, 2, 5, or 10 times a
    power of ten — spanning the values' range, so the labels read as plain
    numbers; the multi view needs this on both axes because its rate values
    reach the tens of thousands, where plotext's default switches to scientific
    notation. Each label uses a thousands separator and only the decimals the
    value needs.
    """
    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.1, 0.5)
        lower -= padding
        upper += padding
    step = _nice_step(upper - lower, _TICK_TARGET)
    first = math.floor(lower / step) * step
    last = math.ceil(upper / step) * step
    count = round((last - first) / step) + 1
    positions = [first + index * step for index in range(count)]
    return positions, [_format_tick(position) for position in positions]


def _x_ticks(count: int) -> tuple[list[int], list[str]]:
    """Return integer x tick positions and labels for ``count`` finished trials.

    Trial numbers are always whole, so every tick is an integer: a short sweep
    ticks every number, while a sweep past :data:`_X_TICK_CAP` is thinned to an
    integer step so the labels never crowd each other. The step starts at the
    first trial and is not padded out to the last, so the gaps stay even rather
    than ending with a lone tick a single trial from its neighbour; the last
    trial is ticked only when the step happens to land on it.
    """
    if count <= 0:
        return [], []
    if count <= _X_TICK_CAP:
        positions = list(range(1, count + 1))
    else:
        step = math.ceil((count - 1) / (_X_TICK_CAP - 1))
        positions = list(range(1, count + 1, step))
    return positions, [str(position) for position in positions]


def _nice_step(span: float, target: int) -> float:
    """Return a round tick step covering ``span`` in about ``target`` ticks."""
    raw = span / (target - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    if residual <= 1:
        factor = 1
    elif residual <= 2:
        factor = 2
    elif residual <= 5:
        factor = 5
    else:
        factor = 10
    return factor * magnitude


def _format_tick(value: float) -> str:
    """Render a tick as a thousands-separated decimal, never in scientific form."""
    text = f"{value:,.10f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _points(x: Sequence[int], y: Sequence[float | None]) -> tuple[list[int], list[float]]:
    """Pair each x with its non-``None`` y, dropping the gaps.

    A failed trial's missing value is skipped rather than plotted as zero, so
    the raw series shows a gap while the best-so-far line simply spans it.
    """
    xs = [position for position, value in zip(x, y, strict=True) if value is not None]
    ys = [value for value in y if value is not None]
    return xs, ys


def _pair_points(
    pairs: Sequence[tuple[float, float] | None],
) -> tuple[list[float], list[float]]:
    """Split the non-``None`` ``(x, y)`` pairs into two axis series.

    A failed multi trial is recorded as a ``None`` entry so it still counts
    toward the trial budget; it simply contributes no scatter point.
    """
    present = [pair for pair in pairs if pair is not None]
    return [pair[0] for pair in present], [pair[1] for pair in present]


def _trial_pair(trial: optuna.trial.FrozenTrial) -> tuple[float, float] | None:
    """Return a completed multi trial's ``(prefill, decode)`` pair, or ``None``.

    A failed trial carries no objective values, so it contributes no scatter
    point while still counting toward the trial budget.
    """
    values = trial.values
    if values is None:
        return None
    return values[0], values[1]


def _front_points(study: optuna.Study) -> list[tuple[float, float]]:
    """Return the study's Pareto front as ``(prefill, decode)`` pairs, by prefill.

    The front is Optuna's own ``best_trials`` set, so the highlighted trade-off
    line is exactly the non-dominated frontier the sweep has reached — including
    the trials of an earlier invocation when a study is resumed. Sorting by
    prefill makes the connected line read as the trade-off curve; a study whose
    trials all failed has an empty front.
    """
    points = [
        (values[0], values[1])
        for trial in study.best_trials
        if (values := trial.values) is not None
    ]
    return sorted(points)


def _front_rates(front: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Return the best prefill and decode rate on a Pareto ``front``.

    The per-phase maxima over the front are the per-phase maxima over every
    completed trial — a point that beat the front on one phase would itself be
    non-dominated — so the status line reports the sweep's best rates. ``None``
    when the front is empty, which is the fresh or all-failed sweep case.
    """
    if not front:
        return None
    return max(prefill for prefill, _ in front), max(decode for _, decode in front)


def _rates_text(prefill: float, decode: float) -> str:
    """Render a phase-rate pair as the labelled columns the status line uses."""
    return f"prefill {prefill:.2f} tps  decode {decode:.2f} tps"


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
