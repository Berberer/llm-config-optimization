"""The sweep's stop budgets, and the remaining-time estimate they imply.

Both live reporters show an ``eta`` column, and the two must not disagree about
the number or its absence, so the arithmetic that produces it lives here rather
than in either of them. :class:`Budget` holds the settings that can end a sweep
— the effective trial count, the wall-clock timeout, and whether trials run in
parallel — and projects the remaining seconds from the progress so far.
:class:`TrialTimings` supplies the mean duration of this invocation's
successful trials, the projection's only history input, so the TUI's ``avg
trial`` column and the estimate it feeds are measured once rather than twice.

The projection has two terms. The deadline term is ``max(0, timeout - elapsed)``
and needs no trial history, so a timeout-only sweep has an estimate from its
first frame. The trial term is ``max(0, n_trials - finished) * mean`` where
``mean`` is the mean duration of the successful trials finished in this
invocation; it is unknown until one of those has finished, so a budget that
counts trials has no estimate before its first success. When both terms exist
the smaller wins, because the sweep stops at whichever binds first. When neither
exists — an unbounded sweep — there is no estimate to show.

Trials running in parallel overlap, so the trial term would overstate the
remaining time by roughly the job count; a parallel budget suppresses it and
only a deadline can produce an estimate. The trial count still shapes the
reporters' ``i/N`` progress column.

The result is clamped at zero: Optuna checks its deadline between trials, so the
last trial may run past the deadline, and the estimate reports that state as
``0s`` rather than a negative remainder.
"""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Budget", "TrialTimings"]


@dataclass(frozen=True, slots=True)
class Budget:
    """The sweep's stop budgets and the remaining time they project.

    ``n_trials`` is the effective trial budget — the requested count, or one
    trial when the model-aware assessment left no dimensions — and ``None``
    means the sweep is unbounded by trial count. ``timeout`` is the wall-clock
    limit in seconds, also unbounded when ``None``. ``parallel`` marks a sweep
    whose trials overlap, which makes the mean duration useless for
    extrapolation.

    The value is immutable and carries no clock: callers pass their own elapsed
    time and success mean, so both reporters project from the same inputs and
    the object stays testable without a terminal or real time.
    """

    n_trials: int | None = None
    timeout: float | None = None
    parallel: bool = False

    def remaining_seconds(
        self,
        finished: int,
        *,
        mean: float | None,
        elapsed: float,
    ) -> float | None:
        """Project the sweep's remaining seconds, or ``None`` when unknown.

        ``finished`` counts every trial this invocation has finished, including
        failed ones, because a failed trial consumes one of the budget's trials;
        ``mean`` is the mean duration of the successful ones, or ``None`` until
        one has finished; and ``elapsed`` is the wall-clock seconds since the
        sweep started.

        A timeout-only budget returns its deadline remainder immediately. A
        budget that counts trials returns ``None`` until its first successful
        trial, even when a timeout is also set, because the smaller of the two
        terms cannot be known yet. With both known, the smaller term wins. A
        parallel budget suppresses the trial term entirely, so it projects only
        from a timeout.
        """
        deadline = None if self.timeout is None else max(0.0, self.timeout - elapsed)
        if self.n_trials is None or self.parallel:
            return deadline
        if mean is None:
            return None
        trials = max(0, self.n_trials - finished) * mean
        return trials if deadline is None else min(deadline, trials)


class TrialTimings:
    """Measure this invocation's successful trial durations from their start.

    The TUI shows the mean as ``avg trial`` and both reporters feed it to the
    budget's ``eta``, so the start-to-finish bookkeeping lives here rather than
    in either of them and the two modes cannot disagree about it. Every reading
    comes from the reporter's own injectable clock, so the value is testable
    without real time.

    A trial's duration runs from the instant :meth:`trial_started` recorded —
    the optimizer's trial-start hook — to the moment it completes. When a
    caller drives a reporter as a string producer without that hook, the
    reporter's own :meth:`start` instant stands in, so the average stays
    defined. Only successful trials contribute, because a failed trial says
    nothing about how long a good one takes, and :meth:`start` forgets both the
    in-flight start and the recorded durations so a restart cannot replay an
    earlier sweep.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started_at = clock()
        self._trial_started_at: float | None = None
        self._durations: list[float] = []

    def start(self) -> None:
        """Restart the timing, forgetting the previous sweep's trials."""
        self._started_at = self._clock()
        self._trial_started_at = None
        self._durations.clear()

    def trial_started(self) -> None:
        """Record when the in-flight trial started."""
        self._trial_started_at = self._clock()

    def mean(self, *, successful: bool) -> float | None:
        """Return the success mean once the in-flight trial completes.

        ``successful`` says whether the trial being rendered adds its own
        duration to the recorded ones; a failed trial contributes nothing and
        the mean stays over the earlier successes. ``None`` before any
        successful trial, because there is nothing to average yet.
        """
        durations = self._durations_with(successful)
        return sum(durations) / len(durations) if durations else None

    def record(self, *, successful: bool) -> None:
        """Add the completed trial's duration when it succeeded."""
        if successful:
            self._durations.append(self._current())

    def _durations_with(self, successful: bool) -> list[float]:
        """Return the recorded durations, plus the in-flight one when it succeeded."""
        if not successful:
            return list(self._durations)
        return [*self._durations, self._current()]

    def _current(self) -> float:
        """Return the in-flight trial's seconds so far.

        The start is the one :meth:`trial_started` recorded; without that hook
        the reporter's own start time stands in, so a string-producer driver
        still gets a defined duration.
        """
        started_at = self._started_at if self._trial_started_at is None else self._trial_started_at
        return self._clock() - started_at
