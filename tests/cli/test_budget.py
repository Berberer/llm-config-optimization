"""Unit tests for the CLI's remaining-time budget and trial timing.

Both live reporters derive their ``eta`` column from :class:`Budget`, so the
deadline term, the successful-trial extrapolation, their minimum, the
first-success availability rule, parallel suppression, and the zero clamp are
pinned here independently of either reporter. The success mean the projection
consumes comes from :class:`TrialTimings`, so its start-to-finish measurement,
its success-only mean, and its restart rule are pinned here too. A fake clock
stands in for real time in the timing tests; the projection itself takes its
elapsed time and mean as arguments and needs no clock at all.
"""

from llm_config_optimization.cli._budget import Budget, TrialTimings


class FakeClock:
    """A settable monotonic clock, so measured durations are exact."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_a_timeout_only_budget_projects_the_deadline_from_the_first_frame() -> None:
    assert Budget(timeout=600).remaining_seconds(0, mean=None, elapsed=100) == 500


def test_a_passed_deadline_projects_zero_rather_than_a_negative_remainder() -> None:
    assert Budget(timeout=600).remaining_seconds(3, mean=None, elapsed=700) == 0


def test_a_deadline_reached_exactly_projects_zero() -> None:
    assert Budget(timeout=600).remaining_seconds(3, mean=None, elapsed=600) == 0


def test_a_trial_budget_extrapolates_the_remaining_trials_from_the_success_mean() -> None:
    assert Budget(n_trials=10).remaining_seconds(1, mean=12, elapsed=12) == 108


def test_a_consumed_trial_budget_projects_zero() -> None:
    assert Budget(n_trials=3).remaining_seconds(3, mean=12, elapsed=36) == 0


def test_a_trial_budget_past_its_count_is_clamped_to_zero() -> None:
    assert Budget(n_trials=3).remaining_seconds(5, mean=12, elapsed=60) == 0


def test_a_trial_budget_has_no_projection_before_its_first_success() -> None:
    assert Budget(n_trials=10).remaining_seconds(0, mean=None, elapsed=5) is None


def test_a_trial_budget_with_a_deadline_waits_for_its_first_success() -> None:
    """The minimum cannot be known before the trial term is, so neither is shown."""
    assert Budget(n_trials=10, timeout=600).remaining_seconds(0, mean=None, elapsed=5) is None


def test_the_trial_extrapolation_wins_when_it_binds_before_the_deadline() -> None:
    budget = Budget(n_trials=10, timeout=100)
    assert budget.remaining_seconds(3, mean=12, elapsed=10) == 84


def test_the_deadline_wins_when_it_binds_before_the_trial_extrapolation() -> None:
    budget = Budget(n_trials=10, timeout=100)
    assert budget.remaining_seconds(3, mean=12, elapsed=50) == 50


def test_a_parallel_budget_projects_only_from_its_timeout() -> None:
    budget = Budget(n_trials=10, timeout=600, parallel=True)
    assert budget.remaining_seconds(3, mean=12, elapsed=100) == 500


def test_a_parallel_budget_without_a_timeout_has_no_projection() -> None:
    assert Budget(n_trials=10, parallel=True).remaining_seconds(3, mean=12, elapsed=100) is None


def test_an_unbounded_budget_has_no_projection() -> None:
    assert Budget().remaining_seconds(3, mean=12, elapsed=100) is None


def test_a_first_successful_trial_supplies_its_own_duration() -> None:
    """The trial being rendered has already finished, so it joins the mean."""
    clock = FakeClock()
    timings = TrialTimings(clock)
    timings.trial_started()
    clock.now = 12

    assert timings.mean(successful=True) == 12


def test_a_successful_trial_is_recorded_for_the_next_mean() -> None:
    clock = FakeClock()
    timings = TrialTimings(clock)
    timings.trial_started()
    clock.now = 10
    timings.record(successful=True)
    timings.trial_started()
    clock.now = 30

    assert timings.mean(successful=True) == 15


def test_a_failed_trial_contributes_neither_its_duration_nor_a_mean() -> None:
    clock = FakeClock()
    timings = TrialTimings(clock)
    timings.trial_started()
    clock.now = 90

    assert timings.mean(successful=False) is None
    timings.record(successful=False)
    assert timings.mean(successful=False) is None


def test_a_failed_trial_does_not_distort_an_earlier_success() -> None:
    clock = FakeClock()
    timings = TrialTimings(clock)
    timings.trial_started()
    clock.now = 10
    timings.record(successful=True)
    timings.trial_started()
    clock.now = 100

    assert timings.mean(successful=False) == 10


def test_the_reporters_own_start_stands_in_without_the_trial_start_hook() -> None:
    """A string-producer driver skips the hook, so the duration stays defined."""
    clock = FakeClock()
    timings = TrialTimings(clock)
    clock.now = 7

    assert timings.mean(successful=True) == 7


def test_a_restart_forgets_the_running_start_and_the_recorded_mean() -> None:
    """The fresh start stands in, never the forgotten in-flight instant."""
    clock = FakeClock()
    timings = TrialTimings(clock)
    timings.trial_started()
    clock.now = 10
    timings.record(successful=True)
    clock.now = 100
    timings.trial_started()
    clock.now = 500

    timings.start()

    assert timings.mean(successful=False) is None
    assert timings.mean(successful=True) == 0
