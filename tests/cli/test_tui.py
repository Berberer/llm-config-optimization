"""Unit tests for the TUI-mode reporter.

``TuiReporter``'s observable contract is the frame it renders: a plotext graph
with the best-so-far line and the raw per-trial points in the scalar view, or
the prefill-versus-decode scatter with the highlighted Pareto front in the multi
view, followed by a status line. These tests drive it as a string producer with
an injectable clock and stream, using real Optuna studies and frozen trials, so
no sweep, terminal, or real time is involved. They assert on the status line and
the graph's legend and labels rather than on plotext's exact drawing, so the
frame can be restyled without breaking them. The status-line tests cover the
time group — the humanized ``avg trial`` and ``elapsed`` columns and the
budget's ``eta`` estimate — including a timeout-only running frame before any
trial has finished, the smaller-of-two-budgets rule, parallel suppression, and
the clamp at a passed deadline.
"""

import io
import re

import optuna

from llm_config_optimization.cli._budget import Budget
from llm_config_optimization.cli.tui import TuiReporter
from llm_config_optimization.optimization import ObjectiveMode


class FakeClock:
    """A settable monotonic clock, so elapsed time is exact in assertions."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def completed_trial(value: float) -> optuna.trial.FrozenTrial:
    """A frozen completed trial carrying ``value`` as its objective."""
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=value,
        params={},
        distributions={},
    )


def failed_trial(error: str | None = None) -> optuna.trial.FrozenTrial:
    """A frozen failed trial, optionally carrying a recorded error message."""
    user_attrs = {} if error is None else {"error": error}
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.FAIL,
        params={},
        distributions={},
        user_attrs=user_attrs,
    )


def study_with(*trials: optuna.trial.FrozenTrial) -> optuna.Study:
    """An in-memory maximizing study holding ``trials``."""
    study = optuna.create_study(direction="maximize")
    for trial in trials:
        study.add_trial(trial)
    return study


def completed_multi_trial(prefill: float, decode: float) -> optuna.trial.FrozenTrial:
    """A frozen completed trial carrying the two phase rates as its objectives."""
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[prefill, decode],
        params={},
        distributions={},
    )


def multi_study_with(*trials: optuna.trial.FrozenTrial) -> optuna.Study:
    """An in-memory two-objective maximizing study holding ``trials``."""
    study = optuna.create_study(directions=["maximize", "maximize"])
    for trial in trials:
        study.add_trial(trial)
    return study


def status_line(frame: str) -> str:
    """Return the last line of a frame, where the status line lives."""
    return frame.splitlines()[-1]


def graph_body(frame: str) -> str:
    """Return the frame without its trailing status line."""
    return frame.rstrip("\n").rsplit("\n", 1)[0]


def x_tick_labels(frame: str) -> list[str]:
    """Return the x-axis tick labels, read from the line under the plot."""
    lines = frame.splitlines()
    axis_line = next(index for index, line in enumerate(lines) if line.startswith("tps "))
    return lines[axis_line - 1].split()


def y_tick_labels(frame: str) -> list[str]:
    """Return the y-axis tick labels, read from the left edge of the plot."""
    return re.findall(r"^\s*([\d,\.]+)┤", frame, re.MULTILINE)


def multi_x_tick_labels(frame: str) -> list[str]:
    """Return the multi view's x tick labels, above its ``decode tps`` caption."""
    lines = frame.splitlines()
    axis_line = next(index for index, line in enumerate(lines) if line.startswith("decode tps"))
    return lines[axis_line - 1].split()


def final_frame(*values: float, budget: Budget | None = None) -> str:
    """Drive the reporter over ``values`` and return the last frame it drew."""
    stream = io.StringIO()
    reporter = TuiReporter(budget=budget, stream=stream, clock=FakeClock())
    study = optuna.create_study(direction="maximize")
    for value in values:
        trial = completed_trial(value)
        study.add_trial(trial)
        stream.truncate(0)
        stream.seek(0)
        reporter(study, trial)
    return stream.getvalue()


def test_a_frame_plots_both_series_with_a_status_line() -> None:
    reporter = TuiReporter(budget=Budget(n_trials=10), clock=FakeClock())
    trial = completed_trial(42.0)
    study = study_with(trial)

    frame = reporter.render(study, trial)

    assert "tokens per second" in graph_body(frame)
    assert "trial" in graph_body(frame)
    assert "tps" in graph_body(frame)
    assert "best" in graph_body(frame)
    assert "trials" in graph_body(frame)
    assert (
        status_line(frame)
        == "trials 1/10 | failed 0% | best 42.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_the_status_line_reports_count_best_and_elapsed() -> None:
    clock = FakeClock()
    reporter = TuiReporter(budget=Budget(n_trials=10), clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    clock.now = 3.5

    assert status_line(reporter.render(study, trial)) == (
        "trials 1/10 | failed 0% | best 42.00 tps | avg trial 4s | elapsed 4s | eta 32s"
    )


def test_the_average_is_omitted_until_a_trial_succeeds() -> None:
    clock = FakeClock()
    reporter = TuiReporter(budget=Budget(n_trials=10), clock=clock)
    trial = failed_trial("oom")
    study = study_with(trial)
    reporter.trial_started()
    clock.now = 5.0

    status = status_line(reporter.render(study, trial))

    assert "avg" not in status
    assert status == "trials 1/10 | failed 100% | best n/a | elapsed 5s"


def test_failed_trials_do_not_contribute_to_the_average() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=clock)
    good = completed_trial(50.0)
    study = study_with(good)
    reporter.trial_started()
    clock.now = 10.0
    reporter(study, good)
    bad = failed_trial("oom")
    study.add_trial(bad)
    reporter.trial_started()
    clock.now = 100.0
    reporter(study, bad)

    assert (
        status_line(stream.getvalue())
        == "trials 2/10 | failed 50% | best 50.00 tps | avg trial 10s | elapsed 1m 40s | eta 1m 20s"
    )


def test_the_average_is_the_mean_of_successful_trial_durations() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=clock)
    study = optuna.create_study(direction="maximize")
    for value, elapsed in ((40.0, 10.0), (50.0, 20.0)):
        trial = completed_trial(value)
        study.add_trial(trial)
        reporter.trial_started()
        clock.now += elapsed
        reporter(study, trial)

    assert (
        status_line(stream.getvalue())
        == "trials 2/10 | failed 0% | best 50.00 tps | avg trial 15s | elapsed 30s | eta 2m"
    )


def test_an_unbounded_sweep_omits_the_denominator() -> None:
    reporter = TuiReporter(clock=FakeClock())
    trial = completed_trial(42.0)
    study = study_with(trial)

    assert status_line(reporter.render(study, trial)) == (
        "trials 1 | failed 0% | best 42.00 tps | avg trial 0s | elapsed 0s"
    )


def test_a_failed_trial_advances_the_axis_with_the_best_carried_forward() -> None:
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=io.StringIO(), clock=FakeClock())
    good = completed_trial(50.0)
    study = study_with(good)
    reporter(study, good)
    bad = failed_trial("oom")
    study.add_trial(bad)

    frame = reporter.render(study, bad)

    assert (
        status_line(frame)
        == "trials 2/10 | failed 50% | best 50.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )
    assert "best" in graph_body(frame)


def test_a_study_with_no_completed_trial_plots_no_best_line() -> None:
    reporter = TuiReporter(budget=Budget(n_trials=10), clock=FakeClock())
    trial = failed_trial("oom")
    study = study_with(trial)

    frame = reporter.render(study, trial)

    assert status_line(frame) == "trials 1/10 | failed 100% | best n/a | elapsed 0s"
    assert "best" not in graph_body(frame)


def test_the_failure_share_rounds_a_mixed_sweep() -> None:
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())
    study = optuna.create_study(direction="maximize")
    for trial in (completed_trial(50.0), failed_trial("oom"), completed_trial(60.0)):
        study.add_trial(trial)
        reporter(study, trial)

    assert (
        status_line(stream.getvalue())
        == "trials 3/10 | failed 33% | best 60.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_the_failure_share_counts_only_this_invocations_trials() -> None:
    """A resumed study's earlier failures must not distort the current run."""
    earlier = failed_trial("oom")
    trial = completed_trial(42.0)
    study = study_with(earlier, trial)
    reporter = TuiReporter(budget=Budget(n_trials=10), clock=FakeClock())

    frame = reporter.render(study, trial)

    assert (
        status_line(frame)
        == "trials 1/10 | failed 0% | best 42.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_the_pre_first_trial_frame_omits_the_failure_share() -> None:
    """Nothing has finished, so there is no rate to report yet."""
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())

    reporter.trial_started()

    assert "failed" not in status_line(stream.getvalue())


def test_start_announces_preparation_without_drawing_a_frame() -> None:
    """The cold start must not leave a blank plot on screen."""
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())

    reporter.start()

    output = stream.getvalue()
    assert "preparing experiments" in output
    assert "trials 0/10" not in output
    assert "tps" not in output


def test_the_running_frame_marks_the_trial_in_flight() -> None:
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())

    reporter.trial_started()

    frame = stream.getvalue()
    assert "tokens per second" in graph_body(frame)
    assert status_line(frame) == "trials 0/10 | running trial 1 | best n/a | elapsed 0s"


def test_the_running_frame_keeps_the_summary_columns() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=clock)
    good = completed_trial(42.0)
    study = study_with(good)
    reporter.trial_started()
    clock.now = 10.0
    reporter(study, good)
    clock.now = 12.0
    stream.truncate(0)
    stream.seek(0)

    reporter.trial_started()

    assert status_line(stream.getvalue()) == (
        "trials 1/10 | failed 0% | running trial 2 | best 42.00 tps | avg trial 10s | elapsed 12s | eta 1m 30s"
    )


def test_a_timeout_only_budget_shows_the_estimate_before_the_first_trial() -> None:
    """The deadline is known from the first frame, so no success mean is needed."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(timeout=100.0), stream=stream, clock=clock)
    clock.now = 30.0

    reporter.trial_started()

    assert status_line(stream.getvalue()) == (
        "trials 0 | running trial 1 | best n/a | elapsed 30s | eta 1m 10s"
    )


def test_a_trial_count_budget_projects_once_a_mean_exists() -> None:
    """Nine trials remain, each expected to take the twelve seconds just spent."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    reporter.trial_started()
    clock.now = 12.0

    reporter(study, trial)

    assert status_line(stream.getvalue()).endswith("| avg trial 12s | elapsed 12s | eta 1m 48s")


def test_the_estimate_is_the_smaller_of_the_two_budgets() -> None:
    """A near deadline binds before the remaining trials would finish."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10, timeout=100.0), stream=stream, clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    reporter.trial_started()
    clock.now = 30.0

    reporter(study, trial)

    assert status_line(stream.getvalue()).endswith("| elapsed 30s | eta 1m 10s")


def test_a_parallel_budget_suppresses_the_trial_estimate() -> None:
    """Overlapping trials make the mean useless for extrapolation."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10, parallel=True), stream=stream, clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    reporter.trial_started()
    clock.now = 12.0

    reporter(study, trial)

    assert "eta" not in status_line(stream.getvalue())


def test_a_passed_deadline_clamps_the_estimate_at_zero() -> None:
    clock = FakeClock()
    reporter = TuiReporter(budget=Budget(timeout=10.0), clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    clock.now = 25.0

    frame = reporter.render(study, trial)

    assert status_line(frame).endswith("| elapsed 25s | eta 0s")


def test_the_first_completed_trial_draws_the_graph() -> None:
    """The plot appears once there is something to plot."""
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())
    reporter.start()
    stream.truncate(0)
    stream.seek(0)
    trial = completed_trial(42.0)
    study = study_with(trial)

    reporter(study, trial)

    frame = stream.getvalue()
    assert "tokens per second" in graph_body(frame)
    assert (
        status_line(frame)
        == "trials 1/10 | failed 0% | best 42.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_a_short_sweep_ticks_every_integer_trial() -> None:
    frame = final_frame(*[float(value) for value in range(10)], budget=Budget(n_trials=10))

    labels = x_tick_labels(frame)

    assert labels == [str(value) for value in range(1, 11)]


def test_a_long_sweep_thins_the_x_axis_to_integer_ticks() -> None:
    frame = final_frame(*[float(value) for value in range(20)], budget=Budget(n_trials=20))

    labels = x_tick_labels(frame)

    assert all(label.isdigit() for label in labels)
    assert labels == ["1", "4", "7", "10", "13", "16", "19"]


def test_the_y_axis_labels_are_thousands_separated_and_never_scientific() -> None:
    frame = final_frame(1_000_000.0, 1_500_000.0, 2_000_000.0, budget=Budget(n_trials=3))

    labels = y_tick_labels(frame)

    assert labels == ["2,000,000", "1,500,000", "1,000,000"]
    assert all("e" not in label.lower() for label in labels)


def test_start_resets_the_series_so_a_restart_does_not_replay_points() -> None:
    stream = io.StringIO()
    reporter = TuiReporter(budget=Budget(n_trials=10), stream=stream, clock=FakeClock())
    good = completed_trial(50.0)
    study = study_with(good)
    reporter(study, good)
    reporter.start()
    stream.truncate(0)
    stream.seek(0)

    reporter(study, good)

    assert (
        status_line(stream.getvalue())
        == "trials 1/10 | failed 0% | best 50.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_a_multi_frame_draws_the_trade_off_with_the_front_highlighted() -> None:
    trial = completed_multi_trial(1200.0, 50.0)
    study = multi_study_with(trial)
    reporter = TuiReporter(budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, clock=FakeClock())

    frame = reporter.render(study, trial)

    graph = graph_body(frame)
    assert "prefill vs decode" in graph
    assert "prefill tps" in graph
    assert "decode tps" in graph
    assert "front" in graph
    assert "trials" in graph
    assert status_line(frame) == (
        "trials 1/10 | failed 0% | best prefill 1200.00 tps  decode 50.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_the_multi_status_reports_the_best_rate_of_each_phase() -> None:
    """The two maxima can come from different trials, so neither hides the other."""
    high_prefill = completed_multi_trial(1200.0, 50.0)
    high_decode = completed_multi_trial(800.0, 90.0)
    study = multi_study_with(high_prefill, high_decode)
    reporter = TuiReporter(budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, clock=FakeClock())

    frame = reporter.render(study, high_decode)

    assert status_line(frame) == (
        "trials 1/10 | failed 0% | best prefill 1200.00 tps  decode 90.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_a_failed_multi_trial_keeps_the_front_and_the_best_rates() -> None:
    completed = completed_multi_trial(1000.0, 40.0)
    study = multi_study_with(completed)
    reporter = TuiReporter(
        budget=Budget(n_trials=10),
        mode=ObjectiveMode.MULTI,
        stream=io.StringIO(),
        clock=FakeClock(),
    )
    reporter(study, completed)
    failed = failed_trial("simulated out-of-memory")
    study.add_trial(failed)

    frame = reporter.render(study, failed)

    assert "front" in graph_body(frame)
    assert status_line(frame) == (
        "trials 2/10 | failed 50% | best prefill 1000.00 tps  decode 40.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_a_multi_study_with_no_completed_trial_plots_no_front() -> None:
    failed = failed_trial("simulated out-of-memory")
    study = multi_study_with(failed)
    reporter = TuiReporter(budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, clock=FakeClock())

    frame = reporter.render(study, failed)

    assert status_line(frame) == "trials 1/10 | failed 100% | best n/a | elapsed 0s"
    assert "front" not in graph_body(frame)


def test_the_running_multi_frame_keeps_the_front_and_the_best_rates() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    reporter = TuiReporter(
        budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, stream=stream, clock=clock
    )
    trial = completed_multi_trial(1200.0, 50.0)
    study = multi_study_with(trial)
    reporter.trial_started()
    clock.now = 10.0
    reporter(study, trial)
    clock.now = 12.0
    stream.truncate(0)
    stream.seek(0)

    reporter.trial_started()

    frame = stream.getvalue()
    assert "front" in graph_body(frame)
    assert status_line(frame) == (
        "trials 1/10 | failed 0% | running trial 2 | best prefill 1200.00 tps  decode 50.00 tps"
        " | avg trial 10s | elapsed 12s | eta 1m 30s"
    )


def test_a_scalar_mode_keeps_the_historical_line_plot() -> None:
    """The weighted mode is one scalar objective, so its view is unchanged."""
    trial = completed_trial(42.0)
    study = study_with(trial)
    reporter = TuiReporter(
        budget=Budget(n_trials=10), mode=ObjectiveMode.WEIGHTED, clock=FakeClock()
    )

    frame = reporter.render(study, trial)

    assert "tokens per second" in graph_body(frame)
    assert "front" not in graph_body(frame)
    assert (
        status_line(frame)
        == "trials 1/10 | failed 0% | best 42.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_start_resets_the_multi_series_so_a_restart_does_not_replay_points() -> None:
    stream = io.StringIO()
    reporter = TuiReporter(
        budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, stream=stream, clock=FakeClock()
    )
    trial = completed_multi_trial(1200.0, 50.0)
    study = multi_study_with(trial)
    reporter(study, trial)
    reporter.start()
    stream.truncate(0)
    stream.seek(0)

    reporter(study, trial)

    assert status_line(stream.getvalue()) == (
        "trials 1/10 | failed 0% | best prefill 1200.00 tps  decode 50.00 tps | avg trial 0s | elapsed 0s | eta 0s"
    )


def test_the_multi_x_axis_labels_are_thousands_separated_and_never_scientific() -> None:
    """Prefill rates reach the tens of thousands, where plotext uses scientific form."""
    lower = completed_multi_trial(1_200_000.0, 50.0)
    higher = completed_multi_trial(1_500_000.0, 40.0)
    study = multi_study_with(lower, higher)
    reporter = TuiReporter(budget=Budget(n_trials=10), mode=ObjectiveMode.MULTI, clock=FakeClock())

    frame = reporter.render(study, higher)

    labels = multi_x_tick_labels(frame)
    assert labels == ["1,200,000", "1,300,000", "1,400,000", "1,500,000"]
    assert all("e" not in label.lower() for label in labels)
