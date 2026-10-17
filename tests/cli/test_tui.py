"""Unit tests for the TUI-mode reporter.

``TuiReporter``'s observable contract is the frame it renders: a plotext graph
with the best-so-far line and the raw per-trial points, followed by a status
line. These tests drive it as a string producer with an injectable clock and
stream, using real Optuna studies and frozen trials, so no sweep, terminal, or
real time is involved. They assert on the status line and the graph's legend and
labels rather than on plotext's exact drawing, so the frame can be restyled
without breaking them.
"""

import io

import optuna

from llm_config_optimization.cli.tui import TuiReporter


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


def status_line(frame: str) -> str:
    """Return the last line of a frame, where the status line lives."""
    return frame.splitlines()[-1]


def graph_body(frame: str) -> str:
    """Return the frame without its trailing status line."""
    return frame.rstrip("\n").rsplit("\n", 1)[0]


def test_a_frame_plots_both_series_with_a_status_line() -> None:
    reporter = TuiReporter(n_trials=10, clock=FakeClock())
    trial = completed_trial(42.0)
    study = study_with(trial)

    frame = reporter.render(study, trial)

    assert "tokens per second" in graph_body(frame)
    assert "trial" in graph_body(frame)
    assert "tps" in graph_body(frame)
    assert "best" in graph_body(frame)
    assert "trials" in graph_body(frame)
    assert status_line(frame) == "trials 1/10  best 42.00 tps  (0.0s)"


def test_the_status_line_reports_count_best_and_elapsed() -> None:
    clock = FakeClock()
    reporter = TuiReporter(n_trials=10, clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    clock.now = 3.5

    assert status_line(reporter.render(study, trial)) == "trials 1/10  best 42.00 tps  (3.5s)"


def test_an_unbounded_sweep_omits_the_denominator() -> None:
    reporter = TuiReporter(clock=FakeClock())
    trial = completed_trial(42.0)
    study = study_with(trial)

    assert status_line(reporter.render(study, trial)) == "trials 1  best 42.00 tps  (0.0s)"


def test_a_failed_trial_advances_the_axis_with_the_best_carried_forward() -> None:
    reporter = TuiReporter(n_trials=10, stream=io.StringIO(), clock=FakeClock())
    good = completed_trial(50.0)
    study = study_with(good)
    reporter(study, good)
    bad = failed_trial("oom")
    study.add_trial(bad)

    frame = reporter.render(study, bad)

    assert status_line(frame) == "trials 2/10  best 50.00 tps  (0.0s)"
    assert "best" in graph_body(frame)


def test_a_study_with_no_completed_trial_plots_no_best_line() -> None:
    reporter = TuiReporter(n_trials=10, clock=FakeClock())
    trial = failed_trial("oom")
    study = study_with(trial)

    frame = reporter.render(study, trial)

    assert status_line(frame) == "trials 1/10  best n/a  (0.0s)"
    assert "best" not in graph_body(frame)


def test_start_announces_preparation_without_drawing_a_frame() -> None:
    """The cold start must not leave a blank plot on screen."""
    stream = io.StringIO()
    reporter = TuiReporter(n_trials=10, stream=stream, clock=FakeClock())

    reporter.start()

    output = stream.getvalue()
    assert "preparing experiments" in output
    assert "trials 0/10" not in output
    assert "tps" not in output


def test_the_first_completed_trial_draws_the_graph() -> None:
    """The plot appears once there is something to plot."""
    stream = io.StringIO()
    reporter = TuiReporter(n_trials=10, stream=stream, clock=FakeClock())
    reporter.start()
    stream.truncate(0)
    stream.seek(0)
    trial = completed_trial(42.0)
    study = study_with(trial)

    reporter(study, trial)

    frame = stream.getvalue()
    assert "tokens per second" in graph_body(frame)
    assert status_line(frame) == "trials 1/10  best 42.00 tps  (0.0s)"


def test_start_resets_the_series_so_a_restart_does_not_replay_points() -> None:
    stream = io.StringIO()
    reporter = TuiReporter(n_trials=10, stream=stream, clock=FakeClock())
    good = completed_trial(50.0)
    study = study_with(good)
    reporter(study, good)
    reporter.start()
    stream.truncate(0)
    stream.seek(0)

    reporter(study, good)

    assert status_line(stream.getvalue()) == "trials 1/10  best 50.00 tps  (0.0s)"
