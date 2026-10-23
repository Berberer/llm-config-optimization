"""Unit tests for the log-mode reporter.

``LogReporter``'s observable contract is the line it renders for a finished
trial and the sequence of lines it writes as it is called. These tests drive it
as a string producer with an injectable clock and stream, using real Optuna
studies and frozen trials, so no sweep, terminal, or real time is involved. They
assert on the documented line format for successful and failed trials, with and
without a budget and recorded phase rates, in scalar and multi-objective
studies, and on the best-so-far value coming from the study so a resumed sweep
reflects it.
"""

import io

import optuna

from llm_config_optimization.cli.log import LogReporter


class FakeClock:
    """A settable monotonic clock, so elapsed time is exact in assertions."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def completed_trial(
    value: float,
    *,
    prefill: float | None = None,
    decode: float | None = None,
) -> optuna.trial.FrozenTrial:
    """A frozen completed trial carrying ``value`` as its objective.

    When either phase rate is given, both are recorded under the ``result``
    user attribute the optimizer uses, so the reporter can render the pair the
    engine reported alongside the scalar objective.
    """
    user_attrs: dict[str, object] = {}
    if prefill is not None or decode is not None:
        user_attrs["result"] = {
            "prefill_tokens_per_second": prefill,
            "decode_tokens_per_second": decode,
        }
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=value,
        params={},
        distributions={},
        user_attrs=user_attrs,
    )


def completed_multi_trial(prefill: float, decode: float) -> optuna.trial.FrozenTrial:
    """A frozen completed trial carrying the two phase rates as its objectives."""
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[prefill, decode],
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


def multi_study_with(*trials: optuna.trial.FrozenTrial) -> optuna.Study:
    """An in-memory two-objective maximizing study holding ``trials``."""
    study = optuna.create_study(directions=["maximize", "maximize"])
    for trial in trials:
        study.add_trial(trial)
    return study


def test_a_successful_trial_renders_the_documented_line() -> None:
    clock = FakeClock()
    reporter = LogReporter(n_trials=10, clock=clock)
    trial = completed_trial(42.0)
    study = study_with(trial)
    clock.now = 3.5

    assert reporter.render(study, trial) == "trial 1/10  COMPLETE  42.00 tps  best 42.00  (3.5s)"


def test_a_scalar_trial_with_recorded_phase_rates_shows_both_on_the_line() -> None:
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = completed_trial(42.0, prefill=1200.0, decode=50.0)
    study = study_with(trial)

    assert reporter.render(study, trial) == (
        "trial 1/10  COMPLETE  42.00 tps  prefill 1200.00 tps  decode 50.00 tps  best 42.00  (0.0s)"
    )


def test_a_scalar_trial_with_only_one_phase_rate_keeps_the_historical_line() -> None:
    """The pair is shown together or not at all, so the grammar stays regular."""
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = completed_trial(42.0, prefill=1200.0)
    study = study_with(trial)

    assert reporter.render(study, trial) == "trial 1/10  COMPLETE  42.00 tps  best 42.00  (0.0s)"


def test_an_unbounded_sweep_omits_the_denominator() -> None:
    reporter = LogReporter(clock=FakeClock())
    trial = completed_trial(42.0)
    study = study_with(trial)

    assert reporter.render(study, trial) == "trial 1  COMPLETE  42.00 tps  best 42.00  (0.0s)"


def test_a_failed_trial_prints_its_recorded_error_in_place_of_throughput() -> None:
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = failed_trial("simulated out-of-memory")
    study = study_with(trial)

    line = reporter.render(study, trial)

    assert line == "trial 1/10  FAIL  simulated out-of-memory  best n/a  (0.0s)"


def test_a_multiline_error_is_collapsed_to_a_single_line() -> None:
    """An engine traceback attached to the error must not break the log format."""
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = failed_trial("container exited with code 1\nAssertionError: boom")
    study = study_with(trial)

    line = reporter.render(study, trial)

    assert line == (
        "trial 1/10  FAIL  container exited with code 1 AssertionError: boom  best n/a  (0.0s)"
    )


def test_a_failure_without_a_recorded_message_uses_a_placeholder() -> None:
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = failed_trial()
    study = study_with(trial)

    line = reporter.render(study, trial)

    assert "  no result  " in line
    assert "best n/a" in line


def test_a_multi_trial_renders_both_objectives_as_phase_rates() -> None:
    reporter = LogReporter(n_trials=10, clock=FakeClock())
    trial = completed_multi_trial(1200.0, 50.0)
    study = multi_study_with(trial)

    assert reporter.render(study, trial) == (
        "trial 1/10  COMPLETE  prefill 1200.00 tps  decode 50.00 tps"
        "  best prefill 1200.00 tps  decode 50.00 tps  (0.0s)"
    )


def test_a_multi_best_column_reports_the_high_water_mark_of_each_phase() -> None:
    """The two rates can come from different trials, so neither hides the other."""
    high_prefill = completed_multi_trial(1200.0, 50.0)
    high_decode = completed_multi_trial(800.0, 90.0)
    study = multi_study_with(high_prefill, high_decode)
    reporter = LogReporter(n_trials=10, clock=FakeClock())

    assert "best prefill 1200.00 tps  decode 90.00 tps" in reporter.render(study, high_decode)


def test_a_failed_multi_trial_still_reports_its_error_and_the_best_rates() -> None:
    completed = completed_multi_trial(1000.0, 40.0)
    failed = failed_trial("simulated out-of-memory")
    study = multi_study_with(completed, failed)
    reporter = LogReporter(n_trials=10, clock=FakeClock())

    assert reporter.render(study, failed) == (
        "trial 1/10  FAIL  simulated out-of-memory"
        "  best prefill 1000.00 tps  decode 40.00 tps  (0.0s)"
    )


def test_a_multi_study_without_a_completed_trial_has_no_best() -> None:
    failed = failed_trial("simulated out-of-memory")
    study = multi_study_with(failed)
    reporter = LogReporter(n_trials=10, clock=FakeClock())

    line = reporter.render(study, failed)

    assert "  simulated out-of-memory  best n/a  " in line


def test_best_so_far_is_read_from_the_study_so_a_resumed_sweep_reflects_it() -> None:
    prior = completed_trial(50.0)
    current = completed_trial(42.0)
    study = study_with(prior, current)
    reporter = LogReporter(n_trials=10, clock=FakeClock())

    assert "best 50.00" in reporter.render(study, current)


def test_calling_the_reporter_writes_each_line_and_advances_the_index() -> None:
    stream = io.StringIO()
    clock = FakeClock()
    reporter = LogReporter(n_trials=2, stream=stream, clock=clock)
    first = completed_trial(10.0)
    study = study_with(first)
    reporter(study, first)
    second = completed_trial(20.0)
    study.add_trial(second)
    clock.now = 1.0
    reporter(study, second)

    assert stream.getvalue().splitlines() == [
        "trial 1/2  COMPLETE  10.00 tps  best 10.00  (0.0s)",
        "trial 2/2  COMPLETE  20.00 tps  best 20.00  (1.0s)",
    ]


def test_start_restarts_the_elapsed_clock() -> None:
    clock = FakeClock()
    reporter = LogReporter(clock=clock)
    clock.now = 5.0
    reporter.start()
    trial = completed_trial(1.0)
    study = study_with(trial)
    clock.now = 7.0

    assert "(2.0s)" in reporter.render(study, trial)
