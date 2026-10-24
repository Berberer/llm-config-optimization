"""Unit tests for the CLI's shared failure-rate formatting.

The reporters and the summary all render the same percentage, so its rounding
rule and ``<1%`` floor are pinned here independently of those call sites. The
tests cover a clean sweep, a mixed sweep, the floor at a small nonzero rate, an
all-failed sweep, and the no-finished-trials case the live TUI omits.
"""

from llm_config_optimization.cli._formatting import failure_text


def test_no_failures_render_zero_percent() -> None:
    assert failure_text(0, 3) == "0%"


def test_one_failure_in_three_renders_a_whole_percentage() -> None:
    assert failure_text(1, 3) == "33%"


def test_a_small_nonzero_rate_renders_the_floor_rather_than_zero() -> None:
    assert failure_text(1, 200) == "<1%"


def test_every_trial_failing_renders_one_hundred_percent() -> None:
    assert failure_text(200, 200) == "100%"


def test_no_finished_trials_have_no_rate() -> None:
    assert failure_text(0, 0) is None
