"""Unit tests for the CLI's shared failure-rate and duration formatting.

The reporters and the summary all render the same percentage, so its rounding
rule and ``<1%`` floor are pinned here independently of those call sites. The
tests cover a clean sweep, a mixed sweep, the floor at a small nonzero rate, an
all-failed sweep, and the no-finished-trials case the live TUI omits.

The duration format is shared by the ``avg trial``, ``elapsed``, and ``eta``
columns, so its two-unit, half-up nearest-second, and no-zero-padding rules are
pinned here too: zero, sub-minute rounding, the exact half, minutes with
seconds, an exact hour, hours with minutes, days, a value that rounds to zero, a
negative value, and the dropped third unit.

The byte-size format renders the summary's memory peaks, so its binary units,
one-decimal rule, half-up rounding, plain-byte floor, and ``n/a`` for an
unmeasured figure are pinned here as well.
"""

from llm_config_optimization.cli._formatting import byte_size_text, duration_text, failure_text


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


def test_zero_duration_renders_zero_seconds() -> None:
    assert duration_text(0) == "0s"


def test_a_sub_minute_duration_renders_whole_seconds() -> None:
    assert duration_text(42) == "42s"


def test_a_sub_second_duration_rounds_to_zero_seconds() -> None:
    assert duration_text(0.4) == "0s"


def test_an_exact_half_second_rounds_up_rather_than_to_even() -> None:
    """Half-up keeps a one-second change monotonic, unlike banker's rounding."""
    assert duration_text(2.5) == "3s"


def test_a_negative_duration_is_clamped_to_zero() -> None:
    assert duration_text(-4.0) == "0s"


def test_a_duration_rounding_up_to_a_minute_switches_unit() -> None:
    assert duration_text(59.6) == "1m"


def test_minutes_and_seconds_render_together() -> None:
    assert duration_text(130) == "2m 10s"


def test_an_exact_hour_omits_the_zero_minutes() -> None:
    assert duration_text(3600) == "1h"


def test_hours_and_minutes_render_together() -> None:
    assert duration_text(3900) == "1h 5m"


def test_days_and_hours_render_together() -> None:
    assert duration_text(2 * 86_400 + 4 * 3_600) == "2d 4h"


def test_a_third_unit_is_dropped_rather_than_padded() -> None:
    assert duration_text(86_400 + 3_600 + 60 + 1) == "1d 1h"


def test_single_digit_units_are_not_zero_padded() -> None:
    assert duration_text(65) == "1m 5s"


def test_an_unmeasured_size_renders_not_available() -> None:
    """Absence must never read as an observed zero."""
    assert byte_size_text(None) == "n/a"


def test_a_size_below_one_kibibyte_renders_whole_bytes() -> None:
    assert byte_size_text(512) == "512 B"


def test_zero_bytes_render_a_measured_zero() -> None:
    assert byte_size_text(0) == "0 B"


def test_a_negative_size_is_clamped_to_zero() -> None:
    assert byte_size_text(-1) == "0 B"


def test_a_kibibyte_is_the_smallest_decimal_unit() -> None:
    assert byte_size_text(1536) == "1.5 KiB"


def test_a_mibibyte_uses_the_next_unit() -> None:
    assert byte_size_text(1024 * 1024) == "1.0 MiB"


def test_a_gibibyte_sized_peak_renders_one_decimal() -> None:
    assert byte_size_text(round(20.4 * 1024**3)) == "20.4 GiB"


def test_the_tenths_digit_rounds_half_up_rather_than_to_even() -> None:
    """Half-up keeps a one-byte change monotonic, unlike banker's rounding."""
    assert byte_size_text(1024**3 + 1024**3 // 4) == "1.3 GiB"
    assert byte_size_text(2 * 1024**3 + 1024**3 // 4) == "2.3 GiB"


def test_a_size_rounding_up_to_a_whole_unit_keeps_the_decimal() -> None:
    assert byte_size_text(10 * 1024**3) == "10.0 GiB"


def test_the_units_are_binary_rather_than_decimal() -> None:
    """A ``GiB`` is 1024 ``MiB``, so 1000 MiB stays in ``MiB``."""
    assert byte_size_text(1000 * 1024**2) == "1000.0 MiB"
