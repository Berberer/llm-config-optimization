"""Shared rendering rules for the CLI's reporters.

The live reporters and the final summary all report how often trials fail and
how long the sweep has taken, so the percentage format, the duration format, and
the live segment separator live here rather than in any one of them. Keeping
them in one place means log mode, TUI mode, and the summary cannot disagree
about the rounding rule or about the ``<1%`` floor, and the time columns cannot
drift apart.

A failure rate is rendered as a whole percentage. A nonzero failure count whose
rounded percentage is zero renders as ``<1%`` so a small but real failure is
never hidden by rounding, while a genuinely clean sweep renders ``0%``. When no
trial has finished there is no rate to report, so the helper returns ``None``;
callers that need a segment omit it instead of inventing a measurement.

A duration is rendered as at most two units, rounded to the nearest second,
half up, and without zero padding: ``0s``, ``42s``, ``2m 10s``, ``1h``,
``1h 5m``, ``2d 4h``. The largest unit is always shown, and the next smaller
unit joins it only when it is nonzero, so an exact hour stays ``1h`` rather
than ``1h 0m``. Rounding is spelled out rather than delegated to
:func:`round`, whose banker's rounding would turn ``2.5`` into ``2`` and
``3.5`` into ``4`` and make a one-second change look inconsistent.
"""

__all__ = ["LIVE_SEPARATOR", "duration_text", "failure_text"]

LIVE_SEPARATOR = " | "
"""The separator between top-level segments of the live log and TUI lines."""

_DURATION_UNITS = ((86_400, "d"), (3_600, "h"), (60, "m"), (1, "s"))
"""The duration units, largest first, with their seconds per unit."""


def failure_text(failed: int, finished: int) -> str | None:
    """Render a failed-trial rate as whole-percent text.

    ``failed`` is the number of failed trials and ``finished`` the number of
    finished trials the rate is measured against. Both counts come from the
    same scope — an invocation for the live reporters, the whole study for the
    summary. ``None`` is returned when no trial has finished, because there is
    no rate to display yet.
    """
    if finished == 0:
        return None
    percent = round(failed * 100 / finished)
    if percent == 0 and failed > 0:
        return "<1%"
    return f"{percent}%"


def duration_text(seconds: float) -> str:
    """Render a duration as at most two units, rounded to the nearest second.

    Fractions round half up and a non-positive duration renders as ``0s``, so
    callers never see a negative or sub-second count.
    """
    remaining = max(int(seconds + 0.5), 0)
    if remaining == 0:
        return "0s"
    parts: list[str] = []
    for size, suffix in _DURATION_UNITS:
        if len(parts) == 2:
            break
        if remaining < size:
            if parts:
                break
            continue
        count, remaining = divmod(remaining, size)
        parts.append(f"{count}{suffix}")
    return " ".join(parts)
