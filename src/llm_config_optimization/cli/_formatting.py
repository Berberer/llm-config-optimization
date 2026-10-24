"""Shared rendering rules for the CLI's reporters.

The live reporters and the final summary all report how often trials fail, so
the percentage format and the live segment separator live here rather than in
any one of them. Keeping them in one place means log mode, TUI mode, and the
summary cannot disagree about the rounding rule or about the ``<1%`` floor.

A failure rate is rendered as a whole percentage. A nonzero failure count whose
rounded percentage is zero renders as ``<1%`` so a small but real failure is
never hidden by rounding, while a genuinely clean sweep renders ``0%``. When no
trial has finished there is no rate to report, so the helper returns ``None``;
callers that need a segment omit it instead of inventing a measurement.
"""

__all__ = ["LIVE_SEPARATOR", "failure_text"]

LIVE_SEPARATOR = " | "
"""The separator between top-level segments of the live log and TUI lines."""


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
