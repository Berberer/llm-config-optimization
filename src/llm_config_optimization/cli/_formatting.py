"""Shared rendering rules for the CLI's reporters.

The live reporters and the final summary all report how often trials fail and
how long the sweep has taken, so the percentage format, the duration format, and
the live segment separator live here rather than in any one of them. Keeping
them in one place means log mode, TUI mode, and the summary cannot disagree
about the rounding rule or about the ``<1%`` floor, and the time columns cannot
drift apart. The summary's memory figures use the byte-size format below, so a
lifetime and a measured peak are rendered by the same rule.

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

A byte size is rendered in the largest binary unit it reaches — ``GiB``,
``MiB``, ``KiB``, then plain bytes — with one decimal place and no zero padding:
``512 B``, ``1.5 KiB``, ``6.2 GiB``. The tenths digit rounds half up for the same
reason a duration does, and a size that rounds up to a whole unit keeps the
decimal (``10.0 GiB``) so the column width stays stable. Unmeasured memory
renders as ``n/a`` rather than ``0 B``, because a hardware requirement nobody
observed must never read as an observed zero.
"""

__all__ = ["LIVE_SEPARATOR", "byte_size_text", "duration_text", "failure_text"]

LIVE_SEPARATOR = " | "
"""The separator between top-level segments of the live log and TUI lines."""

_DURATION_UNITS = ((86_400, "d"), (3_600, "h"), (60, "m"), (1, "s"))
"""The duration units, largest first, with their seconds per unit."""

_BYTE_UNITS = ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB"))
"""The binary byte units, largest first, with their bytes per unit."""

_UNMEASURED_TEXT = "n/a"
"""What an unmeasured figure renders as, so absence never reads as a zero."""


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


def byte_size_text(size: int | None) -> str:
    """Render a byte count in the largest binary unit it reaches.

    Units above plain bytes keep one decimal place, rounded half up, so two
    memory peaks are directly comparable; a count below one ``KiB`` renders as
    whole bytes. ``None`` is an unmeasured figure — the sampler found no vendor
    probe, no container statistic, or no sample at all — and renders as ``n/a``
    rather than ``0 B``, because the two mean different things.
    """
    if size is None:
        return _UNMEASURED_TEXT
    remaining = max(size, 0)
    for unit_size, suffix in _BYTE_UNITS:
        if remaining < unit_size:
            continue
        tenths = (remaining * 10 + unit_size // 2) // unit_size
        whole, fraction = divmod(tenths, 10)
        return f"{whole}.{fraction} {suffix}"
    return f"{remaining} B"


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
