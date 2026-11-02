"""Final summary and winning-configuration writer for the optimization runner.

This module closes out a sweep: it renders the human-readable best-configuration
summary printed when the budget is reached or the user interrupts, and it writes
the recommended configuration to a YAML file when ``--output-config`` is given.

The summary follows the objective the study was run under, which its maximize
directions record: a scalar study prints the best trial and its throughput,
while a multi-objective study prints the Pareto front as a table — trial
number, both phase rates, the recorded combined throughput, and a marker on the
recommended row — under a headline naming the recommended point. Both modes
report how many trials the whole study holds, how many of them failed, and the
largest memory peak any evaluated trial reached, then end with the recommended
configuration, prefaced by that configuration's own measured footprint.

The memory figures are read from the study's per-trial result projections — the
same source as the front table's combined column — so a printed number cannot
contradict the persisted study. The hardware line carries the study-wide
maximums: the largest lifetime peak over a server container's whole life and the
largest peak over measured runs' windows, for VRAM and RAM alike, because a
configuration's hardware requirement is a ceiling that a user weighs against a
throughput gain. The subsection under the winning configuration carries the
recommended point's own two peaks, so the number a user deploys with is never
the sweep's worst case. Memory is informational: it is reported, never scored. A
legacy study whose projection lacks the keys, or a host whose probes found
nothing, renders ``n/a`` for a field it does not have rather than a fabricated
zero, and omits the subsection entirely when the recommended trial measured
nothing at all.

Both the summary and the written file describe the same document, so they
cannot disagree. The document is rendered by the output package's YAML writer,
which owns the three domains of an experiment:

- a curated ``container`` section holding only the host-facing settings a user
  would re-supply — ``target``, ``image``, ``shm_size``, ``mem_limit``, and
  ``pull_policy`` — rather than the backend-filled mounts and environment;
- the full ``benchmark`` section; and
- the full ``llm`` section.

This module keeps the human-readable summary and delegates the document's
construction and serialisation to the registered YAML writer, so the printed and
written configuration share one renderer with the other output formats.

The recommended config reconstructed by the optimizer deliberately leaves an
auto-detected ``target`` and an image resolved by the backend as ``None``. Both
outputs therefore take the resolved values the preparation stage found and fill
them in, so the summary and the file name the hardware and image that actually
ran and the file is re-runnable without re-deriving the environment. Every other
value stays as the recommended trial reconstructed it.

Writing overwrites silently, matching how the rest of the CLI treats an output
path, and never creates parent directories.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from ..inference.types import ExperimentConfig, HardwareTarget
from ..output import OutputFormat, RenderContext, writer_for
from ._formatting import byte_size_text, failure_text

if TYPE_CHECKING:
    from ..optimization.types import OptimizationResult

__all__ = ["render_summary", "write_config"]

_RESULT_ATTRIBUTE = "result"
"""The trial user attribute the optimizer records a successful run's metrics under."""

_COMBINED_KEY = "tokens_per_second"
"""The result-projection key holding the aggregate combined tokens per second."""

_LIFETIME_VRAM_KEY = "lifetime_peak_vram_bytes"
"""The result-projection key holding the whole-container VRAM peak."""

_LIFETIME_RAM_KEY = "lifetime_peak_ram_bytes"
"""The result-projection key holding the whole-container RAM peak."""

_RUN_VRAM_KEY = "peak_vram_bytes"
"""The per-run projection key holding a measured window's VRAM peak."""

_RUN_RAM_KEY = "peak_ram_bytes"
"""The per-run projection key holding a measured window's RAM peak."""

_RUNS_KEY = "runs"
"""The result-projection key holding each successful run's record."""

_FRONT_HEADER = ("trial", "prefill tps", "decode tps", "combined tps", "recommended")
"""The Pareto front table's columns, in render order."""


def _yaml_text(
    config: ExperimentConfig,
    *,
    target: HardwareTarget,
    image: str,
) -> str:
    """Render the shared winning-configuration document through the YAML writer."""
    writer = writer_for(OutputFormat.YAML)
    return writer.render(RenderContext(config=config, target=target, image=image))


def render_summary(
    result: OptimizationResult,
    *,
    target: HardwareTarget,
    image: str,
) -> str:
    """Render the recommended-configuration summary for a finished or interrupted sweep.

    The study's maximize directions decide the shape, because they are the
    definitive record of the objective the stored scores were produced under:
    a multi-objective study gets the Pareto front table, anything else the
    historical single-winner headline. Either way the summary names how many
    trials the study holds, how many of those failed, and ends with the
    recommended point's configuration as the same YAML document
    :func:`write_config` produces. ``target`` and ``image`` are the values the
    preparation stage resolved, and replace the ``None`` the optimizer left in
    the reconstructed config so the printed configuration is explicit.
    """
    if len(result.study.directions) > 1:
        return _multi_summary(result, target=target, image=image)
    return _scalar_summary(result, target=target, image=image)


def _scalar_summary(
    result: OptimizationResult,
    *,
    target: HardwareTarget,
    image: str,
) -> str:
    """Render the historical single-winner headline and the recommended configuration."""
    recommended = result.recommended
    return (
        f"best trial: {recommended.trial_number}\n"
        f"best throughput: {recommended.value:.2f} tps\n"
        f"trials run: {result.trial_count}\n"
        f"{_failure_line(result)}\n"
        f"{_hardware_line(result)}\n"
        f"\n{_configuration_section(result, target=target, image=image)}"
    )


def _multi_summary(
    result: OptimizationResult,
    *,
    target: HardwareTarget,
    image: str,
) -> str:
    """Render the recommended point, the Pareto front table, and its configuration.

    The headline repeats the recommended point because the table alone does not
    say which trade-off the sweep picked, and the width of the front makes the
    row easy to lose. The recommended configuration is the point
    ``--output-config`` writes, so a user can read the front and still get one
    re-runnable file.
    """
    recommended = result.recommended
    prefill, decode = _phase_cells(recommended.value)
    return (
        f"recommended trial: {recommended.trial_number}\n"
        f"recommended rates: prefill {prefill} tps  decode {decode} tps\n"
        f"trials run: {result.trial_count}\n"
        f"{_failure_line(result)}\n"
        f"{_hardware_line(result)}\n"
        f"\npareto front:\n{_front_table(result)}\n"
        f"\n{_configuration_section(result, target=target, image=image)}"
    )


def _failure_line(result: OptimizationResult) -> str:
    """Render how many of the study's trials failed, over the whole study.

    The denominator is the same whole-study count as ``trials run``, so a
    resumed sweep's summary covers every trial it holds rather than only the
    ones the last invocation added. Failures are the trials whose terminal state
    is ``FAIL``; pruned trials and a successful run's ``failed_runs`` field are
    different signals and are not counted. The percentage uses the shared
    formatting rule, including its ``<1%`` floor.
    """
    failed = sum(1 for trial in result.study.trials if trial.state is optuna.trial.TrialState.FAIL)
    finished = len(result.study.trials)
    rate = failure_text(failed, finished) or "0%"
    return f"failed trials: {failed}/{finished} ({rate})"


def _hardware_line(result: OptimizationResult) -> str:
    """Render the largest measured footprint any evaluated trial reached.

    The numbers are the maxima over every successful trial's recorded
    projection, so the line answers "what did this sweep need at its heaviest"
    rather than what the recommended point needs. The lifetime peak is the
    largest whole-container peak a trial recorded, and the measured peak is the
    largest over all runs of all trials — the same per-run maximum
    :func:`_measured_bytes` derives within a trial, taken across the study.

    Every field is rendered independently, so a failed trial, a legacy study
    without the keys, or a CPU target's missing VRAM leaves that field ``n/a``
    instead of a zero. Memory is informational, so the line always appears, and
    the recommended configuration's own footprint follows it in
    :func:`_winning_memory_section`.
    """
    return (
        f"hardware (all trials): {byte_size_text(_study_peak(result, _LIFETIME_VRAM_KEY))} vram peak "
        f"(measured {byte_size_text(_study_measured_peak(result, _RUN_VRAM_KEY))}), "
        f"{byte_size_text(_study_peak(result, _LIFETIME_RAM_KEY))} ram peak "
        f"(measured {byte_size_text(_study_measured_peak(result, _RUN_RAM_KEY))})"
    )


def _configuration_section(
    result: OptimizationResult,
    *,
    target: HardwareTarget,
    image: str,
) -> str:
    """Render the winning configuration, prefaced by its measured footprint.

    Memory belongs to the configuration it describes, so it is a subsection of
    this section rather than a section of its own: there is no way to print a
    footprint without printing the configuration that carries it, and a run
    that produced no winning configuration prints neither.
    """
    body = _yaml_text(result.recommended.config, target=target, image=image).rstrip("\n")
    memory = _memory_subsection(result)
    if memory is None:
        return f"winning configuration:\n{body}"
    return f"winning configuration:\n{memory}\n{body}"


def _memory_subsection(result: OptimizationResult) -> str | None:
    """Render the recommended configuration's own measured footprint, or ``None``.

    The subject is the single point the summary recommends and whose
    configuration follows it, so the number a user takes away for deployment is
    never the sweep's worst case by accident. Both figures come from that
    trial's recorded projection, like the study-wide maxima in the hardware
    line, and an unmeasured half renders ``n/a``.

    A trial that recorded no measurement at all — a legacy study, or a host
    that could read neither the vendor tooling nor the container's cgroup —
    yields ``None`` so the configuration prints without an empty subsection.
    """
    recorded = _recorded_result(result, result.recommended.trial_number)
    lifetime_vram = _projected_bytes(recorded, _LIFETIME_VRAM_KEY)
    lifetime_ram = _projected_bytes(recorded, _LIFETIME_RAM_KEY)
    measured_vram = _measured_bytes(recorded, _RUN_VRAM_KEY)
    measured_ram = _measured_bytes(recorded, _RUN_RAM_KEY)
    if all(value is None for value in (lifetime_vram, lifetime_ram, measured_vram, measured_ram)):
        return None
    return (
        "memory:\n"
        f"  vram: {byte_size_text(lifetime_vram)} lifetime peak "
        f"({byte_size_text(measured_vram)} while serving)\n"
        f"  ram:  {byte_size_text(lifetime_ram)} lifetime peak "
        f"({byte_size_text(measured_ram)} while serving)"
    )


def _recorded_projections(result: OptimizationResult) -> list[dict[str, object]]:
    """Every successful trial's recorded result projection, in trial order.

    Memory is recorded per trial, so a whole-study figure is a maximum over
    these records. A trial that never recorded one — a failed trial, or a
    hand-built study — contributes nothing rather than a zero.
    """
    records = (trial.user_attrs.get(_RESULT_ATTRIBUTE) for trial in result.study.trials)
    return [record for record in records if isinstance(record, dict)]


def _study_peak(result: OptimizationResult, key: str) -> int | None:
    """The largest value a single projected byte count reaches across the study."""
    peaks = [_projected_bytes(record, key) for record in _recorded_projections(result)]
    measured = [peak for peak in peaks if peak is not None]
    return max(measured) if measured else None


def _study_measured_peak(result: OptimizationResult, key: str) -> int | None:
    """The largest per-run byte count any trial's runs recorded."""
    peaks = [_measured_bytes(record, key) for record in _recorded_projections(result)]
    measured = [peak for peak in peaks if peak is not None]
    return max(measured) if measured else None


def _front_table(result: OptimizationResult) -> str:
    """Render the Pareto front as aligned columns with the recommended row marked.

    The rows are the result's solutions in trial order, so the table reads the
    same however Optuna ordered the front. Each row carries the two phase rates
    the solution submitted, the combined throughput the optimizer recorded on
    the trial, and ``yes`` in the recommended column for the point the caller
    should re-run. A trial whose recorded combined rate is missing renders
    ``n/a`` rather than dropping the row, so the front stays complete.
    """
    rows: list[tuple[str, ...]] = [_FRONT_HEADER]
    for solution in sorted(result.solutions, key=lambda item: item.trial_number):
        prefill, decode = _phase_cells(solution.value)
        combined = _recorded_combined(result, solution.trial_number)
        rows.append(
            (
                str(solution.trial_number),
                prefill,
                decode,
                "n/a" if combined is None else f"{combined:.2f}",
                "yes" if solution.trial_number == result.recommended.trial_number else "",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(_FRONT_HEADER))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )


def _phase_cells(value: float | tuple[float, float]) -> tuple[str, str]:
    """Format a solution's objective as the front table's two phase-rate cells.

    A multi-objective solution's value is its ``(prefill, decode)`` pair, which
    is what a multi-direction study always stores; the scalar branch only keeps
    the renderer total for a hand-built result whose study and points disagree,
    a construction error rather than a state the CLI produces.
    """
    if isinstance(value, tuple):
        return f"{value[0]:.2f}", f"{value[1]:.2f}"
    return "n/a", "n/a"


def _recorded_result(result: OptimizationResult, trial_number: int) -> dict[str, object] | None:
    """Return the result projection the optimizer recorded on a trial.

    An :class:`OptimizationSolution` carries only the objective point itself,
    so every figure the summary reports beyond the objective — the combined
    throughput and the memory peaks — is read from the ``result`` user
    attribute the optimizer writes on every successful trial. A trial without
    a recorded projection — a hand-built study, or one whose projection failed
    to record it — yields ``None``.
    """
    for trial in result.study.trials:
        if trial.number == trial_number:
            recorded = trial.user_attrs.get(_RESULT_ATTRIBUTE)
            return recorded if isinstance(recorded, dict) else None
    return None


def _projected_bytes(recorded: dict[str, object] | None, key: str) -> int | None:
    """Read one recorded byte count, or ``None`` when the projection lacks it.

    A study recorded before memory was measured simply has no such key, so an
    absent or non-integer value is an unmeasured figure rather than an error.
    """
    if recorded is None:
        return None
    value = recorded.get(key)
    return value if isinstance(value, int) else None


def _measured_bytes(recorded: dict[str, object] | None, key: str) -> int | None:
    """Take the maximum of a per-run byte count over a projected trial.

    The measured windows are adjacent but not contiguous, so the trial's
    measured peak is the maximum of the runs' peaks rather than a peak over
    their union, and a run that measured nothing is skipped. ``None`` means no
    run recorded a value, which renders as ``n/a``.
    """
    if recorded is None:
        return None
    runs = recorded.get(_RUNS_KEY)
    if not isinstance(runs, list):
        return None
    peaks = [_projected_bytes(run, key) for run in runs if isinstance(run, dict)]
    measured = [peak for peak in peaks if peak is not None]
    return max(measured) if measured else None


def _recorded_combined(result: OptimizationResult, trial_number: int) -> float | None:
    """Return the combined throughput the optimizer recorded on a trial.

    The front table's combined column is read from the same projection the
    hardware line uses; a trial without a recorded number yields ``None``.
    """
    recorded = _recorded_result(result, trial_number)
    if recorded is None:
        return None
    value = recorded.get(_COMBINED_KEY)
    if isinstance(value, int | float):
        return float(value)
    return None


def write_config(
    path: Path,
    config: ExperimentConfig,
    *,
    target: HardwareTarget,
    image: str,
) -> None:
    """Write the recommended configuration to ``path`` as YAML, overwriting silently.

    ``target`` and ``image`` are the resolved values the preparation stage
    found; they replace the ``None`` an auto-detected target and a
    backend-resolved image leave in ``config`` so the file is self-contained.
    An existing file is truncated and replaced, and the parent directory is
    assumed to exist.
    """
    path.write_text(
        _yaml_text(config, target=target, image=image),
        encoding="utf-8",
    )
