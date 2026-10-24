"""Final summary and winning-configuration writer for the optimization runner.

This module closes out a sweep: it renders the human-readable best-configuration
summary printed when the budget is reached or the user interrupts, and it writes
the recommended configuration to a YAML file when ``--output-config`` is given.

The summary follows the objective the study was run under, which its maximize
directions record: a scalar study prints the best trial and its throughput,
while a multi-objective study prints the Pareto front as a table — trial
number, both phase rates, the recorded combined throughput, and a marker on the
recommended row — under a headline naming the recommended point. Both modes
report how many trials the whole study holds and how many of them failed, then
end with the recommended configuration.

Both outputs describe the same document, so they cannot disagree. The document
has the three domains of an experiment:

- a curated ``container`` section holding only the host-facing settings a user
  would re-supply — ``target``, ``image``, ``shm_size``, ``mem_limit``, and
  ``pull_policy`` — rather than the backend-filled mounts and environment;
- the full ``benchmark`` section; and
- the full ``llm`` section.

The benchmark and LLM sections are emitted from the config's own fields, so a
new field appears here automatically and an unset value is written as an
explicit ``null`` rather than omitted, keeping the file self-describing. Enum
values are written in their engine-neutral spellings, so the document never
carries one engine's vocabulary.

The recommended config reconstructed by the optimizer deliberately leaves an
auto-detected ``target`` and an image resolved by the backend as ``None``. Both
outputs therefore take the resolved values the preparation stage found and fill
them in, so the summary and the file name the hardware and image that actually
ran and the file is re-runnable without re-deriving the environment. Every other
value stays as the recommended trial reconstructed it.

Writing overwrites silently, matching how the rest of the CLI treats an output
path, and never creates parent directories.
"""

from dataclasses import fields
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import optuna
import yaml

from ..inference.types import BenchmarkConfig, ExperimentConfig, HardwareTarget, LLMConfig
from ._formatting import failure_text

if TYPE_CHECKING:
    from ..optimization.types import OptimizationResult

__all__ = ["render_summary", "write_config"]

_RESULT_ATTRIBUTE = "result"
"""The trial user attribute the optimizer records a successful run's metrics under."""

_COMBINED_KEY = "tokens_per_second"
"""The result-projection key holding the aggregate combined tokens per second."""

_FRONT_HEADER = ("trial", "prefill tps", "decode tps", "combined tps", "recommended")
"""The Pareto front table's columns, in render order."""

_CONTAINER_FIELDS = ("shm_size", "mem_limit", "pull_policy")
"""Container settings written besides the resolved ``target`` and ``image``.

The container domain is curated rather than exhaustive: ``selinux_opts``,
``env``, and ``volumes`` are backend- or runtime-filled and would only add noise
to a file a user is meant to re-run, so they are left out.
"""


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
    body = _dump(_config_document(recommended.config, target=target, image=image)).rstrip("\n")
    return (
        f"best trial: {recommended.trial_number}\n"
        f"best throughput: {recommended.value:.2f} tps\n"
        f"trials run: {result.trial_count}\n"
        f"{_failure_line(result)}\n"
        f"\nwinning configuration:\n{body}"
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
    body = _dump(_config_document(recommended.config, target=target, image=image)).rstrip("\n")
    return (
        f"recommended trial: {recommended.trial_number}\n"
        f"recommended rates: prefill {prefill} tps  decode {decode} tps\n"
        f"trials run: {result.trial_count}\n"
        f"{_failure_line(result)}\n"
        f"\npareto front:\n{_front_table(result)}\n"
        f"\nwinning configuration:\n{body}"
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


def _recorded_combined(result: OptimizationResult, trial_number: int) -> float | None:
    """Return the combined throughput the optimizer recorded on a trial.

    An :class:`OptimizationSolution` carries only the objective point itself,
    so the front table's combined column is read from the ``result`` user
    attribute the optimizer writes on every successful trial. A trial without a
    recorded number — a hand-built study, or one whose projection failed to
    record it — yields ``None``.
    """
    for trial in result.study.trials:
        if trial.number != trial_number:
            continue
        recorded = trial.user_attrs.get(_RESULT_ATTRIBUTE)
        if isinstance(recorded, dict):
            value = recorded.get(_COMBINED_KEY)
            if isinstance(value, int | float):
                return float(value)
        return None
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
        _dump(_config_document(config, target=target, image=image)),
        encoding="utf-8",
    )


def _config_document(
    config: ExperimentConfig,
    *,
    target: HardwareTarget,
    image: str,
) -> dict[str, dict[str, object]]:
    """Build the three-domain mapping the summary and the file share.

    The container section is curated and always carries the resolved target and
    image; the benchmark and LLM sections are built from every config field, in
    declaration order, with enum members reduced to their string values and
    unset values kept as ``None`` for YAML's explicit ``null``.
    """
    container: dict[str, object] = {"target": _scalar(target), "image": image}
    container.update({name: _scalar(getattr(config.container, name)) for name in _CONTAINER_FIELDS})
    return {
        "container": container,
        "benchmark": _section(config.benchmark),
        "llm": _section(config.llm),
    }


def _section(config: BenchmarkConfig | LLMConfig) -> dict[str, object]:
    """Return every field of a frozen config dataclass, keyed by field name."""
    return {field.name: _scalar(getattr(config, field.name)) for field in fields(config)}


def _scalar(value: object) -> object:
    """Reduce a config value to something PyYAML's safe dumper can represent.

    A :class:`~enum.StrEnum` member is unwrapped to its engine-neutral string
    value and a :class:`~pathlib.Path` is rendered as its string form; every
    other value passes through unchanged. This keeps the document carrying
    ``cuda``, ``float16``, and a plain path string rather than Python tags
    PyYAML's safe dumper could not represent.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _dump(document: dict[str, dict[str, object]]) -> str:
    """Serialise the document as block-style YAML, preserving field order."""
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
