"""Final summary and winning-configuration writer for the optimization runner.

This module closes out a sweep: it renders the human-readable best-configuration
summary printed when the budget is reached or the user interrupts, and it writes
the winning configuration to a YAML file when ``--output-config`` is given.

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

The winning config reconstructed by the optimizer deliberately leaves an
auto-detected ``target`` and an image resolved by the backend as ``None``. Both
outputs therefore take the resolved values the preparation stage found and fill
them in, so the summary and the file name the hardware and image that actually
ran and the file is re-runnable without re-deriving the environment. Every other
value stays as the winning trial reconstructed it.

Writing overwrites silently, matching how the rest of the CLI treats an output
path, and never creates parent directories.
"""

from dataclasses import fields
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..inference.types import BenchmarkConfig, ExperimentConfig, HardwareTarget, LLMConfig

if TYPE_CHECKING:
    from ..optimization.types import OptimizationResult

__all__ = ["render_summary", "write_config"]

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
    """Render the best-configuration summary for a finished or interrupted sweep.

    The summary names the winning trial, its throughput, and how many trials the
    study holds, then the full winning configuration as the same YAML document
    :func:`write_config` produces. ``target`` and ``image`` are the values the
    preparation stage resolved, and replace the ``None`` the optimizer left in
    the reconstructed config so the printed configuration is explicit.
    """
    document = _config_document(result.best_config, target=target, image=image)
    body = _dump(document).rstrip("\n")
    return (
        f"best trial: {result.best_trial_number}\n"
        f"best throughput: {result.best_value:.2f} tps\n"
        f"trials run: {result.trial_count}\n"
        f"\nwinning configuration:\n{body}"
    )


def write_config(
    path: Path,
    config: ExperimentConfig,
    *,
    target: HardwareTarget,
    image: str,
) -> None:
    """Write the winning configuration to ``path`` as YAML, overwriting silently.

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
