"""The YAML experiment-description writer.

This writer renders the winning configuration as the three-domain experiment
document the runner has always written: a curated ``container`` section, the
full ``benchmark`` section, and the full ``llm`` section. It describes an
experiment rather than an executable container, so it needs no planned server
and keeps the historical config-only input path.

The document is built from the config's own fields, so a new field appears here
automatically and an unset value is written as an explicit ``null`` rather than
omitted. Enum values are reduced to their engine-neutral spellings, so the file
never carries one engine's vocabulary, and blocks are written in declaration
order.
"""

from dataclasses import fields
from enum import StrEnum
from pathlib import Path

import yaml

from ..inference.types import BenchmarkConfig, ExperimentConfig, HardwareTarget, LLMConfig
from .types import RenderContext

__all__ = ["YamlWriter"]

_CONTAINER_FIELDS = ("shm_size", "mem_limit", "pull_policy")
"""Container settings written besides the resolved ``target`` and ``image``.

The container domain is curated rather than exhaustive: ``selinux_opts``,
``env``, and ``volumes`` are backend- or runtime-filled and would only add noise
to a file a user is meant to re-run, so they are left out.
"""


class YamlWriter:
    """Render the winning configuration as the engine-neutral YAML document.

    The artifact describes an experiment, not a running server, so planning is
    never required: the recommended config and the resolved target and image
    are all the writer reads.
    """

    requires_plan = False

    def render(self, context: RenderContext) -> str:
        """Render ``context`` as a block-style three-domain YAML document."""
        document = _config_document(context.config, target=context.target, image=context.image)
        return _dump(document)


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
