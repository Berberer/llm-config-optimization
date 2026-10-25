"""The writer registry that maps each output format to its renderer.

The registry is the only place that knows which formats have a writer, so the
composition root selects one by :class:`~llm_config_optimization.output.OutputFormat`
and never imports a renderer class. Adding a fourth artifact style is a small,
isolated change: implement the writer, add one entry here, and extend the enum.

The writers are shared instances, because a renderer is stateless and pure. The
:data:`WRITERS` mapping is exhaustive over the enum, so a format can never be
selected without a registered writer.
"""

from collections.abc import Mapping
from types import MappingProxyType

from ._compose import ComposeWriter
from ._quadlet import QuadletWriter
from ._yaml import YamlWriter
from .types import ConfigWriter, OutputFormat

__all__ = ["WRITERS", "writer_for"]

WRITERS: Mapping[OutputFormat, ConfigWriter] = MappingProxyType(
    {
        OutputFormat.YAML: YamlWriter(),
        OutputFormat.QUADLET: QuadletWriter(),
        OutputFormat.COMPOSE: ComposeWriter(),
    }
)
"""The registered renderer for every output format.

The mapping is immutable and exhaustive over :class:`OutputFormat`, so selection
is a plain lookup with no fallback that could silently choose the wrong artifact.
"""


def writer_for(format: OutputFormat) -> ConfigWriter:
    """Return the renderer registered for ``format``.

    This is the selection entry point: the composition root chooses a format
    from the command line and receives the writer whose ``requires_plan`` tells
    it whether the backend must plan the server first.
    """
    return WRITERS[format]
