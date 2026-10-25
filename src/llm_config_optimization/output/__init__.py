"""Engine-neutral configuration output writers.

This subpackage is the public import path for turning a finished sweep's
recommended configuration into a file::

    from llm_config_optimization.output import ConfigWriter, OutputFormat, RenderContext

The surface is deliberately small and engine-free: :class:`OutputFormat` names
the artifact styles, :class:`RenderContext` carries the recommended experiment
and — for executable formats — the planned server, and :class:`ConfigWriter` is
the protocol every renderer implements. Writers are pure renderers, so the
composition root owns planning and writing the file. :class:`YamlWriter` renders
the historical three-domain experiment document, :class:`QuadletWriter` renders a
planned server as a Podman systemd unit, and :class:`ComposeWriter` renders the same
server as a Docker/Podman Compose document. :func:`writer_for` selects the
registered renderer for an :class:`OutputFormat`, so callers never import a
concrete writer class.
"""

from ._compose import ComposeWriter
from ._quadlet import QuadletWriter
from ._registry import WRITERS, writer_for
from ._yaml import YamlWriter
from .types import ConfigWriter, OutputFormat, RenderContext

__all__ = [
    "WRITERS",
    "ComposeWriter",
    "ConfigWriter",
    "OutputFormat",
    "QuadletWriter",
    "RenderContext",
    "YamlWriter",
    "writer_for",
]
