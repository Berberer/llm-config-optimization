"""Engine-neutral contracts for rendering a winning configuration to a file.

The output layer turns the recommended experiment of a finished sweep into a
text artifact. It knows the engine-neutral experiment description and the
engine-neutral server plan, and nothing else: a renderer never imports an
inference backend, an engine module, or a container runtime, so adding a new
artifact format cannot pull engine vocabulary into this package.

A writer is a pure renderer. It receives a :class:`RenderContext` and returns
the document as a string; the composition root owns planning and the filesystem
write. The context therefore carries both halves a writer may need: the
recommended :class:`~llm_config_optimization.inference.types.ExperimentConfig`
with the resolved target and image, and the :class:`ServerSpec` a trial planned
when the format reproduces the server that actually ran. A writer that only
needs the experiment — the YAML description — declares that it does not require
a plan and may be handed a context whose ``spec`` is ``None``.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..inference.types import ExperimentConfig, HardwareTarget, ServerSpec

__all__ = ["ConfigWriter", "OutputFormat", "RenderContext"]


class OutputFormat(StrEnum):
    """The artifact a winning configuration is rendered to.

    The members are the explicit ``--output-format`` choices. ``yaml`` is the
    three-domain experiment document, while ``quadlet`` and ``compose`` are
    executable container descriptions and therefore need a planned server. A
    format's conventional filename suffix is documentation, not validation:
    the selected format stays authoritative whatever path the user writes.
    """

    YAML = "yaml"
    QUADLET = "quadlet"
    COMPOSE = "compose"


@dataclass(frozen=True, slots=True)
class RenderContext:
    """Everything a writer needs to render the recommended configuration.

    ``config`` is the recommended experiment, with ``target`` and ``image``
    filled in from preparation so the artifact names the hardware and image
    that actually ran rather than the ``None`` the optimizer reconstructed.
    ``spec`` is the server plan for the same configuration; it is only present
    for formats that reproduce the container, and a writer whose
    ``requires_plan`` is ``True`` is handed a context that carries one.
    """

    config: ExperimentConfig
    target: HardwareTarget
    image: str
    spec: ServerSpec | None = None


@runtime_checkable
class ConfigWriter(Protocol):
    """The seam between an output format and the code that writes its file.

    ``requires_plan`` tells the composition root whether it must invoke the
    backend's planning hook before rendering; a writer that is ``False`` is
    handed only the recommended configuration. ``render`` is pure: it returns
    the complete document and never touches the filesystem, so a writer can be
    unit-tested against a hand-built context and a caller decides where the
    text goes.
    """

    requires_plan: bool

    def render(self, context: RenderContext) -> str:
        """Render ``context`` as this format's complete document."""
        ...
