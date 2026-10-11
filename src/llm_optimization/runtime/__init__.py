"""Container runtime types and backends.

This subpackage is the stable public import path for running containers::

    from llm_optimization.runtime import ContainerConfig, run_container

The shared types are backend-agnostic; the Podman backend is re-exported here
so callers need not know its module layout.
"""

from ._podman import run_container
from .types import ContainerConfig, ContainerResult, ContainerRunError, Mount

__all__ = [
    "ContainerConfig",
    "ContainerResult",
    "ContainerRunError",
    "Mount",
    "run_container",
]
