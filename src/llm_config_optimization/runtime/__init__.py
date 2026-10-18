"""Container runtime types and backends.

This subpackage is the stable public import path for running containers::

    from llm_config_optimization.runtime import ContainerConfig, run_container

The shared types are backend-agnostic; the Podman backend is re-exported here
so callers need not know its module layout.
"""

from ._podman import pull_image, run_container, start_container
from .types import (
    ContainerConfig,
    ContainerResult,
    ContainerRunError,
    Mount,
    RunningContainer,
)

__all__ = [
    "ContainerConfig",
    "ContainerResult",
    "ContainerRunError",
    "Mount",
    "RunningContainer",
    "pull_image",
    "run_container",
    "start_container",
]
