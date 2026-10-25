"""Helpers for integration tests that shell out to real artifact tooling.

The default suite is hermetic, so only ``integration``-marked tests import this
module. Every helper locates a binary first and skips the calling test when it is
not installed, which keeps a host with partial tooling runnable. Commands always
run with captured output so a failure can be reported instead of leaking to the
test log, and environment overrides are merged over the current environment
rather than replacing it.
"""

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

__all__ = ["docker_is_podman_backed", "find_tool", "require_tool", "run"]


def find_tool(*names: str) -> Path | None:
    """Return the first executable named by ``names``, or ``None``.

    A name may be a bare command found on ``PATH`` or an absolute path such as
    the quadlet generator installed outside ``PATH``.
    """
    for name in names:
        located = shutil.which(name)
        if located is not None:
            return Path(located)
        candidate = Path(name)
        if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def require_tool(*names: str, purpose: str) -> Path:
    """Return the first available tool, or skip because ``purpose`` needs it."""
    tool = find_tool(*names)
    if tool is None:
        searched = " or ".join(names)
        pytest.skip(f"{purpose} requires {searched}, which is not installed")
    return tool


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` with captured text output and optional environment overrides."""
    merged = None if env is None else {**os.environ, **env}
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=merged,
        capture_output=True,
        text=True,
    )


def docker_is_podman_backed(docker: Path) -> bool:
    """Whether the ``docker`` CLI on this host is Podman's emulation.

    Podman's Docker-compatible shim answers Podman-specific ``docker info``
    fields, while a real Docker daemon cannot render ``.Host.BuildahVersion``.
    The CDI compatibility matrix needs the distinction: Podman-backed Docker
    accepts CDI device strings, while real Docker Compose does not.
    """
    result = run([docker, "info", "--format", "{{.Host.BuildahVersion}}"])
    return result.returncode == 0 and bool(result.stdout.strip())
