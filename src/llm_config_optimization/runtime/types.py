"""Backend-agnostic configuration, result, and error types for container runs.

These types deliberately avoid importing any container backend so that callers
and future backends can depend on them without pulling in Podman (or Docker, or
Apptainer) bindings.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

PullPolicy = Literal["missing", "always", "never", "newer"]


@dataclass(frozen=True, slots=True)
class Mount:
    """A host directory bind-mounted into the container."""

    source: str
    target: str
    read_only: bool = False
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContainerConfig:
    """Declarative description of a container run.

    ``entrypoint`` overrides the image's own ``ENTRYPOINT`` when set; ``None``
    keeps it, so the image decides how its command is invoked. It is separate
    from ``command`` because a container runtime applies the two independently:
    ``command`` replaces the image's ``CMD`` while the entrypoint still
    prefixes it.

    ``selinux_opts`` carries Podman SELinux label options (for example
    ``disable``, without the CLI's ``label=`` prefix). ``None`` leaves the
    choice to the runtime's automatic default; an empty tuple asks for no label
    options and a populated tuple asks for exactly those.

    ``ports`` maps container port specifications to host ports. A key is either
    a bare number (``"8000"``) or ``port/protocol`` (``"8000/tcp"``); a value
    is the host port to publish on, or ``None`` to ask the runtime for an
    unused one. An empty mapping publishes nothing, so a blocking run is
    unaffected.
    """

    image: str
    command: tuple[str, ...] = ()
    entrypoint: tuple[str, ...] | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: tuple[Mount, ...] = ()
    devices: tuple[str, ...] = ()
    ports: Mapping[str, int | None] = field(default_factory=dict)
    mem_limit: int | str | None = None
    shm_size: int | str | None = None
    selinux_opts: tuple[str, ...] | None = None
    pull_policy: PullPolicy = "missing"
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Store ``env`` and ``ports`` as read-only views so the config is immutable."""
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))

    def __hash__(self) -> int:
        """Hash the config, ordering the mappings so equal configs hash equally."""
        return hash(
            (
                self.image,
                self.command,
                self.entrypoint,
                tuple(sorted(self.env.items())),
                self.volumes,
                self.devices,
                tuple(sorted(self.ports.items())),
                self.mem_limit,
                self.shm_size,
                self.selinux_opts,
                self.pull_policy,
                self.timeout,
            )
        )


@dataclass(frozen=True, slots=True)
class ContainerResult:
    """Outcome of a container run, with output kept as raw bytes."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    container_id: str | None = None

    @property
    def stdout_text(self) -> str:
        """Decode stdout as UTF-8, replacing undecodable bytes."""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        """Decode stderr as UTF-8, replacing undecodable bytes."""
        return self.stderr.decode("utf-8", errors="replace")


class RunningContainer(Protocol):
    """Handle to a container started without waiting for it to exit.

    A detached run cannot use :func:`run_container`, because that helper owns
    the container for the whole blocking call. Instead the backend starts the
    container, reads back the host ports the runtime assigned, and returns this
    handle, which the caller drives until it is done.

    ``stop`` finishes the run: it stops the container, captures the logs as a
    :class:`ContainerResult`, and removes the container. ``remove`` is the
    cleanup path for a run that should not be stopped gracefully (for example
    when a caller raised before reaching ``stop``).
    """

    @property
    def container_id(self) -> str:
        """The runtime's identifier for the running container."""
        ...

    @property
    def ports(self) -> Mapping[str, int]:
        """Resolved host ports, keyed by the container specification asked for."""
        ...

    def logs(self) -> tuple[bytes, bytes]:
        """Read the container's current logs as ``(stdout, stderr)``."""
        ...

    def is_running(self) -> bool:
        """Report whether the container is still running.

        The health wait polls this alongside the server's health endpoint, so a
        container that crashed during startup — a model-load failure, an
        out-of-memory kill — ends the run as soon as it exits rather than
        waiting out the whole startup timeout. A runtime that cannot be queried
        reports the container as running, leaving the timeout as the backstop.
        """
        ...

    def memory_usage(self) -> int | None:
        """Report the container's current cgroup memory usage in bytes.

        The figure includes page cache, so file-backed pages such as an
        mmap'd model count as memory the container holds. The read is a
        best-effort diagnostic: a runtime that cannot report stats, a
        rejected request, or an unreadable payload reports ``None`` rather
        than raising or failing the run.
        """
        ...

    def stop(self, timeout: float | None = None) -> ContainerResult:
        """Stop the container, capture its result, and remove it.

        ``timeout`` is the grace period before the runtime kills the container.
        A runtime whose stop request takes whole seconds rounds it up, so the
        container is never stopped earlier than the caller asked for; ``None``
        leaves the runtime's own default in place.
        """
        ...

    def remove(self) -> None:
        """Force-remove the container without capturing a result."""
        ...


class ContainerRunError(RuntimeError):
    """Raised for every container-run failure mode.

    Covers non-zero exits, timeouts, and backend API errors. The partial
    ``result`` is attached when output was captured, and ``timed_out``
    distinguishes a timeout from an ordinary failure. The originating backend
    exception, when there is one, is preserved as ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        result: ContainerResult | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.timed_out = timed_out
