"""The Podman quadlet writer.

A quadlet is a systemd unit that describes a container, so this writer
translates the engine-neutral server a trial actually ran into the unit a user
can drop into their systemd user directory. It reads only the planned
:class:`~llm_config_optimization.inference.types.ServerSpec`, which already
carries the image, command, mounts, devices, and resource settings the backend
chose, so no engine vocabulary enters this layer.

The unit is deliberately minimal: a description, the container section with a
curl-based healthcheck derived from the spec's health endpoint, and a restart
policy. It has no ``[Install]`` section, because a benchmark artifact must not
become a service that starts at boot.
"""

import json
import shlex
from collections.abc import Mapping

from ..runtime.types import Mount
from .types import RenderContext

__all__ = ["QuadletWriter"]

_SAFE_ENVIRONMENT_VALUE = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./:@,+="
)
"""Characters that need no quoting inside an ``Environment=`` value.

Anything else — whitespace, quotes, or a shell metacharacter — is wrapped in
double quotes so the unit's value survives both systemd's parser and Podman's.
"""

_HEALTH_INTERVAL = "10s"
"""The fixed healthcheck probing interval, so generated units stay deterministic."""

_HEALTH_RETRIES = "3"
"""The fixed number of consecutive probe failures before a container is unhealthy."""


class QuadletWriter:
    """Render a planned server as a Podman ``.container`` quadlet.

    The format reproduces a container, so it requires a plan: the recommended
    configuration alone does not carry the engine command, environment, or host
    mounts the trial ran with.
    """

    requires_plan = True

    def render(self, context: RenderContext) -> str:
        """Render ``context``'s planned server as a minimal quadlet unit."""
        spec = context.spec
        if spec is None:
            raise ValueError("the quadlet writer requires a planned server spec")

        container = spec.container
        lines = ["[Unit]", "Description=Benchmarked inference server", "", "[Container]"]
        lines.append(f"Image={container.image}")

        if container.command:
            lines.append(f"Exec={shlex.join(container.command)}")
        if container.entrypoint is not None:
            lines.append(f"Entrypoint={_entrypoint(container.entrypoint)}")

        for name, value in container.env.items():
            lines.append(f"Environment={name}={_environment_value(value)}")

        for mount in container.volumes:
            lines.append(f"Volume={_volume(mount)}")

        for container_port, host_port in container.ports.items():
            lines.append(f"PublishPort={_published_port(container_port, host_port)}")

        for device in container.devices:
            lines.append(f"AddDevice={device}")

        if container.mem_limit is not None:
            lines.append(f"Memory={container.mem_limit}")
        if container.shm_size is not None:
            lines.append(f"ShmSize={container.shm_size}")

        lines.append(f"Pull={container.pull_policy}")
        lines.extend(_selinux(container.selinux_opts))
        lines.extend(_healthcheck(spec.health_path, spec.health_timeout, container.ports))

        lines.extend(["", "[Service]", "Restart=on-failure", ""])
        return "\n".join(lines)


def _entrypoint(arguments: tuple[str, ...]) -> str:
    """Render an entrypoint as the JSON string quadlet requires.

    Quadlet forwards ``Entrypoint=`` as a single ``--entrypoint=<value>``
    argument, and Podman only splits a multi-option entrypoint when the value is
    a JSON array. A shell-joined value would become one executable name instead.
    """
    return json.dumps(list(arguments), separators=(",", ":"))


def _environment_value(value: str) -> str:
    """Quote an environment value only when its characters require it."""
    if all(character in _SAFE_ENVIRONMENT_VALUE for character in value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _volume(mount: Mount) -> str:
    """Render a bind mount in quadlet's ``source:target[:options]`` form."""
    options = list(mount.options)
    if mount.read_only and "ro" not in options:
        options.insert(0, "ro")
    rendered = f"{mount.source}:{mount.target}"
    if options:
        rendered = f"{rendered}:{','.join(options)}"
    return rendered


def _published_port(container_port: str, host_port: int | None) -> str:
    """Render a port mapping, defaulting the host port to the container port.

    A trial asks the runtime for an unused host port, so the plan carries
    ``None``. A generated artifact must instead be self-contained and
    predictable, so the container port itself becomes the host port.
    """
    container_number = container_port.split("/", 1)[0]
    host = container_number if host_port is None else str(host_port)
    return f"{host}:{container_port}"


def _selinux(opts: tuple[str, ...] | None) -> list[str]:
    """Map the plan's SELinux label options onto quadlet's native keys.

    ``disable`` has a dedicated ``SecurityLabelDisable=`` key. Every other
    option is passed through as a ``--security-opt label=...`` Podman argument,
    so no label configuration is silently dropped.
    """
    lines: list[str] = []
    for option in opts or ():
        if option == "disable":
            lines.append("SecurityLabelDisable=true")
        else:
            lines.append(f"PodmanArgs=--security-opt label={option}")
    return lines


def _healthcheck(
    health_path: str,
    health_timeout: float,
    ports: Mapping[str, int | None],
) -> list[str]:
    """Render the curl-based healthcheck against the server's container port.

    The start period is the configured startup timeout, so a slow model load is
    not marked unhealthy while it is still starting. The probe asks the engine's
    real health path, so readiness means it is serving rather than merely
    accepting a socket.
    """
    port_keys = list(ports)
    if not port_keys:
        return []
    container_port = str(port_keys[0]).split("/", 1)[0]
    url = f"http://127.0.0.1:{container_port}{health_path}"
    return [
        f"HealthCmd=curl -fsS {url}",
        f"HealthInterval={_HEALTH_INTERVAL}",
        f"HealthRetries={_HEALTH_RETRIES}",
        f"HealthStartPeriod={_seconds(health_timeout)}",
    ]


def _seconds(value: float) -> str:
    """Render a timeout in seconds as a systemd duration without trailing zeros."""
    return f"{value:g}s"
