"""The Docker/Podman Compose writer.

A Compose file describes the same container a trial ran, so this writer reads
the engine-neutral :class:`~llm_config_optimization.inference.types.ServerSpec`
the backend planned and translates it into a modern single-service document.
Like the quadlet writer it knows no engine vocabulary: the image, command,
mounts, devices, and resource settings all arrive on the plan.

The document omits the obsolete ``version:`` key and keys its only service
``server``, so the file stays engine-neutral and easy to rename. The one
non-faithful value is Podman's ``newer`` pull policy, which Compose does not
understand: it is written as ``always`` with an adjacent comment explaining the
approximation.
"""

from collections.abc import Mapping

import yaml

from ..runtime.types import Mount
from .types import RenderContext

__all__ = ["ComposeWriter"]

_HEALTH_INTERVAL = "10s"
"""The fixed healthcheck probing interval, so generated files stay deterministic."""

_HEALTH_RETRIES = 3
"""The fixed number of consecutive probe failures before a container is unhealthy."""

_PULL_POLICY_NOTE = (
    '    # Podman\'s "newer" pull policy has no Compose equivalent; "always" keeps',
    "    # the same registry check without Podman's local image comparison.",
)
"""The one comment a generated file carries, explaining the pull-policy translation."""


class ComposeWriter:
    """Render a planned server as a modern Docker/Podman Compose document.

    The format reproduces a container, so it requires a plan: the recommended
    configuration alone does not carry the engine command, environment, or host
    mounts the trial ran with.
    """

    requires_plan = True

    def render(self, context: RenderContext) -> str:
        """Render ``context``'s planned server as a Compose file."""
        spec = context.spec
        if spec is None:
            raise ValueError("the compose writer requires a planned server spec")

        container = spec.container
        lines = ["services:", "  server:", f"    image: {_scalar(container.image)}"]

        if container.command:
            lines.append("    command:")
            lines.extend(f"      - {_scalar(argument)}" for argument in container.command)
        if container.entrypoint is not None:
            lines.append("    entrypoint:")
            lines.extend(f"      - {_scalar(argument)}" for argument in container.entrypoint)

        if container.env:
            lines.append("    environment:")
            lines.extend(
                f"      {_scalar(name)}: {_scalar(value)}" for name, value in container.env.items()
            )

        if container.volumes:
            lines.append("    volumes:")
            lines.extend(f"      - {_scalar(_volume(mount))}" for mount in container.volumes)

        if container.ports:
            lines.append("    ports:")
            lines.extend(
                f"      - {_scalar(_published_port(container_port, host_port))}"
                for container_port, host_port in container.ports.items()
            )

        if container.devices:
            lines.append("    devices:")
            lines.extend(f"      - {_scalar(device)}" for device in container.devices)

        if container.shm_size is not None:
            lines.append(f"    shm_size: {_scalar(str(container.shm_size))}")
        if container.mem_limit is not None:
            lines.append(f"    mem_limit: {_scalar(str(container.mem_limit))}")

        if container.selinux_opts:
            lines.append("    security_opt:")
            lines.extend(
                f"      - {_scalar(f'label={option}')}" for option in container.selinux_opts
            )

        pull_policy = "always" if container.pull_policy == "newer" else container.pull_policy
        if pull_policy != container.pull_policy:
            lines.extend(_PULL_POLICY_NOTE)
        lines.append(f"    pull_policy: {_scalar(pull_policy)}")

        lines.extend(_healthcheck(spec.health_path, spec.health_timeout, container.ports))

        return "\n".join([*lines, ""])


def _volume(mount: Mount) -> str:
    """Render a bind mount in Compose's ``source:target[:options]`` form."""
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
        "    healthcheck:",
        "      test:",
        "        - CMD",
        "        - curl",
        "        - -fsS",
        f"        - {_scalar(url)}",
        f"      interval: {_HEALTH_INTERVAL}",
        f"      retries: {_HEALTH_RETRIES}",
        f"      start_period: {_seconds(health_timeout)}",
    ]


def _scalar(value: str) -> str:
    """Render a string as a YAML scalar, quoted so it stays a string."""
    return yaml.safe_dump(value, default_style='"', width=0).rstrip("\n")


def _seconds(value: float) -> str:
    """Render a timeout in seconds as a Compose duration without trailing zeros."""
    return f"{value:g}s"
