"""Podman backend for :func:`run_container`.

This module is the only place that imports the ``podman`` bindings.
"""

import contextlib

from podman import PodmanClient
from podman.domain.containers import Container
from podman.errors import APIError

from ._selinux import selinux_enforcing
from .types import ContainerConfig, ContainerResult, ContainerRunError, Mount


def _volume_specs(volumes: tuple[Mount, ...]) -> dict[str, dict[str, object]]:
    """Translate :class:`Mount` entries into podman-py's ``volumes`` mapping.

    The ``volumes`` mapping is used rather than ``mounts`` because podman-py
    5.8 rebuilds every ``mounts`` entry from the options it recognises and
    silently drops the rest, while ``volumes`` forwards ``extended_mode``
    entries verbatim. A host path that does not look like a named volume is
    turned into a bind mount by podman-py.

    The mapping is keyed by ``mount.source``, so at most one mount per source is
    supported: a repeated source silently overwrites the earlier entry.
    """
    specs: dict[str, dict[str, object]] = {}
    for mount in volumes:
        specs[mount.source] = {
            "bind": mount.target,
            "mode": "ro" if mount.read_only else "rw",
            "extended_mode": list(mount.options),
        }
    return specs


def _security_opt(config: ContainerConfig) -> list[str]:
    """Resolve the SELinux label options to pass to Podman for ``config``.

    ``None`` means "choose automatically": disable labeling when the host is
    SELinux-enforcing *and* the run maps devices, and pass no options
    otherwise, so CPU and device-less containers keep their confinement. An
    explicit tuple always wins, so a caller can force confinement on, force it
    off, or pin a specific label.
    """
    opts = config.selinux_opts
    if opts is None:
        opts = ("disable",) if selinux_enforcing() and config.devices else ()
    return list(opts)


def _demux_logs(data: bytes) -> tuple[bytes, bytes]:
    """Split a multiplexed (non-TTY) log stream into stdout and stderr bytes.

    The stream is a sequence of frames, each an 8-byte header followed by its
    payload: the header's first byte is the stream type (1 = stdout, 2 = stderr)
    and its last 4 bytes are the payload length as a big-endian unsigned
    integer. Frames of any other type, and a truncated trailing frame, are
    ignored so that partial logs captured from a killed container still parse.
    """
    stdout = bytearray()
    stderr = bytearray()
    offset = 0
    while offset + 8 <= len(data):
        stream_type = data[offset]
        length = int.from_bytes(data[offset + 4 : offset + 8], "big")
        payload_start = offset + 8
        payload_end = payload_start + length
        if payload_end > len(data):
            break
        payload = data[payload_start:payload_end]
        if stream_type == 1:
            stdout += payload
        elif stream_type == 2:
            stderr += payload
        offset = payload_end
    return bytes(stdout), bytes(stderr)


def _read_logs(container: Container) -> tuple[bytes, bytes]:
    """Fetch the container's complete log stream and split it by stream.

    The high-level ``Container.logs`` helper discards the frame headers, so the
    raw logs endpoint is used here to keep stdout and stderr distinguishable.
    """
    response = container.client.get(
        f"/containers/{container.id}/logs",
        params={"stdout": True, "stderr": True, "tail": "all"},
    )
    response.raise_for_status()
    return _demux_logs(response.content)


def _read_partial_logs(container: Container) -> tuple[bytes, bytes]:
    """Read whatever logs exist, treating a failure as no output.

    Used on the timeout path, where the container may still be mid-write or the
    logs endpoint may be unavailable; a missing log stream must not mask the
    timeout itself.
    """
    try:
        return _read_logs(container)
    except APIError:
        return b"", b""


def _force_remove(container: Container) -> None:
    """Remove ``container``, ignoring failures so cleanup never masks an outcome."""
    with contextlib.suppress(APIError):
        container.remove(force=True)


def run_container(
    config: ContainerConfig, *, client: PodmanClient | None = None
) -> ContainerResult:
    """Run ``config`` to completion and return its captured result.

    The container is created without a TTY (so logs stay multiplexed) and
    without daemon-side auto-removal (so logs survive until they are read), then
    force-removed once the run is over.

    For the ``"always"`` and ``"newer"`` pull policies the image is pulled
    explicitly first, because ``create`` only auto-pulls a missing image and
    would otherwise ignore those policies. ``"missing"`` and ``"never"`` rely on
    ``create``'s own behavior.

    Args:
        config: The declarative description of the run.
        client: An optional pre-configured :class:`PodmanClient`. When omitted, a
            client is discovered from the environment and closed once the run is
            over; an injected client is left open for the caller to manage.

    Raises:
        ContainerRunError: If the container exits non-zero (the result is
            attached), if it does not finish within ``config.timeout``
            (``timed_out`` is set and the partial result is attached), or if
            Podman fails to pull, create, or start it (the original ``APIError``
            is preserved as ``__cause__``).
    """
    if client is not None:
        return _run(config, client)
    with PodmanClient.from_env() as owned_client:
        return _run(config, owned_client)


def _run(config: ContainerConfig, client: PodmanClient) -> ContainerResult:
    """Execute ``config`` against ``client``, which remains owned by the caller."""
    if config.pull_policy in ("always", "newer"):
        try:
            client.images.pull(config.image, policy=config.pull_policy)
        except APIError as exc:
            raise ContainerRunError(
                f"failed to pull image {config.image!r} with policy {config.pull_policy!r}: {exc}"
            ) from exc
    try:
        container = client.containers.create(
            config.image,
            command=list(config.command) or None,
            entrypoint=list(config.entrypoint) if config.entrypoint is not None else None,
            environment=dict(config.env),
            volumes=_volume_specs(config.volumes),
            devices=list(config.devices),
            security_opt=_security_opt(config),
            mem_limit=config.mem_limit,
            shm_size=config.shm_size,
            tty=False,
            auto_remove=False,
        )
    except APIError as exc:
        raise ContainerRunError(
            f"failed to create container from image {config.image!r}: {exc}"
        ) from exc
    try:
        try:
            container.start()
        except APIError as exc:
            raise ContainerRunError(f"failed to start container {container.id}: {exc}") from exc
        try:
            exit_code = container.wait(timeout=config.timeout)
        except APIError as exc:
            if config.timeout is None:
                raise ContainerRunError(
                    f"failed while waiting for container {container.id}: {exc}"
                ) from exc
            stdout, stderr = _read_partial_logs(container)
            with contextlib.suppress(APIError):
                container.kill()
            result = ContainerResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                container_id=container.id,
            )
            raise ContainerRunError(
                f"container {container.id} timed out after {config.timeout} seconds",
                result=result,
                timed_out=True,
            ) from exc
        stdout, stderr = _read_logs(container)
        result = ContainerResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            container_id=container.id,
        )
        if exit_code != 0:
            raise ContainerRunError(
                f"container {container.id} exited with code {exit_code}",
                result=result,
            )
        return result
    finally:
        _force_remove(container)
