"""Integration tests for the Podman container runner.

These run against a real Podman daemon and a small Alpine image, confirming the
exit code and stdout/stderr separation end to end. They are marked
``integration`` and deselected by default; run them with
``uv run pytest -m integration``.

The whole module skips when Podman or its socket is unavailable. The image is
pulled on demand and the tests skip if it cannot be obtained (for example when
the registry is unreachable).
"""

import time

import pytest
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.runtime import (
    ContainerConfig,
    ContainerRunError,
    run_container,
    start_container,
)

pytestmark = pytest.mark.integration

ALPINE_IMAGE = "docker.io/library/alpine:latest"

_SHM_SIZE = "128m"
"""The container's ``/dev/shm`` size, sized well above the allocation below."""

_ALLOCATED_MIB = 64
"""Anonymous shared memory the measured container holds, in MiB.

A tmpfs file is shmem, so the cgroup is charged for it and the kernel cannot
reclaim it while the file exists: the figure the probe must report is a floor
rather than a hint. Anonymous rather than page-cache-backed memory is what makes
the assertion hold whether or not the host pages anything out.
"""

_HOST_WIDE_MIB = 1024
"""An upper bound a per-container figure stays under.

A host-wide memory reading is the realistic wrong answer for this seam, and on
any machine that runs this suite it sits far above one container's charge. The
bound is deliberately loose so only a category error trips it.
"""


@pytest.fixture(scope="module")
def podman_client():
    """A client for the local Podman daemon, or skip when it is unreachable."""
    try:
        client = PodmanClient.from_env()
    except (APIError, OSError) as exc:
        pytest.skip(f"Podman is unavailable: {exc}")
    try:
        try:
            client.ping()
        except (APIError, OSError) as exc:
            pytest.skip(f"Podman is unavailable: {exc}")
        yield client
    finally:
        client.close()


@pytest.fixture(scope="module")
def alpine_image(podman_client: PodmanClient) -> str:
    """The Alpine image, pulled on demand, or skip when it cannot be obtained."""
    try:
        podman_client.images.get(ALPINE_IMAGE)
    except APIError:
        try:
            podman_client.images.pull(ALPINE_IMAGE)
        except (APIError, OSError) as exc:
            pytest.skip(f"{ALPINE_IMAGE} is unavailable and could not be pulled: {exc}")
    return ALPINE_IMAGE


def test_success_separates_stdout_and_stderr(
    podman_client: PodmanClient, alpine_image: str
) -> None:
    result = run_container(
        ContainerConfig(
            image=alpine_image,
            command=("/bin/sh", "-c", "echo to-stdout; echo to-stderr >&2"),
        ),
        client=podman_client,
    )

    assert result.exit_code == 0
    assert "to-stdout" in result.stdout_text
    assert "to-stderr" not in result.stdout_text
    assert "to-stderr" in result.stderr_text
    assert "to-stdout" not in result.stderr_text
    assert result.container_id


def test_non_zero_exit_raises_with_captured_streams(
    podman_client: PodmanClient, alpine_image: str
) -> None:
    with pytest.raises(ContainerRunError) as excinfo:
        run_container(
            ContainerConfig(
                image=alpine_image,
                command=("/bin/sh", "-c", "echo to-stdout; echo to-stderr >&2; exit 7"),
            ),
            client=podman_client,
        )

    error = excinfo.value
    assert error.timed_out is False
    assert error.result is not None
    assert error.result.exit_code == 7
    assert "to-stdout" in error.result.stdout_text
    assert "to-stderr" in error.result.stderr_text


def test_detached_liveness_sees_a_short_lived_container_exit(
    podman_client: PodmanClient, alpine_image: str
) -> None:
    handle = start_container(
        ContainerConfig(image=alpine_image, command=("/bin/sh", "-c", "exit 0")),
        client=podman_client,
    )
    try:
        deadline = time.monotonic() + 30.0
        while handle.is_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert handle.is_running() is False
    finally:
        handle.remove()


def test_detached_memory_usage_reports_the_live_cgroup_charge(
    podman_client: PodmanClient, alpine_image: str
) -> None:
    """A real container's stats payload parses to its own cgroup figure.

    The container allocates a known amount of anonymous shared memory and then
    sleeps, so the charge is held for as long as the measurement takes. Polling
    tolerates the runtime settling; the assertion is the reported figure, not
    the time it took to appear.
    """
    handle = start_container(
        ContainerConfig(
            image=alpine_image,
            command=(
                "/bin/sh",
                "-c",
                f"dd if=/dev/zero of=/dev/shm/blob bs=1M count={_ALLOCATED_MIB} "
                "2>/dev/null && sleep 60",
            ),
            shm_size=_SHM_SIZE,
        ),
        client=podman_client,
    )
    try:
        floor = _ALLOCATED_MIB * 1024 * 1024
        deadline = time.monotonic() + 30.0
        usage = handle.memory_usage()
        while (usage is None or usage < floor) and time.monotonic() < deadline:
            time.sleep(0.2)
            usage = handle.memory_usage()

        assert usage is not None
        assert floor <= usage < _HOST_WIDE_MIB * 1024 * 1024
    finally:
        handle.remove()


def test_detached_memory_usage_is_unknown_once_the_container_is_gone(
    podman_client: PodmanClient, alpine_image: str
) -> None:
    """A stats read for a removed container reports unknown instead of raising.

    This is the last read a sampler makes if it is stopped after the container:
    it must not turn a finished trial into a failed one.
    """
    handle = start_container(
        ContainerConfig(image=alpine_image, command=("/bin/sh", "-c", "exit 0")),
        client=podman_client,
    )
    handle.stop(timeout=10.0)

    assert handle.memory_usage() is None
