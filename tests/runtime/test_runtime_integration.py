"""Integration tests for the Podman container runner.

These run against a real Podman daemon and a small Alpine image, confirming the
exit code and stdout/stderr separation end to end. They are marked
``integration`` and deselected by default; run them with
``uv run pytest -m integration``.

The whole module skips when Podman or its socket is unavailable. The image is
pulled on demand and the tests skip if it cannot be obtained (for example when
the registry is unreachable).
"""

import pytest
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.runtime import ContainerConfig, ContainerRunError, run_container

pytestmark = pytest.mark.integration

ALPINE_IMAGE = "docker.io/library/alpine:latest"


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
