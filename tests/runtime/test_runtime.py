"""Unit tests for the Podman container runner, driven by a hand-written fake client.

These tests exercise the runner's observable contract — the calls it makes, the
result it returns, the exception it raises, and the cleanup it performs — without
a running Podman daemon.
"""

import pytest
from podman.errors import APIError

from fakes import FakePodmanClient, _frame
from llm_optimization.runtime import (
    ContainerConfig,
    ContainerResult,
    ContainerRunError,
    Mount,
    run_container,
)


def test_create_maps_config_fields_to_podman_arguments() -> None:
    client = FakePodmanClient()
    config = ContainerConfig(
        image="example/runner:latest",
        command=("python", "-c", "print('hi')"),
        entrypoint=("vllm",),
        env={"FOO": "bar", "BAZ": "qux"},
        volumes=(
            Mount(source="/host/data", target="/data"),
            Mount(source="/host/models", target="/models", read_only=True, options=("z",)),
        ),
        devices=("nvidia.com/gpu=all",),
        mem_limit="4g",
        shm_size="2g",
    )

    run_container(config, client=client)

    assert len(client.create_calls) == 1
    image, kwargs = client.create_calls[0]
    assert image == "example/runner:latest"
    assert kwargs["command"] == ["python", "-c", "print('hi')"]
    assert kwargs["entrypoint"] == ["vllm"]
    assert kwargs["environment"] == {"FOO": "bar", "BAZ": "qux"}
    assert kwargs["volumes"] == {
        "/host/data": {"bind": "/data", "mode": "rw", "extended_mode": []},
        "/host/models": {"bind": "/models", "mode": "ro", "extended_mode": ["z"]},
    }
    assert kwargs["devices"] == ["nvidia.com/gpu=all"]
    assert kwargs["mem_limit"] == "4g"
    assert kwargs["shm_size"] == "2g"
    assert kwargs["tty"] is False
    assert kwargs["auto_remove"] is False


def test_defaults_defer_to_image_entrypoint_and_empty_collections() -> None:
    client = FakePodmanClient()

    run_container(ContainerConfig(image="example/runner:latest"), client=client)

    _, kwargs = client.create_calls[0]
    assert kwargs["command"] is None
    assert kwargs["entrypoint"] is None
    assert kwargs["environment"] == {}
    assert kwargs["volumes"] == {}
    assert kwargs["devices"] == []
    assert kwargs["mem_limit"] is None
    assert kwargs["shm_size"] is None


def test_success_returns_result_and_force_removes_container() -> None:
    client = FakePodmanClient(logs=_frame(1, b"hello\n") + _frame(2, b"warn\n"))

    result = run_container(ContainerConfig(image="img", command=("true",)), client=client)

    assert isinstance(result, ContainerResult)
    assert result.exit_code == 0
    assert result.stdout == b"hello\n"
    assert result.stderr == b"warn\n"
    assert result.stdout_text == "hello\n"
    assert result.stderr_text == "warn\n"
    assert result.container_id == "fake-container-id"
    assert client.container is not None
    assert client.container.started is True
    assert client.container.killed is False
    assert client.container.remove_calls == [True]


def test_demux_separates_streams_and_ignores_unknown_and_truncated_frames() -> None:
    logs = (
        _frame(1, b"out-1")
        + _frame(2, b"err-1")
        + _frame(1, b"out-2")
        + _frame(3, b"ignored")
        + _frame(1, b"truncated")[:-2]
    )
    client = FakePodmanClient(logs=logs)

    result = run_container(ContainerConfig(image="img"), client=client)

    assert result.stdout == b"out-1out-2"
    assert result.stderr == b"err-1"


def test_non_zero_exit_raises_with_result_attached_and_cleans_up() -> None:
    client = FakePodmanClient(exit_code=3, logs=_frame(1, b"boom") + _frame(2, b"trace"))

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img"), client=client)

    error = excinfo.value
    assert error.timed_out is False
    assert error.__cause__ is None
    assert error.result is not None
    assert error.result.exit_code == 3
    assert error.result.stdout == b"boom"
    assert error.result.stderr == b"trace"
    assert client.container is not None
    assert client.container.remove_calls == [True]


def test_timeout_raises_with_partial_result_kill_and_removal() -> None:
    wait_error = APIError("read timed out")
    client = FakePodmanClient(wait_error=wait_error, logs=_frame(1, b"partial"))

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img", timeout=5.0), client=client)

    error = excinfo.value
    assert error.timed_out is True
    assert error.__cause__ is wait_error
    assert error.result is not None
    assert error.result.exit_code == -1
    assert error.result.stdout == b"partial"
    assert client.container is not None
    assert client.container.wait_calls == [5.0]
    assert client.container.killed is True
    assert client.container.remove_calls == [True]


def test_wait_error_without_timeout_is_an_ordinary_error() -> None:
    wait_error = APIError("podman exploded")
    client = FakePodmanClient(wait_error=wait_error)

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img"), client=client)

    assert excinfo.value.timed_out is False
    assert excinfo.value.result is None
    assert excinfo.value.__cause__ is wait_error
    assert client.container is not None
    assert client.container.remove_calls == [True]


@pytest.mark.parametrize("policy", ["always", "newer"])
def test_explicit_pull_policies_pre_pull_the_image(policy: str) -> None:
    client = FakePodmanClient()

    run_container(ContainerConfig(image="img", pull_policy=policy), client=client)

    assert client.pull_calls == [("img", policy)]
    assert len(client.create_calls) == 1


@pytest.mark.parametrize("policy", ["missing", "never"])
def test_implicit_pull_policies_do_not_pre_pull(policy: str) -> None:
    client = FakePodmanClient()

    run_container(ContainerConfig(image="img", pull_policy=policy), client=client)

    assert client.pull_calls == []


def test_pull_failure_is_wrapped_and_create_is_skipped() -> None:
    pull_error = APIError("pull access denied")
    client = FakePodmanClient(pull_error=pull_error)

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img", pull_policy="always"), client=client)

    assert excinfo.value.__cause__ is pull_error
    assert client.create_calls == []


def test_create_failure_is_wrapped_with_original_cause() -> None:
    create_error = APIError("no such image")
    client = FakePodmanClient(create_error=create_error)

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img"), client=client)

    assert excinfo.value.__cause__ is create_error
    assert client.container is None


def test_start_failure_is_wrapped_with_original_cause_and_cleans_up() -> None:
    start_error = APIError("failed to start container")
    client = FakePodmanClient(start_error=start_error)

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img"), client=client)

    assert excinfo.value.__cause__ is start_error
    assert excinfo.value.timed_out is False
    assert excinfo.value.result is None
    assert client.container is not None
    assert client.container.started is False
    assert client.container.remove_calls == [True]


def test_timeout_with_unavailable_logs_still_reports_timeout() -> None:
    wait_error = APIError("read timed out")
    client = FakePodmanClient(wait_error=wait_error, get_error=APIError("logs unavailable"))

    with pytest.raises(ContainerRunError) as excinfo:
        run_container(ContainerConfig(image="img", timeout=5.0), client=client)

    error = excinfo.value
    assert error.timed_out is True
    assert error.__cause__ is wait_error
    assert error.result is not None
    assert error.result.stdout == b""
    assert error.result.stderr == b""
    assert client.container is not None
    assert client.container.killed is True
    assert client.container.remove_calls == [True]


def test_config_env_is_immutable_and_config_is_hashable() -> None:
    config = ContainerConfig(image="img", env={"FOO": "bar", "BAZ": "qux"})

    assert config.env == {"FOO": "bar", "BAZ": "qux"}
    with pytest.raises(TypeError):
        config.env["FOO"] = "changed"
    assert isinstance(hash(config), int)
    assert config == ContainerConfig(image="img", env={"FOO": "bar", "BAZ": "qux"})
    assert hash(config) == hash(ContainerConfig(image="img", env={"BAZ": "qux", "FOO": "bar"}))


def test_injected_client_is_left_open() -> None:
    client = FakePodmanClient()

    run_container(ContainerConfig(image="img"), client=client)

    assert client.closed is False


def test_self_created_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakePodmanClient] = []

    class FromEnvClient(FakePodmanClient):
        @classmethod
        def from_env(cls):
            instance = cls()
            created.append(instance)
            return instance

    monkeypatch.setattr("llm_optimization.runtime._podman.PodmanClient", FromEnvClient)

    run_container(ContainerConfig(image="img"))

    assert len(created) == 1
    assert created[0].closed is True
