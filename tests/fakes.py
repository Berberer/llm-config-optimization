"""Shared test doubles for the runtime, inference, and optimization test packages.

``FakePodmanClient`` is a hand-written stand-in for the surface of
``podman.PodmanClient`` that :func:`llm_optimization.runtime.run_container` uses,
so tests can drive the runner's observable behaviour without a daemon.
``FakeBackend`` is a recording stand-in for an
:class:`~llm_optimization.inference.types.InferenceBackend` that returns canned
plans and results, so the inference and optimization layers can be driven
without an engine or GPU. Both live beside the root ``conftest.py`` so every
test package can import them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from podman.errors import APIError

from llm_optimization.inference import DetectedHardware, ExperimentConfig, ExperimentResult
from llm_optimization.runtime import ContainerConfig, ContainerResult
from llm_optimization.search_space import ExperimentSpace


def _frame(stream_type: int, payload: bytes) -> bytes:
    """Encode one frame of Podman's multiplexed (non-TTY) log stream."""
    return bytes([stream_type, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


@dataclass
class FakeResponse:
    """Stand-in for the ``requests.Response`` returned by ``client.get``."""

    content: bytes

    def raise_for_status(self) -> None:
        return None


class FakeContainer:
    """Stand-in for ``podman.domain.containers.Container``."""

    def __init__(
        self,
        client: FakePodmanClient,
        *,
        container_id: str,
        exit_code: int,
        wait_error: APIError | None,
        start_error: APIError | None,
    ) -> None:
        self.client = client
        self.id = container_id
        self._exit_code = exit_code
        self._wait_error = wait_error
        self._start_error = start_error
        self.started = False
        self.killed = False
        self.wait_calls: list[float | None] = []
        self.remove_calls: list[bool] = []

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self._wait_error is not None:
            raise self._wait_error
        return self._exit_code

    def kill(self) -> None:
        self.killed = True

    def remove(self, force: bool = False) -> None:
        self.remove_calls.append(force)


class FakeImageManager:
    """Records ``images.pull`` calls and optionally raises a canned error."""

    def __init__(self, client: FakePodmanClient) -> None:
        self._client = client

    def pull(self, image: str, policy: str = "missing") -> None:
        self._client.pull_calls.append((image, policy))
        if self._client.pull_error is not None:
            raise self._client.pull_error


class FakeContainerManager:
    """Records ``containers.create`` calls and returns a canned container."""

    def __init__(self, client: FakePodmanClient) -> None:
        self._client = client

    def create(self, image: str, **kwargs: object) -> FakeContainer:
        self._client.create_calls.append((image, kwargs))
        if self._client.create_error is not None:
            raise self._client.create_error
        container = FakeContainer(
            self._client,
            container_id=self._client.container_id,
            exit_code=self._client.exit_code,
            wait_error=self._client.wait_error,
            start_error=self._client.start_error,
        )
        self._client.container = container
        return container


class FakePodmanClient:
    """Minimal fake of the ``podman.PodmanClient`` surface the runner uses."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        logs: bytes = b"",
        container_id: str = "fake-container-id",
        wait_error: APIError | None = None,
        start_error: APIError | None = None,
        create_error: APIError | None = None,
        pull_error: APIError | None = None,
        get_error: APIError | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.logs = logs
        self.container_id = container_id
        self.wait_error = wait_error
        self.start_error = start_error
        self.create_error = create_error
        self.pull_error = pull_error
        self.get_error = get_error
        self.images = FakeImageManager(self)
        self.containers = FakeContainerManager(self)
        self.pull_calls: list[tuple[str, str]] = []
        self.create_calls: list[tuple[str, dict[str, object]]] = []
        self.get_calls: list[tuple[str, dict[str, object] | None]] = []
        self.container: FakeContainer | None = None
        self.closed = False

    def get(self, path: str, params: dict[str, object] | None = None) -> FakeResponse:
        self.get_calls.append((path, params))
        if self.get_error is not None:
            raise self.get_error
        return FakeResponse(self.logs)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakePodmanClient:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


class FakeBackend:
    """A recording stand-in for an inference backend.

    It returns a canned container plan and a canned typed result, so an
    experiment can be driven end to end without an engine: :meth:`plan` records
    every experiment it is asked to plan and :meth:`parse_result` records the
    captured container outcome, while ``parse_error`` makes every parse fail.
    The inference orchestration tests and the optimization lifecycle tests share
    it so neither needs a daemon or a GPU.
    """

    def __init__(
        self,
        plan_config: ContainerConfig,
        result: ExperimentResult,
        events: list[str],
        *,
        parse_error: Exception | None = None,
    ) -> None:
        self.plan_config = plan_config
        self.result = result
        self.events = events
        self.parse_error = parse_error
        self.plan_calls: list[tuple[ExperimentConfig, DetectedHardware, Path | None]] = []
        self.parse_calls: list[tuple[ContainerResult, ContainerConfig, Path | None]] = []

    def validate(self, space: ExperimentSpace) -> Sequence[str]:
        self.events.append("validate")
        return ()

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        output_dir: Path | None = None,
    ) -> ContainerConfig:
        self.events.append("plan")
        self.plan_calls.append((experiment, hardware, output_dir))
        return self.plan_config

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        self.events.append("parse_result")
        self.parse_calls.append((result, plan, output_dir))
        if self.parse_error is not None:
            raise self.parse_error
        return self.result
