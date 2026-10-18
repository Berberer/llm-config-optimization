"""Shared test doubles for the runtime, inference, and optimization test packages.

``FakePodmanClient`` is a hand-written stand-in for the surface of
``podman.PodmanClient`` that :func:`llm_config_optimization.runtime.run_container` uses,
so tests can drive the runner's observable behaviour without a daemon.
``FakeBackend`` is a recording stand-in for an
:class:`~llm_config_optimization.inference.types.InferenceBackend` that returns canned
plans and results, so the inference and optimization layers can be driven
without an engine or GPU. Both live beside the root ``conftest.py`` so every
test package can import them.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from podman.errors import APIError

from llm_config_optimization.inference import (
    BenchmarkConfig,
    BenchmarkOutcome,
    DetectedHardware,
    ExperimentConfig,
    ExperimentResult,
    HardwareTarget,
    HubCheckpoint,
    LocalCheckpoint,
    ModelRequest,
    ServerSpec,
)
from llm_config_optimization.runtime import ContainerResult
from llm_config_optimization.search_space import TUNABLE_PARAMETERS, ExperimentSpace


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
        self.stopped = False
        self.killed = False
        self.ports: dict[str, object] = {}
        self.wait_calls: list[float | None] = []
        self.remove_calls: list[bool] = []
        self.stop_calls: list[dict[str, object]] = []
        self.reload_calls = 0

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    def stop(self, **kwargs: object) -> None:
        self.stop_calls.append(kwargs)
        if self.client.stop_error is not None:
            raise self.client.stop_error
        self.stopped = True

    def reload(self) -> None:
        self.reload_calls += 1
        if self.client.reload_error is not None:
            raise self.client.reload_error
        self.ports = dict(self.client.resolved_ports)

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
        reload_error: APIError | None = None,
        stop_error: APIError | None = None,
        resolved_ports: dict[str, object] | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.logs = logs
        self.container_id = container_id
        self.wait_error = wait_error
        self.start_error = start_error
        self.create_error = create_error
        self.pull_error = pull_error
        self.get_error = get_error
        self.reload_error = reload_error
        self.stop_error = stop_error
        self.resolved_ports = (
            resolved_ports
            if resolved_ports is not None
            else {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8000"}]}
        )
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

    It returns a canned server plan and a canned typed result, so an experiment
    can be driven end to end without an engine: :meth:`plan` records every
    experiment it is asked to plan and :meth:`parse_result` records the measured
    outcome and captured container outcome, while ``parse_error`` makes every
    parse fail. The inference orchestration tests and the optimization lifecycle
    tests share it so neither needs a daemon or a GPU.
    """

    def __init__(
        self,
        plan_config: ServerSpec,
        result: ExperimentResult,
        events: list[str],
        *,
        parse_error: Exception | None = None,
        applicable_parameters: Iterable[str] | None = None,
    ) -> None:
        self.plan_config = plan_config
        self.result = result
        self.events = events
        self.parse_error = parse_error
        self._applicable_parameters = (
            frozenset(TUNABLE_PARAMETERS)
            if applicable_parameters is None
            else frozenset(applicable_parameters)
        )
        self.plan_calls: list[tuple[ExperimentConfig, DetectedHardware]] = []
        self.parse_calls: list[tuple[BenchmarkOutcome, ServerSpec, ContainerResult]] = []
        self.model_request_calls: list[str] = []

    @property
    def applicable_parameters(self) -> frozenset[str]:
        """The tunable names the fake declares, every registered one by default."""
        return self._applicable_parameters

    def validate(
        self, space: ExperimentSpace, hardware: DetectedHardware | None = None
    ) -> Sequence[str]:
        self.events.append("validate")
        return ()

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> Sequence[str]:
        self.events.append("validate_model")
        return ()

    def model_request(self, model: str) -> ModelRequest:
        """Resolve an existing host path as local, anything else as a Hub id.

        Records every model it is asked about, so a test can pin that the CLI
        consulted the backend's resolution hook rather than deciding itself.
        """
        self.model_request_calls.append(model)
        path = Path(model).expanduser()
        if path.exists():
            return LocalCheckpoint(path=path.resolve())
        return HubCheckpoint(repo_id=model)

    def image_for(self, target: HardwareTarget) -> str:
        return self.plan_config.container.image

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
    ) -> ServerSpec:
        self.events.append("plan")
        self.plan_calls.append((experiment, hardware))
        return self.plan_config

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        self.events.append("parse_result")
        self.parse_calls.append((outcome, spec, result))
        if self.parse_error is not None:
            raise self.parse_error
        return self.result


class FakeBenchmarkClient:
    """A recording stand-in for the host benchmark client.

    It returns a canned :class:`BenchmarkOutcome` and records every server it is
    asked to measure, so the orchestrator's lifecycle can be driven without a
    server, a daemon, or a GPU. ``measure_error`` makes every measurement fail.
    """

    def __init__(
        self,
        outcome: BenchmarkOutcome | None = None,
        *,
        measure_error: Exception | None = None,
    ) -> None:
        self.outcome = outcome if outcome is not None else BenchmarkOutcome(runs=(), failed_runs=0)
        self.measure_error = measure_error
        self.calls: list[tuple[str, str, ServerSpec, BenchmarkConfig]] = []

    def measure_sync(
        self,
        base_url: str,
        model: str,
        spec: ServerSpec,
        benchmark: BenchmarkConfig,
        *,
        concurrency: int | None = None,
    ) -> BenchmarkOutcome:
        self.calls.append((base_url, model, spec, benchmark))
        if self.measure_error is not None:
            raise self.measure_error
        return self.outcome
