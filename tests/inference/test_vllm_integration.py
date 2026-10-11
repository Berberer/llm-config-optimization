"""Integration tests for the vLLM inference backend.

These run a real benchmark end to end: the layer detects the host, plans a
container run, executes it through Podman against vLLM's official CPU image,
and parses the engine's result JSON. The experiment forces the CPU target and
uses a sub-1B model, so no accelerator is assumed and the suite is runnable on
any machine with Podman and network access.

The tests are marked ``integration`` and deselected by default; run them with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the vLLM CPU image cannot be obtained, or when the model is neither
cached nor downloadable. The first run pulls a multi-gigabyte image and
downloads the model into the mounted Hugging Face cache, so it is slow.
"""

import socket
from pathlib import Path

import pytest
from podman import PodmanClient
from podman.errors import APIError

from llm_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    detect_hardware,
    run_experiment,
)

pytestmark = pytest.mark.integration

VLLM_CPU_IMAGE = "docker.io/vllm/vllm-openai-cpu:latest"
MODEL = "facebook/opt-125m"

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_RUN_TIMEOUT_SECONDS = 900.0
_RESULT_FILENAME = "result.json"

# opt-125m ships float16 weights, whose support on torch CPU is unstable; the
# CPU backend asks for an explicit portable dtype instead. On CPU the
# ``gpu_memory_utilization`` field controls the fraction of system memory the
# engine reserves (despite its name), and its 0.92 default exceeds this host.
_CPU_DTYPE = DType.FLOAT32
_CPU_MEMORY_UTILIZATION = 0.5

_MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--opt-125m"


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
def vllm_cpu_image(podman_client: PodmanClient) -> str:
    """The vLLM CPU image, pulled on demand, or skip when it cannot be obtained."""
    try:
        podman_client.images.get(VLLM_CPU_IMAGE)
    except APIError:
        try:
            podman_client.images.pull(VLLM_CPU_IMAGE)
        except (APIError, OSError) as exc:
            pytest.skip(f"{VLLM_CPU_IMAGE} is unavailable and could not be pulled: {exc}")
    return VLLM_CPU_IMAGE


@pytest.fixture(scope="module")
def model(vllm_cpu_image: str) -> str:
    """The model under test, or skip when it is neither cached nor downloadable."""
    if _MODEL_CACHE.is_dir():
        return MODEL
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError as exc:
        pytest.skip(f"{MODEL} is not cached and huggingface.co is unreachable: {exc}")
    return MODEL


def test_hardware_detection_resolves_the_real_host() -> None:
    """The real vendor probes return a target and a usable device count."""
    hardware = detect_hardware()

    assert isinstance(hardware.target, HardwareTarget)
    assert hardware.device_count >= 1


def test_cpu_benchmark_runs_end_to_end(vllm_cpu_image: str, model: str, tmp_path: Path) -> None:
    """A full experiment plans, runs, and parses against the real engine."""
    experiment = ExperimentConfig(
        container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=vllm_cpu_image),
        benchmark=BenchmarkConfig(
            model=model,
            dataset_name="random",
            num_prompts=_NUM_PROMPTS,
            input_len=_INPUT_LEN,
            output_len=_OUTPUT_LEN,
            timeout=_RUN_TIMEOUT_SECONDS,
        ),
        llm=LLMConfig(dtype=_CPU_DTYPE, gpu_memory_utilization=_CPU_MEMORY_UTILIZATION),
    )

    result = run_experiment(experiment, output_dir=tmp_path)

    assert result.raw_result.exit_code == 0
    assert result.target == HardwareTarget.CPU
    assert result.image == vllm_cpu_image
    assert result.devices == ()
    assert result.tokens_per_second > 0
    assert result.requests_per_second > 0
    assert result.elapsed_time > 0
    assert result.num_prompts == _NUM_PROMPTS
    assert result.raw_json["model_id"] == model
    assert (tmp_path / _RESULT_FILENAME).is_file()
