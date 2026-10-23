"""Integration test for the vLLM ROCm backend on a real AMD GPU.

This runs a small benchmark end to end on a host with an AMD ROCm stack: the
layer detects the host, plans a one-container/one-server ROCm run, executes it
through Podman against vLLM's official ROCm image, and parses the repeated
online benchmark runs into a typed result. The experiment sets no SELinux
option, so the run exercises the runtime's automatic default: on an
SELinux-enforcing host labeling is disabled for the device-mapping container,
while on any other host the option resolves to nothing. The same assertions
hold either way, so the test protects the zero-config path without depending on
the host's SELinux state.

The test is marked ``integration`` and deselected by default; run it with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the vLLM ROCm image cannot be obtained, when no AMD GPU with a usable ROCm
stack is present, or when the model is neither cached nor downloadable. The
first run pulls a multi-gigabyte image and downloads the model into the mounted
Hugging Face cache, so it is slow.
"""

import socket
from pathlib import Path

import pytest
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.inference import (
    BenchmarkClient,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    detect_hardware,
    run_experiment,
)

pytestmark = pytest.mark.integration

VLLM_ROCM_IMAGE = "docker.io/vllm/vllm-openai-rocm:latest"
MODEL = "facebook/opt-125m"

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_REPETITIONS = 2
_WARMUP = 1
_CONTAINER_TIMEOUT_SECONDS = 900.0

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
def rocm_host() -> None:
    """Skip unless the host exposes a usable AMD ROCm stack.

    Detection is the same probe the experiment layer uses, so the test runs
    exactly when the automatic ROCm plan has real devices to map.
    """
    if detect_hardware().target != HardwareTarget.ROCM:
        pytest.skip("no AMD GPU with a usable ROCm stack is available")


@pytest.fixture(scope="module")
def vllm_rocm_image(podman_client: PodmanClient) -> str:
    """The vLLM ROCm image, pulled on demand, or skip when it cannot be obtained."""
    try:
        podman_client.images.get(VLLM_ROCM_IMAGE)
    except APIError:
        try:
            podman_client.images.pull(VLLM_ROCM_IMAGE)
        except (APIError, OSError) as exc:
            pytest.skip(f"{VLLM_ROCM_IMAGE} is unavailable and could not be pulled: {exc}")
    return VLLM_ROCM_IMAGE


@pytest.fixture(scope="module")
def model(vllm_rocm_image: str) -> str:
    """The model under test, or skip when it is neither cached nor downloadable."""
    if _MODEL_CACHE.is_dir():
        return MODEL
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError as exc:
        pytest.skip(f"{MODEL} is not cached and huggingface.co is unreachable: {exc}")
    return MODEL


def test_rocm_benchmark_runs_with_the_automatic_selinux_default(
    podman_client: PodmanClient,
    rocm_host: None,
    vllm_rocm_image: str,
    model: str,
) -> None:
    """A ROCm experiment with no SELinux option plans, runs, and parses.

    ``selinux_opts`` is deliberately left at its default so the runtime resolves
    the automatic choice against the real host; the result must be the same
    whether or not the host is enforcing.
    """
    experiment = ExperimentConfig(
        container=ContainerRuntimeConfig(target=HardwareTarget.ROCM, image=vllm_rocm_image),
        benchmark=BenchmarkConfig(
            model=model,
            dataset_name="random",
            num_prompts=_NUM_PROMPTS,
            input_len=_INPUT_LEN,
            output_len=_OUTPUT_LEN,
            repetitions=_REPETITIONS,
            warmup=_WARMUP,
            timeout=_CONTAINER_TIMEOUT_SECONDS,
        ),
        llm=LLMConfig(),
    )

    result = run_experiment(experiment, client=podman_client, benchmark_client=BenchmarkClient())

    assert result.raw_result.exit_code == 0
    assert result.target == HardwareTarget.ROCM
    assert result.image == vllm_rocm_image
    assert result.devices == ("/dev/kfd", "/dev/dri")
    assert len(result.runs) == _REPETITIONS
    assert result.failed_runs == 0
    assert result.tokens_per_second > 0
    assert result.prefill_tokens_per_second > 0
    assert result.decode_tokens_per_second > 0
    assert result.requests_per_second > 0
    assert result.elapsed_time > 0
    assert result.num_prompts == _NUM_PROMPTS * _REPETITIONS
    assert all(run.num_prompts == _NUM_PROMPTS for run in result.runs)
    assert len(result.raw_json) == _REPETITIONS
