"""Integration tests for the llama.cpp inference backend.

These run a real benchmark end to end: the layer detects the host, plans a
one-container/one-server run, executes it through Podman against llama.cpp's
official CPU image, and parses the repeated online benchmark runs into a typed
result. The experiment forces the CPU target and uses the tiny GGUF checkpoint
llama.cpp's own test suite uses, so no accelerator is assumed and the suite is
runnable on any machine with Podman and network access.

The tests are marked ``integration`` and deselected by default; run them with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the llama.cpp CPU image cannot be obtained, or when the checkpoint is
neither cached nor downloadable. The first run pulls the image and downloads
the checkpoint, so it is slow.
"""

import socket
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.inference import (
    BenchmarkClient,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LlamaCppBackend,
    LLMConfig,
    detect_hardware,
    run_experiment,
)

pytestmark = pytest.mark.integration

LLAMACPP_CPU_IMAGE = "ghcr.io/ggml-org/llama.cpp:server"
MODEL_REPO = "ggml-org/test-model-stories260K"
MODEL_FILE = "stories260K-f32.gguf"
MODEL = f"{MODEL_REPO}:F32"
"""The tiny GGUF checkpoint, named the way a user selects a Hub quant."""

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_REPETITIONS = 2
_WARMUP = 1
_CONTAINER_TIMEOUT_SECONDS = 900.0

_MAX_MODEL_LEN = 512
_MAX_NUM_SEQS = 2

_VOCAB_SIZE = 512
"""The checkpoint's tokenizer vocabulary, which the random workload samples from.

The host client's 32000-token default would draw token ids this small model
does not have, so the tests narrow the bound to what the checkpoint actually
serves.
"""

_MODEL_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--ggml-org--test-model-stories260K"
)


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
def llamacpp_cpu_image(podman_client: PodmanClient) -> str:
    """The llama.cpp CPU image, pulled on demand, or skip when it cannot be obtained."""
    try:
        podman_client.images.get(LLAMACPP_CPU_IMAGE)
    except APIError:
        try:
            podman_client.images.pull(LLAMACPP_CPU_IMAGE)
        except (APIError, OSError) as exc:
            pytest.skip(f"{LLAMACPP_CPU_IMAGE} is unavailable and could not be pulled: {exc}")
    return LLAMACPP_CPU_IMAGE


@pytest.fixture(scope="module")
def checkpoint(llamacpp_cpu_image: str) -> Path:
    """The tiny GGUF checkpoint, downloaded on demand, or skip when unavailable.

    A checkpoint already in the Hub cache is used without touching the network;
    otherwise the Hub must be reachable and the file downloaded, so the module
    skips when the checkpoint is neither cached nor downloadable.
    """
    cached = _MODEL_CACHE.is_dir()
    if not cached:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5).close()
        except OSError as exc:
            pytest.skip(f"{MODEL} is not cached and huggingface.co is unreachable: {exc}")
    try:
        return Path(hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=cached))
    except Exception as exc:
        pytest.skip(f"{MODEL} is not available locally and could not be downloaded: {exc}")


def test_hardware_detection_resolves_the_real_host() -> None:
    """The real vendor probes return a target and a usable device count."""
    hardware = detect_hardware()

    assert isinstance(hardware.target, HardwareTarget)
    assert hardware.device_count >= 1


def test_cpu_benchmark_runs_end_to_end(llamacpp_cpu_image: str, checkpoint: Path) -> None:
    """A full experiment plans, runs, and parses against the real engine."""
    experiment = ExperimentConfig(
        container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=llamacpp_cpu_image),
        benchmark=BenchmarkConfig(
            model=MODEL,
            resolved_path=checkpoint,
            dataset_name="random",
            num_prompts=_NUM_PROMPTS,
            input_len=_INPUT_LEN,
            output_len=_OUTPUT_LEN,
            repetitions=_REPETITIONS,
            warmup=_WARMUP,
            timeout=_CONTAINER_TIMEOUT_SECONDS,
        ),
        llm=LLMConfig(max_model_len=_MAX_MODEL_LEN, max_num_seqs=_MAX_NUM_SEQS),
    )

    result = run_experiment(
        experiment,
        backend=LlamaCppBackend(),
        benchmark_client=BenchmarkClient(vocab_size=_VOCAB_SIZE),
    )

    assert result.raw_result.exit_code == 0
    assert result.target == HardwareTarget.CPU
    assert result.image == llamacpp_cpu_image
    assert result.devices == ()
    assert len(result.runs) == _REPETITIONS
    assert result.failed_runs == 0
    assert result.tokens_per_second > 0
    assert result.requests_per_second > 0
    assert result.elapsed_time > 0
    assert result.num_prompts == _NUM_PROMPTS * _REPETITIONS
    assert all(run.num_prompts == _NUM_PROMPTS for run in result.runs)
    assert len(result.raw_json) == _REPETITIONS


def test_cpu_benchmark_with_a_server_fetched_checkpoint(llamacpp_cpu_image: str) -> None:
    """Without a resolved path, ``llama-server`` fetches the selected GGUF itself.

    This is the path a direct library caller takes when it never prefetched the
    checkpoint; the production CLI resolves and mounts the file instead, which
    the sibling test covers. The container needs the Hub, so the test skips
    when huggingface.co is unreachable.
    """
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError as exc:
        pytest.skip(f"huggingface.co is unreachable: {exc}")

    experiment = ExperimentConfig(
        container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=llamacpp_cpu_image),
        benchmark=BenchmarkConfig(
            model=MODEL,
            dataset_name="random",
            num_prompts=_NUM_PROMPTS,
            input_len=_INPUT_LEN,
            output_len=_OUTPUT_LEN,
            repetitions=_REPETITIONS,
            warmup=_WARMUP,
            timeout=_CONTAINER_TIMEOUT_SECONDS,
        ),
        llm=LLMConfig(max_model_len=_MAX_MODEL_LEN, max_num_seqs=_MAX_NUM_SEQS),
    )

    result = run_experiment(
        experiment,
        backend=LlamaCppBackend(),
        benchmark_client=BenchmarkClient(vocab_size=_VOCAB_SIZE),
    )

    assert result.raw_result.exit_code == 0
    assert result.target == HardwareTarget.CPU
    assert result.image == llamacpp_cpu_image
    assert result.devices == ()
    assert len(result.runs) == _REPETITIONS
    assert result.failed_runs == 0
    assert result.tokens_per_second > 0
    assert result.num_prompts == _NUM_PROMPTS * _REPETITIONS
