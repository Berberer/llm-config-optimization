"""Integration test for the optimization runner CLI's real llama.cpp path.

This drives the real command-line entry point end to end: ``main`` parses an
argument vector, loads and validates the search-space file, resolves the CPU
target and the llama.cpp CPU image, resolves a Hub GGUF selector and downloads
the checkpoint, pulls the image, runs a two-trial sweep as
one-container/one-server runs through the real llama.cpp backend and Podman,
prints one log line per completed trial, and writes the winning configuration.
It asserts the observable contract — the exit code, the log lines, and the
written YAML — rather than the throughput, which is hardware-dependent.

The search space declares only the five tunables llama.cpp forwards, and no
engine-scoped flag is passed, so the sweep exercises the engine-scoped CLI rules
that reject ``--quantization`` and ``--gpu-memory-utilization`` for this engine
from the accepting side.

The test is marked ``integration`` and deselected by default; run it with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the llama.cpp CPU image cannot be obtained, or when the checkpoint is
neither cached nor downloadable. The first run pulls the image and downloads
the checkpoint, so it is slow.
"""

import socket
from io import StringIO
from pathlib import Path

import pytest
import yaml
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.cli import main
from llm_config_optimization.inference import BenchmarkClient, HardwareTarget

pytestmark = pytest.mark.integration

LLAMACPP_CPU_IMAGE = "ghcr.io/ggml-org/llama.cpp:server"
MODEL = "ggml-org/test-model-stories260K:F32"
"""The tiny GGUF checkpoint, selected from its Hub repository by quant tag."""

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_REPETITIONS = 2
_WARMUP = 1
_CONTAINER_TIMEOUT_SECONDS = 900.0
_N_TRIALS = 2

_MAX_MODEL_LEN = 512
_MAX_NUM_BATCHED_TOKENS = 512
_MAX_NUM_SEQS = (1, 2)

_VOCAB_SIZE = 512
"""The checkpoint's tokenizer vocabulary, which the random workload samples from.

The host client's 32000-token default would draw token ids this small model
does not have, so the test narrows the bound to what the checkpoint serves.
"""

SPACE = {
    "space": {
        "kv_cache_dtype": {"fixed": "float16"},
        "max_model_len": {"fixed": _MAX_MODEL_LEN},
        "max_num_seqs": {"categorical": list(_MAX_NUM_SEQS)},
        "max_num_batched_tokens": {"fixed": _MAX_NUM_BATCHED_TOKENS},
        "attention_backend": {"default": True},
    }
}
"""A mostly-fixed CPU-safe space with only llama.cpp's five tunables declared."""

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
def model(llamacpp_cpu_image: str) -> str:
    """The model under test, or skip when it is neither cached nor downloadable."""
    if _MODEL_CACHE.is_dir():
        return MODEL
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError as exc:
        pytest.skip(f"{MODEL} is not cached and huggingface.co is unreachable: {exc}")
    return MODEL


def test_two_trial_cpu_sweep_reports_progress_and_writes_the_winner(
    podman_client: PodmanClient, llamacpp_cpu_image: str, model: str, tmp_path: Path
) -> None:
    """The real CLI runs a bounded llama.cpp sweep and persists the winning config."""
    space_path = tmp_path / "space.yaml"
    space_path.write_text(yaml.safe_dump(SPACE), encoding="utf-8")
    config_path = tmp_path / "best.yaml"
    out, err = StringIO(), StringIO()

    code = main(
        [
            model,
            str(space_path),
            "llamacpp",
            "--target",
            "cpu",
            "--n-trials",
            str(_N_TRIALS),
            "--output",
            "log",
            "--output-config",
            str(config_path),
            "--llm-seed",
            "0",
            "--sampler-seed",
            "0",
            "--dataset-name",
            "random",
            "--num-prompts",
            str(_NUM_PROMPTS),
            "--input-len",
            str(_INPUT_LEN),
            "--output-len",
            str(_OUTPUT_LEN),
            "--repetitions",
            str(_REPETITIONS),
            "--warmup-runs",
            str(_WARMUP),
            "--timeout",
            str(_CONTAINER_TIMEOUT_SECONDS),
        ],
        client=podman_client,
        benchmark_client=BenchmarkClient(vocab_size=_VOCAB_SIZE),
        stdout=out,
        stderr=err,
    )

    assert code == 0
    assert "error:" not in err.getvalue()

    stdout = out.getvalue()
    assert f"model: {model}" in stdout
    assert "engine: llamacpp" in stdout
    assert "resolved target: cpu" in stdout
    assert f"pulling container image: {llamacpp_cpu_image}" in stdout
    assert f"trial 1/{_N_TRIALS}  COMPLETE" in stdout
    assert f"trial 2/{_N_TRIALS}  COMPLETE" in stdout
    assert "best throughput:" in stdout
    assert "winning configuration:" in stdout

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert set(document) == {"container", "benchmark", "llm"}
    assert document["container"]["target"] == HardwareTarget.CPU.value
    assert document["container"]["image"] == llamacpp_cpu_image
    assert document["container"]["shm_size"] == "8g"
    assert document["benchmark"]["model"] == model
    assert document["benchmark"]["dataset_name"] == "random"
    assert document["benchmark"]["num_prompts"] == _NUM_PROMPTS
    assert document["benchmark"]["input_len"] == _INPUT_LEN
    assert document["benchmark"]["output_len"] == _OUTPUT_LEN
    assert document["benchmark"]["repetitions"] == _REPETITIONS
    assert document["benchmark"]["warmup"] == _WARMUP
    assert document["benchmark"]["startup_timeout"] == 600.0
    assert document["benchmark"]["timeout"] == _CONTAINER_TIMEOUT_SECONDS
    assert document["llm"]["dtype"] is None
    assert document["llm"]["quantization"] is None
    assert document["llm"]["kv_cache_dtype"] == "float16"
    assert document["llm"]["max_model_len"] == _MAX_MODEL_LEN
    assert document["llm"]["max_num_seqs"] in _MAX_NUM_SEQS
    assert document["llm"]["max_num_batched_tokens"] == _MAX_NUM_BATCHED_TOKENS
    assert document["llm"]["tensor_parallel_size"] is None
    assert document["llm"]["gpu_memory_utilization"] is None
    assert document["llm"]["seed"] == 0
    assert document["llm"]["attention_backend"] is None
