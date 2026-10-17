"""Integration test for the optimization runner CLI's real CPU path.

This drives the real command-line entry point end to end: ``main`` parses an
argument vector, loads and validates the search-space file, resolves the CPU
target and the vLLM CPU image, prefetches the model, pulls the image, runs a
two-trial sweep through the real vLLM CPU backend and Podman, prints one log
line per completed trial, and writes the winning configuration. It asserts the
observable contract — the exit code, the log lines, and the written YAML —
rather than the throughput, which is hardware-dependent.

The test is marked ``integration`` and deselected by default; run it with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the vLLM CPU image cannot be obtained, or when the model is neither cached
nor downloadable. The first run pulls a multi-gigabyte image and downloads the
model into the mounted Hugging Face cache, so it is slow.
"""

import socket
from io import StringIO
from pathlib import Path

import pytest
import yaml
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.cli import main
from llm_config_optimization.inference import HardwareTarget, VLLMBackend

pytestmark = pytest.mark.integration

VLLM_CPU_IMAGE = "docker.io/vllm/vllm-openai-cpu:latest"
MODEL = "facebook/opt-125m"

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_RUN_TIMEOUT_SECONDS = 900.0
_N_TRIALS = 2

# opt-125m ships float16 weights, whose support on torch CPU is unstable; the
# CPU backend asks for an explicit portable dtype instead. On CPU the
# ``gpu_memory_utilization`` field controls the fraction of system memory the
# engine reserves (despite its name), and its 0.92 default exceeds this host.
_CPU_DTYPE = "float32"
_CPU_MEMORY_UTILIZATION = 0.5
_CPU_SEED = 0
_MAX_MODEL_LEN = 2048
_MAX_NUM_BATCHED_TOKENS = 2048
_MAX_NUM_SEQS = (2, 4)

SPACE = {
    "space": {
        "dtype": {"fixed": _CPU_DTYPE},
        "kv_cache_dtype": {"fixed": "auto"},
        "max_model_len": {"fixed": _MAX_MODEL_LEN},
        "max_num_seqs": {"categorical": list(_MAX_NUM_SEQS)},
        "max_num_batched_tokens": {"fixed": _MAX_NUM_BATCHED_TOKENS},
        "tensor_parallel_size": {"fixed": 1},
        "attention_backend": {"default": True},
    }
}
"""A mostly-fixed CPU-safe space with one dimension the sweep varies."""

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


def test_two_trial_cpu_sweep_reports_progress_and_writes_the_winner(
    podman_client: PodmanClient, vllm_cpu_image: str, model: str, tmp_path: Path
) -> None:
    """The real CLI runs a bounded CPU sweep and persists the winning config."""
    space_path = tmp_path / "space.yaml"
    space_path.write_text(yaml.safe_dump(SPACE), encoding="utf-8")
    config_path = tmp_path / "best.yaml"
    out, err = StringIO(), StringIO()

    code = main(
        [
            model,
            str(space_path),
            "--target",
            "cpu",
            "--n-trials",
            str(_N_TRIALS),
            "--output",
            "log",
            "--output-config",
            str(config_path),
            "--gpu-memory-utilization",
            str(_CPU_MEMORY_UTILIZATION),
            "--llm-seed",
            str(_CPU_SEED),
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
            "--run-timeout",
            str(_RUN_TIMEOUT_SECONDS),
        ],
        backend=VLLMBackend(cpu_image=vllm_cpu_image),
        client=podman_client,
        stdout=out,
        stderr=err,
    )

    assert code == 0
    assert "error:" not in err.getvalue()

    stdout = out.getvalue()
    assert f"model: {model}" in stdout
    assert "resolved target: cpu" in stdout
    assert f"pulling container image: {vllm_cpu_image}" in stdout
    assert f"trial 1/{_N_TRIALS}  COMPLETE" in stdout
    assert f"trial 2/{_N_TRIALS}  COMPLETE" in stdout
    assert "best throughput:" in stdout
    assert "winning configuration:" in stdout

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert set(document) == {"container", "benchmark", "llm"}
    assert document["container"]["target"] == HardwareTarget.CPU.value
    assert document["container"]["image"] == vllm_cpu_image
    assert document["container"]["shm_size"] == "8g"
    assert document["benchmark"]["model"] == model
    assert document["benchmark"]["dataset_name"] == "random"
    assert document["benchmark"]["num_prompts"] == _NUM_PROMPTS
    assert document["benchmark"]["input_len"] == _INPUT_LEN
    assert document["benchmark"]["output_len"] == _OUTPUT_LEN
    assert document["benchmark"]["timeout"] == _RUN_TIMEOUT_SECONDS
    assert document["llm"]["dtype"] == _CPU_DTYPE
    assert document["llm"]["quantization"] is None
    assert document["llm"]["kv_cache_dtype"] == "auto"
    assert document["llm"]["max_model_len"] == _MAX_MODEL_LEN
    assert document["llm"]["max_num_seqs"] in _MAX_NUM_SEQS
    assert document["llm"]["max_num_batched_tokens"] == _MAX_NUM_BATCHED_TOKENS
    assert document["llm"]["tensor_parallel_size"] == 1
    assert document["llm"]["gpu_memory_utilization"] == _CPU_MEMORY_UTILIZATION
    assert document["llm"]["seed"] == _CPU_SEED
    assert document["llm"]["attention_backend"] is None
