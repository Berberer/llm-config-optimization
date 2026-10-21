"""Integration test for the optimization adapter's real sweep path.

This runs a real two-trial sweep end to end: ``optimize`` samples a
mostly-fixed space, rebuilds an experiment per trial, executes each as a
one-container/one-server run through the real vLLM CPU backend and Podman,
records the repeated-run outcome on the study, and returns the winning
configuration. It asserts the result and study shape — the trial count, that
every trial completed, and that the returned configuration carries the fixed
and inherited values it should — rather than the throughput, which is
hardware-dependent.

The test is marked ``integration`` and deselected by default; run it with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the vLLM CPU image cannot be obtained, or when the model is neither cached
nor downloadable. The first run pulls a multi-gigabyte image and downloads the
model into the mounted Hugging Face cache, so it is slow.
"""

import socket
from pathlib import Path

import optuna
import pytest
from podman import PodmanClient
from podman.errors import APIError

from llm_config_optimization.inference import (
    BenchmarkClient,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    HardwareTarget,
    KVCacheDType,
    LLMConfig,
    VLLMBackend,
)
from llm_config_optimization.optimization import OptimizationResult, optimize
from llm_config_optimization.search_space import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    validate_space,
)

pytestmark = pytest.mark.integration

VLLM_CPU_IMAGE = "docker.io/vllm/vllm-openai-cpu:latest"
MODEL = "facebook/opt-125m"

_NUM_PROMPTS = 4
_INPUT_LEN = 32
_OUTPUT_LEN = 8
_REPETITIONS = 2
_WARMUP = 1
_CONTAINER_TIMEOUT_SECONDS = 900.0
_N_TRIALS = 2

# opt-125m ships float16 weights, whose support on torch CPU is unstable; the
# CPU backend asks for an explicit portable dtype instead. On CPU the
# ``gpu_memory_utilization`` field controls the fraction of system memory the
# engine reserves (despite its name), and its 0.92 default exceeds this host.
_CPU_DTYPE = DType.FLOAT32
_CPU_MEMORY_UTILIZATION = 0.5
_CPU_SEED = 0
_MAX_MODEL_LEN = 2048
_MAX_NUM_BATCHED_TOKENS = 2048
_MAX_NUM_SEQS = (2, 4)

BASE = ExperimentConfig(
    container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=VLLM_CPU_IMAGE),
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
    llm=LLMConfig(
        dtype=_CPU_DTYPE,
        gpu_memory_utilization=_CPU_MEMORY_UTILIZATION,
        seed=_CPU_SEED,
    ),
)
"""The fixed container and workload setup the sweep varies the LLM config of."""

SPACE = ExperimentSpace(
    dimensions=(
        Dimension("dtype", Fixed(_CPU_DTYPE)),
        Dimension("kv_cache_dtype", Fixed(KVCacheDType.AUTO)),
        Dimension("max_model_len", Fixed(_MAX_MODEL_LEN)),
        Dimension("max_num_seqs", Categorical(_MAX_NUM_SEQS)),
        Dimension("max_num_batched_tokens", Fixed(_MAX_NUM_BATCHED_TOKENS)),
        Dimension("tensor_parallel_size", Fixed(1)),
        Dimension("attention_backend", Default()),
    )
)
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


def test_two_trial_sweep_returns_a_usable_best_configuration(
    podman_client: PodmanClient, vllm_cpu_image: str, model: str
) -> None:
    """A real sweep completes its trials and reports the winning configuration."""
    backend = VLLMBackend(cpu_image=vllm_cpu_image)
    validate_space(SPACE, backend)

    result = optimize(
        SPACE,
        BASE,
        n_trials=_N_TRIALS,
        seed=0,
        backend=backend,
        client=podman_client,
        benchmark_client=BenchmarkClient(),
    )

    assert isinstance(result, OptimizationResult)
    assert result.trial_count == _N_TRIALS
    assert [trial.state for trial in result.study.trials] == [
        optuna.trial.TrialState.COMPLETE
    ] * _N_TRIALS
    assert result.best_trial_number == result.study.best_trial.number
    assert result.best_value == result.study.best_trial.value > 0

    best = result.best_config
    assert best.container is BASE.container
    assert best.benchmark is BASE.benchmark
    assert best.llm.dtype is _CPU_DTYPE
    assert best.llm.kv_cache_dtype is KVCacheDType.AUTO
    assert best.llm.max_model_len == _MAX_MODEL_LEN
    assert best.llm.max_num_seqs == result.study.best_trial.params["max_num_seqs"]
    assert best.llm.max_num_batched_tokens == _MAX_NUM_BATCHED_TOKENS
    assert best.llm.tensor_parallel_size == 1
    assert best.llm.quantization is None
    assert best.llm.attention_backend is None
    assert best.llm.seed == _CPU_SEED
    assert best.llm.gpu_memory_utilization == _CPU_MEMORY_UTILIZATION

    recorded = result.study.best_trial.user_attrs["result"]
    assert recorded["tokens_per_second"] == result.best_value
    assert recorded["target"] == HardwareTarget.CPU.value
    assert recorded["image"] == vllm_cpu_image
    assert recorded["devices"] == []
    assert recorded["failed_runs"] == 0
    per_run = recorded["runs"]
    assert len(per_run) == _REPETITIONS
    assert recorded["tokens_per_second"] == pytest.approx(
        sum(run["tokens_per_second"] for run in per_run) / len(per_run)
    )
