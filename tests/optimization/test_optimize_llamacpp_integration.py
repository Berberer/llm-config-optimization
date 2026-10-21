"""Integration test for the optimization adapter's real llama.cpp sweep path.

This runs a real two-trial sweep end to end: ``optimize`` samples a
mostly-fixed space, rebuilds an experiment per trial, executes each as a
one-container/one-server run through the real llama.cpp CPU backend and Podman,
records the repeated-run outcome on the study, and returns the winning
configuration. It asserts the result and study shape — the trial count, that
every trial completed, and that the returned configuration carries the fixed
and inherited values it should — rather than the throughput, which is
hardware-dependent.

The test is marked ``integration`` and deselected by default; run it with
``uv run pytest -m integration``. The module skips when Podman is unavailable,
when the llama.cpp CPU image cannot be obtained, or when the checkpoint is
neither cached nor downloadable. The first run pulls the image and downloads
the checkpoint, so it is slow.
"""

import socket
from dataclasses import replace
from pathlib import Path

import optuna
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
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
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
_N_TRIALS = 2

_MAX_MODEL_LEN = 512
_MAX_NUM_BATCHED_TOKENS = 512
_MAX_NUM_SEQS = (1, 2)
_SEED = 0

_VOCAB_SIZE = 512
"""The checkpoint's tokenizer vocabulary, which the random workload samples from.

The host client's 32000-token default would draw token ids this small model
does not have, so the client narrows the bound to what the checkpoint serves.
"""

BASE = ExperimentConfig(
    container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=LLAMACPP_CPU_IMAGE),
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
    llm=LLMConfig(seed=_SEED),
)
"""The fixed container and workload setup the sweep varies the LLM config of.

The checkpoint's resolved path is filled in per test from the module fixture.
"""

SPACE = ExperimentSpace(
    dimensions=(
        Dimension("kv_cache_dtype", Fixed(KVCacheDType.FLOAT16)),
        Dimension("max_model_len", Fixed(_MAX_MODEL_LEN)),
        Dimension("max_num_seqs", Categorical(_MAX_NUM_SEQS)),
        Dimension("max_num_batched_tokens", Fixed(_MAX_NUM_BATCHED_TOKENS)),
        Dimension("attention_backend", Default()),
    )
)
"""A mostly-fixed CPU-safe space declaring exactly llama.cpp's applicable set."""

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


def test_two_trial_sweep_returns_a_usable_best_configuration(
    podman_client: PodmanClient, llamacpp_cpu_image: str, checkpoint: Path
) -> None:
    """A real sweep completes its trials and reports the winning configuration."""
    backend = LlamaCppBackend()
    validate_space(SPACE, backend)
    base = replace(BASE, benchmark=replace(BASE.benchmark, resolved_path=checkpoint))

    result = optimize(
        SPACE,
        base,
        n_trials=_N_TRIALS,
        seed=0,
        backend=backend,
        client=podman_client,
        benchmark_client=BenchmarkClient(vocab_size=_VOCAB_SIZE),
    )

    assert isinstance(result, OptimizationResult)
    assert result.trial_count == _N_TRIALS
    assert [trial.state for trial in result.study.trials] == [
        optuna.trial.TrialState.COMPLETE
    ] * _N_TRIALS
    assert result.best_trial_number == result.study.best_trial.number
    assert result.best_value == result.study.best_trial.value > 0

    best = result.best_config
    assert best.container is base.container
    assert best.benchmark is base.benchmark
    assert best.llm.dtype is None
    assert best.llm.quantization is None
    assert best.llm.kv_cache_dtype is KVCacheDType.FLOAT16
    assert best.llm.max_model_len == _MAX_MODEL_LEN
    assert best.llm.max_num_seqs == result.study.best_trial.params["max_num_seqs"]
    assert best.llm.max_num_batched_tokens == _MAX_NUM_BATCHED_TOKENS
    assert best.llm.tensor_parallel_size is None
    assert best.llm.gpu_memory_utilization is None
    assert best.llm.attention_backend is None
    assert best.llm.seed == _SEED

    recorded = result.study.best_trial.user_attrs["result"]
    assert recorded["tokens_per_second"] == result.best_value
    assert recorded["target"] == HardwareTarget.CPU.value
    assert recorded["image"] == llamacpp_cpu_image
    assert recorded["devices"] == []
    assert recorded["failed_runs"] == 0
    per_run = recorded["runs"]
    assert len(per_run) == _REPETITIONS
    assert recorded["tokens_per_second"] == pytest.approx(
        sum(run["tokens_per_second"] for run in per_run) / len(per_run)
    )
