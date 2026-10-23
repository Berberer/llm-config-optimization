"""Integration tests for the optimization adapter's sweep paths.

The first test runs a real two-trial sweep end to end: ``optimize`` samples a
mostly-fixed space, rebuilds an experiment per trial, executes each as a
one-container/one-server run through the real vLLM CPU backend and Podman,
records the repeated-run outcome on the study, and returns the winning
configuration. It asserts the result and study shape — the trial count, that
every trial completed, the phase rates the engine reported, and that the
returned configuration carries the fixed and inherited values it should —
rather than the throughput, which is hardware-dependent.

The second test drives the same adapter with a fake backend returning a
phase-rate pair per trial, so the multi-objective path — a two-direction
study, a real Pareto front, and the recommended point — is exercised end to
end without Podman or a GPU.

The tests are marked ``integration`` and deselected by default; run them with
``uv run pytest -m integration``. The real sweep skips when Podman is
unavailable, when the vLLM CPU image cannot be obtained, or when the model is
neither cached nor downloadable. The first run pulls a multi-gigabyte image
and downloads the model into the mounted Hugging Face cache, so it is slow.
"""

import socket
from collections.abc import Sequence
from pathlib import Path

import optuna
import pytest
from podman import PodmanClient
from podman.errors import APIError

from fakes import FakeBackend, FakePodmanClient
from llm_config_optimization.inference import (
    BenchmarkClient,
    BenchmarkConfig,
    BenchmarkOutcome,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    ExperimentResult,
    HardwareTarget,
    KVCacheDType,
    LLMConfig,
    RunMetrics,
    ServerSpec,
    VLLMBackend,
)
from llm_config_optimization.optimization import (
    ObjectiveMode,
    ObjectiveWeights,
    OptimizationResult,
    optimize,
)
from llm_config_optimization.runtime import ContainerConfig, ContainerResult
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

_FAKE_PLAN = ServerSpec(
    container=ContainerConfig(image="fake/engine:latest", command=("serve",), ports={"8000": None}),
    health_path="/health",
    health_timeout=600.0,
)
"""The canned server plan the phase-rate fake backend returns."""

_FAKE_CONTAINER = ContainerResult(exit_code=0, stdout=b"", stderr=b"", container_id="fake")
"""The canned container outcome the phase-rate fake backend's results carry."""


def _phased_result(prefill: float, decode: float) -> ExperimentResult:
    """A one-run result carrying the given phase rates."""
    return ExperimentResult(
        runs=(
            RunMetrics(
                tokens_per_second=42.0,
                requests_per_second=4.0,
                elapsed_time=2.5,
                num_prompts=8,
                raw_json={},
                prefill_tokens_per_second=prefill,
                decode_tokens_per_second=decode,
            ),
        ),
        failed_runs=0,
        target=HardwareTarget.CPU,
        image=_FAKE_PLAN.container.image,
        devices=(),
        raw_result=_FAKE_CONTAINER,
    )


class PhasedSequenceBackend(FakeBackend):
    """A fake backend returning one preset phase-rate pair per trial.

    Every trial still runs through the real orchestration, objective, and
    study, so a test can give each sampled point a distinct pair and observe
    the multi-objective result the sweep builds. The last pair repeats when the
    study runs more trials than pairs were given.
    """

    def __init__(self, pairs: Sequence[tuple[float, float]]) -> None:
        super().__init__(_FAKE_PLAN, _phased_result(*pairs[0]), [])
        self.pairs = tuple(pairs)
        self.parses = 0

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        super().parse_result(outcome, spec, result)
        pair = self.pairs[min(self.parses, len(self.pairs) - 1)]
        self.parses += 1
        return _phased_result(*pair)


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
        mode=ObjectiveMode.COMBINED,
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
    assert recorded["prefill_tokens_per_second"] > 0
    assert recorded["decode_tokens_per_second"] > 0


def test_a_fake_backed_multi_sweep_returns_the_front_and_recommends_a_point() -> None:
    """The multi path runs end to end without a container runtime or an engine.

    The fake backend returns a distinct phase-rate pair per trial, so the real
    two-direction study builds a real Pareto front, and the recommended point
    is the front member with the highest weighted-harmonic score under the
    benchmark's own token mix.
    """
    backend = PhasedSequenceBackend(
        (
            (1000.0, 50.0),
            (500.0, 200.0),
            (400.0, 40.0),
            (200.0, 500.0),
        )
    )

    result = optimize(
        SPACE,
        BASE,
        n_trials=4,
        seed=0,
        mode=ObjectiveMode.MULTI,
        weights=ObjectiveWeights(prefill=_INPUT_LEN, decode=_OUTPUT_LEN),
        backend=backend,
        client=FakePodmanClient(),
    )

    assert list(result.study.directions) == [optuna.study.StudyDirection.MAXIMIZE] * 2
    assert [trial.values for trial in result.study.trials] == [
        [1000.0, 50.0],
        [500.0, 200.0],
        [400.0, 40.0],
        [200.0, 500.0],
    ]
    # Trial 2's (400, 40) is dominated by trial 0's (1000, 50), so it drops out.
    assert [solution.trial_number for solution in result.solutions] == [0, 1, 3]
    assert result.recommended.trial_number == 1
    assert result.recommended.value == (500.0, 200.0)
    assert result.best_config is result.recommended.config
    assert result.best_value == result.recommended.value
