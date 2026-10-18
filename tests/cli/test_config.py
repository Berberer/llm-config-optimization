"""Unit tests for base experiment construction from parsed arguments.

``build_base_config``'s observable contract is the
:class:`~llm_config_optimization.inference.types.ExperimentConfig` it produces: the
container, benchmark, and non-tunable LLM values come from the flags, every
tunable serving parameter is left unset for the space to fill, and an omitted
target or image stays ``None`` for the backend to resolve. ``is_local_model``
decides whether the model argument is a host checkpoint or a Hugging Face id.
These tests parse real argument vectors and assert on the resulting config, so
the parser and the builder are exercised together without a backend.
"""

from pathlib import Path

import pytest

from llm_config_optimization.cli.arguments import parse_args
from llm_config_optimization.cli.config import build_base_config, is_local_model
from llm_config_optimization.inference import HardwareTarget, QuantizationMethod


def test_defaults_produce_the_documented_base_config() -> None:
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml"]))

    assert config.container.target is None
    assert config.container.image is None
    assert config.container.shm_size == "8g"
    assert config.container.mem_limit is None
    assert config.container.pull_policy == "missing"
    assert config.benchmark.model == "facebook/opt-125m"
    assert config.benchmark.dataset_name == "random"
    assert config.benchmark.dataset_path is None
    assert config.benchmark.num_prompts == 200
    assert config.benchmark.input_len == 1024
    assert config.benchmark.output_len == 128
    assert config.benchmark.repetitions == 3
    assert config.benchmark.warmup == 1
    assert config.benchmark.startup_timeout == 600
    assert config.benchmark.timeout == 900
    assert config.llm.seed == 0
    assert config.llm.quantization is None
    assert config.llm.gpu_memory_utilization is None


def test_tunable_serving_parameters_are_left_unset_for_the_space() -> None:
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml"]))

    assert config.llm.dtype is None
    assert config.llm.kv_cache_dtype is None
    assert config.llm.max_model_len is None
    assert config.llm.max_num_seqs is None
    assert config.llm.max_num_batched_tokens is None
    assert config.llm.tensor_parallel_size is None
    assert config.llm.attention_backend is None


def test_explicit_flags_override_the_base_config() -> None:
    config = build_base_config(
        parse_args(
            [
                "facebook/opt-125m",
                "space.yaml",
                "--target",
                "rocm",
                "--image",
                "vllm/vllm-rocm:latest",
                "--shm-size",
                "16g",
                "--mem-limit",
                "32g",
                "--pull-policy",
                "always",
                "--dataset-name",
                "sharegpt",
                "--dataset-path",
                "/data/sharegpt.json",
                "--num-prompts",
                "64",
                "--input-len",
                "512",
                "--output-len",
                "64",
                "--repetitions",
                "5",
                "--warmup-runs",
                "2",
                "--startup-timeout",
                "300",
                "--timeout",
                "120",
                "--llm-seed",
                "7",
                "--quantization",
                "awq",
                "--gpu-memory-utilization",
                "0.9",
            ]
        )
    )

    assert config.container.target is HardwareTarget.ROCM
    assert config.container.image == "vllm/vllm-rocm:latest"
    assert config.container.shm_size == "16g"
    assert config.container.mem_limit == "32g"
    assert config.container.pull_policy == "always"
    assert config.benchmark.model == "facebook/opt-125m"
    assert config.benchmark.dataset_name == "sharegpt"
    assert config.benchmark.dataset_path == "/data/sharegpt.json"
    assert config.benchmark.num_prompts == 64
    assert config.benchmark.input_len == 512
    assert config.benchmark.output_len == 64
    assert config.benchmark.repetitions == 5
    assert config.benchmark.warmup == 2
    assert config.benchmark.startup_timeout == 300
    assert config.benchmark.timeout == 120
    assert config.llm.seed == 7
    assert config.llm.quantization is QuantizationMethod.AWQ
    assert config.llm.gpu_memory_utilization == 0.9


def test_out_of_range_measurement_controls_are_rejected() -> None:
    """A nonsensical control reaches ``BenchmarkConfig``'s own invariant check."""
    with pytest.raises(ValueError):
        build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "--repetitions", "0"]))

    with pytest.raises(ValueError):
        build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "--warmup-runs", "-1"]))

    with pytest.raises(ValueError):
        build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "--startup-timeout", "0"]))


def test_an_existing_host_path_is_a_local_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Llama-3.1-8B"
    checkpoint.mkdir()

    assert is_local_model(str(checkpoint))


def test_a_hugging_face_id_is_not_a_local_model() -> None:
    assert not is_local_model("facebook/opt-125m")


def test_a_missing_host_path_is_not_a_local_model(tmp_path: Path) -> None:
    assert not is_local_model(str(tmp_path / "absent"))


def test_a_local_model_argument_reaches_the_benchmark_config(tmp_path: Path) -> None:
    config = build_base_config(parse_args([str(tmp_path), "space.yaml"]))

    assert config.benchmark.model == str(tmp_path)
