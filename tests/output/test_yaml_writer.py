"""Unit tests for the YAML experiment-description writer.

The writer replaced the document construction that used to live in the CLI
summary module, so these tests pin its external document: the three domains,
the curated container section, explicit nulls, engine-neutral enum spellings,
path coercion, and declaration order. The exact-text assertion is the
compatibility guard for the move — a change to the document a user's scripts
consume has to be deliberate.
"""

from pathlib import Path

import yaml

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
)
from llm_config_optimization.output import RenderContext, YamlWriter

TARGET = HardwareTarget.CUDA
IMAGE = "vllm/vllm-openai:latest"


def base_config() -> ExperimentConfig:
    """A winning configuration with an unresolved target and image."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(shm_size="8g", mem_limit="16g", pull_policy="missing"),
        benchmark=BenchmarkConfig(model="facebook/opt-125m", dataset_name="random", num_prompts=10),
        llm=LLMConfig(dtype=DType.FLOAT16, max_model_len=2048),
    )


def render(config: ExperimentConfig) -> str:
    """Render ``config`` with the resolved target and image filled in."""
    return YamlWriter().render(RenderContext(config=config, target=TARGET, image=IMAGE))


_EXPECTED_DOCUMENT = """\
container:
  target: cuda
  image: vllm/vllm-openai:latest
  shm_size: 8g
  mem_limit: 16g
  pull_policy: missing
benchmark:
  model: facebook/opt-125m
  resolved_path: null
  dataset_name: random
  dataset_path: null
  num_prompts: 10
  input_len: null
  output_len: null
  max_request_tokens: null
  repetitions: 3
  warmup: 1
  startup_timeout: 600.0
  timeout: null
llm:
  dtype: float16
  quantization: null
  kv_cache_dtype: null
  max_model_len: 2048
  max_num_seqs: null
  max_num_batched_tokens: null
  gpu_memory_utilization: null
  tensor_parallel_size: null
  seed: null
  attention_backend: null
  speculative_method: null
  draft_model: null
  speculative_mode: null
  num_speculative_tokens: null
  resolved_draft_path: null
"""
"""The document consumers already know; the move must not change a byte of it."""


def test_the_yaml_writer_never_requires_a_plan() -> None:
    """The document describes an experiment, so planning is not part of rendering."""
    assert YamlWriter().requires_plan is False


def test_the_rendered_document_is_byte_compatible_with_the_previous_writer() -> None:
    """The historical three-domain document keeps its content, order, and nulls."""
    assert render(base_config()) == _EXPECTED_DOCUMENT


def test_the_container_section_is_curated() -> None:
    """Only the host-facing settings a user would re-supply are written."""
    container = yaml.safe_load(render(base_config()))["container"]

    assert set(container) == {"target", "image", "shm_size", "mem_limit", "pull_policy"}
    assert container["target"] == "cuda"
    assert container["image"] == IMAGE


def test_paths_are_written_as_plain_strings() -> None:
    """A resolved model or draft path is self-describing, not a Python tag."""
    config = ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(
            model="facebook/opt-125m",
            resolved_path=Path("/models/facebook-opt-125m"),
        ),
        llm=LLMConfig(resolved_draft_path=Path("/models/draft")),
    )

    document = yaml.safe_load(render(config))

    assert document["benchmark"]["resolved_path"] == "/models/facebook-opt-125m"
    assert document["llm"]["resolved_draft_path"] == "/models/draft"
