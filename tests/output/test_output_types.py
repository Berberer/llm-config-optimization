"""Unit tests for the output layer's public contract.

The output package is the seam between the optimization result and a written
artifact. These tests pin its external shape — the format members a caller can
select, the context a writer receives, and the frozen semantics writers rely
on — without asserting anything about a particular renderer.
"""

from dataclasses import FrozenInstanceError

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    ServerSpec,
)
from llm_config_optimization.output import ConfigWriter, OutputFormat, RenderContext
from llm_config_optimization.runtime import ContainerConfig


def _config() -> ExperimentConfig:
    """Build a minimal complete experiment for a render context."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B"),
        llm=LLMConfig(),
    )


def _spec() -> ServerSpec:
    """Build a hand-made server plan, as a renderer test would."""
    return ServerSpec(
        container=ContainerConfig(image="vllm/vllm-openai:latest"),
        health_path="/health",
        health_timeout=600.0,
    )


def test_output_format_members_are_the_selectable_artifacts() -> None:
    """The format enum exposes exactly the YAML, quadlet, and compose choices."""
    assert [member.value for member in OutputFormat] == ["yaml", "quadlet", "compose"]


def test_render_context_holds_the_optional_plan() -> None:
    """A context carries the recommended config and a plan only when one was made."""
    config = _config()
    context = RenderContext(config=config, target=HardwareTarget.CUDA, image="vllm/vllm-openai")

    assert context.config is config
    assert context.target is HardwareTarget.CUDA
    assert context.image == "vllm/vllm-openai"
    assert context.spec is None

    planned = RenderContext(
        config=config,
        target=HardwareTarget.CUDA,
        image="vllm/vllm-openai",
        spec=_spec(),
    )
    assert planned.spec == _spec()


def test_render_context_is_frozen() -> None:
    """A render context is immutable, like every other experiment description."""
    context = RenderContext(config=_config(), target=HardwareTarget.CUDA, image="image")

    with pytest.raises(FrozenInstanceError):
        context.image = "other"  # type: ignore[misc]


def test_config_writer_is_a_structural_protocol() -> None:
    """Any object with a requires_plan flag and a render method is a writer."""

    class Writer:
        requires_plan = False

        def render(self, context: RenderContext) -> str:
            return context.image

    writer = Writer()
    assert isinstance(writer, ConfigWriter)
    assert writer.render(RenderContext(_config(), HardwareTarget.CUDA, "image")) == "image"
