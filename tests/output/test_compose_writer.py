"""Unit tests for the Docker/Podman Compose writer.

The writer renders a hand-built server plan, so these tests pin the document's
external shape without a backend or a container daemon: a modern single-service
file, the container data the plan supplies, the ``newer`` pull-policy
translation, and the curl-based healthcheck. The exact-text assertion is the
compatibility guard, while the parsed checks keep the meaning of the document
explicit.
"""

import pytest
import yaml

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    ServerSpec,
)
from llm_config_optimization.output import ComposeWriter, RenderContext
from llm_config_optimization.runtime import ContainerConfig, Mount


def _config() -> ExperimentConfig:
    """A minimal recommended experiment for the render context."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B"),
        llm=LLMConfig(),
    )


def _spec(
    *,
    entrypoint: tuple[str, ...] | None = None,
    ports: dict[str, int | None] | None = None,
    selinux_opts: tuple[str, ...] | None = ("disable",),
    pull_policy: str = "newer",
) -> ServerSpec:
    """A realistic planned server with every container domain populated."""
    return ServerSpec(
        container=ContainerConfig(
            image="vllm/vllm-openai:cu128",
            command=("vllm", "serve", "meta-llama/Llama-3.1-8B", "--port", "8000"),
            entrypoint=entrypoint,
            env={
                "HF_HOME": "/root/.cache/huggingface",
                "VLLM_FLAGS": "--enable-prefix-caching --disable-log",
            },
            volumes=(
                Mount("/models/llama", "/models/llama", read_only=True),
                Mount(
                    "/home/bench/.cache/huggingface",
                    "/root/.cache/huggingface",
                    options=("z",),
                ),
            ),
            devices=("nvidia.com/gpu=all", "/dev/kfd", "/dev/dri"),
            ports=ports if ports is not None else {"8000/tcp": None},
            mem_limit="16g",
            shm_size="8g",
            selinux_opts=selinux_opts,
            pull_policy=pull_policy,
        ),
        health_path="/health",
        health_timeout=600.0,
    )


def _render(spec: ServerSpec) -> str:
    """Render a hand-built plan through a context carrying it."""
    return ComposeWriter().render(
        RenderContext(
            config=_config(),
            target=HardwareTarget.CUDA,
            image="vllm/vllm-openai:cu128",
            spec=spec,
        )
    )


def _service(spec: ServerSpec) -> dict[str, object]:
    """Render a plan and return its single parsed service."""
    document = yaml.safe_load(_render(spec))
    assert list(document) == ["services"]
    return document["services"]["server"]


_EXPECTED_DOCUMENT = """\
services:
  server:
    image: "vllm/vllm-openai:cu128"
    command:
      - "vllm"
      - "serve"
      - "meta-llama/Llama-3.1-8B"
      - "--port"
      - "8000"
    environment:
      "HF_HOME": "/root/.cache/huggingface"
      "VLLM_FLAGS": "--enable-prefix-caching --disable-log"
    volumes:
      - "/models/llama:/models/llama:ro"
      - "/home/bench/.cache/huggingface:/root/.cache/huggingface:z"
    ports:
      - "8000:8000/tcp"
    devices:
      - "nvidia.com/gpu=all"
      - "/dev/kfd"
      - "/dev/dri"
    shm_size: "8g"
    mem_limit: "16g"
    security_opt:
      - "label=disable"
    # Podman's "newer" pull policy has no Compose equivalent; "always" keeps
    # the same registry check without Podman's local image comparison.
    pull_policy: "always"
    healthcheck:
      test:
        - CMD
        - curl
        - -fsS
        - "http://127.0.0.1:8000/health"
      interval: 10s
      retries: 3
      start_period: 600s
"""
"""The complete file a user would serve with Compose tooling."""


def test_the_compose_writer_requires_a_plan() -> None:
    """The engine command and mounts exist only in the planned server."""
    assert ComposeWriter().requires_plan is True


def test_the_rendered_document_contains_the_planned_container() -> None:
    """Image, command, environment, mounts, devices, and limits are reproduced."""
    assert _render(_spec()) == _EXPECTED_DOCUMENT


def test_the_document_is_a_modern_single_service_file() -> None:
    """No obsolete ``version`` key, and the one service is keyed ``server``."""
    document = yaml.safe_load(_render(_spec()))

    assert "version" not in document
    assert list(document["services"]) == ["server"]


def test_an_entrypoint_is_emitted_only_when_the_plan_sets_one() -> None:
    """A ``None`` entrypoint keeps the image's own, so no key is written."""
    without = _service(_spec())
    with_entrypoint = _service(_spec(entrypoint=("/bin/sh", "-c")))

    assert "entrypoint" not in without
    assert with_entrypoint["entrypoint"] == ["/bin/sh", "-c"]


def test_a_planned_host_port_becomes_the_container_port() -> None:
    """A runtime-assigned port is replaced by a predictable self-contained one."""
    service = _service(_spec(ports={"9000": None, "8000/tcp": 18000}))

    assert service["ports"] == ["9000:9000", "18000:8000/tcp"]


def test_the_newer_pull_policy_is_translated_with_a_comment() -> None:
    """Compose has no ``newer`` policy, so it becomes ``always`` and explains why."""
    document = _render(_spec())

    assert 'pull_policy: "always"' in document
    assert "newer" in document
    assert _service(_spec())["pull_policy"] == "always"


def test_other_pull_policies_are_emitted_unchanged_without_the_comment() -> None:
    """Only the ``newer`` translation needs a note in the generated file."""
    document = _render(_spec(pull_policy="missing"))

    assert 'pull_policy: "missing"' in document
    assert "#" not in document


def test_selinux_options_become_security_opt_entries() -> None:
    """The plan's label options are passed to Compose as security options."""
    service = _service(_spec(selinux_opts=("disable", "type:container_file_t")))

    assert service["security_opt"] == ["label=disable", "label=type:container_file_t"]


def test_the_healthcheck_is_built_from_the_spec_health_path() -> None:
    """Readiness asks the engine's real health endpoint with curl."""
    healthcheck = _service(_spec())["healthcheck"]

    assert healthcheck == {
        "test": ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"],
        "interval": "10s",
        "retries": 3,
        "start_period": "600s",
    }


def test_rendering_without_a_plan_is_rejected() -> None:
    """A context without a server plan cannot describe a container."""
    context = RenderContext(
        config=_config(),
        target=HardwareTarget.CUDA,
        image="vllm/vllm-openai:cu128",
    )

    with pytest.raises(ValueError):
        ComposeWriter().render(context)
