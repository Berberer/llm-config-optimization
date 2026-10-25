"""Unit tests for the Podman quadlet writer.

The writer renders a hand-built server plan, so these tests pin the unit's
external shape without a backend or a container daemon: the three sections, the
container keys the plan supplies, the SELinux mapping, and the curl-based
healthcheck. Quadlet is a line-oriented format, so the exact-text assertions are
the compatibility guard for the generated unit.
"""

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    ServerSpec,
)
from llm_config_optimization.output import QuadletWriter, RenderContext
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
            pull_policy="newer",
        ),
        health_path="/health",
        health_timeout=600.0,
    )


def _render(spec: ServerSpec) -> str:
    """Render a hand-built plan through a context carrying it."""
    return QuadletWriter().render(
        RenderContext(
            config=_config(),
            target=HardwareTarget.CUDA,
            image="vllm/vllm-openai:cu128",
            spec=spec,
        )
    )


_EXPECTED_UNIT = """\
[Unit]
Description=Benchmarked inference server

[Container]
Image=vllm/vllm-openai:cu128
Exec=vllm serve meta-llama/Llama-3.1-8B --port 8000
Environment=HF_HOME=/root/.cache/huggingface
Environment=VLLM_FLAGS="--enable-prefix-caching --disable-log"
Volume=/models/llama:/models/llama:ro
Volume=/home/bench/.cache/huggingface:/root/.cache/huggingface:z
PublishPort=8000:8000/tcp
AddDevice=nvidia.com/gpu=all
AddDevice=/dev/kfd
AddDevice=/dev/dri
Memory=16g
ShmSize=8g
Pull=newer
SecurityLabelDisable=true
HealthCmd=curl -fsS http://127.0.0.1:8000/health
HealthInterval=10s
HealthRetries=3
HealthStartPeriod=600s

[Service]
Restart=on-failure
"""
"""The complete unit a user would drop into their systemd user directory."""


def test_the_quadlet_writer_requires_a_plan() -> None:
    """The engine command and mounts exist only in the planned server."""
    assert QuadletWriter().requires_plan is True


def test_the_rendered_unit_contains_the_planned_container() -> None:
    """Image, command, environment, mounts, devices, and limits are reproduced."""
    assert _render(_spec()) == _EXPECTED_UNIT


def test_an_entrypoint_is_emitted_only_when_the_plan_sets_one() -> None:
    """A ``None`` entrypoint keeps the image's own, so no key is written."""
    without = _render(_spec())
    with_entrypoint = _render(_spec(entrypoint=("/bin/sh", "-c")))

    assert "Entrypoint=" not in without
    assert 'Entrypoint=["/bin/sh","-c"]\n' in with_entrypoint


def test_a_planned_host_port_becomes_the_container_port() -> None:
    """A runtime-assigned port is replaced by a predictable self-contained one."""
    unit = _render(_spec(ports={"9000": None, "8000/tcp": 18000}))

    assert "PublishPort=9000:9000" in unit
    assert "PublishPort=18000:8000/tcp" in unit


def test_non_disable_selinux_options_become_podman_arguments() -> None:
    """Only ``disable`` has a native key; other labels are passed through."""
    unit = _render(_spec(selinux_opts=("disable", "type:container_file_t")))

    assert "SecurityLabelDisable=true" in unit
    assert "PodmanArgs=--security-opt label=type:container_file_t" in unit


def test_a_unit_with_no_install_section_stays_minimal() -> None:
    """A benchmark artifact must not start at boot or be enabled as a service."""
    unit = _render(_spec())

    assert "[Install]" not in unit
    assert "[Service]\nRestart=on-failure" in unit


def test_the_healthcheck_is_built_from_the_spec_health_path() -> None:
    """Readiness asks the engine's real health endpoint with curl."""
    unit = _render(_spec())

    assert "HealthCmd=curl -fsS http://127.0.0.1:8000/health" in unit
    assert "HealthStartPeriod=600s" in unit
    assert "HealthInterval=10s" in unit
    assert "HealthRetries=3" in unit


def test_rendering_without_a_plan_is_rejected() -> None:
    """A context without a server plan cannot describe a container."""
    context = RenderContext(
        config=_config(),
        target=HardwareTarget.CUDA,
        image="vllm/vllm-openai:cu128",
    )

    with pytest.raises(ValueError):
        QuadletWriter().render(context)
