"""Integration tests that validate rendered artifacts with real tooling.

The writer unit tests and the hermetic conformance layer pin the documents'
shape, but only the consuming tool can say whether a plausible document is
accepted. These tests therefore write the rendered artifacts to temporary
directories and run them through the binaries a user would use: the Podman
quadlet generator, ``systemd-analyze verify``, ``docker compose config``, and
``podman-compose config``.

They are marked ``integration`` and deselected by default; run them with
``uv run pytest -m integration``. Each tool is located at test time and the
test skips cleanly when the host does not have it, so a partially tooled
machine stays runnable. No test touches the developer's real systemd user
directory, and every artifact lives under ``tmp_path``.

The CDI compatibility matrix is explicit here: a portable device list must pass
every available Compose provider, while a CDI device such as
``nvidia.com/gpu=all`` passes ``podman-compose`` and is expected to be rejected
by real Docker. That keeps the documented Docker limitation executable instead
of inferred from prose.
"""

from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    ServerSpec,
)
from llm_config_optimization.output import ComposeWriter, QuadletWriter, RenderContext
from llm_config_optimization.runtime import ContainerConfig, Mount
from tooling import docker_is_podman_backed, find_tool, require_tool, run

pytestmark = pytest.mark.integration

_IMAGE = "docker.io/vllm/vllm-openai:cu128"
"""The image a representative trial runs, fully qualified for the quadlet generator."""

_QUADLET_GENERATORS = (
    "podman-system-generator",
    "podman-user-generator",
    "/usr/libexec/podman/quadlet",
    "/usr/lib/systemd/system-generators/podman-system-generator",
    "/usr/lib/systemd/user-generators/podman-user-generator",
)
"""The names and installed paths of Podman's quadlet generator."""

_CDI_DEVICE = "nvidia.com/gpu=all"
_PORTABLE_DEVICES = ("/dev/kfd", "/dev/dri")


def _config() -> ExperimentConfig:
    """A minimal recommended experiment for the render context."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B"),
        llm=LLMConfig(),
    )


def _spec(devices: tuple[str, ...] = (_CDI_DEVICE, *_PORTABLE_DEVICES)) -> ServerSpec:
    """A realistic planned server with every container domain populated."""
    return ServerSpec(
        container=ContainerConfig(
            image=_IMAGE,
            command=("vllm", "serve", "meta-llama/Llama-3.1-8B", "--port", "8000"),
            entrypoint=("/bin/sh", "-c"),
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
            devices=devices,
            ports={"8000/tcp": None},
            mem_limit="16g",
            shm_size="8g",
            selinux_opts=("disable",),
            pull_policy="newer",
        ),
        health_path="/health",
        health_timeout=600.0,
    )


def _context(spec: ServerSpec) -> RenderContext:
    return RenderContext(config=_config(), target=HardwareTarget.CUDA, image=_IMAGE, spec=spec)


def _write_quadlet(directory: Path, spec: ServerSpec) -> Path:
    path = directory / "bench.container"
    path.write_text(QuadletWriter().render(_context(spec)), encoding="utf-8")
    return path


def _write_compose(directory: Path, spec: ServerSpec) -> Path:
    path = directory / "compose.yaml"
    path.write_text(ComposeWriter().render(_context(spec)), encoding="utf-8")
    return path


def _compose_providers(path: Path) -> list[tuple[str, list[str | Path]]]:
    """Every Compose config checker installed on this host, with its command."""
    providers: list[tuple[str, list[str | Path]]] = []
    docker = find_tool("docker")
    if docker is not None:
        providers.append(("docker compose", [docker, "compose", "-f", path, "config", "--quiet"]))
    podman_compose = find_tool("podman-compose")
    if podman_compose is not None:
        providers.append(("podman-compose", [podman_compose, "-f", path, "config"]))
    return providers


def test_the_quadlet_generator_accepts_the_rendered_unit(tmp_path: Path) -> None:
    """The unit a user would drop into their systemd directory actually generates."""
    generator = require_tool(*_QUADLET_GENERATORS, purpose="validating a quadlet")
    units = tmp_path / "units"
    units.mkdir()
    unit = _write_quadlet(units, _spec())

    result = run(
        [generator, "--user", "--dryrun"],
        cwd=tmp_path,
        env={"QUADLET_UNIT_DIRS": str(units)},
    )

    assert result.returncode == 0, result.stderr
    assert "unsupported key" not in result.stderr
    assert f"---{unit.stem}.service---" in result.stdout
    assert "ExecStart=" in result.stdout
    assert f"--device {_CDI_DEVICE}" in result.stdout
    assert "--health-cmd" in result.stdout


def test_systemd_analyze_accepts_the_generated_unit(tmp_path: Path) -> None:
    """The generated service parses as a systemd unit, not merely as a container."""
    generator = require_tool(*_QUADLET_GENERATORS, purpose="validating a quadlet")
    systemd_analyze = require_tool("systemd-analyze", purpose="verifying a generated quadlet")

    units = tmp_path / "units"
    units.mkdir()
    unit = _write_quadlet(units, _spec())
    normal = tmp_path / "normal"
    early = tmp_path / "early"
    late = tmp_path / "late"
    for directory in (normal, early, late):
        directory.mkdir()

    generated = run(
        [generator, "--user", normal, early, late],
        cwd=tmp_path,
        env={"QUADLET_UNIT_DIRS": str(units)},
    )
    assert generated.returncode == 0, generated.stderr

    service = normal / f"{unit.stem}.service"
    assert service.is_file(), generated.stderr
    verified = run([systemd_analyze, "--user", "verify", service], cwd=tmp_path)
    assert verified.returncode == 0, verified.stderr


def test_a_portable_compose_file_passes_every_available_provider(tmp_path: Path) -> None:
    """Host-path devices are portable, so every Compose validator must accept them."""
    path = _write_compose(tmp_path, _spec(devices=_PORTABLE_DEVICES))
    providers = _compose_providers(path)
    if not providers:
        pytest.skip("validating a Compose file requires docker compose or podman-compose")

    for label, command in providers:
        result = run(command, cwd=tmp_path)
        assert result.returncode == 0, f"{label} rejected the Compose file:\n{result.stderr}"


def test_a_cdi_compose_file_passes_podman_compose(tmp_path: Path) -> None:
    """Podman passes CDI device strings through, so its Compose provider accepts them."""
    podman_compose = require_tool("podman-compose", purpose="validating a Compose file")
    path = _write_compose(tmp_path, _spec(devices=(_CDI_DEVICE,)))

    result = run([podman_compose, "-f", path, "config"], cwd=tmp_path)

    assert result.returncode == 0, result.stderr


def test_a_cdi_compose_file_is_a_documented_docker_limitation(tmp_path: Path) -> None:
    """Real Docker Compose does not understand CDI strings, which is expected here."""
    docker = require_tool("docker", purpose="checking the CDI limitation")
    if docker_is_podman_backed(docker):
        pytest.skip("the docker CLI on this host is Podman's emulation, not real Docker")
    path = _write_compose(tmp_path, _spec(devices=(_CDI_DEVICE,)))

    result = run([docker, "compose", "-f", path, "config", "--quiet"], cwd=tmp_path)

    assert result.returncode != 0, "real Docker Compose unexpectedly accepted a CDI device"
