"""Unit tests for the vLLM server plan.

``VLLMBackend.plan``'s observable contract is the
:class:`~llm_config_optimization.inference.types.ServerSpec` it returns: the
container configuration (image, device mapping, mounts, environment, published
port, and the ``vllm serve`` command) plus the health path, cache-reset paths,
and startup timeout the host client uses. These tests assert that spec directly
— no Podman daemon and no inference engine is involved — covering every
accelerator target, both local and Hub models, the serving parameters, and the
measurement controls. The client command is deliberately absent: the host
benchmark client owns the workload now.
"""

from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DetectedHardware,
    DType,
    ExperimentConfig,
    HardwareTarget,
    HubCheckpoint,
    LLMConfig,
    LocalCheckpoint,
    ServerSpec,
    VLLMBackend,
)
from llm_config_optimization.runtime import Mount

HUB_MODEL = "meta-llama/Llama-3.1-8B"

_HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
_HF_CACHE_TARGET = "/root/.cache/huggingface"
_MODEL_TARGET = "/model"
_CACHE_TARGET = "/vllm-cache"


def _experiment(
    *,
    container: ContainerRuntimeConfig | None = None,
    benchmark: BenchmarkConfig | None = None,
    llm: LLMConfig | None = None,
) -> ExperimentConfig:
    """Build an experiment, defaulting to a Hub model and no tuning."""
    return ExperimentConfig(
        container=container if container is not None else ContainerRuntimeConfig(),
        benchmark=benchmark if benchmark is not None else BenchmarkConfig(model=HUB_MODEL),
        llm=llm if llm is not None else LLMConfig(),
    )


def _plan(
    experiment: ExperimentConfig,
    hardware: DetectedHardware,
    *,
    backend: VLLMBackend | None = None,
) -> ServerSpec:
    """Plan a run with the default backend, or a supplied one."""
    return (backend if backend is not None else VLLMBackend()).plan(experiment, hardware)


def _serve(spec: ServerSpec) -> list[str]:
    """Recover the server command's arguments from the planned container."""
    return list(spec.container.command)


def _mount_for(spec: ServerSpec, target: str) -> Mount:
    """Return the single mount at ``target``, failing if it is absent or doubled."""
    matches = [mount for mount in spec.container.volumes if mount.target == target]
    assert len(matches) == 1, f"expected one mount at {target!r}, found {matches!r}"
    return matches[0]


@pytest.mark.parametrize(
    ("target", "image", "devices"),
    [
        (HardwareTarget.CUDA, "docker.io/vllm/vllm-openai:latest", ("nvidia.com/gpu=all",)),
        (HardwareTarget.ROCM, "docker.io/vllm/vllm-openai-rocm:latest", ("/dev/kfd", "/dev/dri")),
        (HardwareTarget.CPU, "docker.io/vllm/vllm-openai-cpu:latest", ()),
    ],
)
def test_each_target_resolves_to_its_image_and_device_mapping(
    target: HardwareTarget, image: str, devices: tuple[str, ...]
) -> None:
    spec = _plan(_experiment(), DetectedHardware(target, 1))

    assert spec.container.image == image
    assert spec.container.devices == devices


def test_cpu_image_is_configurable_for_pinning_a_build() -> None:
    backend = VLLMBackend(cpu_image="registry.example/vllm-cpu:2025.1")

    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), backend=backend)

    assert spec.container.image == "registry.example/vllm-cpu:2025.1"


@pytest.mark.parametrize(
    ("target", "image"),
    [
        (HardwareTarget.CUDA, "docker.io/vllm/vllm-openai:latest"),
        (HardwareTarget.ROCM, "docker.io/vllm/vllm-openai-rocm:latest"),
        (HardwareTarget.CPU, "docker.io/vllm/vllm-openai-cpu:latest"),
    ],
)
def test_image_for_resolves_the_default_image_per_target(
    target: HardwareTarget, image: str
) -> None:
    assert VLLMBackend().image_for(target) == image


def test_image_for_honours_a_configured_cpu_image() -> None:
    backend = VLLMBackend(cpu_image="registry.example/vllm-cpu:2025.1")

    assert backend.image_for(HardwareTarget.CPU) == "registry.example/vllm-cpu:2025.1"
    assert backend.image_for(HardwareTarget.CUDA) == "docker.io/vllm/vllm-openai:latest"


def test_shared_memory_defaults_to_eight_gigabytes() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.shm_size == "8g"


def test_shared_memory_limit_can_be_overridden() -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(shm_size="16g"))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.shm_size == "16g"


def test_explicit_target_overrides_the_detected_target() -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(target=HardwareTarget.CPU))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4))

    assert spec.container.image == "docker.io/vllm/vllm-openai-cpu:latest"
    assert spec.container.devices == ()


def test_explicit_image_overrides_the_target_default() -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(image="custom/vllm:dev"))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.ROCM, 1))

    assert spec.container.image == "custom/vllm:dev"
    assert spec.container.devices == ("/dev/kfd", "/dev/dri")


def test_plan_publishes_the_server_port_and_keeps_the_image_entrypoint() -> None:
    """The server must be reachable from the host, so its port is published.

    vLLM's official image already starts ``vllm serve``, so the plan leaves the
    entrypoint alone and only supplies the serve arguments as the command.
    """
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.container.entrypoint is None
    assert spec.container.ports == {"8000": None}
    assert spec.container.command[0] == "--model"


def test_tensor_parallel_size_defaults_to_the_detected_device_count() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 4))

    serve = _serve(spec)
    assert serve[serve.index("--tensor-parallel-size") + 1] == "4"


def test_explicit_tensor_parallel_size_is_left_alone() -> None:
    experiment = _experiment(llm=LLMConfig(tensor_parallel_size=2))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4))

    serve = _serve(spec)
    assert serve[serve.index("--tensor-parallel-size") + 1] == "2"


def test_dev_mode_is_enabled_so_the_cache_reset_endpoint_is_available() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.container.env["VLLM_SERVER_DEV_MODE"] == "1"


def test_a_callers_dev_mode_choice_takes_precedence() -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(env={"VLLM_SERVER_DEV_MODE": "0"}))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.container.env["VLLM_SERVER_DEV_MODE"] == "0"


def test_serve_command_carries_the_model_served_name_and_host_binding() -> None:
    """The server binds on every interface and is addressed by its served name."""
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1))

    assert _serve(spec)[:8] == [
        "--model",
        HUB_MODEL,
        "--served-model-name",
        HUB_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_serving_parameters_reach_the_server_command() -> None:
    experiment = _experiment(llm=LLMConfig(dtype=DType.BFLOAT16, tensor_parallel_size=2))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 2))

    assert _serve(spec) == [
        "--model",
        HUB_MODEL,
        "--served-model-name",
        HUB_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
    ]


def test_extra_args_land_on_the_server_command() -> None:
    backend = VLLMBackend(extra_args=("--enforce-eager",))
    experiment = _experiment(llm=LLMConfig(dtype=DType.FLOAT16))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), backend=backend)

    assert _serve(spec) == [
        "--model",
        HUB_MODEL,
        "--served-model-name",
        HUB_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
    ]


def test_health_reset_and_startup_timeout_come_from_the_benchmark() -> None:
    experiment = _experiment(
        benchmark=BenchmarkConfig(model=HUB_MODEL, startup_timeout=123.0, timeout=456.0)
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.health_path == "/health"
    assert spec.health_timeout == 123.0
    assert spec.reset_paths == (
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    )


def test_hub_model_is_passed_through_without_a_model_mount() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1))

    serve = _serve(spec)
    assert serve[serve.index("--model") + 1] == HUB_MODEL
    assert all(mount.target != _MODEL_TARGET for mount in spec.container.volumes)


def test_model_request_resolves_an_existing_host_path(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()

    request = VLLMBackend().model_request(str(model_path))

    assert request == LocalCheckpoint(path=model_path.resolve())


def test_model_request_resolves_a_hub_id_to_a_snapshot_request() -> None:
    request = VLLMBackend().model_request(HUB_MODEL)

    assert request == HubCheckpoint(repo_id=HUB_MODEL)


def test_a_resolved_path_is_mounted_and_the_served_name_is_kept(tmp_path: Path) -> None:
    """The composition root's resolved snapshot is mounted without renaming the model."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    experiment = _experiment(benchmark=BenchmarkConfig(model=HUB_MODEL, resolved_path=snapshot))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    mount = _mount_for(spec, _MODEL_TARGET)
    assert mount.source == str(snapshot)
    assert mount.read_only is True
    serve = _serve(spec)
    assert serve[serve.index("--model") + 1] == _MODEL_TARGET
    assert serve[serve.index("--served-model-name") + 1] == HUB_MODEL


def test_a_resolved_path_takes_precedence_over_the_model_string(tmp_path: Path) -> None:
    """The resolved checkpoint is mounted even when the model string names a path."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    experiment = _experiment(benchmark=BenchmarkConfig(model=str(other), resolved_path=snapshot))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert _mount_for(spec, _MODEL_TARGET).source == str(snapshot)


def test_a_hub_snapshot_mounts_its_cache_root_so_symlinks_resolve(tmp_path: Path) -> None:
    """An HF cache snapshot is mounted through its cache root, not on its own.

    A snapshot's files are relative symlinks that can hop through the cache's
    shared ``blobs`` store before reaching the real file (Xet-backed
    downloads), so only mounting the model's repository would still leave the
    weights dangling. The cache hub root must be the mount source, with the
    server pointed at the snapshot inside it.
    """
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    snapshot = repo / "snapshots" / "abc123"
    repo_blobs = repo / "blobs"
    shared_blobs = hub / "blobs" / "01"
    snapshot.mkdir(parents=True)
    repo_blobs.mkdir()
    shared_blobs.mkdir(parents=True)
    (shared_blobs / "deadbeef").write_text("{}")
    (repo_blobs / "cafebabe").symlink_to("../../blobs/01/deadbeef")
    (snapshot / "model.safetensors").symlink_to("../../blobs/cafebabe")
    experiment = _experiment(benchmark=BenchmarkConfig(model=HUB_MODEL, resolved_path=snapshot))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    mount = _mount_for(spec, _MODEL_TARGET)
    assert mount.source == str(hub)
    # Both hops of the weights symlink resolve inside the mounted tree.
    assert (Path(mount.source) / "blobs" / "01" / "deadbeef").is_file()
    serve = _serve(spec)
    assert (
        serve[serve.index("--model") + 1] == f"{_MODEL_TARGET}/models--org--name/snapshots/abc123"
    )
    assert serve[serve.index("--served-model-name") + 1] == HUB_MODEL


def test_local_model_is_mounted_read_only_and_addressed_in_container(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    experiment = _experiment(benchmark=BenchmarkConfig(model=str(model_path)))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.ROCM, 1))

    mount = _mount_for(spec, _MODEL_TARGET)
    assert mount.source == str(model_path.resolve())
    assert mount.read_only is True
    assert mount.options == ("z",)
    serve = _serve(spec)
    assert serve[serve.index("--model") + 1] == _MODEL_TARGET


def test_hugging_face_cache_is_mounted_for_weight_reuse() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1))

    mount = _mount_for(spec, _HF_CACHE_TARGET)
    assert mount.source == str(_HF_CACHE_HOST)
    assert mount.read_only is False
    assert mount.options == ("z",)


def test_a_cache_directory_is_mounted_and_redirected(tmp_path: Path) -> None:
    """A supplied cache dir is mounted and the compile env points into it."""
    cache = tmp_path / "cache"
    backend = VLLMBackend(cache_dir=cache)

    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), backend=backend)

    mount = _mount_for(spec, _CACHE_TARGET)
    assert mount.source == str(cache)
    assert mount.options == ("z",)
    assert cache.is_dir()
    assert spec.container.env["VLLM_CACHE_ROOT"] == "/vllm-cache/vllm"
    assert spec.container.env["TRITON_CACHE_DIR"] == "/vllm-cache/triton"


def test_no_cache_directory_means_no_mount_or_cache_env() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1))

    assert all(mount.target != _CACHE_TARGET for mount in spec.container.volumes)
    assert "VLLM_CACHE_ROOT" not in spec.container.env


def test_a_callers_cache_env_takes_precedence(tmp_path: Path) -> None:
    backend = VLLMBackend(cache_dir=tmp_path / "cache")
    experiment = _experiment(container=ContainerRuntimeConfig(env={"VLLM_CACHE_ROOT": "/custom"}))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), backend=backend)

    assert spec.container.env["VLLM_CACHE_ROOT"] == "/custom"
    assert spec.container.env["TRITON_CACHE_DIR"] == "/vllm-cache/triton"


def test_caller_volumes_precede_the_backend_mounts(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    caller_mount = Mount(source="/host/data", target="/data")
    experiment = _experiment(
        container=ContainerRuntimeConfig(volumes=(caller_mount,)),
        benchmark=BenchmarkConfig(model=str(model_path)),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert [mount.target for mount in spec.container.volumes] == [
        "/data",
        _HF_CACHE_TARGET,
        _MODEL_TARGET,
    ]
    assert spec.container.volumes[0] == caller_mount


def test_selinux_options_are_copied_into_the_planned_container() -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(selinux_opts=("disable",)))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.selinux_opts == ("disable",)


def test_absent_selinux_options_stay_automatic_in_the_planned_container() -> None:
    spec = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.selinux_opts is None


def test_container_limits_environment_and_pull_policy_are_forwarded() -> None:
    experiment = _experiment(
        container=ContainerRuntimeConfig(
            mem_limit="16g",
            env={"HF_TOKEN": "secret"},
            pull_policy="always",
        ),
        benchmark=BenchmarkConfig(model=HUB_MODEL, timeout=600.0),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.mem_limit == "16g"
    assert dict(spec.container.env) == {"HF_TOKEN": "secret", "VLLM_SERVER_DEV_MODE": "1"}
    assert spec.container.pull_policy == "always"
