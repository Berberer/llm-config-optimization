"""Unit tests for the vLLM container plan.

``VLLMBackend.plan``'s observable contract is the
:class:`~llm_config_optimization.runtime.types.ContainerConfig` it returns: the image
and device mapping resolved for a target, the container defaults, the mounts the
run needs, and the exact ``vllm bench throughput`` command assembled. These
tests assert that returned plan directly — no Podman daemon and no inference
engine is involved — covering every accelerator target and both local and Hub
models.
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
    LLMConfig,
    VLLMBackend,
)
from llm_config_optimization.runtime import ContainerConfig, Mount

HUB_MODEL = "meta-llama/Llama-3.1-8B"

_HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
_HF_CACHE_TARGET = "/root/.cache/huggingface"
_MODEL_TARGET = "/model"
_OUTPUT_TARGET = "/output"
_CACHE_TARGET = "/vllm-cache"
_RESULT_PATH = "/output/result.json"


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
    output_dir: Path | None = None,
    backend: VLLMBackend | None = None,
) -> ContainerConfig:
    """Plan a run with the default backend, or a supplied one."""
    return (backend if backend is not None else VLLMBackend()).plan(
        experiment, hardware, output_dir
    )


def _mount_for(config: ContainerConfig, target: str) -> Mount:
    """Return the single mount at ``target``, failing if it is absent or doubled."""
    matches = [mount for mount in config.volumes if mount.target == target]
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
    target: HardwareTarget, image: str, devices: tuple[str, ...], tmp_path: Path
) -> None:
    config = _plan(_experiment(), DetectedHardware(target, 1), output_dir=tmp_path)

    assert config.image == image
    assert config.devices == devices


def test_cpu_image_is_configurable_for_pinning_a_build(tmp_path: Path) -> None:
    backend = VLLMBackend(cpu_image="registry.example/vllm-cpu:2025.1")

    config = _plan(
        _experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path, backend=backend
    )

    assert config.image == "registry.example/vllm-cpu:2025.1"


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


def test_shared_memory_defaults_to_eight_gigabytes(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.shm_size == "8g"


def test_shared_memory_limit_can_be_overridden(tmp_path: Path) -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(shm_size="16g"))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.shm_size == "16g"


def test_explicit_target_overrides_the_detected_target(tmp_path: Path) -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(target=HardwareTarget.CPU))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4), output_dir=tmp_path)

    assert config.image == "docker.io/vllm/vllm-openai-cpu:latest"
    assert config.devices == ()


def test_explicit_image_overrides_the_target_default(tmp_path: Path) -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(image="custom/vllm:dev"))

    config = _plan(experiment, DetectedHardware(HardwareTarget.ROCM, 1), output_dir=tmp_path)

    assert config.image == "custom/vllm:dev"
    assert config.devices == ("/dev/kfd", "/dev/dri")


def test_tensor_parallel_size_defaults_to_the_detected_device_count(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 4), output_dir=tmp_path)

    assert "--tensor-parallel-size" in config.command
    assert config.command[config.command.index("--tensor-parallel-size") + 1] == "4"


def test_explicit_tensor_parallel_size_is_left_alone(tmp_path: Path) -> None:
    experiment = _experiment(llm=LLMConfig(tensor_parallel_size=2))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4), output_dir=tmp_path)

    assert config.command[config.command.index("--tensor-parallel-size") + 1] == "2"


def test_image_entrypoint_is_overridden_for_the_benchmark_subcommand(tmp_path: Path) -> None:
    """vLLM images start ``vllm serve``; the plan must run the bench instead."""
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.entrypoint == ("vllm",)
    assert config.command[:2] == ("bench", "throughput")


def test_bench_command_assembles_workload_and_llm_arguments(tmp_path: Path) -> None:
    experiment = _experiment(
        benchmark=BenchmarkConfig(
            model=HUB_MODEL,
            dataset_name="sharegpt",
            dataset_path="/data/prompts.jsonl",
            num_prompts=100,
            input_len=128,
            output_len=64,
        ),
        llm=LLMConfig(dtype=DType.BFLOAT16, tensor_parallel_size=2),
    )

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 2), output_dir=tmp_path)

    assert config.command == (
        "bench",
        "throughput",
        "--model",
        HUB_MODEL,
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        "/data/prompts.jsonl",
        "--num-prompts",
        "100",
        "--input-len",
        "128",
        "--output-len",
        "64",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--output-json",
        _RESULT_PATH,
    )


def test_random_dataset_uses_the_random_length_flags(tmp_path: Path) -> None:
    """vLLM's random dataset reads --random-*-len and ignores the generic flags."""
    experiment = _experiment(
        benchmark=BenchmarkConfig(
            model=HUB_MODEL,
            dataset_name="random",
            num_prompts=8,
            input_len=32,
            output_len=8,
        ),
    )

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.command == (
        "bench",
        "throughput",
        "--model",
        HUB_MODEL,
        "--dataset-name",
        "random",
        "--num-prompts",
        "8",
        "--random-input-len",
        "32",
        "--random-output-len",
        "8",
        "--tensor-parallel-size",
        "1",
        "--output-json",
        _RESULT_PATH,
    )


def test_omitted_dataset_name_uses_the_random_length_flags(tmp_path: Path) -> None:
    """vLLM defaults to the random dataset when no dataset name is given."""
    experiment = _experiment(benchmark=BenchmarkConfig(model=HUB_MODEL, input_len=32, output_len=8))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert "--random-input-len" in config.command
    assert "--random-output-len" in config.command
    assert "--input-len" not in config.command
    assert "--output-len" not in config.command


def test_unset_workload_fields_are_omitted(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.command == (
        "bench",
        "throughput",
        "--model",
        HUB_MODEL,
        "--tensor-parallel-size",
        "1",
        "--output-json",
        _RESULT_PATH,
    )


def test_extra_args_are_appended_after_the_mapped_parameters(tmp_path: Path) -> None:
    backend = VLLMBackend(extra_args=("--enforce-eager",))
    experiment = _experiment(llm=LLMConfig(dtype=DType.FLOAT16))

    config = _plan(
        experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path, backend=backend
    )

    assert config.command == (
        "bench",
        "throughput",
        "--model",
        HUB_MODEL,
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
        "--output-json",
        _RESULT_PATH,
    )


def test_hub_model_is_passed_through_without_a_model_mount(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.command[config.command.index("--model") + 1] == HUB_MODEL
    assert all(mount.target != _MODEL_TARGET for mount in config.volumes)


def test_local_model_is_mounted_read_only_and_addressed_in_container(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    experiment = _experiment(benchmark=BenchmarkConfig(model=str(model_path)))

    config = _plan(experiment, DetectedHardware(HardwareTarget.ROCM, 1), output_dir=tmp_path)

    mount = _mount_for(config, _MODEL_TARGET)
    assert mount.source == str(model_path.resolve())
    assert mount.read_only is True
    assert mount.options == ("z",)
    assert config.command[config.command.index("--model") + 1] == _MODEL_TARGET


def test_hugging_face_cache_is_mounted_for_weight_reuse(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    mount = _mount_for(config, _HF_CACHE_TARGET)
    assert mount.source == str(_HF_CACHE_HOST)
    assert mount.read_only is False
    assert mount.options == ("z",)


def test_a_cache_directory_is_mounted_and_redirected(tmp_path: Path) -> None:
    """A supplied cache dir is mounted and the compile env points into it."""
    cache = tmp_path / "cache"
    backend = VLLMBackend(cache_dir=cache)

    config = _plan(
        _experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path, backend=backend
    )

    mount = _mount_for(config, _CACHE_TARGET)
    assert mount.source == str(cache)
    assert mount.options == ("z",)
    assert cache.is_dir()
    assert config.env["VLLM_CACHE_ROOT"] == "/vllm-cache/vllm"
    assert config.env["TRITON_CACHE_DIR"] == "/vllm-cache/triton"


def test_no_cache_directory_means_no_mount_or_cache_env(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert all(mount.target != _CACHE_TARGET for mount in config.volumes)
    assert "VLLM_CACHE_ROOT" not in config.env


def test_a_callers_cache_env_takes_precedence(tmp_path: Path) -> None:
    backend = VLLMBackend(cache_dir=tmp_path / "cache")
    experiment = _experiment(container=ContainerRuntimeConfig(env={"VLLM_CACHE_ROOT": "/custom"}))

    config = _plan(
        experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path, backend=backend
    )

    assert config.env["VLLM_CACHE_ROOT"] == "/custom"
    assert config.env["TRITON_CACHE_DIR"] == "/vllm-cache/triton"


def test_output_directory_is_created_and_mounted(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"

    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=output_dir)

    assert output_dir.is_dir()
    mount = _mount_for(config, _OUTPUT_TARGET)
    assert mount.source == str(output_dir)
    assert mount.options == ("z",)


def test_caller_volumes_precede_the_backend_mounts(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    caller_mount = Mount(source="/host/data", target="/data")
    experiment = _experiment(
        container=ContainerRuntimeConfig(volumes=(caller_mount,)),
        benchmark=BenchmarkConfig(model=str(model_path)),
    )

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert [mount.target for mount in config.volumes] == [
        "/data",
        _HF_CACHE_TARGET,
        _MODEL_TARGET,
        _OUTPUT_TARGET,
    ]
    assert config.volumes[0] == caller_mount


def test_selinux_options_are_copied_into_the_planned_container(tmp_path: Path) -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(selinux_opts=("disable",)))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.selinux_opts == ("disable",)


def test_absent_selinux_options_stay_automatic_in_the_planned_container(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.selinux_opts is None


def test_container_limits_environment_and_pull_policy_are_forwarded(tmp_path: Path) -> None:
    experiment = _experiment(
        container=ContainerRuntimeConfig(
            mem_limit="16g",
            env={"HF_TOKEN": "secret"},
            pull_policy="always",
        ),
        benchmark=BenchmarkConfig(model=HUB_MODEL, timeout=600.0),
    )

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    assert config.mem_limit == "16g"
    assert dict(config.env) == {"HF_TOKEN": "secret"}
    assert config.pull_policy == "always"
    assert config.timeout == 600.0
