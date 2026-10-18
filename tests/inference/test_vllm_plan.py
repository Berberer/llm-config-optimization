"""Unit tests for the vLLM container plan.

``VLLMBackend.plan``'s observable contract is the
:class:`~llm_config_optimization.runtime.types.ContainerConfig` it returns plus the
generated ``runner.sh`` it writes into the output directory: the image and
device mapping resolved for a target, the container defaults, the mounts the run
needs, and the self-contained runner that starts the server and takes the
repeated measurements. These tests assert that returned plan and the runner's
content directly — no Podman daemon and no inference engine is involved —
covering every accelerator target, both local and Hub models, the serve/bench
split, and the measurement controls.
"""

import shlex
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
_RUNNER_FILENAME = "runner.sh"
_MANIFEST_PATH = "/output/manifest.json"
_RUNNER_PATH = "/output/runner.sh"
_SERVER_LOG_PATH = "/output/server.log"
_HEALTH_URL = "http://127.0.0.1:8000/health"
_RESET_URL = "http://127.0.0.1:8000/reset_prefix_cache"


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


def _runner(config: ContainerConfig, output_dir: Path) -> str:
    """Read the generated runner a plan wrote into ``output_dir``."""
    return (output_dir / _RUNNER_FILENAME).read_text(encoding="utf-8")


def _array(runner: str, name: str) -> list[str]:
    """Recover one generated bash array as its list of arguments."""
    start = runner.index(f"{name}=(\n") + len(f"{name}=(\n")
    end = runner.index("\n)\n", start)
    return shlex.split(runner[start:end])


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

    serve = _array(_runner(config, tmp_path), "SERVE_ARGS")
    assert serve[serve.index("--tensor-parallel-size") + 1] == "4"


def test_explicit_tensor_parallel_size_is_left_alone(tmp_path: Path) -> None:
    experiment = _experiment(llm=LLMConfig(tensor_parallel_size=2))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4), output_dir=tmp_path)

    serve = _array(_runner(config, tmp_path), "SERVE_ARGS")
    assert serve[serve.index("--tensor-parallel-size") + 1] == "2"


def test_plan_runs_the_generated_runner_with_the_image_entrypoint_overridden(
    tmp_path: Path,
) -> None:
    """vLLM images start ``vllm serve``; the plan must run the generated runner."""
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.entrypoint == ("/bin/bash",)
    assert config.command == (_RUNNER_PATH,)
    assert (tmp_path / _RUNNER_FILENAME).is_file()


def test_dev_mode_is_enabled_so_the_cache_reset_endpoint_is_available(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.env["VLLM_SERVER_DEV_MODE"] == "1"


def test_a_callers_dev_mode_choice_takes_precedence(tmp_path: Path) -> None:
    experiment = _experiment(container=ContainerRuntimeConfig(env={"VLLM_SERVER_DEV_MODE": "0"}))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert config.env["VLLM_SERVER_DEV_MODE"] == "0"


def test_serve_command_carries_the_model_and_loopback_binding(tmp_path: Path) -> None:
    """The server is addressed on container loopback, never on a host port."""
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    serve = _array(_runner(config, tmp_path), "SERVE_ARGS")
    assert serve[:6] == ["--model", HUB_MODEL, "--host", "127.0.0.1", "--port", "8000"]


def test_serve_and_bench_arguments_reach_their_own_command(tmp_path: Path) -> None:
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
    runner = _runner(config, tmp_path)

    assert _array(runner, "SERVE_ARGS") == [
        "--model",
        HUB_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
    ]
    assert _array(runner, "BENCH_ARGS") == [
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
    ]


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

    assert _array(_runner(config, tmp_path), "BENCH_ARGS") == [
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
    ]


def test_omitted_dataset_name_uses_the_random_length_flags(tmp_path: Path) -> None:
    """vLLM defaults to the random dataset when no dataset name is given."""
    experiment = _experiment(benchmark=BenchmarkConfig(model=HUB_MODEL, input_len=32, output_len=8))

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    bench = _array(_runner(config, tmp_path), "BENCH_ARGS")
    assert "--random-input-len" in bench
    assert "--random-output-len" in bench
    assert "--input-len" not in bench
    assert "--output-len" not in bench


def test_unset_workload_fields_are_omitted(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert _array(_runner(config, tmp_path), "BENCH_ARGS") == ["--model", HUB_MODEL]


def test_extra_args_land_on_the_server_command(tmp_path: Path) -> None:
    backend = VLLMBackend(extra_args=("--enforce-eager",))
    experiment = _experiment(llm=LLMConfig(dtype=DType.FLOAT16))

    config = _plan(
        experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path, backend=backend
    )
    runner = _runner(config, tmp_path)

    assert _array(runner, "SERVE_ARGS") == [
        "--model",
        HUB_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
    ]
    assert "--enforce-eager" not in _array(runner, "BENCH_ARGS")


def test_runner_embeds_the_measurement_controls(tmp_path: Path) -> None:
    experiment = _experiment(
        benchmark=BenchmarkConfig(
            model=HUB_MODEL,
            repetitions=5,
            warmup=2,
            startup_timeout=123.0,
            timeout=456.0,
        )
    )

    config = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)
    runner = _runner(config, tmp_path)

    assert "REPETITIONS=5" in runner
    assert "WARMUP=2" in runner
    assert "STARTUP_TIMEOUT=123" in runner
    assert "OVERALL_TIMEOUT=456" in runner


def test_runner_defaults_to_three_measured_and_one_warmup_run(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)
    runner = _runner(config, tmp_path)

    assert "REPETITIONS=3" in runner
    assert "WARMUP=1" in runner
    assert "STARTUP_TIMEOUT=600" in runner


def test_unbounded_run_leaves_the_runner_without_a_deadline(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert "OVERALL_TIMEOUT=\n" in _runner(config, tmp_path)


def test_runner_polls_health_and_resets_the_caches(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)
    runner = _runner(config, tmp_path)

    assert _HEALTH_URL in runner
    assert _RESET_URL in runner
    assert _SERVER_LOG_PATH in runner
    assert "vllm serve" in runner
    assert "vllm bench serve" in runner


def test_runner_writes_a_manifest_and_enforces_a_strict_majority(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)
    runner = _runner(config, tmp_path)

    assert _MANIFEST_PATH in runner
    assert '"failed_runs"' in runner
    assert '"runs"' in runner
    assert "successes * 2" in runner


def test_runner_guards_the_empty_success_list_under_set_u(tmp_path: Path) -> None:
    """A trial where every run failed still writes a manifest without tripping ``set -u``."""
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)

    assert '${success_files[@]+"${success_files[@]}"}' in _runner(config, tmp_path)


def test_runner_saves_each_run_with_a_per_index_seed(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CPU, 1), output_dir=tmp_path)
    runner = _runner(config, tmp_path)

    assert "--save-result" in runner
    assert "--result-filename" in runner
    assert "run-${run_index}.json" in runner
    assert '"${run_index}"' in runner


def test_hub_model_is_passed_through_without_a_model_mount(tmp_path: Path) -> None:
    config = _plan(_experiment(), DetectedHardware(HardwareTarget.CUDA, 1), output_dir=tmp_path)

    serve = _array(_runner(config, tmp_path), "SERVE_ARGS")
    assert serve[serve.index("--model") + 1] == HUB_MODEL
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
    serve = _array(_runner(config, tmp_path), "SERVE_ARGS")
    assert serve[serve.index("--model") + 1] == _MODEL_TARGET


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
    assert (output_dir / _RUNNER_FILENAME).is_file()


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
    assert dict(config.env) == {"HF_TOKEN": "secret", "VLLM_SERVER_DEV_MODE": "1"}
    assert config.pull_policy == "always"
    assert config.timeout == 600.0
