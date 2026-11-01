"""Unit tests for the llama.cpp server plan and image selection.

``LlamaCppBackend.plan``'s observable contract is the
:class:`~llm_config_optimization.inference.types.ServerSpec` it returns: the container
configuration (image, device mapping, mounts, published port, and the
``llama-server`` command) plus the health path, reset paths, and startup timeout
the host client uses. llama.cpp serves a single GGUF file rather than a snapshot
directory and exposes no cache-reset endpoint, so the plan's file mount and its
empty ``reset_paths`` are llama.cpp-specific. These tests drive the backend
directly with a temporary checkpoint and an injected repository-file lister — no
engine, no GPU, and no network are involved.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from llm_config_optimization.inference import (
    AttentionBackend,
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DetectedHardware,
    ExperimentConfig,
    HardwareTarget,
    HubCheckpoint,
    KVCacheDType,
    LlamaCppBackend,
    LLMConfig,
    LocalCheckpoint,
    ServerSpec,
    SpeculativeMethod,
    SpeculativeMode,
)
from llm_config_optimization.runtime import Mount

REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
"""The repository id the fake file lister serves."""

_MODEL_TARGET = "/models/model.gguf"
_DRAFT_TARGET = "/models/draft.gguf"


def _repo(files: Sequence[str]) -> Callable[[str], Sequence[str]]:
    """Build a lister that serves ``files`` and pins the repository id."""

    def list_repo_files(repo_id: str) -> Sequence[str]:
        assert repo_id == REPO
        return list(files)

    return list_repo_files


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A local checkpoint the backend can mount; the bytes are irrelevant."""
    path = tmp_path / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    path.write_bytes(b"GGUF")
    return path


def _experiment(
    model: str,
    *,
    container: ContainerRuntimeConfig | None = None,
    benchmark: BenchmarkConfig | None = None,
    llm: LLMConfig | None = None,
) -> ExperimentConfig:
    """Build an experiment around ``model``, leaving the other domains fixed."""
    return ExperimentConfig(
        container=container if container is not None else ContainerRuntimeConfig(),
        benchmark=benchmark if benchmark is not None else BenchmarkConfig(model=model),
        llm=llm if llm is not None else LLMConfig(),
    )


def _plan(
    experiment: ExperimentConfig,
    hardware: DetectedHardware,
    *,
    backend: LlamaCppBackend | None = None,
) -> ServerSpec:
    """Plan a run with the default backend, or a supplied one."""
    return (backend if backend is not None else LlamaCppBackend()).plan(experiment, hardware)


def _serve(spec: ServerSpec) -> list[str]:
    """Recover the server command's arguments from the planned container."""
    return list(spec.container.command)


def _mount_for(spec: ServerSpec, target: str) -> Mount:
    """Return the single mount at ``target``, failing if it is absent or doubled."""
    matches = [mount for mount in spec.container.volumes if mount.target == target]
    assert len(matches) == 1, f"expected one mount at {target!r}, found {matches!r}"
    return matches[0]


def _random_benchmark(
    model: str, *, input_len: int = 1024, output_len: int = 128
) -> BenchmarkConfig:
    """A random-workload benchmark whose request length the slot cap can use."""
    return BenchmarkConfig(
        model=model, dataset_name="random", input_len=input_len, output_len=output_len
    )


@pytest.mark.parametrize(
    ("target", "image", "devices"),
    [
        (HardwareTarget.CUDA, "ghcr.io/ggml-org/llama.cpp:server-cuda", ("nvidia.com/gpu=all",)),
        (HardwareTarget.ROCM, "ghcr.io/ggml-org/llama.cpp:server-rocm", ("/dev/kfd", "/dev/dri")),
        (HardwareTarget.CPU, "ghcr.io/ggml-org/llama.cpp:server", ()),
    ],
)
def test_each_target_resolves_to_its_image_and_device_mapping(
    target: HardwareTarget, image: str, devices: tuple[str, ...], model_file: Path
) -> None:
    spec = _plan(_experiment(str(model_file)), DetectedHardware(target, 1))

    assert spec.container.image == image
    assert spec.container.devices == devices


@pytest.mark.parametrize(
    ("target", "image"),
    [
        (HardwareTarget.CUDA, "ghcr.io/ggml-org/llama.cpp:server-cuda"),
        (HardwareTarget.ROCM, "ghcr.io/ggml-org/llama.cpp:server-rocm"),
        (HardwareTarget.CPU, "ghcr.io/ggml-org/llama.cpp:server"),
    ],
)
def test_image_for_resolves_the_official_server_image_per_target(
    target: HardwareTarget, image: str
) -> None:
    assert LlamaCppBackend().image_for(target) == image


def test_explicit_image_overrides_the_target_default(model_file: Path) -> None:
    experiment = _experiment(
        str(model_file), container=ContainerRuntimeConfig(image="custom/llama.cpp:dev")
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.ROCM, 1))

    assert spec.container.image == "custom/llama.cpp:dev"
    assert spec.container.devices == ("/dev/kfd", "/dev/dri")


def test_explicit_target_overrides_the_detected_target(model_file: Path) -> None:
    experiment = _experiment(
        str(model_file), container=ContainerRuntimeConfig(target=HardwareTarget.CPU)
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 4))

    assert spec.container.image == "ghcr.io/ggml-org/llama.cpp:server"
    assert spec.container.devices == ()


def test_plan_publishes_the_server_port_and_keeps_the_image_entrypoint(model_file: Path) -> None:
    """The server must be reachable from the host, so its port is published.

    The official server image already runs ``llama-server``, so the plan leaves
    the entrypoint alone and only supplies the server arguments as the command.
    """
    spec = _plan(_experiment(str(model_file)), DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.container.entrypoint is None
    assert spec.container.ports == {"8080": None}
    assert spec.container.command[0] == "--model"


def test_local_gguf_is_mounted_read_only_and_addressed_in_container(model_file: Path) -> None:
    spec = _plan(_experiment(str(model_file)), DetectedHardware(HardwareTarget.CPU, 1))

    mount = _mount_for(spec, _MODEL_TARGET)
    assert mount.source == str(model_file.resolve())
    assert mount.read_only is True
    assert mount.options == ("z",)
    assert _serve(spec)[:2] == ["--model", _MODEL_TARGET]


def test_a_resolved_checkpoint_is_mounted_and_the_served_name_is_pinned(model_file: Path) -> None:
    """The composition root's resolved file is mounted without renaming the model."""
    model = f"{REPO}:Q4_K_M"
    experiment = _experiment(
        model, benchmark=BenchmarkConfig(model=model, resolved_path=model_file)
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    mount = _mount_for(spec, _MODEL_TARGET)
    assert mount.source == str(model_file)
    serve = _serve(spec)
    assert serve[serve.index("--model") + 1] == _MODEL_TARGET
    assert serve[serve.index("--alias") + 1] == model


def test_a_resolved_path_takes_precedence_over_the_model_string(model_file: Path) -> None:
    """The resolved checkpoint is mounted even when the model string names a path."""
    other = model_file.parent / "other.gguf"
    other.write_bytes(b"GGUF")
    experiment = _experiment(
        str(other), benchmark=BenchmarkConfig(model=str(other), resolved_path=model_file)
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert _mount_for(spec, _MODEL_TARGET).source == str(model_file)


def test_a_resolved_local_draft_is_mounted_at_its_own_path(
    model_file: Path, tmp_path: Path
) -> None:
    """The draft never contends with the main model's fixed container path."""
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF")
    experiment = _experiment(
        str(model_file),
        llm=LLMConfig(
            speculative_method=SpeculativeMethod.DRAFT_MODEL,
            draft_model="org/draft",
            resolved_draft_path=draft,
        ),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    draft_mount = _mount_for(spec, _DRAFT_TARGET)
    assert draft_mount.source == str(draft)
    assert draft_mount.read_only is True
    assert draft_mount.options == ("z",)
    assert _mount_for(spec, _MODEL_TARGET).source == str(model_file.resolve())
    serve = _serve(spec)
    assert serve[serve.index("--spec-draft-model") + 1] == _DRAFT_TARGET


def test_an_unresolved_local_draft_is_mounted_too(model_file: Path, tmp_path: Path) -> None:
    """A direct caller that never prefetched still gets a mounted local draft."""
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF")
    experiment = _experiment(
        str(model_file),
        llm=LLMConfig(
            speculative_method=SpeculativeMethod.DRAFT_MODEL,
            draft_model=str(draft),
        ),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert _mount_for(spec, _DRAFT_TARGET).source == str(draft.resolve())
    serve = _serve(spec)
    assert serve[serve.index("--spec-draft-model") + 1] == _DRAFT_TARGET


def test_an_unresolved_draft_repository_is_fetched_by_the_server(model_file: Path) -> None:
    """A Hub draft goes through llama.cpp's own draft-repository flag."""
    backend = LlamaCppBackend(list_repo_files=_repo(["draft-Q4_K_M.gguf"]))
    draft = f"{REPO}:Q4_K_M"
    experiment = _experiment(
        str(model_file),
        llm=LLMConfig(
            speculative_method=SpeculativeMethod.EAGLE3,
            draft_model=draft,
        ),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1), backend=backend)

    assert all(mount.target != _DRAFT_TARGET for mount in spec.container.volumes)
    serve = _serve(spec)
    assert "--spec-draft-model" not in serve
    assert serve[serve.index("--spec-draft-hf") + 1] == draft


def test_a_disabled_mode_keeps_the_draft_mount_but_drops_its_flags(
    model_file: Path, tmp_path: Path
) -> None:
    """The mount is harmless when the engine is told not to load the draft."""
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF")
    experiment = _experiment(
        str(model_file),
        llm=LLMConfig(
            speculative_method=SpeculativeMethod.DRAFT_MODEL,
            draft_model="org/draft",
            speculative_mode=SpeculativeMode.DISABLED,
            resolved_draft_path=draft,
        ),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert _mount_for(spec, _DRAFT_TARGET).source == str(draft)
    serve = _serve(spec)
    assert "--spec-type" not in serve
    assert "--spec-draft-model" not in serve


def test_a_hub_model_without_a_resolved_path_is_fetched_by_the_server() -> None:
    """A direct caller that never prefetched leaves llama-server to fetch the file."""
    backend = LlamaCppBackend(list_repo_files=_repo(["Llama-3.2-1B-Instruct-Q4_K_M.gguf"]))

    spec = _plan(_experiment(REPO), DetectedHardware(HardwareTarget.CUDA, 1), backend=backend)

    serve = _serve(spec)
    assert "--model" not in serve
    assert serve[serve.index("--hf-repo") + 1] == REPO
    assert serve[serve.index("--hf-file") + 1] == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    assert serve[serve.index("--alias") + 1] == REPO
    assert serve[serve.index("--host") + 1] == "0.0.0.0"
    assert serve[serve.index("--port") + 1] == "8080"
    assert all(mount.target != _MODEL_TARGET for mount in spec.container.volumes)


def test_a_hub_request_without_a_selected_file_asks_the_server_for_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository-wide request emits only ``--hf-repo``, for the engine's default quant.

    :func:`model_request` always selects a GGUF file for llama.cpp, so this is
    the fallback for a neutral :class:`HubCheckpoint` that names no file.
    """
    monkeypatch.setattr(
        "llm_config_optimization.inference._llamacpp.model_request",
        lambda model, *, list_repo_files: HubCheckpoint(repo_id=model),
    )

    spec = _plan(_experiment(REPO), DetectedHardware(HardwareTarget.CPU, 1))

    serve = _serve(spec)
    assert "--hf-file" not in serve
    assert serve[serve.index("--hf-repo") + 1] == REPO
    assert serve[serve.index("--alias") + 1] == REPO
    assert all(mount.target != _MODEL_TARGET for mount in spec.container.volumes)


def test_serve_command_carries_the_model_alias_and_host_binding(model_file: Path) -> None:
    """The server binds every interface on the port the host client will reach.

    The published port is only usable because the command binds ``0.0.0.0`` on
    the same port the plan publishes, and the alias pins the name the
    OpenAI-compatible endpoint reports. The local checkpoint's name stands in
    for the served name here, exactly as the composition root passes it.
    """
    spec = _plan(_experiment(str(model_file)), DetectedHardware(HardwareTarget.CPU, 1))

    assert _serve(spec)[:8] == [
        "--model",
        _MODEL_TARGET,
        "--alias",
        str(model_file),
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]


def test_a_hub_selector_picks_the_quant_the_server_will_fetch() -> None:
    files = ["Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q8_0.gguf"]
    backend = LlamaCppBackend(list_repo_files=_repo(files))

    spec = _plan(
        _experiment(f"{REPO}:Q8_0"), DetectedHardware(HardwareTarget.CPU, 1), backend=backend
    )

    serve = _serve(spec)
    assert serve[serve.index("--hf-file") + 1] == "Llama-3.2-1B-Instruct-Q8_0.gguf"


def test_every_mapped_tunable_reaches_the_server_command(model_file: Path) -> None:
    llm = LLMConfig(
        kv_cache_dtype=KVCacheDType.BFLOAT16,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        ubatch_size=2048,
        attention_backend=AttentionBackend.FLASH_ATTENTION,
    )

    spec = _plan(_experiment(str(model_file), llm=llm), DetectedHardware(HardwareTarget.CUDA, 1))

    serve = _serve(spec)
    assert serve[serve.index("--cache-type-k") :] == [
        "--cache-type-k",
        "bf16",
        "--cache-type-v",
        "bf16",
        "--ctx-size",
        "2048",
        "--parallel",
        "4",
        "--batch-size",
        "4096",
        "--ubatch-size",
        "2048",
        "--flash-attn",
        "on",
    ]


@pytest.mark.parametrize(
    ("max_model_len", "max_num_seqs", "expected_parallel"),
    [
        (4096, 64, "3"),
        (8192, 4, "4"),
        (1152, 8, "1"),
        (1024, 8, "1"),
    ],
)
def test_parallel_is_capped_to_what_the_context_can_serve(
    model_file: Path,
    max_model_len: int,
    max_num_seqs: int,
    expected_parallel: str,
) -> None:
    """``--parallel`` never exceeds the slots ``--ctx-size`` can give a request.

    llama.cpp divides the context across the slots, so a request longer than a
    slot's share is rejected; the plan cuts the slot count to the largest one
    whose share still fits the workload. The sampled ``max_model_len`` itself is
    forwarded unchanged, even where it is too small to serve the workload and
    the engine is left to reject the request.
    """
    experiment = _experiment(
        str(model_file),
        benchmark=_random_benchmark(str(model_file)),
        llm=LLMConfig(max_model_len=max_model_len, max_num_seqs=max_num_seqs),
    )

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert serve[serve.index("--ctx-size") + 1] == str(max_model_len)
    assert serve[serve.index("--parallel") + 1] == expected_parallel


def test_declared_parallel_is_kept_when_the_workload_length_is_unknown(
    model_file: Path,
) -> None:
    """An unset request length leaves the declared slot count alone."""
    experiment = _experiment(str(model_file), llm=LLMConfig(max_model_len=2048, max_num_seqs=8))

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert serve[serve.index("--parallel") + 1] == "8"


def test_an_unmeasured_dataset_workload_uses_one_slot(model_file: Path) -> None:
    """A JSONL workload with no measured token length falls back to one slot."""
    benchmark = BenchmarkConfig(
        model=str(model_file),
        dataset_path="/data/prompts.jsonl",
        input_len=1024,
        output_len=128,
    )
    experiment = _experiment(
        str(model_file), benchmark=benchmark, llm=LLMConfig(max_model_len=2048, max_num_seqs=8)
    )

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert serve[serve.index("--parallel") + 1] == "1"


def test_a_measured_dataset_request_caps_the_slot_count(model_file: Path) -> None:
    """A tokenized dataset caps --parallel from its longest request."""
    benchmark = BenchmarkConfig(
        model=str(model_file),
        dataset_path="/data/prompts.jsonl",
        max_request_tokens=256,
    )
    experiment = _experiment(
        str(model_file), benchmark=benchmark, llm=LLMConfig(max_model_len=2048, max_num_seqs=8)
    )

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert serve[serve.index("--parallel") + 1] == "8"


def test_a_measured_dataset_request_can_reduce_the_slot_count(model_file: Path) -> None:
    """The measured request length is what the context is divided by."""
    benchmark = BenchmarkConfig(
        model=str(model_file),
        dataset_path="/data/prompts.jsonl",
        max_request_tokens=1024,
    )
    experiment = _experiment(
        str(model_file), benchmark=benchmark, llm=LLMConfig(max_model_len=2048, max_num_seqs=8)
    )

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert serve[serve.index("--parallel") + 1] == "2"


def test_declared_parallel_is_kept_when_the_context_is_deferred(model_file: Path) -> None:
    """Without a declared context there is nothing to cap the slot count against."""
    experiment = _experiment(
        str(model_file),
        benchmark=_random_benchmark(str(model_file)),
        llm=LLMConfig(max_num_seqs=8),
    )

    serve = _serve(_plan(experiment, DetectedHardware(HardwareTarget.CPU, 1)))

    assert "--ctx-size" not in serve
    assert serve[serve.index("--parallel") + 1] == "8"


def test_extra_args_land_on_the_server_command(model_file: Path) -> None:
    backend = LlamaCppBackend(extra_args=("--n-gpu-layers", "99"))
    experiment = _experiment(str(model_file), llm=LLMConfig(max_model_len=4096))

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1), backend=backend)

    serve = _serve(spec)
    assert serve[serve.index("--ctx-size") :] == ["--ctx-size", "4096", "--n-gpu-layers", "99"]


def test_health_reset_and_startup_timeout_come_from_the_benchmark(model_file: Path) -> None:
    experiment = _experiment(
        str(model_file),
        benchmark=BenchmarkConfig(model=str(model_file), startup_timeout=123.0, timeout=456.0),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert spec.health_path == "/health"
    assert spec.health_timeout == 123.0
    assert spec.reset_paths == ()


def test_caller_volumes_precede_the_backend_mount(model_file: Path) -> None:
    caller_mount = Mount(source="/host/data", target="/data")
    experiment = _experiment(
        str(model_file), container=ContainerRuntimeConfig(volumes=(caller_mount,))
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CPU, 1))

    assert [mount.target for mount in spec.container.volumes] == ["/data", _MODEL_TARGET]
    assert spec.container.volumes[0] == caller_mount


def test_selinux_options_and_container_limits_are_forwarded(model_file: Path) -> None:
    experiment = _experiment(
        str(model_file),
        container=ContainerRuntimeConfig(
            selinux_opts=("disable",),
            mem_limit="16g",
            shm_size="4g",
            env={"HF_TOKEN": "secret"},
            pull_policy="always",
        ),
    )

    spec = _plan(experiment, DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.selinux_opts == ("disable",)
    assert spec.container.mem_limit == "16g"
    assert spec.container.shm_size == "4g"
    assert dict(spec.container.env) == {"HF_TOKEN": "secret"}
    assert spec.container.pull_policy == "always"


def test_absent_selinux_options_stay_automatic_and_shm_keeps_its_default(
    model_file: Path,
) -> None:
    spec = _plan(_experiment(str(model_file)), DetectedHardware(HardwareTarget.CUDA, 1))

    assert spec.container.selinux_opts is None
    assert spec.container.shm_size == "8g"


def test_model_request_resolves_an_existing_host_file(model_file: Path) -> None:
    request = LlamaCppBackend().model_request(str(model_file))

    assert request == LocalCheckpoint(path=model_file.resolve())


def test_model_request_resolves_a_hub_id_through_the_injected_lister() -> None:
    backend = LlamaCppBackend(list_repo_files=_repo(["model-Q8_0.gguf"]))

    request = backend.model_request(REPO)

    assert request == HubCheckpoint(repo_id=REPO, filename="model-Q8_0.gguf")
