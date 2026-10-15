"""Engine-neutral types for the inference experiment layer.

These types describe *what* to evaluate and *on which class of hardware*, never
*how* a particular inference engine is driven. The vLLM backend (and any future
backend) imports them; they import no engine and no container backend, so the
experiment surface stays free of engine-shaped details.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..runtime.types import ContainerConfig, ContainerResult, Mount, PullPolicy

if TYPE_CHECKING:
    from ..search_space.types import ExperimentSpace


class HardwareTarget(StrEnum):
    """The class of accelerator an experiment is planned against.

    Detection resolves the host to one of these, and the container runtime
    config may override it. Values are lower-case so they serialise cleanly into
    logs and image/tag lookups.
    """

    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


class DType(StrEnum):
    """Model weight and activation precision, named engine-neutrally.

    A backend translates these members to its own spelling; vLLM, for example,
    spells :attr:`BFLOAT16` ``bfloat16`` and additionally accepts the ``half``
    and ``float`` aliases this curated enum does not expose.
    """

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class QuantizationMethod(StrEnum):
    """Weight quantization method, named engine-neutrally.

    The set is curated, not exhaustive: engines register quantization methods
    dynamically, so a backend maps only the members it supports and the long
    tail is reached through the backend's raw-argument escape hatch.
    """

    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"


class KVCacheDType(StrEnum):
    """Precision of the KV cache, named engine-neutrally.

    Only the portable core of an engine's cache precisions is modelled;
    engine-specific formats (vLLM's ``turboquant_*`` and ``nvfp4*``, for
    instance) are left to the backend's raw-argument escape hatch.
    """

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class AttentionBackend(StrEnum):
    """Attention implementation to use, named engine-neutrally.

    Attention backends are engine- and platform-specific and the set grows over
    time, so only widely available implementations are modelled. A backend maps
    these to its own names; vLLM spells :attr:`FLASH_ATTENTION` ``FLASH_ATTN``.
    """

    FLASH_ATTENTION = "flash_attention"
    FLASH_INFER = "flash_infer"
    XFORMERS = "xformers"


@dataclass(frozen=True, slots=True)
class DetectedHardware:
    """The accelerator class and usable device count resolved for a host.

    ``device_count`` supplies the default tensor-parallel size for a run; it is
    one for CPU hosts and for an overridden target whose vendor cannot be
    probed.
    """

    target: HardwareTarget
    device_count: int


@dataclass(frozen=True, slots=True)
class ContainerRuntimeConfig:
    """Engine-neutral, host-facing settings for containerising a benchmark run.

    This domain covers *how* an experiment is containerised independently of the
    inference engine: which accelerator class to target, which image to run,
    resource limits, and the mounts and environment the run needs. The backend
    fills in everything it requires beyond this (device mapping, the engine
    command, engine-specific mounts).

    ``target`` and ``image`` are overrides: ``None`` leaves the choice to
    hardware detection and the backend's default image. ``shm_size`` defaults to
    ``"8g"`` because vLLM needs a large ``/dev/shm`` to avoid crashes.

    ``selinux_opts`` carries Podman SELinux label options (for example
    ``disable``, without the CLI's ``label=`` prefix) that the backend copies
    into the planned container. ``None`` leaves the choice to the runtime's
    automatic default, an empty tuple asks for no label options, and a
    populated tuple asks for exactly those.
    """

    target: HardwareTarget | None = None
    image: str | None = None
    mem_limit: int | str | None = None
    shm_size: int | str | None = "8g"
    selinux_opts: tuple[str, ...] | None = None
    pull_policy: PullPolicy = "missing"
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: tuple[Mount, ...] = ()

    def __post_init__(self) -> None:
        """Store ``env`` as a read-only view so the frozen config is immutable."""
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))

    def __hash__(self) -> int:
        """Hash the config, ordering ``env`` so equal configs hash equally."""
        return hash(
            (
                self.target,
                self.image,
                self.mem_limit,
                self.shm_size,
                self.selinux_opts,
                self.pull_policy,
                tuple(sorted(self.env.items())),
                self.volumes,
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Engine-neutral description of the evaluation workload.

    This domain covers *what* is measured independently of the serving
    parameters being varied: the model under test and the shape of the workload
    it is driven with. ``model`` is required and accepts either a Hugging Face
    identifier or a host path to a local checkpoint.

    The layer deliberately imposes no workload defaults of its own: every
    optional field is ``None`` until the caller sets it, so a run never silently
    measures something other than what was configured. The backend decides how
    these fields reach the engine and which engine flag each one targets.

    ``timeout`` is a wall-clock limit for the whole run, in seconds; ``None``
    leaves the run unbounded.
    """

    model: str
    dataset_name: str | None = None
    dataset_path: str | None = None
    num_prompts: int | None = None
    input_len: int | None = None
    output_len: int | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Engine-neutral serving parameters, the part varied between experiments.

    This domain is flat, strictly typed, and carries no free-form passthrough:
    every field is a first-class parameter of the experiment, so an experiment
    reads clearly and cannot silently smuggle engine-shaped junk into a config.
    The backend decides how each field reaches the engine and which CLI flag it
    targets; adding a parameter is an explicit two-step of adding a field here
    and a mapping entry there.

    Every field is optional and ``None`` means "leave it to the engine's own
    default", so the layer never asserts a serving value the caller did not ask
    for. ``tensor_parallel_size`` is additionally resolved by the backend to the
    detected device count when left unset. A series of experiments is expressed
    by holding the container and benchmark domains fixed and varying this one
    with :func:`dataclasses.replace`.

    The categorical fields (:attr:`dtype`, :attr:`quantization`,
    :attr:`kv_cache_dtype`, :attr:`attention_backend`) hold engine-neutral enums
    rather than engine strings, so an experiment never carries one engine's
    vocabulary. Each backend translates the members it supports to its own
    spellings and rejects the rest.
    """

    dtype: DType | None = None
    quantization: QuantizationMethod | None = None
    kv_cache_dtype: KVCacheDType | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    seed: int | None = None
    attention_backend: AttentionBackend | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, immutable description of one benchmark experiment.

    It composes the three domains — container runtime, benchmark workload, and
    LLM serving parameters — as separate types rather than flattening them, so
    the fixed and varied parts of an experiment stay obvious at the type level.
    A series holds ``container`` and ``benchmark`` fixed and varies ``llm`` with
    :func:`dataclasses.replace`.

    The config is hashable and its sub-configs are frozen, so an experiment can
    be stored, logged, compared, and used as a cache key.
    """

    container: ContainerRuntimeConfig
    benchmark: BenchmarkConfig
    llm: LLMConfig


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The typed outcome of one benchmark run.

    The metrics are parsed into first-class fields so experiments can be
    compared directly, while ``raw_json`` keeps the engine's full result payload
    and ``raw_result`` keeps the container's raw output, so details the typed
    result omits remain inspectable.

    ``tokens_per_second`` is the engine's *total* token throughput: prompt and
    generated tokens combined, divided by elapsed wall-clock time. It is not
    generation-only, so a workload with a large prompt-to-output ratio is
    dominated by prefill. The name mirrors the engine's own result key, which
    exposes no output-only throughput.

    ``target``, ``image``, and ``devices`` record what actually ran rather than
    what was requested, so a result is reproducible even when the image or
    device mapping was resolved by default.
    """

    tokens_per_second: float
    requests_per_second: float
    elapsed_time: float
    num_prompts: int
    target: HardwareTarget
    image: str
    devices: tuple[str, ...]
    raw_json: Mapping[str, object]
    raw_result: ContainerResult

    def __post_init__(self) -> None:
        """Store ``raw_json`` as a read-only view so the result is immutable."""
        object.__setattr__(self, "raw_json", MappingProxyType(dict(self.raw_json)))

    def __hash__(self) -> int:
        """Hash the result, ordering ``raw_json`` so equal results hash equally."""
        return hash(
            (
                self.tokens_per_second,
                self.requests_per_second,
                self.elapsed_time,
                self.num_prompts,
                self.target,
                self.image,
                self.devices,
                tuple(sorted(self.raw_json.items())),
                self.raw_result,
            )
        )


class ExperimentError(RuntimeError):
    """Raised for every failure of the inference experiment layer.

    A single type covers the three ways a benchmark run can fail: the container
    run itself failed, the engine never wrote its result file, or the result
    file could not be parsed. Callers therefore handle benchmark failures
    uniformly instead of catching a different exception per stage.

    The raw container result is attached as ``raw_result`` whenever output was
    captured, so a failure can be debugged from what the engine actually
    printed. The originating exception, when there is one (a
    :class:`~llm_optimization.runtime.ContainerRunError` for a failed run, or a
    decoding/parsing error for bad output), is preserved as ``__cause__``.
    """

    def __init__(self, message: str, *, raw_result: ContainerResult | None = None) -> None:
        super().__init__(message)
        self.raw_result = raw_result


@runtime_checkable
class InferenceBackend(Protocol):
    """The engine seam: how an experiment becomes a run and a result is read.

    The experiment layer above is engine-agnostic; every engine-specific detail
    lives behind this protocol. The orchestrator resolves the host hardware once
    and passes it to :meth:`plan`, so a backend never probes the host itself and
    can be unit-tested with injected hardware.

    Before a series runs, :meth:`validate` checks a search-space definition
    against the backend, so a parameter or value the engine does not support is
    rejected up front rather than midway through an expensive sweep.

    An output directory is optional because not every engine writes a result
    file. A backend that does not need one ignores the argument; a backend that
    does is responsible for creating it when the caller supplies ``None`` and
    for reading from the same directory in :meth:`parse_result`. The planned
    container configuration is passed back into :meth:`parse_result` as well, so
    a backend can record what actually ran (resolved image, devices, and so on)
    without retaining state between the two calls.
    """

    def validate(self, space: ExperimentSpace) -> Sequence[str]:
        """Report every way this backend cannot support ``space``.

        The space is assumed to have already passed engine-agnostic validation;
        this is the engine-specific half, asking whether the backend can forward
        each declared parameter and value. Problems are returned as
        human-readable strings rather than raised, so the caller can aggregate
        them across the whole space and report them at once. An empty sequence
        means the backend supports every dimension.

        A :class:`~llm_optimization.search_space.types.Default` dimension carries
        no value to support-check and is accepted as long as the backend knows
        the parameter itself.
        """
        ...

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        output_dir: Path | None = None,
    ) -> ContainerConfig:
        """Turn an experiment and the detected hardware into a container run.

        Returns a concrete
        :class:`~llm_optimization.runtime.ContainerConfig` — the resolved image,
        device mapping, mounts, and the engine's benchmark command — which the
        orchestrator executes through the runtime layer unchanged.

        Planning is not required to be side-effect free: a backend may create the
        host directories it mounts (for example an output directory and a cache
        directory) while planning, because they must exist before the run can be
        planned to use them.
        """
        ...

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        """Turn a captured container outcome into a typed benchmark result.

        ``plan`` is the container configuration :meth:`plan` produced for the
        run, so the result can record what actually ran rather than what was
        requested. Raises :class:`ExperimentError` when the engine produced no
        usable result, carrying the raw container result where one was
        captured.
        """
        ...
