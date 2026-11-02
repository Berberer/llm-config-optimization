"""Engine-neutral types for the inference experiment layer.

These types describe *what* to evaluate and *on which class of hardware*, never
*how* a particular inference engine is driven. The vLLM backend (and any future
backend) imports them; they import no engine and no container backend, so the
experiment surface stays free of engine-shaped details.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..runtime.types import ContainerConfig, ContainerResult, Mount, PullPolicy

if TYPE_CHECKING:
    from ..search_space.types import ExperimentSpace, SpaceAssessment


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

    A checkpoint is produced with exactly one method, so this is a property of
    the model rather than a value a sweep varies. The set is curated, not
    exhaustive: a backend maps only the members it can load, and the long tail
    of engine-specific methods is reached through the backend's raw-argument
    escape hatch.
    """

    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"


class KVCacheDType(StrEnum):
    """Precision of the KV cache, named engine-neutrally.

    The float members are the portable element formats both engines serve; the
    quantized members are llama.cpp's cache formats, which trade cache
    precision for memory. No single engine serves the whole set, so a backend
    maps only the members it can load and every other member is rejected as
    unsupported. vLLM's engine-specific formats (``turboquant_*`` and
    ``nvfp4*``, for instance) stay behind its raw-argument escape hatch.
    """

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FLOAT32 = "float32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    IQ4_NL = "iq4_nl"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"


class GpuLayersMode(StrEnum):
    """How many layers llama.cpp offloads to the GPU, named engine-neutrally.

    The modes are the engine's two non-numeric choices: :attr:`AUTO` lets
    llama.cpp pick a count, and :attr:`ALL` offloads every layer. An exact
    count is a plain integer rather than a member, so a GPU-layer dimension is
    an int-or-enum union. vLLM has no equivalent setting and rejects the field.
    """

    AUTO = "auto"
    ALL = "all"


class CpuMoeMode(StrEnum):
    """How many Mixture-of-Experts layers llama.cpp keeps in system RAM.

    llama.cpp accepts either an exact layer count or :attr:`ALL`, which keeps
    every expert layer on the CPU through the value-less ``--cpu-moe`` flag. An
    exact count is a plain integer rather than a member, so a CPU-expert
    dimension is an int-or-enum union. vLLM has no equivalent setting and
    rejects the field.
    """

    ALL = "all"


class LoadMode(StrEnum):
    """How an engine loads a checkpoint's weights into memory.

    The members name the loading strategy, not any one engine's spelling: a
    backend maps them to its own names, so :attr:`MMAP_MLOCK` may become an
    engine's ``mmap+mlock``. :attr:`AUTO` leaves the choice to the engine, and
    the remaining members select a specific strategy.
    """

    AUTO = "auto"
    NONE = "none"
    MMAP = "mmap"
    MLOCK = "mlock"
    MMAP_MLOCK = "mmap_mlock"
    DIO = "dio"


class FitMode(StrEnum):
    """Whether an engine may rewrite arguments to fit the available memory.

    This is a correctness pin rather than a tuning knob: enabling it lets the
    engine adjust the declared configuration to fit the device, while disabling
    it makes a run measure the arguments the search space declared.
    """

    ON = "on"
    OFF = "off"


class AttentionBackend(StrEnum):
    """Attention implementation to use, named engine-neutrally.

    Attention backends are engine- and platform-specific and the set grows over
    time, so only widely available implementations are modelled. A backend maps
    these to its own names; vLLM spells :attr:`FLASH_ATTENTION` ``FLASH_ATTN``.
    :attr:`AUTO` and :attr:`OFF` are llama.cpp's flash-attention settings and
    have no vLLM equivalent, so vLLM rejects them.
    """

    FLASH_ATTENTION = "flash_attention"
    FLASH_INFER = "flash_infer"
    XFORMERS = "xformers"
    AUTO = "auto"
    OFF = "off"


class All2AllBackend(StrEnum):
    """Transport for vLLM's Mixture-of-Experts all-to-all communication.

    Only the portable transports are modelled: both work in the default vLLM
    image without extra communication libraries. Faster transports that require
    additional packages stay behind the backend's raw-argument escape hatch.
    llama.cpp has no equivalent setting and rejects the field.
    """

    ALLGATHER_REDUCESCATTER = "allgather_reducescatter"
    NAIVE = "naive"


class SpeculativeMethod(StrEnum):
    """Drafting strategy for speculative decoding, named engine-neutrally.

    A backend maps these members to its own spellings; vLLM spells them on its
    ``--speculative-config`` object and llama.cpp on ``--spec-type``. The set is
    curated to the methods both engines serve: engine-only variants and
    per-model aliases remain reachable through the backend's raw-argument
    escape hatch.

    :attr:`MTP` uses the target model's own multi-token-prediction heads, so it
    needs no separate draft checkpoint; the other members take a draft model,
    except :attr:`NGRAM`, which is model-free.
    """

    DRAFT_MODEL = "draft_model"
    EAGLE3 = "eagle3"
    MTP = "mtp"
    DFLASH = "dflash"
    DSPARK = "dspark"
    NGRAM = "ngram"


class SpeculativeMode(StrEnum):
    """Whether a trial runs with the configured speculative setup.

    The members are spelled ``enabled``/``disabled`` rather than ``on``/``off``
    because PyYAML parses bare ``on`` and ``off`` as booleans, which the search
    space loader's string coercion would reject.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"


class SpecDraftSampling(StrEnum):
    """How a speculative draft model samples the tokens it proposes.

    :attr:`GREEDY` drafts the most likely token at each step, while
    :attr:`PROBABILISTIC` samples from the draft distribution. The choice trades
    the acceptance rate against the diversity of the proposals.
    """

    GREEDY = "greedy"
    PROBABILISTIC = "probabilistic"


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
class LocalCheckpoint:
    """A model that already exists on the host.

    A backend's model-resolution hook returns this when the model argument
    names an existing path, so the composition root fetches nothing and the run
    mounts the path directly.
    """

    path: Path


@dataclass(frozen=True, slots=True)
class HubCheckpoint:
    """A model to fetch from the Hugging Face Hub.

    ``repo_id`` names the repository and ``filename`` optionally restricts the
    fetch to one file inside it — a single GGUF quant, for example — with
    ``None`` fetching the whole snapshot. The backend decides which repository
    and file a model argument refers to; the composition root performs the
    download.
    """

    repo_id: str
    filename: str | None = None


type ModelRequest = LocalCheckpoint | HubCheckpoint
"""An engine-neutral description of the checkpoint a run needs.

A backend's :meth:`InferenceBackend.model_request` returns one of these, so the
composition root can resolve it to a local path — downloading when necessary —
without learning an engine's model-argument rules.
"""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Engine-neutral description of the evaluation workload.

    This domain covers *what* is measured independently of the serving
    parameters being varied: the model under test, the shape of the workload it
    is driven with, and how many times that workload is measured. ``model`` is
    required and accepts either a Hugging Face identifier or a host path to a
    local checkpoint.

    ``resolved_path`` is where the composition root put the checkpoint after
    asking the backend's model-resolution hook what to fetch; it is ``None``
    until then, and a backend plans a run from ``model`` alone when it is
    unset. It is kept separate from ``model`` so the user-facing identifier
    still names the served model even when the weights were fetched into the
    Hub cache.

    The workload-shape fields impose no defaults of their own: each is ``None``
    until the caller sets it, so a run never silently measures something other
    than what was configured. The measurement controls are different: a trial
    takes several measurements so its objective reflects the configuration
    rather than a single noisy sample, so ``repetitions``, ``warmup``, and
    ``startup_timeout`` carry the documented defaults. The backend decides how
    these fields reach the engine and which engine flag each one targets.

    ``repetitions`` is the number of measured runs taken after warmup, and
    ``warmup`` the number of discarded runs taken first. ``startup_timeout``
    bounds the wait for the served model to become healthy, in seconds.
    ``timeout`` is the overall wall-clock limit for the whole run — startup,
    warmup, and every measured run together — in seconds; ``None`` leaves the
    run unbounded. ``max_request_tokens`` is the longest request the workload
    will send — input plus generation — when that is known before planning, for
    example after a JSONL dataset has been tokenized by the engine. It is
    ``None`` until then, so a backend that needs the length can choose a safe
    fallback rather than guessing. The constructor rejects a ``repetitions``
    below one, a negative ``warmup``, and a non-positive ``startup_timeout``.
    """

    model: str
    resolved_path: Path | None = None
    dataset_name: str | None = None
    dataset_path: str | None = None
    num_prompts: int | None = None
    input_len: int | None = None
    output_len: int | None = None
    max_request_tokens: int | None = None
    repetitions: int = 3
    warmup: int = 1
    startup_timeout: float = 600.0
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Reject measurement controls outside their valid range.

        A repetition count below one would leave a trial with no measured run,
        a negative warmup is meaningless, and a non-positive startup timeout
        could never see a model load. These are configuration errors rather
        than runtime ones, so they raise ``ValueError`` at construction time.
        """
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be at least 1, got {self.repetitions}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")
        if self.startup_timeout <= 0:
            raise ValueError(f"startup_timeout must be positive, got {self.startup_timeout}")


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """Engine-neutral description of the server a trial measures against.

    A trial no longer runs a self-contained benchmark container: it starts the
    engine's server and drives it from the host. This type is the seam between
    the two, so the orchestrator and the host client never learn an engine's
    vocabulary. The backend plans the container the server runs in and
    describes how the host reaches and resets it.

    ``container`` carries the server's command, published ports, mounts,
    environment, and devices. ``health_path`` is the HTTP path the client polls
    until the server is ready, and ``health_timeout`` is how long it waits
    before giving up, in seconds. ``reset_paths`` lists the endpoints the
    client calls between measured runs to drop engine-side caches; an empty
    tuple means the engine exposes none and the client skips resetting.

    ``metrics_path`` is the HTTP path the client scrapes for the engine's
    Prometheus metrics around each measured run. It is plain engine-neutral
    data: ``None`` means the engine exposes no scrapeable metrics and the
    client does nothing extra, while a declared path is the client's
    instruction to snapshot it. Which series the engine publishes and how they
    are interpreted stay in the backend that owns the engine.

    ``tokenize_path`` is the optional HTTP path that tokenizes text. When it is
    declared, a preflight run can ask the server for a dataset's real token
    lengths before the sweep. ``tokenize_prompt_field`` and
    ``tokenize_model_field`` describe the request body: the prompt is sent under
    the former and, when the latter is not ``None``, the served model name under
    it. A response is expected to carry a ``tokens`` array. ``None`` for the
    path means the engine has no tokenizer endpoint this layer can use, so the
    preflight is skipped.
    """

    container: ContainerConfig
    health_path: str
    health_timeout: float
    reset_paths: tuple[str, ...] = ()
    metrics_path: str | None = None
    tokenize_path: str | None = None
    tokenize_prompt_field: str = "prompt"
    tokenize_model_field: str | None = None


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

    The speculative fields describe the speculative-decoding setup. The method
    and the draft model are fixed for a run and are not tunable: they are
    checkpoint-specific, so sweeping them would mostly produce server-startup
    failures. The mode and the depth are the sweep's genuine questions, so
    :attr:`speculative_mode` and :attr:`num_speculative_tokens` are filled by
    the search space. An unset mode means the configured setup's default, which
    is enabled; an unset depth leaves the engine's own default in place. A
    disabled mode starts the engine with no speculative flags at all, so the
    draft model's memory is genuinely freed.

    The draft-only fields shape a separate draft model and apply exactly when
    one is configured: :attr:`spec_draft_n_min` and :attr:`spec_draft_p_split`
    control the depth and split probability of drafting,
    :attr:`spec_draft_sampling` selects greedy or probabilistic drafting, and
    :attr:`n_gpu_layers_draft`, :attr:`n_cpu_moe_draft`, and
    :attr:`draft_kv_cache_dtype` place the draft's layers and cache. They are
    llama.cpp-only, and like the main offload counts the two draft offload
    fields are int-or-enum unions.

    :attr:`prefix_caching` is a boolean toggle, forwarded as the enabling flag
    or its disabling counterpart so both sides of the switch are explicit.
    :attr:`n_gpu_layers` is an int-or-enum union: an exact offloaded-layer count
    or one of llama.cpp's :class:`GpuLayersMode` choices.

    :attr:`expert_parallel` is vLLM's MoE expert-parallelism toggle, forwarded
    as the enabling flag or its disabling counterpart, and
    :attr:`all2all_backend` selects the transport its expert parallelism uses.
    Both are vLLM-only and are the MoE counterpart of the dense-model fields;
    llama.cpp rejects them. A backend that does not forward them never sees a
    value, because the search space only requires the parameters it applies.

    :attr:`n_cpu_moe` is llama.cpp's MoE CPU-offload setting: an exact count of
    expert layers to keep in system RAM, or :class:`CpuMoeMode`'s ``all`` to
    keep every one of them there. It is an int-or-enum union like
    :attr:`n_gpu_layers`, and it is llama.cpp-only, so vLLM rejects it.

    :attr:`n_cpu_ffn` keeps the dense feed-forward weights of an exact number of
    layers in system RAM, :attr:`kv_offload` offloads the KV cache to the device,
    and :attr:`op_offload` lets the engine run tensor operations there.
    :attr:`host_buffer` and :attr:`repack` toggle the host buffer and weight
    repacking, :attr:`swa_full` selects a full-size sliding-window cache, and
    :attr:`load_mode` chooses how the checkpoint's weights are loaded. All of
    them are llama.cpp-only and use positive semantics, so the backend maps a
    ``True`` to the enabling flag and a ``False`` to the disabling one.

    The cache and scheduling fields are llama.cpp-only as well.
    :attr:`cache_ram` bounds the host prompt cache in MiB, where ``-1`` means no
    limit and ``0`` disables it; :attr:`ctx_checkpoints` sizes the per-slot
    context-checkpoint cache and :attr:`checkpoint_min_step` the minimum token
    spacing between checkpoints, with ``0`` meaning no minimum. The booleans
    :attr:`kv_unified`, :attr:`continuous_batching`, and :attr:`cache_idle_slots`
    toggle a unified KV buffer, continuous batching, and idle-slot caching, again
    with positive semantics.

    :attr:`max_num_batched_tokens` and :attr:`ubatch_size` split llama.cpp's
    logical and physical batches: the first is the logical batch size and the
    second the micro-batch the engine processes at once, so prefill batching can
    be tuned without changing the logical batch. vLLM forwards only its own
    batching cap and omits :attr:`ubatch_size`.

    The CPU and process fields are llama.cpp-only as well. :attr:`threads` and
    :attr:`threads_batch` size the generation and prompt-processing thread
    pools, while :attr:`poll` selects the engine's busy-wait polling level and
    :attr:`prio` the process priority.

    The fit family is a correctness pin rather than a tuning knob, so it stays
    out of the search space: :attr:`fit` chooses whether llama.cpp may rewrite
    arguments to fit memory, and :attr:`fit_target` and :attr:`fit_ctx` give
    its optional target margin and minimum context. They are llama.cpp-only,
    fixed for a run, and suppressed unless :attr:`fit` is enabled.

    :attr:`resolved_draft_path` is a resolution artifact like
    :class:`BenchmarkConfig`'s ``resolved_path``: it holds the local path the
    composition root fetched the draft to, and a backend never forwards it to
    the engine. It is set on the base configuration before backend validation
    and planning.
    """

    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_active_seqs: int | None = None
    max_num_scheduled_tokens: int | None = None
    ubatch_size: int | None = None
    dtype: DType | None = None
    quantization: QuantizationMethod | None = None
    kv_cache_dtype: KVCacheDType | None = None
    attention_backend: AttentionBackend | None = None
    n_gpu_layers: int | GpuLayersMode | None = None
    n_cpu_moe: int | CpuMoeMode | None = None
    n_cpu_ffn: int | None = None
    kv_offload: bool | None = None
    op_offload: bool | None = None
    host_buffer: bool | None = None
    repack: bool | None = None
    swa_full: bool | None = None
    load_mode: LoadMode | None = None
    prefix_caching: bool | None = None
    cache_ram: int | None = None
    ctx_checkpoints: int | None = None
    checkpoint_min_step: int | None = None
    kv_unified: bool | None = None
    continuous_batching: bool | None = None
    cache_idle_slots: bool | None = None
    tensor_parallel_size: int | None = None
    expert_parallel: bool | None = None
    all2all_backend: All2AllBackend | None = None
    speculative_method: SpeculativeMethod | None = None
    draft_model: str | None = None
    speculative_mode: SpeculativeMode | None = None
    num_speculative_tokens: int | None = None
    spec_draft_n_min: int | None = None
    spec_draft_p_split: float | None = None
    spec_draft_sampling: SpecDraftSampling | None = None
    n_gpu_layers_draft: int | GpuLayersMode | None = None
    n_cpu_moe_draft: int | CpuMoeMode | None = None
    draft_kv_cache_dtype: KVCacheDType | None = None
    threads: int | None = None
    threads_batch: int | None = None
    poll: int | None = None
    prio: int | None = None
    fit: FitMode | None = None
    fit_target: int | None = None
    fit_ctx: int | None = None
    seed: int | None = None
    gpu_memory_utilization: float | None = None
    resolved_draft_path: Path | None = None


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


def _hashable(value: object) -> object:
    """Return a hashable canonical form of one engine payload value.

    The engine's payload is JSON, so it is built from mappings, lists, and
    scalars. A mapping becomes a tuple of its key/value pairs sorted by key, and
    a list a tuple of its items, so nested structures — such as a list-valued
    field in a client payload — are hashable and two payloads that differ only
    in key order hash equally. Scalars pass through unchanged.
    """
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


class VramSource(StrEnum):
    """How a VRAM measurement attributed memory to a trial.

    :attr:`PROCESS` sums the GPU processes whose cgroup names the trial's
    container, so another workload sharing the device does not inflate the
    number. :attr:`DEVICE` is the device-level used-memory delta against the
    idle baseline, used when no process can be attributed. It is a measurement
    provenance, not a vendor spelling, so the experiment surface can carry it.
    """

    PROCESS = "process"
    DEVICE = "device"


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """One peak memory observation for a trial or a measured run.

    Both fields are *peaks*, not averages, because a memory requirement is a
    ceiling: a consumer that averaged them would understate what the
    configuration needs. Each is ``None`` when nothing was measured — never
    ``0`` — so "unmeasured" and "used nothing" stay distinguishable.

    ``peak_vram_bytes`` is the peak accelerator memory attributed to the
    observation; it is ``None`` on a CPU target, when no vendor memory probe was
    available, or when the sample source failed. ``peak_ram_bytes`` is the peak
    system memory the server container held, which deliberately includes page
    cache so an mmap-based engine's weights count as memory the configuration
    occupies.

    How a VRAM number was obtained — per-process attribution or a device-level
    delta against a baseline — is recorded beside the run's raw samples rather
    than here, so this value stays a plain measurement and the engine-neutral
    experiment surface carries no vendor vocabulary.
    """

    peak_vram_bytes: int | None = None
    peak_ram_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One timestamped memory observation behind a :class:`ResourceUsage`.

    The raw samples are kept for audit: a peak alone cannot say how it was
    obtained or whether the series was stable. ``timestamp`` is the sampler
    clock's reading when the sample was taken, and ``vram_source`` records
    whether the VRAM figure came from the container's own processes or from a
    device-level delta against the idle baseline. Each value is ``None`` when
    that half of the observation was unavailable, so an unmeasured field stays
    distinguishable from a measured zero.
    """

    timestamp: float
    vram_bytes: int | None = None
    vram_source: VramSource | None = None
    ram_bytes: int | None = None


type ResourceMarker = int
"""An opaque marker for a point in a sampler's observation series.

A consumer gets one from :meth:`ResourceSampler.mark` and hands it back to
:meth:`ResourceSampler.peak` or :meth:`ResourceSampler.samples`. The value
means nothing outside the sampler that produced it, so no consumer interprets
it as a time or an index.
"""


@runtime_checkable
class ResourceSampler(Protocol):
    """Best-effort memory measurement of a running server container.

    A sampler observes a container's memory on a fixed cadence while the
    container runs and answers peak queries over that series. A consumer
    brackets a measured window with :meth:`mark` and :meth:`peak`, or reads
    :meth:`lifetime_peak` for the whole-container figure. The marker
    :meth:`mark` returns is opaque, so the consumer learns nothing about how the
    sampler obtains its numbers.

    Sampling is a diagnostic and is always best-effort: an absent vendor probe,
    an unreadable container stat, or a failed sample leaves a field ``None``
    rather than raising, and every query is safe to call whether or not the
    sampler ever ran. The engine-neutral experiment surface therefore carries
    no vendor, engine, or container vocabulary.
    """

    @property
    def baseline_vram_bytes(self) -> int | None:
        """The idle device memory the device-level fallback subtracts."""
        ...

    def start(self) -> None:
        """Begin sampling; a sampler that is already running is left alone."""
        ...

    def stop(self) -> None:
        """Stop sampling and wait for the observation to finish."""
        ...

    def mark(self) -> ResourceMarker:
        """Record a point in the observation series and return its marker."""
        ...

    def samples(self, since: ResourceMarker) -> Sequence[ResourceSample]:
        """The raw observations recorded at or after ``since``, oldest first."""
        ...

    def peak(self, since: ResourceMarker) -> ResourceUsage:
        """The peak of the samples recorded at or after ``since``.

        A field no sample measured stays ``None``, including when the window
        saw no sample at all.
        """
        ...

    def lifetime_peak(self) -> ResourceUsage:
        """The peak over every sample the sampler recorded."""
        ...


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """The typed metrics of one measured benchmark run.

    One trial takes several measured runs; each becomes a :class:`RunMetrics`
    and the trial's :class:`ExperimentResult` aggregates them. ``raw_json``
    keeps the engine's full payload for that run, so details the typed metrics
    omit remain inspectable, and it is stored as a read-only view so a run is
    immutable and hashable like the result that holds it.

    ``tokens_per_second`` is the engine's *total* token throughput: prompt and
    generated tokens combined, divided by elapsed wall-clock time. It is not
    generation-only, so a workload with a large prompt-to-output ratio is
    dominated by prefill. The host client computes it from the engine's usage
    payload, which exposes no output-only throughput.

    ``prefill_tokens_per_second`` and ``decode_tokens_per_second`` separate the
    combined number into the two phases of inference. They are phase-work
    rates: total phase tokens divided by the total phase seconds the engine
    reported, not a partition of the run's wall-clock time — continuous
    batching and chunked prefill interleave the phases, so only the engine can
    attribute work to one. Either is ``None`` when the run's engine did not
    report what the rate needs, so a consumer that requires a phase must check
    rather than fall back to the combined number.

    ``resources`` is the peak memory observed during this run's measured window,
    or ``None`` when the run was not bracketed by a resource sampler or nothing
    could be measured. It describes the window only; the whole-container peak is
    kept on :attr:`ExperimentResult.lifetime_resources`.
    """

    tokens_per_second: float
    requests_per_second: float
    elapsed_time: float
    num_prompts: int
    raw_json: Mapping[str, object]
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    resources: ResourceUsage | None = None

    def __post_init__(self) -> None:
        """Store ``raw_json`` as a read-only view so the run is immutable."""
        object.__setattr__(self, "raw_json", MappingProxyType(dict(self.raw_json)))

    def __hash__(self) -> int:
        """Hash the run, canonicalising ``raw_json`` so equal runs hash equally."""
        return hash(
            (
                self.tokens_per_second,
                self.requests_per_second,
                self.elapsed_time,
                self.num_prompts,
                self.prefill_tokens_per_second,
                self.decode_tokens_per_second,
                self.resources,
                _hashable(self.raw_json),
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkOutcome:
    """The runs a host-side benchmark client produced for one trial.

    The client measures the server from outside the container and hands back
    what it observed, so the orchestrator can stop the server and let the
    backend assemble the final :class:`ExperimentResult` from data that no
    longer depends on a running container. ``runs`` holds one
    :class:`RunMetrics` per successful measured run, in run order, and each
    run's ``raw_json`` keeps the client payloads behind its typed metrics.
    ``failed_runs`` counts the measured runs whose requests all failed; a run
    that completed at least one request is successful even when some of its
    requests failed, mirroring the engine's own client.
    """

    runs: tuple[RunMetrics, ...]
    failed_runs: int

    @property
    def raw_payloads(self) -> tuple[Mapping[str, object], ...]:
        """The raw client payload of each successful run, in run order."""
        return tuple(run.raw_json for run in self.runs)


def _max_present(values: Iterable[int | None]) -> int | None:
    """Largest of the values that are present, or ``None`` when none is.

    A peak aggregate takes a maximum rather than a mean, and it ignores absent
    measurements instead of treating them as zero, so a run that measured one
    field cannot drag another run's number down.
    """
    present: list[int] = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return None
    return max(present)


def _mean_present(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are present, or ``None`` when none is.

    A phase rate is optional on a run, so an aggregate averages only the runs
    that reported one rather than treating a missing measurement as zero, and
    a series with no measurement at all is ``None``.
    """
    present: list[float] = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return None
    return sum(present) / len(present)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The typed outcome of one benchmark trial.

    A trial takes one or more measured runs, so the result stores the
    successful runs as ``runs`` and the number that failed as ``failed_runs``.
    ``runs`` is the single source of truth: the scalar metrics are derived from
    it rather than stored alongside it, so the aggregate cannot drift from the
    runs it summarizes. ``tokens_per_second``, ``requests_per_second``, and
    ``elapsed_time`` are the means over the successful runs, ``num_prompts`` is
    their summed measured prompt count, and ``raw_json`` is the tuple of
    per-run payloads. ``prefill_tokens_per_second`` and
    ``decode_tokens_per_second`` are likewise means, taken over the runs that
    reported a phase rate; each is ``None`` when no run did, mirroring the
    optional field on :class:`RunMetrics`.

    ``raw_result`` keeps the container's raw output, so details the typed
    result omits remain inspectable. ``target``, ``image``, and ``devices``
    record what actually ran rather than what was requested, so a result is
    reproducible even when the image or device mapping was resolved by default.

    ``lifetime_resources`` is the peak memory observed over the server
    container's whole life — model load, health wait, warmup, and the measured
    runs — so it is the headline hardware requirement of the configuration. It
    defaults to an all-``None`` :class:`ResourceUsage`, which keeps results
    built before measurements existed valid.

    A result must carry at least one successful run: an empty ``runs`` tuple
    cannot describe a trial, so the constructor rejects it.
    """

    runs: tuple[RunMetrics, ...]
    failed_runs: int
    target: HardwareTarget
    image: str
    devices: tuple[str, ...]
    raw_result: ContainerResult
    lifetime_resources: ResourceUsage = field(default_factory=ResourceUsage)

    def __post_init__(self) -> None:
        """Reject a result that carries no successful run."""
        if not self.runs:
            raise ValueError("an experiment result requires at least one successful run")

    @property
    def tokens_per_second(self) -> float:
        """Mean total token throughput over the successful runs."""
        return sum(run.tokens_per_second for run in self.runs) / len(self.runs)

    @property
    def requests_per_second(self) -> float:
        """Mean request throughput over the successful runs."""
        return sum(run.requests_per_second for run in self.runs) / len(self.runs)

    @property
    def elapsed_time(self) -> float:
        """Mean elapsed wall-clock time over the successful runs."""
        return sum(run.elapsed_time for run in self.runs) / len(self.runs)

    @property
    def prefill_tokens_per_second(self) -> float | None:
        """Mean prefill throughput over the runs that reported it, or ``None``."""
        return _mean_present(run.prefill_tokens_per_second for run in self.runs)

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Mean decode throughput over the runs that reported it, or ``None``."""
        return _mean_present(run.decode_tokens_per_second for run in self.runs)

    @property
    def measured_peak_resources(self) -> ResourceUsage:
        """Element-wise maximum of the runs' measured-window peaks.

        The measured windows are adjacent but not contiguous, so this is the
        maximum of the per-run peaks rather than a peak over their union; every
        run is the single source of truth for its own window. A field no run
        measured stays ``None`` rather than becoming zero.
        """
        usages = [run.resources for run in self.runs if run.resources is not None]
        return ResourceUsage(
            peak_vram_bytes=_max_present(usage.peak_vram_bytes for usage in usages),
            peak_ram_bytes=_max_present(usage.peak_ram_bytes for usage in usages),
        )

    @property
    def num_prompts(self) -> int:
        """Total measured prompt count across the successful runs."""
        return sum(run.num_prompts for run in self.runs)

    @property
    def raw_json(self) -> tuple[Mapping[str, object], ...]:
        """The engine payload of each successful run, in run order."""
        return tuple(run.raw_json for run in self.runs)

    def __hash__(self) -> int:
        """Hash the result, so equal results hash equally."""
        return hash(
            (
                self.runs,
                self.failed_runs,
                self.target,
                self.image,
                self.devices,
                self.raw_result,
                self.lifetime_resources,
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
    :class:`~llm_config_optimization.runtime.ContainerRunError` for a failed run, or a
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
    rejected up front rather than midway through an expensive sweep. The
    composition root asks :meth:`model_request` what the model argument refers
    to, downloads it when necessary, and records the resolved path on the run's
    :class:`BenchmarkConfig`. Once the model has been resolved to a local
    directory, :meth:`validate_model` checks it against the configuration, so a
    checkpoint that cannot serve the experiment is rejected before anything is
    pulled or run. Finally, :meth:`assess_space` turns the declared space into
    the effective one for that checkpoint, so a dimension the resolved model or
    hardware makes meaningless is dropped instead of sampled.

    A trial is host-driven: the backend plans a :class:`ServerSpec`, the
    orchestrator starts that server container and drives it with the host
    benchmark client, then hands the measured :class:`BenchmarkOutcome` and the
    captured server result back to :meth:`parse_result`. The spec is passed back
    into :meth:`parse_result` as well, so a backend can record what actually ran
    (resolved image, devices, and so on) without retaining state between the two
    calls, and the orchestrator never learns an engine's vocabulary.

    The search space is total over the backend's *applicable* parameters rather
    than over every tunable, so :meth:`applicable_parameters` tells the loader
    which parameters the selected engine requires. A definition declares exactly
    those, and a value can still never silently fall back to an engine default.
    Applicability is asked for the run's fixed :class:`LLMConfig`, because a
    parameter can be applicable only under a configured setup.
    """

    def applicable_parameters(self, llm: LLMConfig) -> frozenset[str]:
        """The tunable parameter names this engine forwards for ``llm``.

        The composition root passes this set to the search-space loader as the
        required set, so a definition may omit a parameter the engine does not
        support and must declare every one it does. The names are the engine-
        neutral YAML names in
        :data:`~llm_config_optimization.search_space.registry.TUNABLE_PARAMETERS`.
        A backend that forwards the whole registry returns all of it; one that
        lacks a parameter (llama.cpp has no tensor parallelism, for example)
        omits it. A parameter that is only meaningful under a configured setup
        is included exactly then, so the same space can stay unchanged for a
        run that does not use that setup. Declaring a parameter outside this set
        is a separate, backend-specific problem reported by :meth:`validate`.
        """
        ...

    def validate(
        self,
        space: ExperimentSpace,
        llm: LLMConfig,
        hardware: DetectedHardware | None = None,
    ) -> Sequence[str]:
        """Report every way this backend cannot support ``space`` for ``llm``.

        The space is assumed to have already passed engine-agnostic validation;
        this is the engine-specific half, asking whether the backend can forward
        each declared parameter and value. ``llm`` is the run's fixed
        configuration, so a declared parameter is support-checked against
        :meth:`applicable_parameters` for the same run rather than against the
        raw forwarding table; a gated tunable declared without its setup is
        therefore reported as unsupported. Problems are returned as
        human-readable strings rather than raised, so the caller can aggregate
        them across the whole space and report them at once. An empty sequence
        means the backend supports every dimension.

        A :class:`~llm_config_optimization.search_space.types.Default` dimension carries
        no value to support-check and is accepted as long as the backend knows
        the parameter itself.

        ``hardware``, when given, narrows the check to the host the run will
        use: a value the engine supports in general but the accelerator class
        does not (a CUDA-only attention backend on ROCm, for example), or one
        the host cannot fit (a tensor-parallel size larger than its device
        count), is rejected before the sweep starts. ``None`` asks the
        host-agnostic question only.
        """
        ...

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> Sequence[str]:
        """Report every way the resolved ``model`` cannot serve ``experiment``.

        This is the model-aware half of backend validation. It runs once the
        checkpoint has been resolved to a local directory and before anything
        is executed, so a configuration the checkpoint cannot satisfy fails
        before a container is pulled. The hook receives the whole experiment
        rather than only the model-dependent field, so further model-aware
        checks can be added without widening the protocol, and ``model`` is the
        same directory the run will mount, so validation and execution inspect
        one checkpoint.

        Problems are returned as human-readable strings, mirroring
        :meth:`validate`, so the caller can aggregate and report them together;
        an empty sequence means the checkpoint supports the configuration. A
        checkpoint whose metadata cannot be read is accepted: absence of
        evidence is not a mismatch, and the engine remains the final authority
        when the model loads.
        """
        ...

    def assess_space(
        self,
        space: ExperimentSpace,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        model: Path,
    ) -> SpaceAssessment:
        """Derive the effective space for the resolved checkpoint ``model``.

        A definition is written before the checkpoint is known, so it may
        declare dimensions the resolved model can never be affected by: a
        Mixture-of-Experts tunable does nothing on a dense checkpoint, and a
        host-dependent parameter can be a no-op on the resolved hardware. The
        composition root asks this once the checkpoint has been resolved and
        validated, and runs the returned effective space instead of the
        declared one.

        The backend owns the rules, because each depends on engine vocabulary
        and on the checkpoint format that engine reads. ``experiment`` is the
        run's full fixed configuration and ``hardware`` the resolved host, so a
        rule can consult a companion parameter — an all-to-all transport that
        only matters when expert parallelism is reachable, for example — rather
        than deciding from the dimension alone. ``model`` is the same
        checkpoint the run will mount, so the assessment and the execution
        inspect one model.

        Model-aware problems are returned rather than raised, mirroring
        :meth:`validate` and :meth:`validate_model`, and the caller reports
        them through the same unsupported-configuration path. A checkpoint
        whose metadata cannot be read keeps every declared dimension: absence
        of evidence is not a reason to change the user's space, and the engine
        remains the final authority when the model loads.
        """
        ...

    def image_for(self, target: HardwareTarget) -> str:
        """Return the container image this backend uses for ``target``.

        This is the public image-resolution seam: the image a run is planned
        with is ``container.image`` when the caller set one, and this method's
        result otherwise. Exposing it lets a caller resolve the same image a
        plan will use — for example to prefetch it before planning — without
        duplicating the backend's default-image mapping.
        """
        ...

    def model_request(self, model: str) -> ModelRequest:
        """Describe what the composition root must resolve for ``model``.

        The backend owns the engine's model-argument rules: whether ``model``
        names a local checkpoint or a Hub repository, and — for an engine that
        can fetch a single file — which file inside the repository is needed.
        The composition root turns the returned request into a local path using
        the shared download and progress machinery, and records it on the
        run's :class:`BenchmarkConfig`, so no engine vocabulary leaks into the
        CLI.
        """
        ...

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
    ) -> ServerSpec:
        """Turn an experiment and the detected hardware into a server run.

        Returns a :class:`ServerSpec` — the server's container configuration
        (command, published ports, mounts, environment, devices), the health
        path the host client polls until the server is ready, the cache-reset
        paths it calls between measured runs, and how long to wait for health.
        The orchestrator starts that container, measures it with the host
        client, stops it, and passes the outcome to :meth:`parse_result`, so
        neither the orchestrator nor the client learns an engine's vocabulary.

        Planning is not required to be side-effect free: a backend may create the
        host directories it mounts (for example a model cache directory) while
        planning, because they must exist before the run can be planned to use
        them.
        """
        ...

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        """Assemble a typed result from a measured outcome and the server run.

        ``outcome`` is what the host benchmark client observed — the successful
        measured runs, each carrying its raw payload, and the failed-run count —
        ``spec`` is the server plan that ran, and ``result`` is the captured
        server container outcome. The backend records what actually ran from
        ``spec`` rather than what was requested. Raises :class:`ExperimentError`
        when the outcome cannot describe a result, carrying the raw container
        result.
        """
        ...
