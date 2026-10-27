"""vLLM backend for the inference experiment layer.

This module is the only place that knows vLLM's serving CLI. It maps the
engine-neutral :class:`~llm_config_optimization.inference.types.LLMConfig` fields onto
``vllm serve`` arguments and plans the server container a trial is measured
against: the host benchmark client drives the server's OpenAI-compatible API,
so this backend never assembles a client command. The planned
:class:`~llm_config_optimization.inference.types.ServerSpec` carries the serve
command, the published port, the health, metrics, and cache-reset endpoints,
and the mounts the server needs.

Measurement is host-driven: one server is started per trial and measured
several times by the shared client, so the model is loaded once and each
repetition is a cheap client pass. The resulting numbers are not comparable to
the earlier offline ``vllm bench throughput`` measurements.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from ..runtime.types import ContainerConfig, ContainerResult, Mount
from ..search_space.registry import (
    SPECULATIVE_PARAMETERS,
    TUNABLE_PARAMETERS,
    speculative_configured,
)
from ..search_space.types import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    Spec,
)
from ._benchmark import _METRICS_AFTER_KEY, _METRICS_BEFORE_KEY, _METRICS_KEY
from ._tunables import (
    Gate,
    Toggle,
    declared_values,
    toggle_arguments,
    translate_value,
    unsupported_values,
)
from .types import (
    AttentionBackend,
    BenchmarkOutcome,
    DetectedHardware,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    HubCheckpoint,
    KVCacheDType,
    LLMConfig,
    LocalCheckpoint,
    ModelRequest,
    QuantizationMethod,
    RunMetrics,
    ServerSpec,
    SpeculativeMethod,
    SpeculativeMode,
)


@dataclass(frozen=True, slots=True)
class FlatFlag:
    """A parameter forwarded as its own ``--flag value`` argument pair.

    The flag name is not stored: it is derived from the field name by
    snake_case-to-kebab-case conversion when arguments are assembled, which is
    exactly how vLLM names its flattened flags (``kv_cache_dtype`` becomes
    ``--kv-cache-dtype``).
    """


@dataclass(frozen=True, slots=True)
class JsonGroup:
    """A parameter forwarded inside a JSON configuration group.

    ``flag`` is the group's CLI flag (for example ``--attention-config``) and
    ``key`` is the field's name inside that group's JSON payload (for example
    ``backend``). Fields that share a ``flag`` are merged into one JSON object,
    so the group is emitted once per run.
    """

    flag: str
    key: str


ParameterTarget = FlatFlag | JsonGroup | Gate | Toggle


PARAMETER_TARGETS: Mapping[str, ParameterTarget] = MappingProxyType(
    {
        "dtype": FlatFlag(),
        "quantization": FlatFlag(),
        "kv_cache_dtype": FlatFlag(),
        "max_model_len": FlatFlag(),
        "max_num_seqs": FlatFlag(),
        "max_num_batched_tokens": FlatFlag(),
        "max_num_active_seqs": FlatFlag(),
        "max_num_scheduled_tokens": FlatFlag(),
        "gpu_memory_utilization": FlatFlag(),
        "tensor_parallel_size": FlatFlag(),
        "seed": FlatFlag(),
        # Routed through the JSON group form on purpose, to exercise the JSON
        # path end to end; vLLM also exposes an equivalent flat --attention-backend.
        "attention_backend": JsonGroup(flag="--attention-config", key="backend"),
        "speculative_method": JsonGroup(flag="--speculative-config", key="method"),
        "draft_model": JsonGroup(flag="--speculative-config", key="model"),
        "speculative_mode": Gate(),
        "num_speculative_tokens": JsonGroup(
            flag="--speculative-config", key="num_speculative_tokens"
        ),
        "prefix_caching": Toggle(
            enabled="--enable-prefix-caching", disabled="--no-enable-prefix-caching"
        ),
    }
)
"""Where each :class:`LLMConfig` field is sent on the vLLM command line.

The table is deliberately code-only and declarative: every field is either
forwarded by exactly one entry or explicitly gated by a :class:`Gate`, and
adding a parameter is an explicit two-step of adding a typed field to
``LLMConfig`` and adding its entry here. A field absent from the table is never
forwarded.
"""


VALUE_MAPS: Mapping[str, Mapping[object, str]] = MappingProxyType(
    {
        "dtype": {
            DType.AUTO: "auto",
            DType.FLOAT16: "float16",
            DType.BFLOAT16: "bfloat16",
            DType.FLOAT32: "float32",
        },
        "quantization": {
            QuantizationMethod.AWQ: "awq",
            QuantizationMethod.GPTQ: "gptq",
            QuantizationMethod.FP8: "fp8",
        },
        "kv_cache_dtype": {
            KVCacheDType.AUTO: "auto",
            KVCacheDType.FLOAT16: "float16",
            KVCacheDType.BFLOAT16: "bfloat16",
            KVCacheDType.FP8_E4M3: "fp8_e4m3",
            KVCacheDType.FP8_E5M2: "fp8_e5m2",
        },
        "attention_backend": {
            AttentionBackend.FLASH_ATTENTION: "FLASH_ATTN",
            AttentionBackend.FLASH_INFER: "FLASHINFER",
        },
        "speculative_method": {
            SpeculativeMethod.DRAFT_MODEL: "draft_model",
            SpeculativeMethod.EAGLE3: "eagle3",
            SpeculativeMethod.MTP: "mtp",
            SpeculativeMethod.DFLASH: "dflash",
            SpeculativeMethod.DSPARK: "dspark",
            SpeculativeMethod.NGRAM: "ngram",
        },
        "speculative_mode": {
            SpeculativeMode.ENABLED: "enabled",
            SpeculativeMode.DISABLED: "disabled",
        },
    }
)
"""How vLLM spells each engine-neutral categorical value.

The enums in :mod:`llm_config_optimization.inference.types` carry no engine
vocabulary; this table is the only place their vLLM spellings live. A member
missing from its field's map is unsupported by this backend and is rejected
rather than silently forwarded, while the long tail of engine-specific values
is reached through :attr:`VLLMBackend.extra_args`. ``XFORMERS`` is deliberately
absent: current vLLM releases no longer ship an xFormers attention backend, so
the member is rejected instead of being forwarded to a flag the engine will
not accept. ``AUTO`` and ``OFF`` are llama.cpp's flash-attention settings and
have no vLLM attention-config spelling.
"""

_CHECKPOINT_QUANTIZATION_METHODS: Mapping[QuantizationMethod, frozenset[str]] = MappingProxyType(
    {
        QuantizationMethod.AWQ: frozenset({"awq", "auto_awq", "awq_marlin"}),
        QuantizationMethod.GPTQ: frozenset({"gptq", "auto_gptq", "gptq_marlin"}),
        QuantizationMethod.FP8: frozenset({"fp8"}),
    }
)
"""Checkpoint spellings of each quantization method vLLM can load.

A checkpoint names its method under ``quantization_config.quant_method``, and
checkpoint authors are inconsistent about the exact spelling (``awq`` against
``auto_awq`` or ``awq_marlin``, for example), so each engine-neutral member
maps to the set of spellings it can serve. The spellings are vLLM-version-
specific and drift with the engine, exactly like :data:`VALUE_MAPS`, so they
live here rather than in the engine-neutral types. A declared method absent
from every set is one this backend does not model, so no explicit quantization
value is accepted for it and the user is steered to auto-detection.
"""

_ONLINE_QUANTIZATION_METHODS: frozenset[QuantizationMethod] = frozenset({QuantizationMethod.FP8})
"""Methods vLLM can apply to an unquantized checkpoint while loading it.

A checkpoint that declares no method carries full-precision weights, so only a
method the engine quantizes on the fly can be requested for it; a checkpoint-
time method such as AWQ has no quantized weights to load.
"""

_ATTENTION_BACKEND_TARGETS: Mapping[AttentionBackend, frozenset[HardwareTarget]] = MappingProxyType(
    {
        AttentionBackend.FLASH_ATTENTION: frozenset({HardwareTarget.CUDA}),
        AttentionBackend.FLASH_INFER: frozenset({HardwareTarget.CUDA}),
    }
)
"""Which accelerator classes vLLM supports each attention backend on.

vLLM's ROCm and CPU platforms select from their own backend sets
(``ROCM_ATTN``, ``TRITON_ATTN``, and so on) that the engine-neutral enum does
not model, so the modelled members are CUDA-only. A member with no entry here
is either unsupported everywhere or has no vLLM spelling at all; the latter is
already reported by :data:`VALUE_MAPS` before a target is consulted.
"""

_FLOAT_CACHE_DTYPES: Mapping[KVCacheDType, DType] = MappingProxyType(
    {
        KVCacheDType.FLOAT16: DType.FLOAT16,
        KVCacheDType.BFLOAT16: DType.BFLOAT16,
    }
)
"""Non-quantized KV-cache formats and the model dtype each must match.

vLLM stores a non-quantized KV cache in the model's dtype, so an explicit
float16 or bfloat16 cache is only safe when the model dtype is pinned to the
same value. The FP8 formats are absent because they are stored differently and
may differ from the model dtype.
"""


def _kebab_case(name: str) -> str:
    """Convert a snake_case field name to vLLM's kebab-case flag spelling."""
    return name.replace("_", "-")


def _normalise(value: object) -> object:
    """Recursively reduce a value to something JSON can serialise.

    vLLM takes most complex configuration as a JSON string, so values that are
    not already JSON-native are converted: enums become their ``value``,
    path-likes become strings, mappings become dicts, and sets and sequences
    become lists. Conversion recurses into containers, so an enum or path nested
    inside a JSON group serialises too. Values JSON already understands are
    returned unchanged.

    Sets are ordered by their string form so a run's arguments are deterministic
    even though the set itself is unordered.
    """
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_normalise(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise(item) for item in value]
    return value


def _encode_value(value: object) -> str:
    """Render a flat flag's value as the single token that carries it.

    Containers — including enums, path-likes, and sets once normalised — are
    JSON-encoded, matching vLLM's convention for complex values. Every other
    value is passed through its natural string form.
    """
    normalised = _normalise(value)
    if isinstance(normalised, (dict, list)):
        return json.dumps(normalised)
    return str(normalised)


def _flat_arguments(name: str, value: object) -> list[str]:
    """Encode one flat flag and its value as CLI arguments.

    Booleans follow vLLM's argparse convention: true is the flag itself and
    false is its ``--no-`` counterpart, neither taking a value. Anything else
    becomes ``--flag value`` with nested values JSON-encoded.
    """
    flag = f"--{_kebab_case(name)}"
    if isinstance(value, bool):
        return [flag] if value else [f"--no-{_kebab_case(name)}"]
    return [flag, _encode_value(value)]


def encode_arguments(
    parameters: Mapping[str, object],
    targets: Mapping[str, ParameterTarget] = PARAMETER_TARGETS,
) -> list[str]:
    """Encode parameter values as vLLM CLI arguments according to ``targets``.

    This is the generic serialisation primitive behind :func:`serve_arguments`.
    It is exposed so the encoding rules — flat
    flags, JSON groups, kebab-case names, boolean negation, and JSON encoding
    of nested values — can be exercised directly with values the typed config
    does not yet model.

    Parameters absent from ``targets`` and parameters whose value is ``None``
    are never forwarded, and a :class:`Gate` member controls its group without
    being serialised itself. Flat flags become ``--flag value``; JSON-group
    fields sharing a flag are merged into a single ``--flag '<json>'`` argument
    emitted at the group's first position, so the group appears once per run.

    Values are normalised before encoding, so enums, path-likes, mappings, sets,
    and arbitrary sequences serialise correctly both as flat flags and inside a
    JSON group.
    """
    arguments: list[str] = []
    emitted_groups: set[str] = set()
    for name, target in targets.items():
        value = parameters.get(name)
        if value is None:
            continue
        if isinstance(target, Gate):
            continue
        if isinstance(target, FlatFlag):
            arguments.extend(_flat_arguments(name, value))
            continue
        if isinstance(target, Toggle):
            arguments.extend(toggle_arguments(value, target))
            continue
        if target.flag in emitted_groups:
            continue
        members = {
            member.key: _normalise(parameters[member_name])
            for member_name, member in targets.items()
            if isinstance(member, JsonGroup)
            and member.flag == target.flag
            and parameters.get(member_name) is not None
        }
        arguments.extend((target.flag, json.dumps(members)))
        emitted_groups.add(target.flag)
    return arguments


_SPECULATIVE_GROUP_FIELDS = frozenset(
    {"speculative_method", "draft_model", "num_speculative_tokens"}
)
"""The fields that make up the ``--speculative-config`` object.

The speculative mode is absent because it gates the group instead of being
forwarded inside it. A disabled mode drops these fields before encoding, so no
speculative flag reaches the engine at all.
"""


def serve_arguments(
    config: LLMConfig,
    *,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Turn an LLM config into the ``vllm serve`` arguments that carry it.

    Unset fields (``None``) are omitted so the engine keeps its own defaults,
    categorical fields are translated from their engine-neutral enums into
    vLLM's spellings, and any ``extra_args`` are appended verbatim after the
    mapped parameters as the backend's escape hatch for unmodeled serving
    flags. The workload is assembled separately by the host benchmark client,
    so the server command carries no workload flags.

    The speculative method, draft model, and depth share one
    ``--speculative-config`` JSON object. A disabled speculative mode suppresses
    that whole object, leaving the engine with no speculative flags; an unset
    mode means the configured setup's default and emits it, and an unset depth
    omits only the depth key so the engine keeps its own default.
    """
    parameters = {
        field.name: translate_value(
            field.name, getattr(config, field.name), VALUE_MAPS, engine="vLLM"
        )
        for field in fields(config)
        if getattr(config, field.name) is not None
    }
    if config.speculative_mode is SpeculativeMode.DISABLED:
        parameters = {
            name: value
            for name, value in parameters.items()
            if name not in _SPECULATIVE_GROUP_FIELDS
        }
    return [*encode_arguments(parameters), *extra_args]


_MODEL_MOUNT_PATH = "/model"
_DRAFT_MODEL_MOUNT_PATH = "/draft-model"
"""Where a resolved draft checkpoint is mounted inside the container.

The main model keeps :data:`_MODEL_MOUNT_PATH`, so a local draft gets its own
fixed target and both checkpoints stay addressable at once. A draft that lives
in the same Hugging Face cache hub root as the main model reuses that root's
mount instead, keeping the draft's symlinks resolvable without a second mount.
"""
_SERVER_PORT = 8000
_SERVER_BIND_HOST = "0.0.0.0"
"""Where the server binds inside the container.

The host benchmark client reaches the server through the container's published
port, so the server must accept connections on every interface rather than
container loopback.
"""
_RESET_CACHE_ENDPOINTS = (
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/reset_encoder_cache",
)
"""The dev-mode endpoints the host client calls to clear the server's caches.

A repetition must not be flattered by prompts a previous run left cached, so
the client resets the caches before every run. ``/reset_prefix_cache`` is the
one that matters for text workloads; the multimodal and encoder caches are
reset too so a run starts from a clean server regardless of the workload.
"""
_METRICS_PATH = "/metrics"
"""The endpoint vLLM serves its Prometheus exposition on.

The host benchmark client scrapes it immediately before and after every
measured run, so this backend can derive the phase rates from the deltas of
the counters and histogram sums the exposition carries. The endpoint is part
of vLLM's OpenAI-compatible server and is served without any flag, so the
planned command and image choice are unchanged by it.
"""
_PREFILL_RATE_SERIES = ("vllm:prompt_tokens_total", "vllm:request_prefill_time_seconds_sum")
"""The ``(tokens, seconds)`` series pair behind the prefill rate.

``vllm:prompt_tokens_total`` counts every prompt token the server has processed
and ``vllm:request_prefill_time_seconds_sum`` the seconds it spent in prefill,
so their deltas across one measured run are that run's processed prompt tokens
and prefill seconds. Prompt tokens served from the prefix cache are excluded
because the host client resets the cache before every run.
"""
_DECODE_RATE_SERIES = ("vllm:generation_tokens_total", "vllm:request_decode_time_seconds_sum")
"""The ``(tokens, seconds)`` series pair behind the decode rate.

``vllm:generation_tokens_total`` counts every generated token the server has
produced and ``vllm:request_decode_time_seconds_sum`` the seconds it spent
decoding, so their deltas across one measured run are that run's generated
tokens and decode seconds.
"""
_DEV_MODE_ENV: Mapping[str, str] = MappingProxyType({"VLLM_SERVER_DEV_MODE": "1"})
"""Environment that enables vLLM's dev-mode endpoints.

The cache-reset endpoints are development endpoints gated behind
``VLLM_SERVER_DEV_MODE``; without it the client's resets are rejected and a
run can benefit from its predecessor's cached prompts. A caller's own
``container.env`` entry still wins, so the gate can be forced off deliberately.
"""
_HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
_HF_CACHE_MOUNT_PATH = "/root/.cache/huggingface"
_SELINUX_OPTIONS = ("z",)
"""Bind-mount options applied to the mounts the backend adds itself.

The ``z`` option asks the container runtime to relabel the host path for
SELinux, so runs work on enforcing hosts. Caller-supplied volumes keep whatever
options they were given.
"""
_CACHE_MOUNT_PATH = "/vllm-cache"
"""Where a caller-supplied compile cache is mounted inside the container."""

_CACHE_ENV: Mapping[str, str] = MappingProxyType(
    {
        "VLLM_CACHE_ROOT": f"{_CACHE_MOUNT_PATH}/vllm",
        "TRITON_CACHE_DIR": f"{_CACHE_MOUNT_PATH}/triton",
        "MIOPEN_CACHE_DIR": f"{_CACHE_MOUNT_PATH}/miopen",
        "MIOPEN_USER_DB_PATH": f"{_CACHE_MOUNT_PATH}/miopen/db",
    }
)
"""Environment that redirects vLLM's compile caches onto the mounted cache.

``VLLM_CACHE_ROOT`` covers vLLM's own caches (the Inductor/AOT compile cache and
the model-info cache), ``TRITON_CACHE_DIR`` covers Triton's kernels, and the
MIOpen variables cover the ROCm library's kernel cache. All four are needed
because a fresh container otherwise rebuilds them from scratch on every trial.
A caller's own ``container.env`` entries take precedence, so an explicit
override still wins.
"""

_DEFAULT_IMAGES: Mapping[HardwareTarget, str] = MappingProxyType(
    {
        HardwareTarget.CUDA: "docker.io/vllm/vllm-openai:latest",
        HardwareTarget.ROCM: "docker.io/vllm/vllm-openai-rocm:latest",
    }
)
"""vLLM's official base image per accelerator class.

The names are registry-qualified because podman refuses to resolve a bare
``vllm/...`` short name when it cannot prompt for a registry (for example with
no TTY, as the runtime client runs). CPU has no entry because its image is
configurable on the backend; see :attr:`VLLMBackend.cpu_image`.
"""

_DEVICE_MAPPINGS: Mapping[HardwareTarget, tuple[str, ...]] = MappingProxyType(
    {
        HardwareTarget.CUDA: ("nvidia.com/gpu=all",),
        HardwareTarget.ROCM: ("/dev/kfd", "/dev/dri"),
        HardwareTarget.CPU: (),
    }
)
"""How each accelerator class reaches the container.

CUDA uses the CDI device name so the runtime supplies the driver; ROCm passes
the KFD and DRM nodes the userspace in the image needs; CPU passes nothing.
"""


def _target_for(
    devices: tuple[str, ...], *, raw_result: ContainerResult | None = None
) -> HardwareTarget:
    """Recover the accelerator class a plan's device mapping was chosen for.

    :meth:`VLLMBackend.plan` fills ``devices`` straight from
    :data:`_DEVICE_MAPPINGS`, so a run's target can be read back from the plan
    that produced it. A mapping the backend never emits cannot be attributed to
    a target and raises rather than guessing; ``raw_result`` is attached to that
    error so every :meth:`VLLMBackend.parse_result` failure carries the captured
    container output uniformly.
    """
    for target, mapping in _DEVICE_MAPPINGS.items():
        if mapping == devices:
            return target
    raise ExperimentError(
        f"cannot determine the accelerator class for devices {devices!r}",
        raw_result=raw_result,
    )


def _snapshot_pair(
    run: RunMetrics,
) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    """Return a run's ``(before, after)`` metric snapshots, or ``None``.

    The host client stores the pair under the ``metrics`` key of a run's raw
    JSON only when both halves were scraped, so a run without the key — or
    with a value that is not the documented pair — carries no phase data and
    the caller derives nothing from it.
    """
    metrics = run.raw_json.get(_METRICS_KEY)
    if not isinstance(metrics, Mapping):
        return None
    before = metrics.get(_METRICS_BEFORE_KEY)
    after = metrics.get(_METRICS_AFTER_KEY)
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    return before, after


def _sample(snapshot: Mapping[str, object], series: str) -> float | None:
    """Return ``snapshot``'s numeric value for ``series``, or ``None``.

    The exposition parser produces floats, but the snapshots live in a run's
    untyped raw JSON, so a value that is missing or not a real number is
    treated as absent rather than coerced.
    """
    value = snapshot.get(series)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _phase_rate(run: RunMetrics, series: tuple[str, str]) -> float | None:
    """Derive one phase's tokens-per-second rate from a run's metric deltas.

    ``series`` pairs the phase's token counter with the sum of its time
    histogram; the rate is the counter's increase over the histogram sum's
    increase, so it counts tokens per second of the phase's own work rather
    than a partition of the run's wall-clock time. The result is ``None``
    whenever the run carries no complete snapshot pair, either series is absent
    from a snapshot, or either delta is not positive — a counter that did not
    move describes no measurable work, and a non-positive time delta cannot be
    divided by.
    """
    snapshots = _snapshot_pair(run)
    if snapshots is None:
        return None
    before, after = snapshots
    tokens_before = _sample(before, series[0])
    tokens_after = _sample(after, series[0])
    seconds_before = _sample(before, series[1])
    seconds_after = _sample(after, series[1])
    if tokens_before is None or tokens_after is None:
        return None
    if seconds_before is None or seconds_after is None:
        return None
    tokens = tokens_after - tokens_before
    seconds = seconds_after - seconds_before
    if tokens <= 0 or seconds <= 0:
        return None
    return tokens / seconds


def _with_phase_rates(run: RunMetrics) -> RunMetrics:
    """Return a run enriched with the phase rates its metric snapshots imply.

    Each rate is derived independently, so a run whose exposition lacked one
    phase's series keeps the other rate rather than losing both. When neither
    rate can be derived the run is returned unchanged, so an engine that
    exposes no metrics keeps the object the client produced.
    """
    prefill = _phase_rate(run, _PREFILL_RATE_SERIES)
    decode = _phase_rate(run, _DECODE_RATE_SERIES)
    if prefill is None and decode is None:
        return run
    return replace(run, prefill_tokens_per_second=prefill, decode_tokens_per_second=decode)


def _ensure_directory(path: Path) -> Path:
    """Create ``path`` if needed and return it, so it can be bind-mounted."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_request(model: str) -> ModelRequest:
    """Return the neutral request for vLLM's ``--model`` argument.

    An existing host path is a local checkpoint and is returned resolved;
    anything else is a Hugging Face repository id fetched as a whole snapshot.
    Existence is the only test, so a Hugging Face identifier that happens to
    resolve to a local file or directory — for example a repository whose owner
    name matches a directory in the working tree — is treated as a local
    checkpoint and mounted rather than downloaded, and the container then fails
    to find the model it was asked for.
    """
    path = Path(model).expanduser()
    if path.exists():
        return LocalCheckpoint(path=path.resolve())
    return HubCheckpoint(repo_id=model)


def _hub_cache_root(path: Path) -> Path | None:
    """Return the Hugging Face cache hub root that contains ``path``.

    ``snapshot_download`` returns a snapshot directory whose files are relative
    symlinks into the cache's ``blobs`` directory, and those links can hop more
    than once — a Xet-backed download stores the real file in the cache's
    *shared* ``hub/blobs`` store, so a snapshot file points at the repository's
    ``blobs`` entry which in turn points at ``../../blobs/<hash>``. Mounting
    only the snapshot, or only the model's repository, would leave those links
    dangling inside the container. The whole hub directory — the parent of
    ``models--*`` and the shared ``blobs`` — must be mounted for every link to
    resolve. Returns ``None`` when ``path`` is not laid out as
    ``<hub>/models--*/snapshots/<rev>`` with a sibling ``blobs`` directory, so a
    plain local checkpoint is mounted directly.
    """
    repo = path.parent.parent
    if (
        path.parent.name == "snapshots"
        and repo.name.startswith("models--")
        and (repo.parent / "blobs").is_dir()
    ):
        return repo.parent
    return None


def _model_mount(path: Path, target: str = _MODEL_MOUNT_PATH) -> Mount:
    """Build the read-only mount that exposes a resolved checkpoint to the server."""
    return Mount(
        source=str(path),
        target=target,
        read_only=True,
        options=_SELINUX_OPTIONS,
    )


def _resolve_model(model: str, resolved_path: Path | None) -> tuple[str, tuple[Mount, ...]]:
    """Resolve the model argument and any mount it needs.

    A ``resolved_path`` — the checkpoint the composition root already fetched —
    is mounted read-only at a fixed container path and the container is pointed
    at that path. A Hugging Face snapshot is special: its files are relative
    symlinks that may hop through the cache's shared ``blobs`` store, so the
    whole cache hub root is mounted instead and the container is pointed at the
    snapshot inside it, keeping every symlink resolvable. Without a
    ``resolved_path``, the backend's own model-resolution hook decides: an
    existing host path is a local checkpoint and is mounted the same way, while
    anything else — a Hugging Face identifier — is passed through unchanged and
    downloaded into the mounted Hugging Face cache.
    """
    if resolved_path is not None:
        hub_root = _hub_cache_root(resolved_path)
        if hub_root is not None:
            relative = resolved_path.relative_to(hub_root)
            return f"{_MODEL_MOUNT_PATH}/{relative}", (_model_mount(hub_root),)
        return _MODEL_MOUNT_PATH, (_model_mount(resolved_path),)
    request = _model_request(model)
    if isinstance(request, LocalCheckpoint):
        return _MODEL_MOUNT_PATH, (_model_mount(request.path),)
    return request.repo_id, ()


def _resolve_draft(
    draft_model: str,
    resolved_draft_path: Path | None,
    model_mounts: tuple[Mount, ...],
) -> tuple[str, tuple[Mount, ...]]:
    """Resolve the draft-model argument and any mount it needs.

    An unresolved draft string is passed through unchanged, just as an
    unresolved main model is, so the engine fetches it itself. A resolved draft
    is mounted read-only and the speculative configuration is pointed at its
    in-container path. When the draft is a Hugging Face snapshot, the cache hub
    root is mounted so its symlinks resolve: the main model's mount is reused
    when it already exposes that root, and otherwise the root is mounted at the
    dedicated draft path. A local draft directory or file is always mounted at
    that dedicated path, so it never collides with the main model's mount.
    """
    if resolved_draft_path is None:
        return draft_model, ()
    hub_root = _hub_cache_root(resolved_draft_path)
    if hub_root is not None:
        shared = next((mount for mount in model_mounts if Path(mount.source) == hub_root), None)
        target = shared.target if shared is not None else _DRAFT_MODEL_MOUNT_PATH
        relative = resolved_draft_path.relative_to(hub_root)
        mounts = () if shared is not None else (_model_mount(hub_root, target),)
        return f"{target}/{relative}", mounts
    return _DRAFT_MODEL_MOUNT_PATH, (_model_mount(resolved_draft_path, _DRAFT_MODEL_MOUNT_PATH),)


_ENGINE_BOUNDS: tuple[tuple[str, str], ...] = (
    ("max_num_batched_tokens", "max_num_seqs"),
    ("max_num_seqs", "max_num_active_seqs"),
    ("max_num_batched_tokens", "max_num_scheduled_tokens"),
)
"""Numeric constraints vLLM imposes between two tunable parameters.

Each pair ``(upper, lower)`` records that ``upper`` must be at least ``lower``.
The engine-agnostic ``>= 1`` floor is already enforced by the loader; these are
the extra limits vLLM's own configuration validation adds.
``SchedulerConfig.verify_max_model_len`` rejects
``max_num_batched_tokens < max_num_seqs`` unconditionally, so a space that
allows a violating pair would fail partway through a series. The scheduler's
optional caps are bounded by the values they narrow:
``max_num_active_seqs`` cannot exceed ``max_num_seqs`` and
``max_num_scheduled_tokens`` cannot exceed ``max_num_batched_tokens``. vLLM's
other scheduler rule, ``max_num_batched_tokens >= max_model_len``, applies only
when chunked prefill is disabled, which the default vLLM v1 scheduler does not
do, so it is deliberately not enforced here.
"""


def _validate_target_support(dimension: Dimension, field: str, target: HardwareTarget) -> list[str]:
    """Report every declared value the target's platform does not support.

    Only attention backends are platform-specific today, and only for values
    that already have a vLLM spelling: a value with no spelling is reported by
    :func:`~llm_config_optimization.inference._tunables.unsupported_values`
    regardless of target, so checking it here too would duplicate the problem.
    """
    if field != "attention_backend":
        return []
    problems: list[str] = []
    for path, value in declared_values(dimension):
        if not isinstance(value, AttentionBackend):
            continue
        targets = _ATTENTION_BACKEND_TARGETS.get(value)
        if targets is not None and target not in targets:
            problems.append(
                f"{path}: vLLM does not support {field}={value.value!r} "
                f"on the {target.value} target"
            )
    return problems


def _validate_device_count(dimension: Dimension, field: str, device_count: int) -> list[str]:
    """Report every tensor-parallel size larger than the host has devices for.

    ``tensor_parallel_size`` must fit the number of devices a run can see, so a
    space that can produce a larger value is rejected rather than failing
    partway through the sweep. An ``Interval`` is checked by its maximum because
    the optimizer may draw any value in the range; a ``Default`` dimension
    carries no value and is handled by the caller.
    """
    if field != "tensor_parallel_size":
        return []
    problems: list[str] = []
    for path, value in declared_values(dimension):
        if isinstance(value, int) and value > device_count:
            problems.append(
                f"{path}: vLLM cannot run tensor_parallel_size={value} "
                f"on a host with {device_count} device(s)"
            )
    if isinstance(dimension.spec, Interval) and dimension.spec.maximum > device_count:
        problems.append(
            f"space.{dimension.name}.interval: vLLM cannot run tensor_parallel_size "
            f"up to {dimension.spec.maximum} on a host with {device_count} device(s)"
        )
    return problems


def _numeric_range(spec: Spec) -> tuple[int, int] | None:
    """Return the smallest and largest integer a spec allows.

    ``None`` when the spec defers to the engine, or when it describes a
    categorical parameter rather than an integer one, so the caller skips the
    engine-bound check instead of comparing incomparable values.
    """
    if isinstance(spec, Fixed):
        values = (spec.value,)
    elif isinstance(spec, Categorical):
        values = spec.values
    elif isinstance(spec, Interval):
        values = (spec.minimum, spec.maximum)
    else:
        return None
    if any(isinstance(value, Enum) for value in values):
        return None
    numeric = [value for value in values if isinstance(value, int)]
    if len(numeric) != len(values):
        return None
    return min(numeric), max(numeric)


def _validate_float_cache_dtype(space: ExperimentSpace) -> list[str]:
    """Report an explicit float KV cache that cannot be proved to match the model dtype.

    vLLM stores a non-quantized KV cache in the model's dtype, so an explicit
    ``float16`` or ``bfloat16`` cache is only safe when the ``dtype`` dimension
    is pinned to exactly that model dtype. A ``Default``, ``auto``, ``float32``,
    or multi-value dtype dimension leaves at least one combination unverifiable
    or mismatched, so it is rejected rather than failing at server startup. The
    FP8 cache formats are exempt because they are stored differently. A space
    missing either dimension is left to the per-dimension checks.
    """
    dimensions = {dimension.name: dimension for dimension in space.dimensions}
    dtype = dimensions.get("dtype")
    cache = dimensions.get("kv_cache_dtype")
    if dtype is None or cache is None:
        return []
    dtype_values = [value for _, value in declared_values(dtype)]
    pinned = dtype_values[0] if len(dtype_values) == 1 else None
    problems: list[str] = []
    for path, value in declared_values(cache):
        expected = _FLOAT_CACHE_DTYPES.get(value) if isinstance(value, KVCacheDType) else None
        if expected is None or pinned is expected:
            continue
        problems.append(
            f"{path}: vLLM stores an explicit float KV cache in the model dtype, "
            f"but the dtype dimension does not pin {expected.value!r}; pin "
            f"dtype={expected.value!r} or use kv_cache_dtype=auto"
        )
    return problems


def _validate_engine_bounds(space: ExperimentSpace) -> list[str]:
    """Report every engine-specific numeric bound the space could violate.

    A bound is only checked when both parameters are pinned or bounded by a
    value the optimizer will draw from; a ``Default`` parameter is unknown, so
    the check is skipped and the engine enforces the bound at run time instead.
    The comparison uses the most permissive pair the space allows — the
    smallest ``upper`` against the largest ``lower`` — so a space is accepted
    only when every point it can produce satisfies the bound.
    """
    specs = {dimension.name: dimension.spec for dimension in space.dimensions}
    problems: list[str] = []
    for upper, lower in _ENGINE_BOUNDS:
        upper_range = _numeric_range(specs[upper]) if upper in specs else None
        lower_range = _numeric_range(specs[lower]) if lower in specs else None
        if upper_range is None or lower_range is None:
            continue
        if upper_range[0] < lower_range[1]:
            problems.append(
                f"space.{upper}: vLLM requires {upper} >= {lower}, but the space "
                f"allows {upper}={upper_range[0]} with {lower}={lower_range[1]}"
            )
    return problems


class _UnreadableCheckpoint:
    """Sentinel type for a checkpoint whose configuration cannot be read."""


_UNREADABLE_CHECKPOINT = _UnreadableCheckpoint()
"""Returned when ``config.json`` is missing, malformed, or not a JSON object.

Kept distinct from ``None``, which means the file was read but declares no
method: an unreadable checkpoint is accepted (fail-open), while a readable one
that declares no method accepts only a method the engine can apply online.
"""


def _read_checkpoint_quantization(model: Path) -> str | _UnreadableCheckpoint | None:
    """Return the quantization method a checkpoint declares, lower-cased.

    The standard ``config.json`` names the method under
    ``quantization_config.quant_method``. A missing or unreadable file,
    malformed JSON, or a payload that is not a JSON object is *unreadable*: the
    caller accepts the configuration rather than rejecting on metadata it could
    not see. A readable object with no ``quantization_config`` or a non-string
    ``quant_method`` declares no method and returns ``None``. The value is
    lower-cased because checkpoint authors are inconsistent about case while
    the engine's spellings are lower-case.
    """
    try:
        payload = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return _UNREADABLE_CHECKPOINT
    if not isinstance(payload, Mapping):
        return _UNREADABLE_CHECKPOINT
    quantization_config = payload.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        return None
    method = quantization_config.get("quant_method")
    if not isinstance(method, str):
        return None
    return method.lower()


@dataclass(frozen=True, slots=True)
class VLLMBackend:
    """The vLLM implementation of the inference backend seam.

    ``extra_args`` is the single escape hatch for vLLM flags the engine-neutral
    config does not model. The sequence is appended verbatim to the assembled
    ``vllm serve`` command, after the mapped parameters, so a caller can
    experiment without widening :class:`LLMConfig`; values passed this way are
    the caller's responsibility.

    ``cpu_image`` is the base image used for CPU runs. vLLM publishes an
    official CPU image, but it is configurable so a caller can pin a different
    build without waiting for a library release. Like the CUDA and ROCm
    defaults it is registry-qualified so podman can resolve it without a TTY.

    ``cache_dir`` is a host directory for vLLM's compile caches. When set, it
    is bind-mounted into the container and the cache environment points at it,
    so a sweep reuses the kernels the first trial compiled instead of
    rebuilding them in every fresh container. ``None`` (the default) mounts
    nothing. The directory's lifetime is the caller's: a sweep driver typically
    creates a temporary one and removes it when the sweep ends, so repeated
    trials within the sweep share it while nothing is left behind afterwards.
    """

    extra_args: tuple[str, ...] = ()
    cpu_image: str = "docker.io/vllm/vllm-openai-cpu:latest"
    cache_dir: Path | None = None

    def applicable_parameters(self, llm: LLMConfig) -> frozenset[str]:
        """Every registered tunable this backend forwards, minus the gated ones.

        vLLM models the whole registry except llama.cpp's GPU-layer dimension,
        which has no vLLM flag, so its applicable set is every tunable present
        in :data:`PARAMETER_TARGETS`. The two speculative dimensions are added
        only when ``llm`` configures a speculative method or draft model. A
        space must therefore declare every applicable parameter, and each value
        is still support-checked by :meth:`validate`.
        """
        applicable = frozenset(
            name
            for name, field in TUNABLE_PARAMETERS.items()
            if field in PARAMETER_TARGETS and name not in SPECULATIVE_PARAMETERS
        )
        if speculative_configured(llm):
            return applicable | SPECULATIVE_PARAMETERS
        return applicable

    def validate(
        self,
        space: ExperimentSpace,
        llm: LLMConfig,
        hardware: DetectedHardware | None = None,
    ) -> Sequence[str]:
        """Report every way vLLM cannot support ``space`` for ``llm``.

        The space is assumed to have passed engine-agnostic validation. Every
        dimension's parameter must be one the backend forwards for this run, and
        every fixed or categorical enum value must have a vLLM spelling.
        ``Default`` dimensions carry no value to check and are accepted once
        their parameter is known to be applicable. The engine's own numeric
        constraints between parameters are checked last. A space that can pair
        an explicit float16 or bfloat16 KV cache with a model dtype that is not
        pinned to the same value is rejected too, because vLLM stores the cache
        in the model dtype. Problems are returned as strings so the caller can
        aggregate them; an empty sequence means vLLM supports every dimension.

        ``hardware``, when given, additionally asks whether each declared value
        is supported on that accelerator class and fits its device count, so a
        CUDA-only attention backend or a tensor-parallel size larger than the
        host is rejected before the sweep starts instead of failing every trial.
        ``None`` skips that host-specific half and checks only whether vLLM can
        forward the parameter and spell the value.
        """
        problems: list[str] = []
        applicable = self.applicable_parameters(llm)
        for dimension in space.dimensions:
            field = TUNABLE_PARAMETERS[dimension.name]
            if field not in PARAMETER_TARGETS or dimension.name not in applicable:
                problems.append(f"space.{dimension.name}: vLLM does not forward {field!r}")
                continue
            if isinstance(dimension.spec, Default):
                continue
            problems.extend(unsupported_values(dimension, field, VALUE_MAPS, engine="vLLM"))
            if hardware is not None:
                problems.extend(_validate_target_support(dimension, field, hardware.target))
                problems.extend(_validate_device_count(dimension, field, hardware.device_count))
        problems.extend(_validate_engine_bounds(space))
        problems.extend(_validate_float_cache_dtype(space))
        return problems

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> Sequence[str]:
        """Report every way the checkpoint at ``model`` cannot serve ``experiment``.

        The only model-dependent check today is quantization. An unset
        ``quantization`` is always accepted because the engine auto-detects the
        method from the checkpoint. An unquantized checkpoint accepts only a
        method the engine can apply online, while a quantized checkpoint
        accepts only the method it was produced with, matched through
        :data:`_CHECKPOINT_QUANTIZATION_METHODS`. A declared method this
        backend does not model accepts no explicit value, steering the user to
        auto-detection. Problems are returned as strings; a checkpoint whose
        configuration cannot be read is accepted (fail-open), so custom
        checkpoints are never blocked by metadata the check could not see.
        """
        quantization = experiment.llm.quantization
        if quantization is None:
            return []
        declared = _read_checkpoint_quantization(model)
        if isinstance(declared, _UnreadableCheckpoint):
            return []
        if declared is None:
            if quantization in _ONLINE_QUANTIZATION_METHODS:
                return []
            return [
                f"the checkpoint at {model} declares no quantization method, but "
                f"quantization={quantization.value!r} was requested; omit quantization "
                f"to let vLLM auto-detect"
            ]
        if declared in _CHECKPOINT_QUANTIZATION_METHODS.get(quantization, frozenset()):
            return []
        return [
            f"the checkpoint at {model} was quantized with {declared!r}, which is "
            f"incompatible with the requested quantization={quantization.value!r}; omit "
            f"quantization to let vLLM auto-detect"
        ]

    def model_request(self, model: str) -> ModelRequest:
        """Decide whether ``model`` is a local checkpoint or a Hub repository.

        The composition root calls this before downloading, so the engine's
        model-argument rules — existence means a local path, anything else a
        Hugging Face id — stay here rather than in the CLI. vLLM fetches whole
        snapshots, so the request never selects a single file.
        """
        return _model_request(model)

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
    ) -> ServerSpec:
        """Turn an experiment and the detected hardware into a server run.

        The accelerator class resolves from the container config's override or,
        failing that, the detected hardware, and selects the base image, the
        device mapping, and — when the LLM config leaves it unset — the
        tensor-parallel size. The Hugging Face cache, the checkpoint itself —
        the resolved path when the composition root set one, otherwise a host
        path — and, when ``cache_dir`` is set, the compile cache are
        bind-mounted. A configured draft model is resolved the same way: a
        resolved draft is mounted (reusing the main model's cache-hub mount
        when it lives in the same root) and the speculative configuration is
        pointed at its in-container path, while an unresolved draft string is
        passed through for the engine to fetch.

        The planned container keeps the image's own ``vllm serve`` entrypoint and
        runs it with ``--model``, the published port, and the serving parameters
        and ``extra_args``; the served model name is pinned to the configured
        model string so the host client addresses the server with the same name
        it was configured with. The dev-mode environment that the cache reset
        requires is added to the container environment. The returned
        :class:`ServerSpec` describes the health endpoint, the metrics and
        cache-reset endpoints, and the startup timeout the host client uses.

        Planning is not side-effect free: the host Hugging Face cache directory
        and the compile cache directory are created on the host when planning,
        because each must exist before it can be bind-mounted. Merely planning a
        run — as the unit tests do — therefore still touches the host
        filesystem.
        """
        container = experiment.container
        target = container.target or hardware.target
        _ensure_directory(_HF_CACHE_HOST)
        model, model_mounts = _resolve_model(
            experiment.benchmark.model, experiment.benchmark.resolved_path
        )
        llm = experiment.llm
        draft_mounts: tuple[Mount, ...] = ()
        if llm.draft_model is not None:
            draft, draft_mounts = _resolve_draft(
                llm.draft_model, llm.resolved_draft_path, model_mounts
            )
            llm = replace(llm, draft_model=draft)
        if llm.tensor_parallel_size is None:
            llm = replace(llm, tensor_parallel_size=hardware.device_count)
        serve_args = (
            "--model",
            model,
            "--served-model-name",
            experiment.benchmark.model,
            "--host",
            _SERVER_BIND_HOST,
            "--port",
            str(_SERVER_PORT),
            *serve_arguments(llm, extra_args=self.extra_args),
        )
        volumes = [
            *container.volumes,
            Mount(
                source=str(_HF_CACHE_HOST),
                target=_HF_CACHE_MOUNT_PATH,
                options=_SELINUX_OPTIONS,
            ),
            *model_mounts,
            *draft_mounts,
        ]
        env = {**_DEV_MODE_ENV, **container.env}
        if self.cache_dir is not None:
            cache_dir = _ensure_directory(self.cache_dir)
            volumes.append(
                Mount(source=str(cache_dir), target=_CACHE_MOUNT_PATH, options=_SELINUX_OPTIONS)
            )
            env = {**_CACHE_ENV, **env}
        return ServerSpec(
            container=ContainerConfig(
                image=container.image or self.image_for(target),
                command=serve_args,
                env=env,
                volumes=tuple(volumes),
                devices=_DEVICE_MAPPINGS[target],
                ports={str(_SERVER_PORT): None},
                mem_limit=container.mem_limit,
                shm_size=container.shm_size,
                selinux_opts=container.selinux_opts,
                pull_policy=container.pull_policy,
            ),
            health_path="/health",
            health_timeout=experiment.benchmark.startup_timeout,
            reset_paths=_RESET_CACHE_ENDPOINTS,
            metrics_path=_METRICS_PATH,
        )

    def image_for(self, target: HardwareTarget) -> str:
        """Return the base image this backend uses for ``target``.

        This is the public image-resolution seam: the container image a run is
        planned with is ``container.image`` when the caller set one, and this
        method's result otherwise. Exposing it lets a caller resolve the same
        image a plan will use — for example to prefetch it before planning —
        without duplicating the default-image mapping.
        """
        if target == HardwareTarget.CPU:
            return self.cpu_image
        return _DEFAULT_IMAGES[target]

    def parse_result(
        self,
        outcome: BenchmarkOutcome,
        spec: ServerSpec,
        result: ContainerResult,
    ) -> ExperimentResult:
        """Assemble a typed result from the host client's measured runs.

        The runs and the failed-run count come straight from the host benchmark
        client, which already enforced the strict majority, so this method only
        records what actually ran: the accelerator class, image, and device
        mapping are recovered from the planned :class:`ServerSpec`, and the
        captured server container outcome is kept as ``raw_result``. Each run
        is enriched with the prefill and decode rates its metric snapshots
        imply, and a run without a usable snapshot pair keeps ``None`` for
        both. An outcome with no successful run raises :class:`ExperimentError`
        rather than letting :class:`ExperimentResult` reject it with a bare
        ``ValueError``.
        """
        if not outcome.runs:
            raise ExperimentError("the benchmark produced no successful run", raw_result=result)
        return ExperimentResult(
            runs=tuple(_with_phase_rates(run) for run in outcome.runs),
            failed_runs=outcome.failed_runs,
            target=_target_for(spec.container.devices, raw_result=result),
            image=spec.container.image,
            devices=spec.container.devices,
            raw_result=result,
        )
