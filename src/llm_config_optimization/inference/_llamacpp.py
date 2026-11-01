"""llama.cpp backend for the inference experiment layer.

This module is the only place that knows llama.cpp's vocabulary: how
``llama-server`` spells the engine-neutral tunables, what the engine's
``--model`` argument refers to, how a GGUF checkpoint's metadata reports its
quantization, and which image, devices, and command a trial plans for each
accelerator class. The engine-neutral surfaces never import it.

Argument translation maps each parameter the engine supports onto its CLI flags
and translates the curated categorical values, so a value with no engine
spelling is rejected rather than silently dropped. The parameters llama.cpp
does not model at all — ``dtype``, ``quantization``, ``tensor_parallel_size``,
and ``gpu_memory_utilization`` — have no target here; declaring one in a space
is the backend's validation problem, not this module's.

Result parsing reads the phase timings llama.cpp attaches to every completion
response, so each measured run carries its prefill and decode rates without a
metrics endpoint: the engine's own ``prompt_n``/``prompt_ms`` and
``predicted_n``/``predicted_ms`` are summed across a run's responses, and
prompt tokens served from the cache are already excluded from ``prompt_n``.

Model resolution decides whether ``--model`` names a local checkpoint or a
Hugging Face repository with an optional ``:QUANT`` selector, and picks the
single GGUF file the server mounts. GGUF metadata makes model-aware validation
possible without the engine: a file that is not GGUF is rejected up front, and
the quantization, block count, expert count, and sliding-window size it declares
are read straight from its header.
"""

import os
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from huggingface_hub import list_repo_files as list_hub_files

from ..runtime.types import ContainerConfig, ContainerResult, Mount
from ..search_space.registry import (
    DRAFT_MODEL_PARAMETERS,
    SPECULATIVE_PARAMETERS,
    TUNABLE_PARAMETERS,
    draft_configured,
    speculative_configured,
)
from ..search_space.types import Default, Dimension, ExperimentSpace, Interval, SpaceAssessment
from ._benchmark import _RESPONSES_KEY
from ._tunables import (
    Gate,
    Toggle,
    declared_values,
    prune_dimensions,
    toggle_arguments,
    translate_value,
    unsupported_values,
)
from .types import (
    AttentionBackend,
    BenchmarkConfig,
    BenchmarkOutcome,
    CpuMoeMode,
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    FitMode,
    GpuLayersMode,
    HardwareTarget,
    HubCheckpoint,
    KVCacheDType,
    LLMConfig,
    LoadMode,
    LocalCheckpoint,
    ModelRequest,
    QuantizationMethod,
    RunMetrics,
    ServerSpec,
    SpecDraftSampling,
    SpeculativeMethod,
    SpeculativeMode,
)

PARAMETER_TARGETS: Mapping[str, tuple[str, ...] | Gate | Toggle] = MappingProxyType(
    {
        "kv_cache_dtype": ("--cache-type-k", "--cache-type-v"),
        "max_model_len": ("--ctx-size",),
        "max_num_seqs": ("--parallel",),
        "max_num_batched_tokens": ("--batch-size",),
        "ubatch_size": ("--ubatch-size",),
        "attention_backend": ("--flash-attn",),
        "speculative_method": ("--spec-type",),
        "draft_model": ("--spec-draft-model",),
        "speculative_mode": Gate(),
        "num_speculative_tokens": ("--spec-draft-n-max",),
        "spec_draft_n_min": ("--spec-draft-n-min",),
        "spec_draft_p_split": ("--spec-draft-p-split",),
        "spec_draft_sampling": ("--spec-draft-sampling",),
        "n_gpu_layers_draft": ("--n-gpu-layers-draft",),
        "n_cpu_moe_draft": ("--n-cpu-moe-draft",),
        "draft_kv_cache_dtype": ("--cache-type-k-draft", "--cache-type-v-draft"),
        "prefix_caching": Toggle(enabled="--cache-prompt", disabled="--no-cache-prompt"),
        "n_gpu_layers": ("--n-gpu-layers",),
        "n_cpu_moe": ("--n-cpu-moe",),
        "n_cpu_ffn": ("--n-cpu-ffn",),
        "kv_offload": Toggle(enabled="--kv-offload", disabled="--no-kv-offload"),
        "op_offload": Toggle(enabled="--op-offload", disabled="--no-op-offload"),
        "host_buffer": Toggle(disabled="--no-host"),
        "repack": Toggle(enabled="--repack", disabled="--no-repack"),
        "swa_full": Toggle(enabled="--swa-full"),
        "load_mode": ("--load-mode",),
        "cache_ram": ("--cache-ram",),
        "ctx_checkpoints": ("--ctx-checkpoints",),
        "checkpoint_min_step": ("--checkpoint-min-step",),
        "kv_unified": Toggle(enabled="--kv-unified", disabled="--no-kv-unified"),
        "continuous_batching": Toggle(enabled="--cont-batching", disabled="--no-cont-batching"),
        "cache_idle_slots": Toggle(enabled="--cache-idle-slots", disabled="--no-cache-idle-slots"),
        "threads": ("--threads",),
        "threads_batch": ("--threads-batch",),
        "poll": ("--poll",),
        "prio": ("--prio",),
        "fit": ("--fit",),
        "fit_target": ("--fit-target",),
        "fit_ctx": ("--fit-ctx",),
    }
)
"""Where each applicable :class:`LLMConfig` field is sent on ``llama-server``.

Unlike vLLM's flags, llama.cpp's names cannot be derived from the field names
(``max_model_len`` becomes ``--ctx-size``), so each one is spelled out. A field
maps to a tuple of flags because the engine splits one value across symmetric
options, as ``kv_cache_dtype`` does for the K and the V cache type, which must
agree for the cache the user asked for. The logical ``max_num_batched_tokens``
and the physical ``ubatch_size`` are independent dimensions rather than one
value sent twice: they map to ``--batch-size`` and ``--ubatch-size``, so
prefill batching can be tuned without changing the logical batch. The
speculative method, draft model, depth, and the draft-only knobs each map to
one ``--spec-*`` or draft-specific flag, while ``speculative_mode`` is a
:class:`Gate`: it is recognised as applicable but never forwarded, because it
only decides whether the other speculative flags are emitted. An empty, absent,
or gate entry means the field is never serialised.

The placement and memory fields enter the same table: ``n_cpu_ffn`` is a
count, ``load_mode`` a translated categorical value, and the boolean pairs are
:tclass:`Toggle` entries whose two sides name the engine's enabling and
disabling flags. ``host_buffer`` and ``swa_full`` expose only one side each —
llama.cpp has ``--no-host`` but no enabling counterpart, and ``--swa-full`` but
no disabling one — so the absent side emits nothing and the engine's own
default applies.

The cache and scheduling fields follow: ``cache_ram``, ``ctx_checkpoints``,
and ``checkpoint_min_step`` are plain values, while ``kv_unified``,
``continuous_batching``, and ``cache_idle_slots`` are two-sided toggles whose
enabling and disabling flags both exist on the engine.

The CPU and process fields close the table: ``threads`` and
``threads_batch`` are plain counts, as are ``poll`` and ``prio``, so each maps
to exactly one ``llama-server`` flag. The fixed-only fit family follows:
``fit`` translates its two engine-neutral modes, while ``fit_target`` and
``fit_ctx`` are plain values forwarded only when fit is enabled.

The insertion order is the backend's canonical argument order, which groups
related flags for readability rather than mirroring the :class:`LLMConfig`
field order; it is fixed, so a run's command is deterministic.
"""

_SPECULATIVE_FIELDS = frozenset(
    {
        "speculative_method",
        "draft_model",
        "num_speculative_tokens",
        "spec_draft_n_min",
        "spec_draft_p_split",
        "spec_draft_sampling",
        "n_gpu_layers_draft",
        "n_cpu_moe_draft",
        "draft_kv_cache_dtype",
    }
)
"""The fields suppressed when the speculative mode is disabled.

``speculative_mode`` is absent because it is the gate rather than a forwarded
field: a disabled mode drops these fields so the engine starts with no
speculative flags and the draft model's memory is genuinely freed.
"""

_FIT_RANGE_FIELDS = frozenset({"fit_target", "fit_ctx"})
"""The fit settings suppressed unless ``fit`` is enabled.

llama.cpp ignores a target margin or minimum context without ``--fit on``, so a
configuration that sets them while fit is unset or off emits neither rather
than forwarding flags the engine would discard.
"""

VALUE_MAPS: Mapping[str, Mapping[object, str]] = MappingProxyType(
    {
        "kv_cache_dtype": {
            KVCacheDType.FLOAT16: "f16",
            KVCacheDType.BFLOAT16: "bf16",
            KVCacheDType.FLOAT32: "f32",
            KVCacheDType.Q8_0: "q8_0",
            KVCacheDType.Q4_0: "q4_0",
            KVCacheDType.Q4_1: "q4_1",
            KVCacheDType.IQ4_NL: "iq4_nl",
            KVCacheDType.Q5_0: "q5_0",
            KVCacheDType.Q5_1: "q5_1",
        },
        "draft_kv_cache_dtype": {
            KVCacheDType.FLOAT16: "f16",
            KVCacheDType.BFLOAT16: "bf16",
            KVCacheDType.FLOAT32: "f32",
            KVCacheDType.Q8_0: "q8_0",
            KVCacheDType.Q4_0: "q4_0",
            KVCacheDType.Q4_1: "q4_1",
            KVCacheDType.IQ4_NL: "iq4_nl",
            KVCacheDType.Q5_0: "q5_0",
            KVCacheDType.Q5_1: "q5_1",
        },
        "attention_backend": {
            AttentionBackend.AUTO: "auto",
            AttentionBackend.FLASH_ATTENTION: "on",
            AttentionBackend.OFF: "off",
        },
        "speculative_method": {
            SpeculativeMethod.DRAFT_MODEL: "draft-simple",
            SpeculativeMethod.EAGLE3: "draft-eagle3",
            SpeculativeMethod.MTP: "draft-mtp",
            SpeculativeMethod.DFLASH: "draft-dflash",
            SpeculativeMethod.DSPARK: "draft-dspark",
            SpeculativeMethod.NGRAM: "ngram-simple",
        },
        "speculative_mode": {
            SpeculativeMode.ENABLED: "enabled",
            SpeculativeMode.DISABLED: "disabled",
        },
        "n_gpu_layers": {
            GpuLayersMode.AUTO: "auto",
            GpuLayersMode.ALL: "all",
        },
        "n_cpu_moe": {
            CpuMoeMode.ALL: "all",
        },
        "n_gpu_layers_draft": {
            GpuLayersMode.AUTO: "auto",
            GpuLayersMode.ALL: "all",
        },
        "n_cpu_moe_draft": {
            CpuMoeMode.ALL: "all",
        },
        "spec_draft_sampling": {
            SpecDraftSampling.GREEDY: "greedy",
            SpecDraftSampling.PROBABILISTIC: "probabilistic",
        },
        "load_mode": {
            LoadMode.AUTO: "auto",
            LoadMode.NONE: "none",
            LoadMode.MMAP: "mmap",
            LoadMode.MLOCK: "mlock",
            LoadMode.MMAP_MLOCK: "mmap+mlock",
            LoadMode.DIO: "dio",
        },
        "fit": {
            FitMode.ON: "on",
            FitMode.OFF: "off",
        },
    }
)
"""How llama.cpp spells each engine-neutral categorical value it can serve.

The enums in :mod:`llm_config_optimization.inference.types` carry no engine
vocabulary; this table is the only place their llama.cpp spellings live. A
member missing from its field's map is unsupported and is rejected rather than
forwarded, while the engine's own default is reached by declaring the parameter
as ``default: true`` instead of naming a value.

``kv_cache_dtype`` covers the cache formats llama.cpp serves: the three float
formats and the quantized ``q*``/``iq4_nl`` formats. ``auto`` and the FP8
members have no equivalent here. ``draft_kv_cache_dtype`` shares those
spellings for the draft model's own cache. For ``attention_backend`` all three
flash-attention settings are modelled: ``auto`` lets the engine choose,
``flash_attention`` maps to the engine's ``on`` setting, and ``off`` disables
it; ``flash_infer`` and ``xformers`` are vLLM-only backends. The six
speculative methods map onto llama.cpp's ``--spec-type`` spellings; the
engine-only n-gram cache, map, and mod variants stay behind ``extra_args``.
``n_gpu_layers`` translates its two modes and passes an exact integer count
through unchanged, so a union dimension samples both branches. ``n_cpu_moe``
does the same for an exact count, but its :attr:`CpuMoeMode.ALL` mode is a
value-less flag of its own rather than a spelling of ``--n-cpu-moe``; that
branch is assembled in :func:`serve_arguments`. ``load_mode`` maps the
engine-neutral loading strategies onto llama.cpp's names, with the compound
:attr:`LoadMode.MMAP_MLOCK` becoming the engine's ``mmap+mlock``. The draft
placement and sampling maps mirror their main-model counterparts, and
``draft_kv_cache_dtype`` follows ``kv_cache_dtype``. ``fit`` maps the two
engine-neutral modes onto the engine's own spellings; the fit target and
context are plain values rather than categoricals, so they need no entry here.
"""

_CPU_MOE_ALL_FLAG = "--cpu-moe"
"""The value-less flag that keeps every expert layer in system RAM.

llama.cpp models the two choices as different flags: ``--n-cpu-moe N`` takes an
exact layer count and ``--cpu-moe`` takes no value at all. The ``all`` mode must
therefore emit this flag rather than ``--n-cpu-moe all``, which the engine would
reject while parsing the count.
"""

_CPU_MOE_DRAFT_ALL_FLAG = "--cpu-moe-draft"
"""The draft-model counterpart of :data:`_CPU_MOE_ALL_FLAG`.

``n_cpu_moe_draft`` reuses the same two-flag shape: an exact count is
``--n-cpu-moe-draft N`` while the ``all`` mode is the value-less
``--cpu-moe-draft``.
"""

_CPU_MOE_PARAMETER = "n_cpu_moe"
"""The tunable that keeps a number of MoE expert layers in system RAM.

It only changes how a Mixture-of-Experts checkpoint's experts are placed, so a
dense checkpoint cannot be affected by it.
"""

_DENSE_CHECKPOINT_REASON = "the checkpoint declares no experts"
"""Why the CPU expert offload is dropped for a checkpoint that is confidently dense."""

_CPU_TARGET_MOE_REASON = "the CPU target keeps every weight in system RAM already"
"""Why the CPU expert offload is dropped for a CPU-only run.

``--n-cpu-moe``/``--cpu-moe`` keep some expert layers out of the accelerator's
memory. On a CPU target no weight ever leaves system RAM, so neither flag can
change how much memory the run uses or where the experts are computed.
"""

_CPU_TARGET_FFN_REASON = "the CPU target keeps every dense FFN weight in system RAM already"
"""Why the dense FFN CPU offload is dropped for a CPU-only run.

``--n-cpu-ffn`` keeps some dense FFN layers out of the accelerator's memory. A
CPU target has no accelerator memory to leave them out of, so the flag cannot
change where the weights live or how they are computed.
"""

_CPU_TARGET_DEVICE_OFFLOAD_REASON = "the CPU target has no device memory to offload from"
"""Why the tensor-operation and KV-cache offloads are dropped for a CPU-only run.

``--kv-offload`` and ``--op-offload`` move KV cache and tensor operations to a
device. On a CPU target every operation already runs on the host, so neither
flag can change the work's placement or cost.
"""

_CPU_TARGET_DRAFT_GPU_LAYERS_REASON = "the CPU target can never offload draft GPU layers"
"""Why the draft model's GPU-layer offload is dropped for a CPU-only run.

``--n-gpu-layers-draft`` offloads draft layers to a GPU, which a CPU target does
not have, so no declared mode or count can take effect there.
"""

_CPU_TARGET_DRAFT_MOE_REASON = "the CPU target keeps every draft weight in system RAM already"
"""Why the draft model's CPU expert offload is dropped for a CPU-only run.

``--n-cpu-moe-draft``/``--cpu-moe-draft`` keep draft expert layers out of the
accelerator's memory. On a CPU target no draft weight ever leaves system RAM,
so the flag cannot change memory use or computation placement.
"""


_CPU_TARGET_GPU_LAYERS_REASON = (
    "the CPU target can never offload GPU layers and every declared value already disables it"
)
"""Why a CPU-valid GPU-layer dimension is dropped for a CPU-only run.

``auto`` may resolve to no offload and ``0`` disables it, so a dimension
confined to those values cannot change anything on a CPU target. ``all`` and
positive counts are rejected by the pre-model validation instead of being
pruned, so a GPU-oriented space never silently changes meaning on a CPU host.
"""

_SWA_PARAMETER = "swa_full"
"""The tunable that selects a full-size sliding-window attention cache.

It only changes a checkpoint that uses sliding-window attention, so a model
with full attention cannot be affected by it.
"""

_SWA_FREE_CHECKPOINT_REASON = "the checkpoint declares no sliding-window attention"
"""Why the full-size SWA cache is dropped for a full-attention checkpoint."""

_MAIN_BLOCK_COUNT_HINTS: Mapping[str, str] = MappingProxyType(
    {
        "n_gpu_layers": "use n_gpu_layers=all to offload every layer",
        _CPU_MOE_PARAMETER: "use n_cpu_moe=all to keep every expert layer on the CPU",
        "n_cpu_ffn": "reduce n_cpu_ffn to the checkpoint's layer count (it has no all mode)",
    }
)
"""How to fix each main-checkpoint layer count that exceeds the block count.

``n_gpu_layers`` and ``n_cpu_moe`` have an ``all`` mode that expresses full
placement unambiguously, so their hints point at it. ``n_cpu_ffn`` has no mode
and its ``0`` disables the offload, so the hint names the count to reduce
instead.
"""

_DRAFT_BLOCK_COUNT_HINTS: Mapping[str, str] = MappingProxyType(
    {
        "n_gpu_layers_draft": "use n_gpu_layers_draft=all to offload every draft layer",
        "n_cpu_moe_draft": ("use n_cpu_moe_draft=all to keep every draft expert layer on the CPU"),
    }
)
"""How to fix each draft-checkpoint layer count that exceeds its block count.

The draft model has the same ``all`` modes as the target model, so those hints
mirror :data:`_MAIN_BLOCK_COUNT_HINTS`; the counts are bounded by the draft
checkpoint's own block count because that is the model they apply to.
"""
_QUANTIZED_CACHE_DTYPES: frozenset[KVCacheDType] = frozenset(
    {
        KVCacheDType.Q8_0,
        KVCacheDType.Q4_0,
        KVCacheDType.Q4_1,
        KVCacheDType.IQ4_NL,
        KVCacheDType.Q5_0,
        KVCacheDType.Q5_1,
    }
)
"""KV-cache formats llama.cpp can only build with flash attention enabled.

llama.cpp rejects a quantized V cache when ``--flash-attn off`` is passed,
because the quantized cache kernels require flash attention. The field's
engine-neutral names are used here; this set is llama.cpp's subset that carries
the constraint.
"""

_NUMERIC_RANGES: Mapping[str, tuple[int | float, int | float | None]] = MappingProxyType(
    {
        "poll": (0, 100),
        "prio": (-1, 3),
        "spec_draft_p_split": (0.0, 1.0),
        "cache_ram": (-1, None),
    }
)
"""Inclusive numeric ranges llama.cpp accepts for a single tunable.

The loader already enforces each parameter's intrinsic floor; these are the
engine's own limits. ``poll`` selects a busy-wait level on a 0..100 scale,
``prio`` is the process nice value llama.cpp documents from ``-1`` to ``3``,
and ``spec_draft_p_split`` is a probability. ``cache_ram`` is one-sided:
``-1`` is the engine's "no limit" marker and ``0`` disables the prompt cache,
so every value below ``-1`` is meaningless and it has no upper bound.
"""


def serve_arguments(
    config: LLMConfig,
    *,
    extra_args: Sequence[str] = (),
    draft_repository: str | None = None,
) -> list[str]:
    """Turn an LLM config into the ``llama-server`` arguments that carry it.

    Unset fields (``None``) are omitted so the engine keeps its own defaults,
    categorical fields are translated from their engine-neutral enums into
    llama.cpp's spellings, and a field that targets two flags emits both with
    the same value. Any ``extra_args`` are appended verbatim after the mapped
    parameters as the backend's escape hatch for unmodeled serving flags. The
    workload is assembled separately by the host benchmark client, so the server
    command carries no workload flags.

    The speculative method, draft model, and depth target their own
    ``--spec-*`` flags, translated into llama.cpp's spellings. A disabled
    speculative mode suppresses all three so no speculative flag reaches the
    engine; an unset mode means the configured setup's default and emits them,
    and an unset depth omits only ``--spec-draft-n-max`` so the engine keeps its
    own draft depth.

    ``draft_repository`` is the repository a plan could not resolve to a local
    file. It replaces the ``--spec-draft-model`` target at the draft model's
    position in the canonical order, because llama.cpp fetches a repository
    through its own ``--spec-draft-hf`` flag rather than through a file path.
    It is suppressed with the other speculative fields when the mode is
    disabled.

    An exact ``n_cpu_moe`` or ``n_cpu_moe_draft`` count is forwarded as its
    ``--n-cpu-moe``/``--n-cpu-moe-draft`` flag, while its ``all`` mode emits the
    value-less ``--cpu-moe``/``--cpu-moe-draft`` flag. The two are different
    flags rather than one flag with two spellings, so the mode is special-cased
    instead of going through the value map.

    Raises :class:`ExperimentError` for an enum value the engine has no spelling
    for.
    """
    suppressed = (
        _SPECULATIVE_FIELDS if config.speculative_mode is SpeculativeMode.DISABLED else frozenset()
    )
    if config.fit is not FitMode.ON:
        suppressed |= _FIT_RANGE_FIELDS
    arguments: list[str] = []
    for field, target in PARAMETER_TARGETS.items():
        if field in suppressed or isinstance(target, Gate):
            continue
        if field == "draft_model" and draft_repository is not None:
            arguments.extend(("--spec-draft-hf", draft_repository))
            continue
        value = getattr(config, field)
        if value is None:
            continue
        if field == "n_cpu_moe" and value is CpuMoeMode.ALL:
            arguments.append(_CPU_MOE_ALL_FLAG)
            continue
        if field == "n_cpu_moe_draft" and value is CpuMoeMode.ALL:
            arguments.append(_CPU_MOE_DRAFT_ALL_FLAG)
            continue
        if isinstance(target, Toggle):
            arguments.extend(toggle_arguments(value, target))
            continue
        translated = translate_value(field, value, VALUE_MAPS, engine="llama.cpp")
        for flag in target:
            arguments.extend((flag, str(translated)))
    return [*arguments, *extra_args]


def _planned_llm(config: LLMConfig, benchmark: BenchmarkConfig) -> LLMConfig:
    """Return the config with ``max_num_seqs`` cut to what the context can serve.

    llama.cpp splits ``--ctx-size`` evenly across the ``--parallel`` slots, so a
    request longer than a slot's share is rejected no matter how many slots the
    engine was asked for. The request length is the random workload's known
    ``input_len + output_len`` or, for a JSONL dataset, the measured
    ``max_request_tokens`` a preflight tokenization recorded. The slot count is
    then capped at ``max_model_len // request_tokens``, at least one, and only
    then is ``--parallel`` forwarded. The sampled ``max_model_len`` still reaches
    the engine as ``--ctx-size`` unchanged; a context smaller than the request
    length cannot be served at any slot count, so its trial fails at the engine.

    A dataset workload whose token length could not be measured — the preflight
    was skipped or failed — conservatively forwards one slot, which gives every
    prompt the whole context instead of letting a large sampled slot count
    reject it. A workload whose request length is genuinely unknown, such as an
    unset random length, leaves the declared slot count alone because there is
    nothing to size it against.
    """
    if config.max_model_len is None or config.max_num_seqs is None:
        return config
    if benchmark.dataset_path is not None:
        request_tokens = benchmark.max_request_tokens
        if request_tokens is None:
            return replace(config, max_num_seqs=1)
    elif benchmark.input_len is not None and benchmark.output_len is not None:
        request_tokens = benchmark.input_len + benchmark.output_len
    else:
        return config
    if request_tokens < 1:
        return config
    return replace(
        config,
        max_num_seqs=min(config.max_num_seqs, max(1, config.max_model_len // request_tokens)),
    )


_GGUF_SUFFIX = ".gguf"
"""The suffix a GGUF checkpoint's file name carries.

Repositories are filtered by this suffix before a file is selected, so the
fetch is narrowed to a servable checkpoint instead of its README or an
adapter.
"""

_RepoFileLister = Callable[[str], Sequence[str]]
"""Lists the file names a Hugging Face repository holds.

The seam is injectable so resolution can be unit-tested without the network;
production passes :func:`huggingface_hub.list_repo_files`.
"""


def model_request(
    model: str,
    *,
    list_repo_files: _RepoFileLister = list_hub_files,
) -> ModelRequest:
    """Describe the checkpoint llama.cpp's ``--model`` argument refers to.

    An existing host file is a local checkpoint and is returned resolved; an
    existing directory is rejected, because llama.cpp serves one GGUF file
    rather than a snapshot directory. Anything else is a Hugging Face
    repository id, optionally followed by ``:`` and a selector — either the
    GGUF file's name or its quantization tag. The repository's files are read
    through ``list_repo_files``, so the single GGUF file is picked automatically
    and an ambiguous repository is rejected instead of guessed at; the returned
    :class:`HubCheckpoint` names the exact file to fetch.

    Existence is the only test for the local case, so a repository id that
    happens to resolve to a path in the working tree is treated as a local
    checkpoint and mounted rather than downloaded. A local file's GGUF-ness is
    not checked here; the checkpoint's metadata is read before the run, so the
    engine-neutral path stays free of file-format details.
    """
    path = Path(model).expanduser()
    if path.exists():
        if not path.is_file():
            raise ExperimentError(
                f"{path} is not a GGUF file; point --model at a local .gguf checkpoint"
            )
        return LocalCheckpoint(path=path.resolve())
    repo_id, separator, raw_selector = model.partition(":")
    selector = raw_selector if separator and raw_selector else None
    files = [name for name in list_repo_files(repo_id) if name.lower().endswith(_GGUF_SUFFIX)]
    if not files:
        raise ExperimentError(f"the repository {repo_id!r} offers no GGUF files")
    return HubCheckpoint(repo_id=repo_id, filename=_select_gguf(files, selector, repo_id))


def _select_gguf(files: Sequence[str], selector: str | None, repo_id: str) -> str:
    """Pick the one GGUF file a repository holds or a selector names.

    With no selector, a repository that holds exactly one GGUF file is used
    directly while one that holds several is rejected: guessing could silently
    measure a different quantization than the user meant. A selector is matched
    as an exact file path first and then as a case-insensitive substring of the
    file path, which is what lets ``:Q4_K_M`` name a file like
    ``Llama-3.2-1B-Instruct-Q4_K_M.gguf``. Zero or several matches are rejected
    with the candidates listed, so the user can narrow the choice.
    """
    if selector is None:
        if len(files) == 1:
            return files[0]
        raise ExperimentError(
            f"the repository {repo_id!r} holds several GGUF files ({', '.join(files)}); "
            f"select one with {repo_id}:QUANT"
        )
    exact = [name for name in files if name == selector]
    if len(exact) == 1:
        return exact[0]
    matches = [name for name in files if selector.casefold() in name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ExperimentError(
            f"no GGUF file in {repo_id!r} matches {selector!r}; available files: {', '.join(files)}"
        )
    raise ExperimentError(
        f"several GGUF files in {repo_id!r} match {selector!r} ({', '.join(matches)}); "
        f"select one with {repo_id}:{matches[0]}"
    )


_GGUF_MAGIC = b"GGUF"
"""The four bytes every GGUF file starts with."""

_GGUF_VERSIONS = frozenset({2, 3})
"""GGUF container versions whose layout this reader understands.

Version 1 used 32-bit counts and is long obsolete. An unknown version is not a
format mismatch but an unreadable header, so the reader fails open rather than
guessing at a layout it does not know.
"""

_GGUF_FILE_TYPE_KEY = "general.file_type"
"""The metadata key that carries the checkpoint's quantization."""

_GGUF_ARCHITECTURE_KEY = "general.architecture"
"""The metadata key that names the checkpoint's architecture.

The block count is architecture-prefixed (``llama.block_count``,
``qwen2.block_count``, and so on), so the reader learns the architecture from
this key rather than hard-coding one model family.
"""

_GGUF_BLOCK_COUNT_SUFFIX = ".block_count"
"""The architecture-independent suffix of the checkpoint's block count."""

_GGUF_EXPERT_COUNT_SUFFIX = ".expert_count"
"""The architecture-independent suffix of the checkpoint's expert count.

A Mixture-of-Experts checkpoint declares how many experts each layer routes
between (``qwen3moe.expert_count`` and so on); a dense checkpoint declares no
such key. The count is architecture-prefixed for the same reason the block
count is.
"""

_GGUF_SLIDING_WINDOW_SUFFIX = ".attention.sliding_window"
"""The architecture-independent suffix of the checkpoint's attention window.

A checkpoint that uses sliding-window attention declares its window size in
tokens (``gemma2.attention.sliding_window`` and so on); llama.cpp reads the key
as optional, so a checkpoint without it uses full attention. The value is
architecture-prefixed for the same reason the block count is.
"""
_GGUF_BOOL_TYPE = 7
"""The value type id of a boolean metadata value."""

_GGUF_STRING_TYPE = 8
"""The value type id of a length-prefixed string metadata value."""

_GGUF_ARRAY_TYPE = 9
"""The value type id of a homogeneous array metadata value."""

_GGUF_SCALAR_FORMATS: Mapping[int, str] = MappingProxyType(
    {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
)
"""Binary format of every fixed-width GGUF metadata value, keyed by type id.

String and array values are variable-length and are handled separately by the
reader. The ids are the ``GGUFValueType`` constants from the GGUF
specification, and every format is little-endian because GGUF is.
"""

_GGUF_FILE_TYPE_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        7: "Q8_0",
        8: "Q5_0",
        9: "Q5_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "IQ2_XXS",
        20: "IQ2_XS",
        21: "Q2_K_S",
        22: "IQ3_XS",
        23: "IQ3_XXS",
        24: "IQ1_S",
        25: "IQ4_NL",
        26: "IQ3_S",
        27: "IQ3_M",
        28: "IQ2_S",
        29: "IQ2_M",
        30: "IQ4_XS",
        31: "IQ1_M",
        32: "BF16",
        36: "TQ1_0",
        37: "TQ2_0",
        38: "MXFP4_MOE",
        39: "NVFP4",
        40: "Q1_0",
        41: "Q2_0",
    }
)
"""Canonical short tag of every quantization ``general.file_type`` can declare.

The numbers are llama.cpp's ``llama_ftype`` values from ``llama.h`` and the tags
are the names checkpoint repositories use in their file names (``Q4_K_M``,
``BF16``, and so on), so a reported quantization reads the same as the one a
user selects. Removed or future values are absent; the reader reports no
quantization for them rather than inventing a name.
"""


class _UnreadableGguf(Exception):
    """Raised internally when the GGUF magic is present but the header is not parseable."""


@dataclass(frozen=True, slots=True)
class GgufMetadata:
    """The model facts a GGUF checkpoint declares in its metadata.

    ``quantization`` is the canonical short tag for the checkpoint's
    ``general.file_type`` (``"Q4_K_M"``, ``"BF16"``, and so on), or ``None``
    when the checkpoint declares no file type or one this library does not
    name. ``general.file_type`` is the value llama.cpp itself reports as the
    model's quantization, so validation speaks the engine's own vocabulary.

    ``block_count`` is the number of transformer blocks the checkpoint's
    architecture declares through its ``<arch>.block_count`` key, or ``None``
    when that key is absent or the header cannot be read. It bounds an exact
    GPU-layer count: offloading more layers than the checkpoint has is a
    typo, and ``--n-gpu-layers all`` is the unambiguous full-offload choice.

    ``expert_count`` is the number of experts the architecture declares
    through its ``<arch>.expert_count`` key, or ``None`` when the key is
    absent or the header cannot be read. A declared zero is reported as zero:
    a readable checkpoint that declares no experts is confidently dense, while
    absent metadata leaves the structure unknown.

    ``sliding_window`` is the attention window in tokens the architecture
    declares through its ``<arch>.attention.sliding_window`` key, or ``None``
    when the key is absent or the header cannot be read. A positive value
    means the checkpoint uses sliding-window attention; absent metadata leaves
    that unknown, while a readable checkpoint with no window or a window of
    zero uses full attention.
    """

    quantization: str | None
    block_count: int | None = None
    expert_count: int | None = None
    sliding_window: int | None = None


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes, or signal that the header is unreadable."""
    data = handle.read(size)
    if len(data) != size:
        raise _UnreadableGguf
    return data


def _read_uint32(handle: BinaryIO) -> int:
    """Read one little-endian unsigned 32-bit integer."""
    return int.from_bytes(_read_exact(handle, 4), "little")


def _read_uint64(handle: BinaryIO) -> int:
    """Read one little-endian unsigned 64-bit integer."""
    return int.from_bytes(_read_exact(handle, 8), "little")


def _read_scalar(handle: BinaryIO, format: str) -> object:
    """Read one fixed-width metadata value in ``format``."""
    return struct.unpack(format, _read_exact(handle, struct.calcsize(format)))[0]


def _read_metadata_string(handle: BinaryIO) -> str:
    """Read one length-prefixed UTF-8 metadata string."""
    return _read_exact(handle, _read_uint64(handle)).decode("utf-8")


def _skip_metadata_value(handle: BinaryIO, value_type: int) -> None:
    """Advance past one metadata value without materialising it.

    Only the file type and the architecture-prefixed facts are ever read, so
    every other entry is skipped.
    A fixed-width array is skipped in one seek, which matters because tokenizer
    metadata can hold a hundred thousand elements; only string elements and
    nested arrays are walked one by one.
    """
    if value_type == _GGUF_STRING_TYPE:
        handle.seek(_read_uint64(handle), os.SEEK_CUR)
        return
    if value_type == _GGUF_ARRAY_TYPE:
        element_type = _read_uint32(handle)
        count = _read_uint64(handle)
        format = _GGUF_SCALAR_FORMATS.get(element_type)
        if format is not None:
            handle.seek(struct.calcsize(format) * count, os.SEEK_CUR)
            return
        for _ in range(count):
            _skip_metadata_value(handle, element_type)
        return
    format = _GGUF_SCALAR_FORMATS.get(value_type)
    if format is None:
        raise _UnreadableGguf
    handle.seek(struct.calcsize(format), os.SEEK_CUR)


def read_gguf_metadata(path: Path) -> GgufMetadata | None:
    """Read the facts a GGUF checkpoint declares, if they can be read.

    A path that is not a regular file, or a file that does not start with the
    GGUF magic, is rejected with :class:`ExperimentError`: llama.cpp cannot
    serve it, so failing before an image is pulled is the clear outcome. A file
    that is GGUF but whose header cannot be parsed returns ``None`` (fail
    open), leaving the engine as the final authority rather than rejecting on
    metadata this library could not see. An unknown container version or
    ``general.file_type`` value is likewise reported as no quantization, never
    as a guess.

    The block count, expert count, and sliding-window size are looked up under
    the architecture named by ``general.architecture`` (``llama.block_count``,
    ``qwen2.block_count``, and so on). A file that declares no architecture
    still reports a fact when exactly one architecture-prefixed key for it is
    present, and reports none when the key is absent or several architectures
    would make the choice ambiguous. An absent key is reported as no fact
    rather than as an error, so the engine stays the final authority.
    """
    if not path.is_file():
        raise ExperimentError(f"{path} is not a GGUF file; llama.cpp serves GGUF checkpoints")
    try:
        with path.open("rb") as handle:
            if _read_exact(handle, len(_GGUF_MAGIC)) != _GGUF_MAGIC:
                raise ExperimentError(
                    f"{path} is not a GGUF file; llama.cpp serves GGUF checkpoints"
                )
            if _read_uint32(handle) not in _GGUF_VERSIONS:
                return None
            _read_uint64(handle)
            metadata_count = _read_uint64(handle)
            file_type: int | None = None
            architecture: str | None = None
            block_counts: dict[str, int] = {}
            expert_counts: dict[str, int] = {}
            sliding_windows: dict[str, int] = {}
            for _ in range(metadata_count):
                key = _read_metadata_string(handle)
                value_type = _read_uint32(handle)
                scalar_format = _GGUF_SCALAR_FORMATS.get(value_type)
                if key == _GGUF_ARCHITECTURE_KEY and value_type == _GGUF_STRING_TYPE:
                    architecture = _read_metadata_string(handle)
                elif key == _GGUF_FILE_TYPE_KEY and value_type != _GGUF_BOOL_TYPE and scalar_format:
                    value = _read_scalar(handle, scalar_format)
                    if isinstance(value, int) and not isinstance(value, bool):
                        file_type = value
                elif key.endswith(_GGUF_BLOCK_COUNT_SUFFIX) and scalar_format:
                    value = _read_scalar(handle, scalar_format)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        block_counts[key] = value
                elif key.endswith(_GGUF_EXPERT_COUNT_SUFFIX) and scalar_format:
                    value = _read_scalar(handle, scalar_format)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        expert_counts[key] = value
                elif key.endswith(_GGUF_SLIDING_WINDOW_SUFFIX) and scalar_format:
                    value = _read_scalar(handle, scalar_format)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        sliding_windows[key] = value
                else:
                    _skip_metadata_value(handle, value_type)
    except ExperimentError:
        raise
    except OSError, _UnreadableGguf, UnicodeDecodeError:
        return None
    quantization = _GGUF_FILE_TYPE_LABELS.get(file_type) if file_type is not None else None
    return GgufMetadata(
        quantization=quantization,
        block_count=_resolve_architecture_value(
            _GGUF_BLOCK_COUNT_SUFFIX, architecture, block_counts
        ),
        expert_count=_resolve_architecture_value(
            _GGUF_EXPERT_COUNT_SUFFIX, architecture, expert_counts
        ),
        sliding_window=_resolve_architecture_value(
            _GGUF_SLIDING_WINDOW_SUFFIX, architecture, sliding_windows
        ),
    )


def _resolve_architecture_value(
    suffix: str, architecture: str | None, values: Mapping[str, int]
) -> int | None:
    """Pick the value the checkpoint's architecture names, if any.

    A declared architecture decides on its own key: a value stored under a
    different architecture is not this checkpoint's, so an absent key reports
    nothing rather than borrowing the only other entry. Without a declared
    architecture, a single architecture-prefixed value is unambiguous and is
    reported; several would make the choice a guess, so none is reported.
    """
    if architecture is not None:
        return values.get(f"{architecture}{suffix}")
    if len(values) == 1:
        return next(iter(values.values()))
    return None


_MODEL_MOUNT_PATH = "/models/model.gguf"
"""Where a resolved GGUF checkpoint is mounted inside the server container.

llama.cpp serves one file rather than a directory, so the checkpoint is mounted
at a fixed target and the command points ``--model`` at it. Mounting the
resolved file itself also follows a Hugging Face cache symlink on the host, so
a snapshot download's relative link cannot dangle inside the container the way
it would if only its parent directory were mounted.
"""

_DRAFT_MOUNT_PATH = "/models/draft.gguf"
"""Where a resolved draft GGUF checkpoint is mounted inside the container.

The draft gets its own fixed target so it never contends with the main model's
``/models/model.gguf``, which also keeps a local target and a local draft
addressable at the same time.
"""

_SERVER_PORT = 8080
"""The container port the server binds and the plan publishes.

The port is explicit rather than left to the engine's own default, so the
command, the published mapping, and the host client's URL all name the same
one. The host port is assigned by the runtime, so parallel trials never
contend for a fixed one.
"""

_SERVER_BIND_HOST = "0.0.0.0"
"""Where the server binds inside the container.

The host benchmark client reaches the server through the container's published
port, so the server must accept connections on every interface rather than
container loopback.
"""

_HEALTH_PATH = "/health"
"""The llama-server endpoint the host client polls until the model is loaded."""

_TOKENIZE_PATH = "/tokenize"
"""The llama-server endpoint that returns a text's token ids.

A preflight run uses it to measure a JSONL dataset's real request lengths, so
slot planning can cap ``--parallel`` from the engine's own tokenizer rather
than guessing from characters.
"""

_TIMINGS_KEY = "timings"
"""The per-response key carrying llama.cpp's own phase timings.

The host client preserves every successful completion payload in a run's raw
JSON, and llama.cpp attaches its timings to each such response, so this engine
needs no scrapeable metrics endpoint: the snapshots the client takes for vLLM
have no counterpart here.
"""

_PREFILL_TIMING_KEYS = ("prompt_n", "prompt_ms")
"""The ``(tokens, milliseconds)`` pair behind the prefill rate.

``prompt_n`` is llama.cpp's ``n_prompt_processed``: the prompt tokens the
server actually processed, with the tokens it served from its cache — reported
separately as ``cache_n`` — already excluded, so the rate measures work
performed. ``prompt_ms`` is the milliseconds it spent on that work.
"""

_DECODE_TIMING_KEYS = ("predicted_n", "predicted_ms")
"""The ``(tokens, milliseconds)`` pair behind the decode rate.

``predicted_n`` counts the tokens generated and ``predicted_ms`` the
milliseconds spent generating them.
"""

_SELINUX_OPTIONS = ("z",)
"""Bind-mount options applied to the checkpoint mount.

The ``z`` option asks the container runtime to relabel the host path for
SELinux, so runs work on enforcing hosts. Caller-supplied volumes keep whatever
options they were given.
"""

_DEFAULT_IMAGES: Mapping[HardwareTarget, str] = MappingProxyType(
    {
        HardwareTarget.CUDA: "ghcr.io/ggml-org/llama.cpp:server-cuda",
        HardwareTarget.ROCM: "ghcr.io/ggml-org/llama.cpp:server-rocm",
        HardwareTarget.CPU: "ghcr.io/ggml-org/llama.cpp:server",
    }
)
"""llama.cpp's official server image per accelerator class.

The names are already registry-qualified, so podman resolves them without a
TTY, and each image contains only ``llama-server`` — exactly what a
host-driven trial runs. ``--image`` still overrides the mapping through the
container configuration.
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

    :meth:`LlamaCppBackend.plan` fills ``devices`` straight from
    :data:`_DEVICE_MAPPINGS`, so a run's target can be read back from the plan
    that produced it. A mapping the backend never emits cannot be attributed to
    a target and raises rather than guessing; ``raw_result`` is attached to that
    error so every :meth:`LlamaCppBackend.parse_result` failure carries the
    captured container output uniformly.
    """
    for target, mapping in _DEVICE_MAPPINGS.items():
        if mapping == devices:
            return target
    raise ExperimentError(
        f"cannot determine the accelerator class for devices {devices!r}",
        raw_result=raw_result,
    )


def _timing_pair(response: object, keys: tuple[str, str]) -> tuple[float, float] | None:
    """Return one response's ``(tokens, milliseconds)`` timing pair, or ``None``.

    The response bodies live in a run's untyped raw JSON, so a payload that is
    not a mapping, carries no ``timings`` object, or holds a value that is not
    a real number is treated as absent rather than coerced. ``keys`` names the
    phase's token count and milliseconds in the engine's own vocabulary.
    """
    if not isinstance(response, Mapping):
        return None
    timings = response.get(_TIMINGS_KEY)
    if not isinstance(timings, Mapping):
        return None
    tokens = timings.get(keys[0])
    milliseconds = timings.get(keys[1])
    if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
        return None
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)):
        return None
    return float(tokens), float(milliseconds)


def _phase_rate(run: RunMetrics, keys: tuple[str, str]) -> float | None:
    """Derive one phase's tokens-per-second rate from the response timings.

    Every response in the run must carry the phase's timing pair, because the
    rate describes the whole run's phase work and a response that went untimed
    would otherwise be silently dropped from the sum. The result is ``None``
    when the run stores no responses, any response lacks the pair, or the
    summed tokens or milliseconds are not positive — a phase with no measured
    work has no rate, and a zero denominator cannot be divided by.
    """
    responses = run.raw_json.get(_RESPONSES_KEY)
    if not isinstance(responses, (list, tuple)) or not responses:
        return None
    tokens = 0.0
    milliseconds = 0.0
    for response in responses:
        pair = _timing_pair(response, keys)
        if pair is None:
            return None
        tokens += pair[0]
        milliseconds += pair[1]
    if tokens <= 0 or milliseconds <= 0:
        return None
    return tokens / (milliseconds / 1000.0)


def _with_phase_rates(run: RunMetrics) -> RunMetrics:
    """Return a run enriched with the phase rates its response timings imply.

    Each rate is derived independently, so a run that reported one phase's
    timings but not the other's keeps the rate it has rather than losing both.
    When neither rate can be derived the run is returned unchanged, so a
    payload that carries no timings keeps the object the client produced.
    """
    prefill = _phase_rate(run, _PREFILL_TIMING_KEYS)
    decode = _phase_rate(run, _DECODE_TIMING_KEYS)
    if prefill is None and decode is None:
        return run
    return replace(run, prefill_tokens_per_second=prefill, decode_tokens_per_second=decode)


def _model_mount(path: Path, target: str = _MODEL_MOUNT_PATH) -> Mount:
    """Build the read-only mount that exposes a checkpoint to the server."""
    return Mount(
        source=str(path),
        target=target,
        read_only=True,
        options=_SELINUX_OPTIONS,
    )


def _model_arguments(
    model: str,
    resolved_path: Path | None,
    list_repo_files: _RepoFileLister,
) -> tuple[list[str], tuple[Mount, ...]]:
    """Resolve the checkpoint arguments and any mount the server needs.

    A ``resolved_path`` — the GGUF file the composition root already fetched —
    is mounted read-only at a fixed container path and the command points at
    it. Without one, the engine's own model-resolution rules decide: a local
    file is mounted the same way, while a Hugging Face repository is left for
    the server to fetch itself through ``--hf-repo``/``--hf-file``, so a direct
    library caller that never prefetched the checkpoint still plans a runnable
    server. The production path is the resolved one: the composition root
    resolves and validates the model before planning, and the server needs no
    network of its own.
    """
    if resolved_path is not None:
        return ["--model", _MODEL_MOUNT_PATH], (_model_mount(resolved_path),)
    request = model_request(model, list_repo_files=list_repo_files)
    if isinstance(request, LocalCheckpoint):
        return ["--model", _MODEL_MOUNT_PATH], (_model_mount(request.path),)
    arguments = ["--hf-repo", request.repo_id]
    if request.filename is not None:
        arguments += ["--hf-file", request.filename]
    return arguments, ()


def _draft_arguments(
    llm: LLMConfig,
    list_repo_files: _RepoFileLister,
) -> tuple[LLMConfig, tuple[Mount, ...], str | None]:
    """Resolve the draft-model argument and any mount the server needs.

    No configured draft leaves the config and the plan unchanged. A
    ``resolved_draft_path`` — the draft file the composition root already
    fetched — is mounted read-only at the fixed draft path and ``--spec-draft-
    model`` is pointed at it. Without a resolved path, the engine's own
    model-resolution rules decide: a local GGUF file is mounted the same way,
    while a Hugging Face repository is returned as the repository string the
    engine fetches itself through ``--spec-draft-hf``. The draft is mounted even
    when the sampled mode disables speculation, but its arguments are suppressed
    there, so the engine never loads it and its memory is genuinely freed.

    The returned config never carries the host draft path into an engine flag: a
    mounted draft's ``draft_model`` is rewritten to the in-container path, and a
    repository draft's ``draft_model`` is cleared because the repository is
    returned separately under the engine's own flag.
    """
    if llm.draft_model is None:
        return llm, (), None
    if llm.resolved_draft_path is not None:
        return (
            replace(llm, draft_model=_DRAFT_MOUNT_PATH),
            (_model_mount(llm.resolved_draft_path, _DRAFT_MOUNT_PATH),),
            None,
        )
    request = model_request(llm.draft_model, list_repo_files=list_repo_files)
    if isinstance(request, LocalCheckpoint):
        return (
            replace(llm, draft_model=_DRAFT_MOUNT_PATH),
            (_model_mount(request.path, _DRAFT_MOUNT_PATH),),
            None,
        )
    return replace(llm, draft_model=None), (), llm.draft_model


def _quantization_problems(
    quantization: QuantizationMethod, model: Path, metadata: GgufMetadata | None
) -> list[str]:
    """Report that llama.cpp takes its quantization from the checkpoint.

    A GGUF file's ``general.file_type`` *is* the engine's quantization and
    llama.cpp has no ``--quantization`` flag, so any configured value is
    rejected no matter what the checkpoint declares. The checkpoint's detected
    quantization is named in the problem so the user can see which quantization
    the run would actually serve.
    """
    detected = metadata.quantization if metadata is not None else None
    detail = f"the checkpoint declares {detected}" if detected else "the checkpoint declares none"
    return [
        f"llama.cpp serves the checkpoint at {model} with its own quantization ({detail}); "
        f"remove quantization={quantization.value!r}"
    ]


def _validate_numeric_range(dimension: Dimension, field: str) -> list[str]:
    """Report every declared value outside the range llama.cpp accepts.

    A ``Fixed`` or ``Categorical`` dimension is checked value by value, so the
    problem points at the exact entry the user wrote. An ``Interval`` is checked
    by both endpoints because the optimizer may draw any value in the range, and
    the problem names the interval rather than a single value. A parameter with
    no engine-imposed range, a non-numeric value, and a ``Default`` dimension
    (which the caller already skips) carry nothing to compare here, so they
    produce no problem.
    """
    bounds = _NUMERIC_RANGES.get(field)
    if bounds is None:
        return []
    minimum, maximum = bounds
    requirement = _range_requirement(field, minimum, maximum)
    problems = [
        f"{path}: {requirement}; got {value}"
        for path, value in declared_values(dimension)
        if _outside_range(value, minimum, maximum)
    ]
    spec = dimension.spec
    if isinstance(spec, Interval) and (
        _outside_range(spec.minimum, minimum, maximum)
        or _outside_range(spec.maximum, minimum, maximum)
    ):
        problems.append(
            f"space.{dimension.name}.interval: {requirement}; "
            f"the space allows {spec.minimum}..{spec.maximum}"
        )
    return problems


def _range_requirement(field: str, minimum: int | float, maximum: int | float | None) -> str:
    """Describe the inclusive range llama.cpp accepts for ``field``."""
    if maximum is None:
        return f"llama.cpp requires {field} >= {minimum}"
    return f"llama.cpp requires {field} in the range {minimum}..{maximum}"


def _outside_range(value: object, minimum: int | float, maximum: int | float | None) -> bool:
    """Whether ``value`` is a number outside ``minimum``..``maximum``.

    Booleans and non-numeric values are never treated as numbers, so an enum
    member or a boolean cannot be compared against a numeric bound.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value < minimum or (maximum is not None and value > maximum)


def _validate_gpu_layers_target(
    dimension: Dimension, field: str, target: HardwareTarget
) -> list[str]:
    """Report every GPU-layer value that needs GPU hardware on a CPU target.

    ``all`` and a positive exact count both ask llama.cpp to offload work to a
    GPU, so a CPU target can never serve them. ``auto`` may still resolve to no
    offload and ``0`` explicitly disables it, so both stay valid for a
    host-agnostic space. An ``Interval`` is rejected when any value it can draw
    is positive, because the optimizer may sample one; a range pinned to zero
    remains acceptable.
    """
    if field != "n_gpu_layers" or target is not HardwareTarget.CPU:
        return []
    problems: list[str] = []
    for path, value in declared_values(dimension):
        if value is GpuLayersMode.ALL or (
            isinstance(value, int) and not isinstance(value, bool) and value > 0
        ):
            problems.append(
                f"{path}: llama.cpp cannot offload GPU layers on the CPU target; "
                f"use {field}=auto or 0"
            )
    if isinstance(dimension.spec, Interval) and dimension.spec.maximum > 0:
        problems.append(
            f"space.{dimension.name}.interval: llama.cpp cannot offload GPU layers "
            f"up to {dimension.spec.maximum} on the CPU target; use {field}=auto or 0"
        )
    return problems


def _validate_quantized_cache_attention(
    space: ExperimentSpace, applicable: frozenset[str]
) -> list[str]:
    """Report a space that can pair a quantized KV cache with flash attention off.

    llama.cpp refuses to create a quantized V cache unless flash attention is
    enabled (``quantized V cache requires flash_attn to be enabled``), so a
    space that can sample both would fail at server startup instead of producing
    a measurement. The rule covers the target model's cache and, when the run
    names a draft model, the draft model's own cache: both are built by the same
    flash-attention-gated kernels. Only declared values create the conflict: a
    ``Default`` dimension defers to the engine, whose own defaults are the float
    cache and ``auto`` attention, and a space missing either dimension is left
    to the per-dimension checks. A cache dimension the backend does not forward
    for this run — the draft cache without a configured draft — is not consulted,
    because it will never be sampled.
    """
    dimensions = {dimension.name: dimension for dimension in space.dimensions}
    attention = dimensions.get("attention_backend")
    if attention is None:
        return []
    disabled = [path for path, value in declared_values(attention) if value is AttentionBackend.OFF]
    if not disabled:
        return []
    problems: list[str] = []
    for field, label in (
        ("kv_cache_dtype", "KV cache"),
        ("draft_kv_cache_dtype", "draft KV cache"),
    ):
        cache = dimensions.get(field)
        if cache is None or field not in applicable:
            continue
        quantized = [
            path
            for path, value in declared_values(cache)
            if isinstance(value, KVCacheDType) and value in _QUANTIZED_CACHE_DTYPES
        ]
        if not quantized:
            continue
        problems.append(
            f"{disabled[0]}: llama.cpp requires flash attention for a quantized {label}, "
            f"but this space can sample {', '.join(quantized)}; remove "
            f"attention_backend=off or restrict {field} to float16, bfloat16, or float32"
        )
    return problems


def _gpu_layers_block_count_problems(
    n_gpu_layers: int | GpuLayersMode | None, model: Path, metadata: GgufMetadata | None
) -> list[str]:
    """Report an exact GPU-layer count larger than the checkpoint has blocks.

    ``all`` offloads every layer unambiguously and ``auto`` defers the choice
    to llama.cpp, so neither is bounded here. An exact integer above the
    architecture's block count is almost always a typo that would silently
    duplicate full offload; the problem points at ``all`` instead. Unreadable
    metadata or a checkpoint that declares no block count fails open, leaving
    the engine as the final authority.
    """
    if n_gpu_layers is None or isinstance(n_gpu_layers, GpuLayersMode):
        return []
    block_count = metadata.block_count if metadata is not None else None
    if block_count is None or n_gpu_layers <= block_count:
        return []
    return [
        f"n_gpu_layers={n_gpu_layers} exceeds the {block_count} layers in {model}; "
        "use n_gpu_layers=all to offload every layer"
    ]


def _is_dense_checkpoint(metadata: GgufMetadata | None) -> bool:
    """Whether the GGUF header confidently declares no experts.

    A readable header with no expert count, or a count of zero, is confidently
    dense. Unreadable metadata is unknown rather than dense, so a checkpoint
    the reader could not see keeps the declared MoE dimensions.
    """
    return metadata is not None and metadata.expert_count in (None, 0)


def _is_swa_free_checkpoint(metadata: GgufMetadata | None) -> bool:
    """Whether the GGUF header confidently declares no sliding-window attention.

    A readable header with no window, or a window of zero, is confidently
    full-attention: llama.cpp reads the key as optional, so a checkpoint that
    omits it has no sliding window. Unreadable metadata is unknown rather than
    full-attention, so a checkpoint the reader could not see keeps the declared
    dimension.
    """
    return metadata is not None and metadata.sliding_window in (None, 0)


def _cpu_valid_gpu_layers(dimension: Dimension) -> bool:
    """Whether every value ``dimension`` can select is valid on a CPU target.

    ``auto`` may resolve to no offload and an exact ``0`` disables it, so both
    are CPU-valid. A ``Default`` dimension declares no value and defers to the
    engine, so it is not judged here, and an ``Interval`` is CPU-valid only when
    its maximum is zero because the optimizer could draw any value in the range.
    """
    spec = dimension.spec
    if isinstance(spec, Default):
        return False
    if isinstance(spec, Interval):
        return spec.maximum <= 0
    values = [value for _, value in declared_values(dimension)]
    return bool(values) and all(
        value is GpuLayersMode.AUTO
        or (isinstance(value, int) and not isinstance(value, bool) and value == 0)
        for value in values
    )


def _prune_cpu_target_dimensions(
    space: ExperimentSpace, target: HardwareTarget
) -> tuple[ExperimentSpace, tuple[str, ...]]:
    """Drop the llama.cpp dimensions a CPU target makes meaningless.

    A CPU run has no device to offload from, so ``n_cpu_moe``, ``n_cpu_ffn``,
    ``kv_offload``, ``op_offload``, and the draft model's ``n_cpu_moe_draft``
    and ``n_gpu_layers_draft`` cannot change anything and are dropped
    unconditionally. ``n_gpu_layers`` is dropped only when every value it can
    select is already CPU-valid, because ``all`` and positive counts are
    rejected by the pre-model validation rather than silently reinterpreted on
    a CPU host. A non-CPU target keeps every dimension, and the remaining
    dimensions keep their canonical order.
    """
    if target is not HardwareTarget.CPU:
        return space, ()
    effective, notes = prune_dimensions(space, {_CPU_MOE_PARAMETER}, reason=_CPU_TARGET_MOE_REASON)
    effective, dropped = prune_dimensions(effective, {"n_cpu_ffn"}, reason=_CPU_TARGET_FFN_REASON)
    notes += dropped
    effective, dropped = prune_dimensions(
        effective, {"kv_offload", "op_offload"}, reason=_CPU_TARGET_DEVICE_OFFLOAD_REASON
    )
    notes += dropped
    effective, dropped = prune_dimensions(
        effective, {"n_cpu_moe_draft"}, reason=_CPU_TARGET_DRAFT_MOE_REASON
    )
    notes += dropped
    effective, dropped = prune_dimensions(
        effective, {"n_gpu_layers_draft"}, reason=_CPU_TARGET_DRAFT_GPU_LAYERS_REASON
    )
    notes += dropped
    layers = next(
        (dimension for dimension in effective.dimensions if dimension.name == "n_gpu_layers"), None
    )
    if layers is not None and _cpu_valid_gpu_layers(layers):
        effective, layer_notes = prune_dimensions(
            effective, {"n_gpu_layers"}, reason=_CPU_TARGET_GPU_LAYERS_REASON
        )
        notes += layer_notes
    return effective, notes


def _layer_count_problems(
    dimension: Dimension, block_count: int, model: Path, hint: str
) -> list[str]:
    """Report every layer count ``dimension`` declares above ``block_count``.

    An exact integer above the checkpoint's block count is almost always a typo
    that would silently duplicate full offload, so the problem points at the
    mode that expresses full offload unambiguously. An ``Interval`` is checked
    by its maximum because the optimizer may draw any value in the range. The
    mode members (``auto``/``all``) declare no count to compare and are never
    bounded here.
    """
    problems: list[str] = []
    for path, value in declared_values(dimension):
        if isinstance(value, int) and not isinstance(value, bool) and value > block_count:
            problems.append(
                f"{path}: {dimension.name}={value} exceeds the {block_count} layers in "
                f"{model}; {hint}"
            )
    if isinstance(dimension.spec, Interval) and dimension.spec.maximum > block_count:
        problems.append(
            f"space.{dimension.name}.interval: {dimension.name} up to "
            f"{dimension.spec.maximum} exceeds the {block_count} layers in {model}; {hint}"
        )
    return problems


def _block_count_problems(
    space: ExperimentSpace,
    model: Path,
    metadata: GgufMetadata | None,
    hints: Mapping[str, str],
) -> list[str]:
    """Report declared layer counts the checkpoint has no blocks for.

    Every parameter named in ``hints`` counts layers, so an exact value or an
    interval maximum above the architecture's ``<arch>.block_count`` is
    reported with that parameter's fix as the hint. Unreadable metadata or a
    checkpoint that declares no block count fails open, leaving the engine as
    the final authority rather than rejecting on metadata this library could
    not see. Dimensions the assessment dropped are not consulted, because the
    run will not sample them.
    """
    block_count = metadata.block_count if metadata is not None else None
    if block_count is None:
        return []
    problems: list[str] = []
    for dimension in space.dimensions:
        hint = hints.get(dimension.name)
        if hint is not None:
            problems.extend(_layer_count_problems(dimension, block_count, model, hint))
    return problems


@dataclass(frozen=True, slots=True)
class LlamaCppBackend:
    """The llama.cpp implementation of the inference backend seam.

    ``extra_args`` is the single escape hatch for ``llama-server`` flags the
    engine-neutral config does not model (``--n-gpu-layers``, ``--split-mode``,
    and the engine-only speculative variants, for example). The sequence is
    appended verbatim
    to the assembled command, after the mapped parameters, so a caller can
    experiment without widening :class:`LLMConfig`; values passed this way are
    the caller's responsibility.

    ``list_repo_files`` is the repository-listing seam model resolution uses to
    pick the GGUF file of a Hugging Face repository. It defaults to
    :func:`huggingface_hub.list_repo_files` and is injectable so a unit test
    needs no network.
    """

    extra_args: tuple[str, ...] = ()
    list_repo_files: _RepoFileLister = list_hub_files

    def applicable_parameters(self, llm: LLMConfig) -> frozenset[str]:
        """Exactly the tunables this backend forwards for ``llm``.

        A tunable is applicable when its field has a ``llama-server`` target in
        :data:`PARAMETER_TARGETS`: that covers the context, slot, batch, cache,
        attention, prefix-caching, GPU-layer, and CPU-resident expert-layer
        dimensions. ``dtype``, the vLLM scheduling caps, and
        ``tensor_parallel_size`` have no engine flag, so a definition may omit
        them and the loader does not require them. The speculative dimensions
        are added exactly when ``llm`` configures a speculative method or draft
        model, and the draft-only dimensions exactly when it names a draft
        model, so a non-speculative space stays unchanged. Both gates add back
        only the members this backend forwards.
        """
        forwarded = frozenset(
            name for name, field in TUNABLE_PARAMETERS.items() if field in PARAMETER_TARGETS
        )
        applicable = forwarded - SPECULATIVE_PARAMETERS - DRAFT_MODEL_PARAMETERS
        if speculative_configured(llm):
            applicable |= SPECULATIVE_PARAMETERS & forwarded
        if draft_configured(llm):
            applicable |= DRAFT_MODEL_PARAMETERS & forwarded
        return applicable

    def validate(
        self,
        space: ExperimentSpace,
        llm: LLMConfig,
        hardware: DetectedHardware | None = None,
    ) -> Sequence[str]:
        """Report every way llama.cpp cannot support ``space`` for ``llm``.

        The space is assumed to have passed engine-agnostic validation. Every
        dimension's parameter must be one the backend forwards for this run,
        and every fixed or categorical enum value must have a spelling in
        :data:`VALUE_MAPS`. A ``Default`` dimension defers its value to the
        engine and carries nothing to support-check. Problems are returned as
        strings so the caller can aggregate them; an empty sequence means
        llama.cpp supports every dimension.

        ``hardware`` is accepted for the backend seam; when given, it also
        rejects a GPU-layer value the resolved accelerator class cannot serve,
        because offloading ``all`` or a positive exact count needs a GPU while
        ``auto`` and ``0`` do not. It also rejects a space that can sample a
        quantized target or draft KV cache together with flash attention off,
        because llama.cpp refuses to create a quantized V cache without flash
        attention, and every declared numeric value outside the range the engine
        accepts for ``poll``, ``prio``, ``spec_draft_p_split``, and
        ``cache_ram``. The other modelled values are not accelerator-specific and
        are either unbounded or clamped by llama.cpp at load time, so nothing
        else is narrowed before a sweep starts.
        """
        problems: list[str] = []
        applicable = self.applicable_parameters(llm)
        for dimension in space.dimensions:
            field = TUNABLE_PARAMETERS[dimension.name]
            if field not in PARAMETER_TARGETS or dimension.name not in applicable:
                problems.append(f"space.{dimension.name}: llama.cpp does not forward {field!r}")
                continue
            if isinstance(dimension.spec, Default):
                continue
            problems.extend(unsupported_values(dimension, field, VALUE_MAPS, engine="llama.cpp"))
            problems.extend(_validate_numeric_range(dimension, field))
            if hardware is not None:
                problems.extend(_validate_gpu_layers_target(dimension, field, hardware.target))
        problems.extend(_validate_quantized_cache_attention(space, applicable))
        return problems

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> Sequence[str]:
        """Report every way the checkpoints cannot serve ``experiment``.

        A non-GGUF target checkpoint is rejected, because llama.cpp can only
        serve GGUF. A configured ``quantization`` is rejected too: llama.cpp
        reads the quantization from the file's metadata rather than from a
        flag, and the problem names the quantization the checkpoint actually
        declares. An exact ``n_gpu_layers`` count above the checkpoint's block
        count is rejected and pointed at ``all``. A resolved draft checkpoint
        is held to the same file-format rule, since llama.cpp loads it through
        the same GGUF reader. A GGUF whose header cannot be parsed is accepted
        (fail-open), leaving the engine as the final authority rather than
        rejecting on metadata this library could not see.
        """
        quantization = experiment.llm.quantization
        problems: list[str] = []
        try:
            metadata = read_gguf_metadata(model)
        except ExperimentError as exc:
            if quantization is not None:
                problems.extend(_quantization_problems(quantization, model, None))
            problems.append(str(exc))
        else:
            if quantization is not None:
                problems.extend(_quantization_problems(quantization, model, metadata))
            problems.extend(
                _gpu_layers_block_count_problems(experiment.llm.n_gpu_layers, model, metadata)
            )
        draft = experiment.llm.resolved_draft_path
        if draft is not None:
            try:
                read_gguf_metadata(draft)
            except ExperimentError as exc:
                problems.append(f"draft model: {exc}")
        return problems

    def assess_space(
        self,
        space: ExperimentSpace,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        model: Path,
    ) -> SpaceAssessment:
        """Derive the effective llama.cpp space for the resolved checkpoint.

        Four rules apply. A checkpoint whose GGUF header confidently declares
        no experts loses ``n_cpu_moe``, because keeping expert layers in system
        RAM cannot change a dense model; a header that cannot be parsed is
        unknown rather than dense, so the dimension is kept and the engine stays
        the final authority. A checkpoint that confidently uses full attention
        loses ``swa_full``, because a full-size sliding-window cache changes
        nothing there. A CPU target then loses ``n_cpu_moe``, ``n_cpu_ffn``,
        ``kv_offload``, ``op_offload``, and the draft model's
        ``n_cpu_moe_draft`` and ``n_gpu_layers_draft`` — no weight ever leaves
        system RAM and no operation runs on a device there — and
        ``n_gpu_layers`` when every value it can select already disables
        offload. Finally, the retained dimensions are bounded by the
        checkpoint's block count: ``n_gpu_layers`` and ``n_cpu_moe`` against
        the target checkpoint and ``n_gpu_layers_draft`` and
        ``n_cpu_moe_draft`` against the draft checkpoint, so an exact count
        above the relevant block count is reported with the ``all`` mode as the
        unambiguous alternative. ``n_cpu_ffn`` is bounded the same way but its
        hint asks for a smaller count, because it has no ``all`` mode. A
        non-GGUF path is already reported by :meth:`validate_model`, so the
        assessment leaves the space untouched rather than raising a second time.
        """
        try:
            metadata = read_gguf_metadata(model)
        except ExperimentError:
            metadata = None
        draft_path = experiment.llm.resolved_draft_path
        try:
            draft_metadata = read_gguf_metadata(draft_path) if draft_path is not None else None
        except ExperimentError:
            draft_metadata = None
        effective = space
        notes: list[str] = []
        if _is_dense_checkpoint(metadata):
            effective, dropped = prune_dimensions(
                effective, {_CPU_MOE_PARAMETER}, reason=_DENSE_CHECKPOINT_REASON
            )
            notes.extend(dropped)
        if _is_swa_free_checkpoint(metadata):
            effective, dropped = prune_dimensions(
                effective, {_SWA_PARAMETER}, reason=_SWA_FREE_CHECKPOINT_REASON
            )
            notes.extend(dropped)
        effective, cpu_notes = _prune_cpu_target_dimensions(effective, hardware.target)
        notes.extend(cpu_notes)
        problems = _block_count_problems(effective, model, metadata, _MAIN_BLOCK_COUNT_HINTS)
        if draft_path is not None:
            problems.extend(
                _block_count_problems(
                    effective, draft_path, draft_metadata, _DRAFT_BLOCK_COUNT_HINTS
                )
            )
        return SpaceAssessment(space=effective, notes=tuple(notes), problems=tuple(problems))

    def model_request(self, model: str) -> ModelRequest:
        """Decide whether ``model`` is a local GGUF file or a Hub repository.

        The composition root calls this before downloading, so llama.cpp's
        model-argument rules — existence means a local file, a repository id
        may carry a ``:QUANT`` selector, and a multi-quant repository requires
        one — stay here rather than in the CLI.
        """
        return model_request(model, list_repo_files=self.list_repo_files)

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
    ) -> ServerSpec:
        """Turn an experiment and the detected hardware into a server run.

        The accelerator class resolves from the container config's override or,
        failing that, the detected hardware, and selects the image and device
        mapping. The checkpoint is mounted read-only at a fixed container path
        (or left for the server to fetch itself when no path was resolved), the
        server binds on every interface on the published port, the served name
        is pinned to the configured model string, and the mapped tunables and
        ``extra_args`` follow. A configured draft is mounted beside it at its
        own fixed path and pointed at through the speculative arguments, or left
        for the server to fetch through its draft-repository flag. The slot count
        is cut beforehand to what the context can serve, so the benchmark's
        request length always fits a slot's share of ``--ctx-size``.

        The returned :class:`ServerSpec` describes the ``/health`` endpoint and
        the startup timeout the host client uses. llama.cpp exposes no
        cache-reset endpoint, so ``reset_paths`` is empty and the client skips
        resetting. Planning touches no host filesystem: unlike vLLM, llama.cpp
        needs no cache directory to exist before the run.
        """
        container = experiment.container
        target = container.target or hardware.target
        benchmark = experiment.benchmark
        model_arguments, model_mounts = _model_arguments(
            benchmark.model, benchmark.resolved_path, self.list_repo_files
        )
        llm, draft_mounts, draft_repository = _draft_arguments(experiment.llm, self.list_repo_files)
        command = (
            *model_arguments,
            "--alias",
            benchmark.model,
            "--host",
            _SERVER_BIND_HOST,
            "--port",
            str(_SERVER_PORT),
            *serve_arguments(
                _planned_llm(llm, benchmark),
                extra_args=self.extra_args,
                draft_repository=draft_repository,
            ),
        )
        return ServerSpec(
            container=ContainerConfig(
                image=container.image or self.image_for(target),
                command=command,
                env=container.env,
                volumes=(*container.volumes, *model_mounts, *draft_mounts),
                devices=_DEVICE_MAPPINGS[target],
                ports={str(_SERVER_PORT): None},
                mem_limit=container.mem_limit,
                shm_size=container.shm_size,
                selinux_opts=container.selinux_opts,
                pull_policy=container.pull_policy,
            ),
            health_path=_HEALTH_PATH,
            health_timeout=benchmark.startup_timeout,
            reset_paths=(),
            tokenize_path=_TOKENIZE_PATH,
            tokenize_prompt_field="content",
        )

    def image_for(self, target: HardwareTarget) -> str:
        """Return the official llama.cpp server image for ``target``.

        This is the public image-resolution seam: the image a run is planned
        with is ``container.image`` when the caller set one, and this method's
        result otherwise.
        """
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
        is enriched with the prefill and decode rates its response timings
        imply, and a run without usable timings keeps ``None`` for both. An
        outcome with no successful run raises :class:`ExperimentError` rather
        than letting :class:`ExperimentResult` reject it with a bare
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
