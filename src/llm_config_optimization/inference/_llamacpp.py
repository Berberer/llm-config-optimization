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
the quantization it declares is read straight from its header.
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
    SPECULATIVE_PARAMETERS,
    TUNABLE_PARAMETERS,
    speculative_configured,
)
from ..search_space.types import Default, ExperimentSpace
from ._benchmark import _RESPONSES_KEY
from ._tunables import Gate, translate_value, unsupported_values
from .types import (
    AttentionBackend,
    BenchmarkConfig,
    BenchmarkOutcome,
    DetectedHardware,
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

PARAMETER_TARGETS: Mapping[str, tuple[str, ...] | Gate] = MappingProxyType(
    {
        "kv_cache_dtype": ("--cache-type-k", "--cache-type-v"),
        "max_model_len": ("--ctx-size",),
        "max_num_seqs": ("--parallel",),
        "max_num_batched_tokens": ("--batch-size", "--ubatch-size"),
        "attention_backend": ("--flash-attn",),
        "speculative_method": ("--spec-type",),
        "draft_model": ("--spec-draft-model",),
        "speculative_mode": Gate(),
        "num_speculative_tokens": ("--spec-draft-n-max",),
    }
)
"""Where each applicable :class:`LLMConfig` field is sent on ``llama-server``.

Unlike vLLM's flags, llama.cpp's names cannot be derived from the field names
(``max_model_len`` becomes ``--ctx-size``), so each one is spelled out. A field
maps to a tuple of flags because the engine either splits one value across
symmetric options — ``kv_cache_dtype`` configures both the K and the V cache
type, which must agree for the cache the user asked for — or applies it at two
levels, as ``max_num_batched_tokens`` does for the logical ``--batch-size`` and
the physical ``--ubatch-size``. The latter defaults to 512, which throttles a
prompt-heavy workload unless the sampled batch raises it. The speculative
method, draft model, and depth each map to one ``--spec-*`` flag, while
``speculative_mode`` is a :class:`Gate`: it is recognised as applicable but
never forwarded, because it only decides whether the other three are emitted.
An empty, absent, or gate entry means the field is never serialised.

The insertion order is the canonical argument order: it follows the fields in
:class:`LLMConfig`, so a run's command is deterministic.
"""

_SPECULATIVE_FIELDS = frozenset({"speculative_method", "draft_model", "num_speculative_tokens"})
"""The fields suppressed when the speculative mode is disabled.

``speculative_mode`` is absent because it is the gate rather than a forwarded
field: a disabled mode drops these fields so the engine starts with no
speculative flags and the draft model's memory is genuinely freed.
"""

VALUE_MAPS: Mapping[str, Mapping[object, str]] = MappingProxyType(
    {
        "kv_cache_dtype": {
            KVCacheDType.FLOAT16: "f16",
            KVCacheDType.BFLOAT16: "bf16",
        },
        "attention_backend": {
            AttentionBackend.FLASH_ATTENTION: "on",
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
    }
)
"""How llama.cpp spells each engine-neutral categorical value it can serve.

The enums in :mod:`llm_config_optimization.inference.types` carry no engine
vocabulary; this table is the only place their llama.cpp spellings live. A
member missing from its field's map is unsupported and is rejected rather than
forwarded, while the engine's own default is reached by declaring the parameter
as ``default: true`` instead of naming a value.

``kv_cache_dtype`` is curated to the portable element formats llama.cpp accepts:
``auto`` and the FP8 members have no equivalent here. For ``attention_backend``
only the flash-attention implementation is modelled, mapped to the engine's
``on`` setting; ``flash_infer`` and ``xformers`` are vLLM-only backends. The six
speculative methods map onto llama.cpp's ``--spec-type`` spellings; the
engine-only n-gram cache, map, and mod variants stay behind ``extra_args``.
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

    Raises :class:`ExperimentError` for an enum value the engine has no spelling
    for.
    """
    suppressed = (
        _SPECULATIVE_FIELDS if config.speculative_mode is SpeculativeMode.DISABLED else frozenset()
    )
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
    """The quantization a GGUF checkpoint declares in its metadata.

    ``quantization`` is the canonical short tag for the checkpoint's
    ``general.file_type`` (``"Q4_K_M"``, ``"BF16"``, and so on), or ``None``
    when the checkpoint declares no file type or one this library does not
    name. ``general.file_type`` is the value llama.cpp itself reports as the
    model's quantization, so validation speaks the engine's own vocabulary.
    """

    quantization: str | None


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

    Only ``general.file_type`` is ever read, so every other entry is skipped.
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
    """Read the quantization a GGUF checkpoint declares, if it can be read.

    A path that is not a regular file, or a file that does not start with the
    GGUF magic, is rejected with :class:`ExperimentError`: llama.cpp cannot
    serve it, so failing before an image is pulled is the clear outcome. A file
    that is GGUF but whose header cannot be parsed returns ``None`` (fail
    open), leaving the engine as the final authority rather than rejecting on
    metadata this library could not see. An unknown container version or
    ``general.file_type`` value is likewise reported as no quantization, never
    as a guess.
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
            for _ in range(metadata_count):
                key = _read_metadata_string(handle)
                value_type = _read_uint32(handle)
                scalar_format = _GGUF_SCALAR_FORMATS.get(value_type)
                if key == _GGUF_FILE_TYPE_KEY and value_type != _GGUF_BOOL_TYPE and scalar_format:
                    value = _read_scalar(handle, scalar_format)
                    if isinstance(value, int):
                        file_type = value
                else:
                    _skip_metadata_value(handle, value_type)
    except ExperimentError:
        raise
    except OSError, _UnreadableGguf, UnicodeDecodeError:
        return None
    quantization = _GGUF_FILE_TYPE_LABELS.get(file_type) if file_type is not None else None
    return GgufMetadata(quantization=quantization)


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

        llama.cpp can vary ``max_model_len``, ``max_num_seqs``,
        ``max_num_batched_tokens``, ``kv_cache_dtype``, and
        ``attention_backend``; ``dtype`` and ``tensor_parallel_size`` have no
        engine flag, so a definition may omit them and the loader does not
        require them. The two speculative dimensions are added exactly when
        ``llm`` configures a speculative method or draft model, so a
        non-speculative space stays unchanged.
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
        """Report every way llama.cpp cannot support ``space`` for ``llm``.

        The space is assumed to have passed engine-agnostic validation. Every
        dimension's parameter must be one the backend forwards for this run,
        and every fixed or categorical enum value must have a spelling in
        :data:`VALUE_MAPS`. A ``Default`` dimension defers its value to the
        engine and carries nothing to support-check. Problems are returned as
        strings so the caller can aggregate them; an empty sequence means
        llama.cpp supports every dimension.

        ``hardware`` is accepted for the backend seam but adds no checks today:
        the modelled values are not accelerator-specific, and the numeric
        values the engine bounds are clamped by llama.cpp at load time rather
        than rejected, so there is nothing to narrow before a sweep starts.
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
        return problems

    def validate_model(self, experiment: ExperimentConfig, model: Path) -> Sequence[str]:
        """Report every way the checkpoints cannot serve ``experiment``.

        A non-GGUF target checkpoint is rejected, because llama.cpp can only
        serve GGUF. A configured ``quantization`` is rejected too: llama.cpp
        reads the quantization from the file's metadata rather than from a
        flag, and the problem names the quantization the checkpoint actually
        declares. A resolved draft checkpoint is held to the same file-format
        rule, since llama.cpp loads it through the same GGUF reader. A GGUF
        whose header cannot be parsed is accepted (fail-open), leaving the
        engine as the final authority rather than rejecting on metadata this
        library could not see.
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
        draft = experiment.llm.resolved_draft_path
        if draft is not None:
            try:
                read_gguf_metadata(draft)
            except ExperimentError as exc:
                problems.append(f"draft model: {exc}")
        return problems

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
        is pinned to the configured model string, and the five mapped tunables
        and ``extra_args`` follow. A configured draft is mounted beside it at its
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
