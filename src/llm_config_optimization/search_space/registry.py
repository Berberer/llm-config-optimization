"""Engine-neutral registry of the parameters an experiment sweep may vary.

The search space has a vocabulary of its own: a YAML definition names
parameters, and the registry says which of those names may be swept and which
:class:`~llm_config_optimization.inference.types.LLMConfig` field each one fills. The
indirection is deliberate — the file's vocabulary can be renamed without
touching the backend, and the backend keeps owning how a field reaches its
engine — while the allow-list stays strict and code-only, so what a sweep may
vary is a single, reviewable edit.

Every other ``LLMConfig`` field is a non-tunable: ``seed`` and
``gpu_memory_utilization`` are managed by the code driving a sweep, and
``quantization`` describes how the checkpoint was produced rather than a
runtime choice, so declaring one is an error rather than a silent no-op. A name
that is not an ``LLMConfig`` field at all is an unknown parameter.

The expected kind of a tunable is not stored here. It is read from the
``LLMConfig`` field's own type annotation by :func:`expected_kind`, so the
registry cannot drift from the config: an enum field resolves to its
engine-neutral enum class, an integer field to :class:`int`, a boolean field to
:class:`bool`, and a field that admits more than one of those (an int-or-enum
union such as a GPU-layer count with ``auto``/``all`` modes) to a
:class:`UnionKind` carrying each branch.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType, NoneType
from typing import get_args, get_type_hints

from ..inference.types import LLMConfig

type AtomKind = type[StrEnum] | type[int] | type[float] | type[bool]
"""A single engine-neutral kind a tunable's ``LLMConfig`` field may expect.

A categorical parameter resolves to its engine-neutral enum class, a numeric
parameter to :class:`int` or :class:`float`, and a boolean parameter to
:class:`bool`.
"""


@dataclass(frozen=True, slots=True)
class UnionKind:
    """The two or more engine-neutral kinds a union-annotated field admits.

    A field such as ``int | GpuLayersMode | None`` has no single kind: a value
    may be an exact integer or one of the enum's mode names, and the loader
    validates a declared value against either branch. The order is the order of
    the annotation. An ``interval`` spec constrains a numeric branch only, so a
    mode name can never masquerade as a numeric bound. A float branch accepts
    integer and fractional bounds and stores them as floats, while an integer
    branch rejects a fractional bound.
    """

    kinds: tuple[AtomKind, ...]

    def __post_init__(self) -> None:
        """Store the branches as a tuple so the kind is immutable and hashable."""
        object.__setattr__(self, "kinds", tuple(self.kinds))


type ParameterKind = AtomKind | UnionKind
"""The engine-neutral kind a tunable's ``LLMConfig`` field expects.

A categorical parameter resolves to its engine-neutral enum class, a numeric
parameter to :class:`int` or :class:`float`, a boolean parameter to
:class:`bool`, and a union field to a :class:`UnionKind`. The validator uses
this to reject a spec that cannot describe the field, such as an
:class:`~llm_config_optimization.search_space.types.Interval` declared for an
enum, a boolean, or a union that carries no numeric branch.
"""


TUNABLE_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "max_model_len": "max_model_len",
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "max_num_active_seqs": "max_num_active_seqs",
        "max_num_scheduled_tokens": "max_num_scheduled_tokens",
        "ubatch_size": "ubatch_size",
        "dtype": "dtype",
        "kv_cache_dtype": "kv_cache_dtype",
        "attention_backend": "attention_backend",
        "n_gpu_layers": "n_gpu_layers",
        "n_cpu_moe": "n_cpu_moe",
        "n_cpu_ffn": "n_cpu_ffn",
        "kv_offload": "kv_offload",
        "op_offload": "op_offload",
        "host_buffer": "host_buffer",
        "repack": "repack",
        "swa_full": "swa_full",
        "load_mode": "load_mode",
        "prefix_caching": "prefix_caching",
        "cache_ram": "cache_ram",
        "ctx_checkpoints": "ctx_checkpoints",
        "checkpoint_min_step": "checkpoint_min_step",
        "kv_unified": "kv_unified",
        "continuous_batching": "continuous_batching",
        "cache_idle_slots": "cache_idle_slots",
        "tensor_parallel_size": "tensor_parallel_size",
        "expert_parallel": "expert_parallel",
        "all2all_backend": "all2all_backend",
        "speculative_mode": "speculative_mode",
        "num_speculative_tokens": "num_speculative_tokens",
        "spec_draft_n_min": "spec_draft_n_min",
        "spec_draft_p_split": "spec_draft_p_split",
        "spec_draft_sampling": "spec_draft_sampling",
        "n_gpu_layers_draft": "n_gpu_layers_draft",
        "n_cpu_moe_draft": "n_cpu_moe_draft",
        "draft_kv_cache_dtype": "draft_kv_cache_dtype",
        "threads": "threads",
        "threads_batch": "threads_batch",
        "poll": "poll",
        "prio": "prio",
    }
)
"""Which serving parameters a sweep may vary, and the ``LLMConfig`` field each fills.

The insertion order is canonical: the parser emits dimensions in this order, so
a space is deterministic regardless of the YAML key order. The mapping is a
strict allow-list, so a parameter absent from it is either a known non-tunable
(an ``LLMConfig`` field such as ``seed``) or unknown entirely. The speculative
dimensions are gated by :data:`SPECULATIVE_PARAMETERS`, and the draft-only ones
by :data:`DRAFT_MODEL_PARAMETERS`.
"""

SPECULATIVE_PARAMETERS: frozenset[str] = frozenset({"speculative_mode", "num_speculative_tokens"})
"""The tunables that are applicable only under a configured speculative setup.

A definition declares these exactly when the run fixes a speculative method or
draft model; otherwise no value would ever reach the engine, so a definition
that declares them is rejected by the backend's support check.
"""

DRAFT_MODEL_PARAMETERS: frozenset[str] = frozenset(
    {
        "spec_draft_n_min",
        "spec_draft_p_split",
        "spec_draft_sampling",
        "n_gpu_layers_draft",
        "n_cpu_moe_draft",
        "draft_kv_cache_dtype",
    }
)
"""The tunables that are applicable only under a configured draft model.

These shape the draft model's own sampling, depth, offload placement, and KV
cache, so they cannot affect a run that names no draft. A backend that forwards
a member adds it to its applicable set exactly when :func:`draft_configured`;
one that does not forward it never requires it.
"""


def speculative_configured(llm: LLMConfig) -> bool:
    """Whether ``llm`` configures a speculative-decoding setup.

    A setup exists as soon as a method or a draft model is fixed for the run;
    a bare draft model implies the generic draft method, so either field alone
    gates the speculative dimensions in.
    """
    return llm.speculative_method is not None or llm.draft_model is not None


def draft_configured(llm: LLMConfig) -> bool:
    """Whether ``llm`` names a separate draft model.

    Draft-only dimensions are gated on the draft model rather than on the
    speculative method, because methods such as n-gram and the target model's
    own MTP heads do not use a separate checkpoint.
    """
    return llm.draft_model is not None


def expected_kind(field: str) -> ParameterKind:
    """Resolve the engine-neutral kind of an ``LLMConfig`` field.

    The kind is read from the field's type annotation rather than duplicated in
    :data:`TUNABLE_PARAMETERS`, so the registry cannot drift from the config.
    Optional annotations (``DType | None``) resolve to their single non-``None``
    member, which is an engine-neutral enum class for categorical fields,
    :class:`int` for numeric ones, and :class:`bool` for boolean ones. An
    annotation with several non-``None`` members (``int | GpuLayersMode |
    None``) resolves to a :class:`UnionKind` in annotation order.

    Raises :class:`KeyError` for a name that is not an ``LLMConfig`` field.
    """
    annotation = get_type_hints(LLMConfig)[field]
    arguments = tuple(argument for argument in get_args(annotation) if argument is not NoneType)
    if len(arguments) > 1:
        return UnionKind(arguments)
    if arguments:
        return arguments[0]
    return annotation
