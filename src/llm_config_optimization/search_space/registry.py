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

type AtomKind = type[StrEnum] | type[int] | type[bool]
"""A single engine-neutral kind a tunable's ``LLMConfig`` field may expect.

A categorical parameter resolves to its engine-neutral enum class, a numeric
parameter to :class:`int`, and a boolean parameter to :class:`bool`.
"""


@dataclass(frozen=True, slots=True)
class UnionKind:
    """The two or more engine-neutral kinds a union-annotated field admits.

    A field such as ``int | GpuLayersMode | None`` has no single kind: a value
    may be an exact integer or one of the enum's mode names, and the loader
    validates a declared value against either branch. The order is the order of
    the annotation. An ``interval`` spec constrains the integer branch only, so
    a mode name can never masquerade as a numeric bound.
    """

    kinds: tuple[AtomKind, ...]

    def __post_init__(self) -> None:
        """Store the branches as a tuple so the kind is immutable and hashable."""
        object.__setattr__(self, "kinds", tuple(self.kinds))


type ParameterKind = AtomKind | UnionKind
"""The engine-neutral kind a tunable's ``LLMConfig`` field expects.

A categorical parameter resolves to its engine-neutral enum class, a numeric
parameter to :class:`int`, a boolean parameter to :class:`bool`, and a union
field to a :class:`UnionKind`. The validator uses this to reject a spec that
cannot describe the field, such as an
:class:`~llm_config_optimization.search_space.types.Interval` declared for an
enum, a boolean, or a union that carries no integer branch.
"""


TUNABLE_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "dtype": "dtype",
        "kv_cache_dtype": "kv_cache_dtype",
        "max_model_len": "max_model_len",
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "max_num_active_seqs": "max_num_active_seqs",
        "max_num_scheduled_tokens": "max_num_scheduled_tokens",
        "tensor_parallel_size": "tensor_parallel_size",
        "attention_backend": "attention_backend",
        "speculative_mode": "speculative_mode",
        "num_speculative_tokens": "num_speculative_tokens",
        "prefix_caching": "prefix_caching",
        "n_gpu_layers": "n_gpu_layers",
    }
)
"""Which serving parameters a sweep may vary, and the ``LLMConfig`` field each fills.

The insertion order is canonical: the parser emits dimensions in this order, so
a space is deterministic regardless of the YAML key order. The mapping is a
strict allow-list, so a parameter absent from it is either a known non-tunable
(an ``LLMConfig`` field such as ``seed``) or unknown entirely. The two
speculative dimensions are gated: see :data:`SPECULATIVE_PARAMETERS`.
"""

SPECULATIVE_PARAMETERS: frozenset[str] = frozenset({"speculative_mode", "num_speculative_tokens"})
"""The tunables that are applicable only under a configured speculative setup.

A definition declares these exactly when the run fixes a speculative method or
draft model; otherwise no value would ever reach the engine, so a definition
that declares them is rejected by the backend's support check.
"""


def speculative_configured(llm: LLMConfig) -> bool:
    """Whether ``llm`` configures a speculative-decoding setup.

    A setup exists as soon as a method or a draft model is fixed for the run;
    a bare draft model implies the generic draft method, so either field alone
    gates the speculative dimensions in.
    """
    return llm.speculative_method is not None or llm.draft_model is not None


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
