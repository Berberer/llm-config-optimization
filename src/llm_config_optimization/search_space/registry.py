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
engine-neutral enum class and an integer field to :class:`int`.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType, NoneType
from typing import get_args, get_type_hints

from ..inference.types import LLMConfig

type ParameterKind = type[StrEnum] | type[int]
"""The engine-neutral kind a tunable's ``LLMConfig`` field expects.

A categorical parameter resolves to its engine-neutral enum class and a numeric
parameter to :class:`int`. The validator uses this to reject a spec that cannot
describe the field, such as an :class:`~llm_config_optimization.search_space.types.Interval`
declared for an enum field.
"""


TUNABLE_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "dtype": "dtype",
        "kv_cache_dtype": "kv_cache_dtype",
        "max_model_len": "max_model_len",
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "tensor_parallel_size": "tensor_parallel_size",
        "attention_backend": "attention_backend",
        "speculative_mode": "speculative_mode",
        "num_speculative_tokens": "num_speculative_tokens",
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
    member, which is an engine-neutral enum class for categorical fields and
    :class:`int` for numeric ones.

    Raises :class:`KeyError` for a name that is not an ``LLMConfig`` field.
    """
    annotation = get_type_hints(LLMConfig)[field]
    for argument in get_args(annotation):
        if argument is not NoneType:
            return argument
    return annotation
