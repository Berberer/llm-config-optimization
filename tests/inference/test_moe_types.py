"""Unit tests for the engine-neutral Mixture-of-Experts vocabulary.

Two string enums name MoE choices without borrowing either engine's spelling:
:class:`All2AllBackend` for vLLM's all-to-all transport and :class:`CpuMoeMode`
for llama.cpp's whole-model expert offload. A space declares these members and
the loader coerces the YAML string back to the member, so these tests pin the
accepted spellings and the round trip from value to member. They import from
the package rather than the private module, because the enums belong to the
shared inference surface. No engine and no daemon is involved.
"""

from enum import StrEnum

import pytest

from llm_config_optimization.inference import All2AllBackend, CpuMoeMode


def test_all2all_backend_exposes_only_the_portable_transports() -> None:
    """Faster transports need extra libraries, so they stay out of the enum."""
    assert [member.value for member in All2AllBackend] == ["allgather_reducescatter", "naive"]


def test_cpu_moe_mode_exposes_the_value_less_all_mode() -> None:
    assert [member.value for member in CpuMoeMode] == ["all"]


@pytest.mark.parametrize("enum_type", [All2AllBackend, CpuMoeMode])
def test_moe_enums_round_trip_through_their_string_value(enum_type: type[StrEnum]) -> None:
    assert issubclass(enum_type, str)
    for member in enum_type:
        assert enum_type(member.value) is member


def test_engine_only_transports_are_not_members() -> None:
    with pytest.raises(ValueError):
        All2AllBackend("deepep")
