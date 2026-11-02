"""Integration tests for the host's vendor memory probes.

The parser unit tests feed captured output to the pure functions, which pins the
parsing contract but not the format: only the installed vendor tool can say
whether what it prints today is still what the parsers read. These tests run the
real NVIDIA and AMD probes with their default runner and assert that a tool which
does report accelerator memory yields a device value, and that a tool which
accounts memory to processes yields a per-process mapping.

They are marked ``integration`` and deselected by default; run them with
``uv run pytest -m integration``. Each test locates its tool first and skips when
it is not installed, and also skips when the tool is installed but answers
nothing — no accelerator bound, no usable driver — because then there is no live
output to parse. The one thing that never skips is a reported answer the probe
fails to parse: that is the divergence these tests exist to catch.
"""

from collections.abc import Sequence

import pytest

from llm_config_optimization.inference.hardware import (
    _probe_amd_memory,
    _probe_nvidia_memory,
)
from tooling import require_tool, run

pytestmark = pytest.mark.integration

_NVIDIA_USED = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
_NVIDIA_PROCESSES = (
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
)
_ROCM_USED = ("rocm-smi", "--showmeminfo", "vram", "--csv")
_ROCM_PROCESSES = ("rocm-smi", "--showpids")


def _rows(command: Sequence[str], purpose: str) -> set[str]:
    """Return the non-empty output lines ``command`` prints, or skip.

    A failing or silent query means the host has nothing for the probe to
    observe, so there is no live answer to hold the parser to.
    """
    answer = run(command)
    if answer.returncode != 0:
        pytest.skip(f"{purpose} is not usable on this host: {answer.stderr.strip()}")
    rows = {line.strip() for line in answer.stdout.splitlines() if line.strip()}
    if not rows:
        pytest.skip(f"{purpose} reported no accelerator on this host")
    return rows


def _accounts_processes(output: str) -> bool:
    """Whether a vendor process query printed data rows rather than boilerplate.

    Both tools exit successfully and print prose when no process holds the
    accelerator — ``rocm-smi`` says ``No KFD PIDs currently running`` and
    ``nvidia-smi`` prints its own notice — so the presence of *output* is not
    evidence of a row. A row is what starts with a process id.
    """
    for line in output.splitlines():
        fields = line.replace(",", " ").split()
        if fields and fields[0].isdigit():
            return True
    return False


def _assert_plausible(used_bytes: int | None, processes: dict[int, int]) -> None:
    """Assert a parsed observation looks like memory rather than a placeholder."""
    assert used_bytes is not None
    assert used_bytes >= 0
    assert all(pid > 0 for pid in processes)
    assert all(used >= 0 for used in processes.values())


def test_nvidia_probe_parses_live_output() -> None:
    require_tool("nvidia-smi", purpose="the live NVIDIA memory probe")
    _rows(_NVIDIA_USED, "nvidia-smi memory query")
    # An idle GPU legitimately accounts memory to no processes, so only a
    # process query that does report rows is held to a non-empty mapping.
    process_rows = run(_NVIDIA_PROCESSES)
    memory = _probe_nvidia_memory()

    _assert_plausible(memory.used_bytes, dict(memory.processes))
    if _accounts_processes(process_rows.stdout):
        assert memory.processes


def test_rocm_probe_parses_live_output() -> None:
    require_tool("rocm-smi", purpose="the live AMD memory probe")
    _rows(_ROCM_USED, "rocm-smi memory query")
    process_rows = run(_ROCM_PROCESSES)
    memory = _probe_amd_memory()

    _assert_plausible(memory.used_bytes, dict(memory.processes))
    if _accounts_processes(process_rows.stdout):
        assert memory.processes
