"""Unit tests for the SELinux enforce-flag probe.

The probe's observable contract is a plain boolean: whether a given enforce-flag
file reports enforcing mode. These tests point it at a temporary file, so every
case is pinned on any host and the real host flag is never read.
"""

from pathlib import Path

from llm_optimization.runtime._selinux import selinux_enforcing


def test_flag_holding_one_reports_enforcing(tmp_path: Path) -> None:
    flag = tmp_path / "enforce"
    flag.write_text("1\n")

    assert selinux_enforcing(enforce_flag=flag) is True


def test_flag_holding_zero_reports_permissive(tmp_path: Path) -> None:
    flag = tmp_path / "enforce"
    flag.write_text("0\n")

    assert selinux_enforcing(enforce_flag=flag) is False


def test_absent_flag_reports_not_enforcing(tmp_path: Path) -> None:
    assert selinux_enforcing(enforce_flag=tmp_path / "missing") is False
