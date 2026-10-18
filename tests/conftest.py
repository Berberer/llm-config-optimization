"""Pytest configuration for the test suite.

The presence of this file puts the ``tests`` directory on ``sys.path`` (pytest
prepends the rootdir of each conftest), so shared test doubles in
``tests/fakes.py`` are importable as ``fakes`` from every test package without
turning the test tree into an installed package.

Unit tests drive the real orchestration with a fake container client, so they
must not reach the network either. The autouse fixture below replaces the
orchestrator's default host benchmark client with the shared fake; a test that
wants to observe the client injects its own, and an integration test that wants
the real HTTP client passes it explicitly.
"""

import pytest

from fakes import FakeBenchmarkClient


@pytest.fixture(autouse=True)
def _fake_benchmark_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the shared fake host client unless a test injects its own."""
    monkeypatch.setattr(
        "llm_config_optimization.inference._orchestrate.BenchmarkClient",
        FakeBenchmarkClient,
    )
