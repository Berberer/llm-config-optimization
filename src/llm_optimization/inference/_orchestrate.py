"""Orchestration for the engine-agnostic inference experiment layer.

This module is the entry point that ties the layers together: it resolves the
host hardware, asks the configured backend to plan a container run, executes
that plan through the runtime layer, and hands the captured outcome back to the
backend for parsing. It knows nothing about any inference engine — every
engine-specific decision lives behind the injected backend.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..runtime import run_container
from ..runtime.types import ContainerRunError
from ._vllm import VLLMBackend
from .hardware import detect_hardware
from .types import (
    DetectedHardware,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    InferenceBackend,
)

if TYPE_CHECKING:
    from podman import PodmanClient

__all__ = ["run_experiment"]


def run_experiment(
    config: ExperimentConfig,
    backend: InferenceBackend | None = None,
    *,
    client: PodmanClient | None = None,
    output_dir: Path | None = None,
) -> ExperimentResult:
    """Run one benchmark experiment and return its typed result.

    The host hardware is detected once — honouring the container config's
    ``target`` override — and passed to the backend's ``plan``, so the backend
    never probes the host itself. The plan is then executed through
    :func:`~llm_optimization.runtime.run_container` and the captured outcome is
    handed back to the backend's ``parse_result``.

    ``backend`` defaults to :class:`~llm_optimization.inference.VLLMBackend`.
    ``client`` is forwarded to ``run_container`` so a caller (or a test) can
    supply a pre-configured Podman client; when it is omitted the runtime
    discovers one from the environment.

    When ``output_dir`` is supplied it is used as-is and left in place. When it
    is omitted a unique temporary directory is created for the run and removed
    once the result has been parsed, so concurrent runs never share a result
    file; the parsed JSON remains available on the result as ``raw_json``.

    Raises:
        ExperimentError: If the container run fails — the underlying
            :class:`~llm_optimization.runtime.ContainerRunError` is preserved as
            ``__cause__`` and its partial result attached — or if the backend
            cannot parse the run's output.
    """
    resolved_backend = backend if backend is not None else VLLMBackend()
    hardware = detect_hardware(config.container.target)
    if output_dir is not None:
        return _run(config, resolved_backend, hardware, output_dir, client)
    with tempfile.TemporaryDirectory(prefix="llm-optimization-") as directory:
        return _run(config, resolved_backend, hardware, Path(directory), client)


def _run(
    config: ExperimentConfig,
    backend: InferenceBackend,
    hardware: DetectedHardware,
    output_dir: Path,
    client: PodmanClient | None,
) -> ExperimentResult:
    """Plan, execute, and parse one run against an already-resolved directory."""
    plan = backend.plan(config, hardware, output_dir)
    try:
        result = run_container(plan, client=client)
    except ContainerRunError as exc:
        raise ExperimentError(
            f"benchmark container for model {config.benchmark.model!r} failed: {exc}",
            raw_result=exc.result,
        ) from exc
    return backend.parse_result(result, plan, output_dir)
