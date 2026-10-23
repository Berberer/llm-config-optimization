"""Command-line argument parser for the optimization runner.

This module owns the CLI's entire flag surface. The command takes three
positionals — a model (a Hugging Face id or a host path), the search-space YAML
file, and the inference engine to benchmark — and every other setting is a flag
with a sensible default, so the simplest invocation is
``llm-config-optimization MODEL SPACE ENGINE``.

The parser is deliberately a pure description of the command line: it loads no
space, touches no hardware, and runs no sweep. :func:`parse_args` returns an
:class:`argparse.Namespace` whose attributes are the documented settings, and
the later CLI stages turn that namespace into an experiment configuration. The
two budget flags are both optional, so an unbounded, Ctrl-C-driven sweep is the
default. ``--output`` accepts exactly ``tui`` and ``log``: there is no ``auto``
value because the mode is selected automatically when the flag is omitted.

``-h``/``--help`` and ``--version`` are handled by argparse itself and exit
zero without side effects, so help is the primary documentation of the command.
"""

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["build_parser", "parse_args"]

_DISTRIBUTION_NAME = "llm-config-optimization"
_PROG = "llm-config-optimization"

_DESCRIPTION = (
    "Search an LLM serving configuration for the best prefill, decode, or combined "
    "token throughput."
)

_EPILOG = """\
examples:
  llm-config-optimization facebook/opt-125m space.yaml vllm
  llm-config-optimization facebook/opt-125m space.yaml vllm --n-trials 50 --output-config best.yaml
  llm-config-optimization /models/Llama-3.1-8B space.yaml vllm --target cuda --sweep-timeout 3600
"""


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Append each flag's default while preserving the epilog's line breaks."""


def _package_version() -> str:
    """Return the installed distribution's version, or ``"unknown"`` if absent."""
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:  # pragma: no cover - only when running uninstalled
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser describing the whole command line.

    The parser is stateless and safe to rebuild; it is exposed so tests can
    inspect the argument surface and render help without running the CLI.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "model",
        metavar="MODEL",
        help="Model to benchmark: a Hugging Face id or a host path to a local checkpoint.",
    )
    parser.add_argument(
        "space",
        metavar="SPACE",
        type=Path,
        help="Path to the search-space YAML file describing the parameters to sweep.",
    )
    parser.add_argument(
        "engine",
        metavar="ENGINE",
        choices=("vllm", "llamacpp"),
        help="Inference engine to benchmark: vllm or llamacpp.",
    )

    container = parser.add_argument_group("container")
    container.add_argument(
        "--target",
        choices=("cuda", "rocm", "cpu"),
        default=None,
        help="Accelerator class to target; auto-detected from the host when omitted.",
    )
    container.add_argument(
        "--image",
        default=None,
        help="Container image to run; the backend's image for the resolved target when omitted.",
    )
    container.add_argument(
        "--shm-size",
        default="8g",
        help="Size of the container's /dev/shm.",
    )
    container.add_argument(
        "--mem-limit",
        default=None,
        help="Container memory limit, such as 16g; unlimited when omitted.",
    )
    container.add_argument(
        "--pull-policy",
        choices=("missing", "always", "never", "newer"),
        default="missing",
        help="When to pull the container image.",
    )

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument(
        "--dataset-name",
        default="random",
        help="Benchmark dataset to serve prompts from.",
    )
    benchmark.add_argument(
        "--dataset-path",
        default=None,
        help="Host path to a custom dataset file; the built-in dataset when omitted.",
    )
    benchmark.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to send per benchmark run.",
    )
    benchmark.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input prompt length, in tokens.",
    )
    benchmark.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of tokens to generate per prompt.",
    )
    benchmark.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of measured benchmark runs taken per trial.",
    )
    benchmark.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Discarded warmup runs taken before the measured runs.",
    )
    benchmark.add_argument(
        "--startup-timeout",
        type=float,
        default=600,
        help="Seconds to wait for the served model to become healthy.",
    )
    benchmark.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Overall time limit for a trial's container, covering startup, warmup, and every measured run, in seconds.",
    )

    engine = parser.add_argument_group("engine")
    engine.add_argument(
        "--llm-seed",
        type=int,
        default=0,
        help="Engine seed, so every trial runs under the same prompt randomness. vLLM only; accepted for llamacpp but not forwarded, because the host benchmark's workload is greedy.",
    )
    engine.add_argument(
        "--quantization",
        choices=("awq", "gptq", "fp8"),
        default=None,
        help="Weight quantization the checkpoint was produced with; checked against the resolved checkpoint. The engine auto-detects when omitted. vLLM only; a usage error for llamacpp, whose GGUF file carries its own quantization.",
    )
    engine.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory the engine may use; the engine's default when omitted. vLLM only; a usage error for llamacpp.",
    )
    engine.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for the optimization sampler; a non-deterministic search when omitted.",
    )

    budget = parser.add_argument_group("budget")
    budget.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Maximum number of trials to run; unbounded when omitted.",
    )
    budget.add_argument(
        "--sweep-timeout",
        type=float,
        default=None,
        help="Wall-clock time limit for the sweep, in seconds; unbounded when omitted.",
    )

    study = parser.add_argument_group("study")
    study.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, such as sqlite:///study.db; an in-memory study when omitted.",
    )
    study.add_argument(
        "--study-name",
        default=None,
        help="Name of the persisted study; derived from the model when omitted.",
    )
    study.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of trials to run in parallel; requires --storage and disjoint devices.",
    )

    objective = parser.add_argument_group("objective")
    objective.add_argument(
        "--objective",
        choices=("multi", "weighted", "combined"),
        default="multi",
        help=(
            "What the sweep maximizes: multi submits prefill and decode rates as two "
            "objectives and reports the Pareto front, weighted scores their "
            "weighted-harmonic effective rate, and combined maximizes total token "
            "throughput."
        ),
    )
    objective.add_argument(
        "--prefill-weight",
        type=float,
        default=None,
        help=(
            "Prompt tokens in the token mix the weighted and multi objectives are scored "
            "under; the benchmark's --input-len when omitted."
        ),
    )
    objective.add_argument(
        "--decode-weight",
        type=float,
        default=None,
        help=(
            "Generated tokens in the token mix the weighted and multi objectives are "
            "scored under; the benchmark's --output-len when omitted."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        choices=("tui", "log"),
        default=None,
        help="Progress output mode; chosen from whether stdout is a terminal when omitted.",
    )
    output.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Write the winning configuration to this YAML path.",
    )
    output.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
        help="Print the version and exit.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (defaulting to ``sys.argv[1:]``) into a settings namespace.

    Invalid input is reported by argparse on stderr and exits with status ``2``;
    ``-h``/``--help`` and ``--version`` print and exit ``0``.
    """
    return build_parser().parse_args(argv)
