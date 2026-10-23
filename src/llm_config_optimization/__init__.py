"""llm-config-optimization: search an LLM serving configuration for the best prefill,
decode, or combined token throughput.

The user-facing entry points are the ``llm-config-optimization`` console script and
``python -m llm_config_optimization``; both delegate to
:func:`llm_config_optimization.cli.main`, the composition root that parses the command
line, prepares the run, drives the optimizer, and maps outcomes onto exit codes.
Importing the package stays cheap, so the CLI subpackage is imported inside
:func:`main` rather than at module load.
"""

__all__ = ["main"]


def main() -> int:
    """Run the command line and return the process exit code."""
    from .cli import main as run

    return run()
