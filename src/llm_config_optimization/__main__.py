"""Entry point for ``python -m llm_config_optimization``.

Mirrors the ``llm-config-optimization`` console script: both call the same
:func:`llm_config_optimization.cli.main` composition root and exit with the code it
returns.
"""

from llm_config_optimization.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
