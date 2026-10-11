# AGENTS.md

## Project

`llm-optimization` is a Python 3.14 library with a `src/` layout, managed with
[uv](https://docs.astral.sh/uv/). Runtime code lives in `src/llm_optimization/`,
tests in `tests/`.

## Code style

Style is defined by the Ruff config in `pyproject.toml`; `.editorconfig` mirrors
it for editors. Ruff is the single source of truth — let it format rather than
hand-formatting.

- Line length 100, double quotes, 4-space indents, LF endings, final newline.
- Target `py314`.
- Lint rules: `E4 E7 E9 F I UP B SIM C4 RUF` (no `E501`; the formatter owns
  line wrapping).

Always run tools through `uv run` so they use the project environment.

## Tools

| Task               | Command                        |
| ------------------ | ------------------------------ |
| Format in place    | `uv run ruff format .`         |
| Lint (report)      | `uv run ruff check .`          |
| Lint (auto-fix)    | `uv run ruff check --fix .`    |
| Unit tests         | `uv run pytest`                |
| Integration tests  | `uv run pytest -m integration` |

- Integration tests are marked `integration` and are **deselected by default**
  (`addopts = "-m 'not integration'"` in `pyproject.toml`). They require a
  running Podman daemon.
- Add dev tooling with `uv add --dev <pkg>` to keep it out of runtime deps.
- `nix develop` provides Python, uv, and Ruff in the dev shell.

## Testing conventions

- Unit tests must not require a daemon: inject the fake Podman client.
- Integration tests carry `@pytest.mark.integration` and skip when Podman is
  unavailable.
