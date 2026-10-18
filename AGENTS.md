# AGENTS.md

## Project

`llm-config-optimization` is a Python 3.14 library with a `src/` layout, managed with
[uv](https://docs.astral.sh/uv/). Runtime code lives in `src/llm_config_optimization/`,
tests in `tests/`.

## Architecture

The code is layered so engine and runtime details stay at the edges. Packages
live under `src/llm_config_optimization/`; each exposes its public surface through its
`__init__.py`, so import from the package, not from a private `_module`.

| Package         | Owns                                                                                                 | Key files |
| --------------- | ---------------------------------------------------------------------------------------------------- | --------- |
| `search_space/` | Engine-neutral definition types, the tunable registry, the YAML loader, engine-agnostic and backend-support validation | `types.py`, `registry.py`, `loader.py`, `validation.py` |
| `inference/`    | Engine-neutral experiment surface (the `InferenceBackend` protocol and `ServerSpec`) plus the vLLM backend, the host benchmark client, and hardware detection | `types.py`, `_orchestrate.py`, `_benchmark.py`, `_vllm.py`, `hardware.py` |
| `runtime/`      | Backend-agnostic container types, the detached start/stop lifecycle, the Podman client, and the SELinux probe | `types.py`, `_podman.py`, `_selinux.py` |
| `optimization/` | The Optuna adapter: maps spec kinds to suggestions and drives a study                                | `types.py`, `_optuna.py` |
| `cli/`          | Composition root: argument parsing, the engine registry, base-config construction, model download progress, log/TUI reporters, summary/writer, exit codes | `arguments.py`, `config.py`, `download.py`, `log.py`, `tui.py`, `summary.py`, `__init__.py` |

### Invariants

- The definition and experiment surfaces (`search_space/`, `inference/types.py`)
  import **no engine and no container backend** — keep engine vocabulary out of
  them.
- `runtime/_podman.py` is the only module that imports the `podman` bindings at
  runtime; elsewhere it is `TYPE_CHECKING`-only. It carries no engine knowledge.
- `optimization/_optuna.py` is the only module that knows Optuna's study and
  trial API.
- Engine spellings live only in `inference/_vllm.py` (`PARAMETER_TARGETS`,
  `VALUE_MAPS`, `_CHECKPOINT_QUANTIZATION_METHODS`); the search space uses
  engine-neutral enum values.
- `InferenceBackend.validate_model` is the model-aware half of backend
  validation: it receives the whole `ExperimentConfig` and the resolved model
  path, returns problems rather than raising, and fails open when the
  checkpoint's metadata cannot be read. Reading the checkpoint and the
  method-to-spelling mapping live in `inference/_vllm.py`, so the
  engine-neutral surfaces carry no checkpoint vocabulary.
- The search space is **total over the backend's applicable set**: every
  parameter the selected backend reports in `applicable_parameters` must be
  declared, so a value never silently falls back to an engine default. The
  registry itself stays engine-neutral; a backend that lacks a parameter (for
  example llama.cpp has no `tensor_parallel_size`) simply omits it.
- A trial is host-driven: the backend plans an engine-neutral `ServerSpec` and
  the orchestrator starts the server container, drives it with the host
  benchmark client, stops it, and hands the measured outcome to
  `parse_result`. Neither the orchestrator nor the client learns an engine's
  vocabulary — health and cache-reset paths come from the spec.

### Extension recipes

- **New inference engine** — implement `InferenceBackend` (`inference/types.py`):
  `applicable_parameters`, `validate`, `validate_model`, `model_request`,
  `image_for`, `plan`, `parse_result`. `plan` returns a `ServerSpec`, so the
  shared host client and orchestrator need no engine-specific code. The file
  format and the layers above are unchanged, so a second engine is an addition,
  not a rewrite.
- **New tunable parameter** — see the checklist below; the search space is
  total over a backend's applicable set, so this touches code, both example
  files, the README, and the tests in one change.
- **New container runtime** — implement `run_container`, `start_container`
  (the detached lifecycle), and `pull_image` against the types in
  `runtime/types.py`; keep `_podman.py` engine-free.

### Checklist: adding a new tunable parameter

A tunable parameter is declared in the registry, must be declared in every
space whose backend forwards it, and must be forwarded by the backend. Because
the search space is **total over a backend's applicable set**, adding one is
not additive for a backend that forwards it: every existing definition for that
backend (examples, README snippets, test fixtures) becomes invalid until it
declares the new parameter, and the loader will report it as a missing tunable.

Code:

- [ ] `inference/types.py` — add a typed, optional (`| None`) field to
  `LLMConfig`. For a categorical parameter, also add an engine-neutral
  `StrEnum` in the same module; never use an engine's own spelling here.
- [ ] `search_space/registry.py` — add an entry to `TUNABLE_PARAMETERS` mapping
  the YAML name to the field name. The loader reads the expected kind from the
  field annotation, so there is no second kind to keep in sync. Insertion order
  is canonical (the parser emits dimensions in it), so append at the end unless
  the canonical order is deliberately being changed.
- [ ] `inference/_vllm.py` — forward the field via `PARAMETER_TARGETS`
  (`FlatFlag()` for a `--flag value`, `JsonGroup(flag=..., key=...)` for a JSON
  group). For a categorical field, add a vLLM spelling for **every** enum member
  to `VALUE_MAPS`; a member left out is rejected as unsupported. If the
  parameter is platform-specific, also record which targets each member is
  supported on (see `_ATTENTION_BACKEND_TARGETS`), so `validate` rejects a value
  the resolved target cannot run. A parameter the engine bounds by the host's
  device count (like `tensor_parallel_size`) needs the equivalent check against
  `DetectedHardware.device_count`.
- [ ] `search_space/loader.py` — if the parameter is an integer with a `>= 1`
  floor, add it to `_POSITIVE_PARAMETERS`.
- [ ] `inference/_vllm.py` — if the engine imposes a cross-parameter rule, add it
  to `_ENGINE_BOUNDS` (or the equivalent engine-specific validation).
- [ ] `cli/arguments.py` + `cli/config.py` — only if the parameter should also
  be settable outside a sweep (like `seed` / `gpu_memory_utilization`, which are
  CLI flags and non-tunables). A pure tuning dimension needs no CLI change.

Docs and examples:

- [ ] `README.md` — add a row to the *Tunable parameters* table, fix the
  parameter count in the surrounding prose ("vLLM forwards all seven parameters
  below"), and add the new dimension to the example YAML snippets so they stay
  complete.
- [ ] `examples/space-full.yaml` — declare the new parameter over its full range.
- [ ] `examples/space-all-fixed.yaml` — declare the new parameter pinned to one
  value.
- [ ] `AGENTS.md` — update this checklist only if the parameter introduces a new
  invariant (for example a new engine-neutral enum, or a new non-tunable field).

Tests:

- [ ] `tests/search_space/test_registry.py` — extend `TUNABLES` and the
  `expected_kind` parametrization so the canonical order and resolved kind are
  pinned.
- [ ] `tests/search_space/test_loader.py` — extend the complete-space helper and
  its expected dimensions; the missing-tunable check makes every loader test fail
  until the new parameter is declared.
- [ ] `tests/inference/test_vllm_arguments.py` — assert the new field is
  forwarded (flat flag or JSON group) and, for a categorical, that each member
  translates and unsupported values are rejected.
- [ ] Any other test that constructs a full `LLMConfig`, a complete
  `ExperimentSpace`, or asserts the canonical dimension order.
- [ ] Run `uv run ruff format .`, `uv run ruff check .`, `uv run ty check`, and `uv run pytest`.

### Data flow

`cli.main` loads and engine-agnostically validates the space against the
selected backend's `applicable_parameters`, asks the backend to `validate` it
against the resolved hardware, builds the base configuration, asks the
backend's `model_request` what to fetch, resolves that to a local path, asks the
backend to `validate_model` it against that checkpoint, pulls the image, then
calls `optimization.optimize`. Each trial: Optuna samples a point → `inference`
plans a `ServerSpec` → `runtime` starts the server container detached and
resolves its host port → the host benchmark client waits for health, takes the
warmup and measured runs, and aggregates them → `runtime` stops the container →
the backend parses the outcome into a result → the reporter callback runs.
`cli/__init__.py` maps outcomes onto exit codes.

## Code style

Style is defined by the Ruff config in `pyproject.toml`; `.editorconfig` mirrors
it for editors. Ruff is the single source of truth — let it format rather than
hand-formatting.

- Line length 100, double quotes, 4-space indents, LF endings, final newline.
- Target `py314`.
- Lint rules: `E4 E7 E9 F I UP B SIM C4 RUF` (no `E501`; the formatter owns
  line wrapping).

Types are checked by [ty](https://docs.astral.sh/ty/) against the library
sources in `src/`; `tests/` is excluded because its fakes do not implement every
protocol member. Keep `uv run ty check` clean when touching `src/`.

Always run tools through `uv run` so they use the project environment.

## Tools

| Task               | Command                        |
| ------------------ | ------------------------------ |
| Format in place    | `uv run ruff format .`         |
| Lint (report)      | `uv run ruff check .`          |
| Lint (auto-fix)    | `uv run ruff check --fix .`    |
| Type check         | `uv run ty check`              |
| Unit tests         | `uv run pytest`                |
| Integration tests  | `uv run pytest -m integration` |

- Integration tests are marked `integration` and are **deselected by default**
  (`addopts = "-m 'not integration'"` in `pyproject.toml`). They require a
  running Podman daemon.
- Add dev tooling with `uv add --dev <pkg>` to keep it out of runtime deps.
- `nix develop` provides Python and uv in the dev shell; `uv sync` installs the Python tooling (ruff, ty, pytest).

## Testing conventions

- Unit tests must not require a daemon: inject the fake Podman client.
- Integration tests carry `@pytest.mark.integration` and skip when Podman is
  unavailable.
