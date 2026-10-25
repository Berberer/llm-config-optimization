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
| `inference/`    | Engine-neutral experiment surface (the `InferenceBackend` protocol, `ServerSpec`, and the optional phase-rate metrics) plus the vLLM and llama.cpp backends, the host benchmark client with its Prometheus snapshots, and hardware detection | `types.py`, `_orchestrate.py`, `_benchmark.py`, `_vllm.py`, `_llamacpp.py`, `hardware.py` |
| `runtime/`      | Backend-agnostic container types, the detached start/stop lifecycle, the Podman client, and the SELinux probe | `types.py`, `_podman.py`, `_selinux.py` |
| `optimization/` | The engine-neutral objective model plus the Optuna adapter: maps spec kinds to suggestions, scores trials, drives a study, and guards study provenance | `types.py`, `_optuna.py` |
| `output/`       | Engine-neutral winning-configuration writers: the historical YAML experiment document, the Podman quadlet, and the Docker/Podman Compose file, plus the pure-renderer contract and the format registry | `types.py`, `_yaml.py`, `_quadlet.py`, `_compose.py`, `_registry.py` |
| `cli/`          | Composition root: argument parsing, the engine registry, base-config construction, model download progress, log/TUI reporters, summary and output-writer selection, exit codes | `arguments.py`, `config.py`, `download.py`, `_formatting.py`, `log.py`, `tui.py`, `summary.py`, `__init__.py` |

### Invariants

- The definition and experiment surfaces (`search_space/`, `inference/types.py`)
  import **no engine and no container backend** — keep engine vocabulary out of
  them.
- `runtime/_podman.py` is the only module that imports the `podman` bindings at
  runtime; elsewhere it is `TYPE_CHECKING`-only. It carries no engine knowledge.
- `optimization/_optuna.py` is the only module that drives Optuna's study and
  trial API; the CLI reporters may read the raw study a result carries — its
  directions, trials, and recorded user attributes — to render progress and
  the summary, and that inspection is the reporting surface, not study logic.
  `optimization/types.py` holds the engine-neutral objective model
  (`ObjectiveMode`, `ObjectiveWeights`, and the scoring functions), and an
  objective that needs a rate a result lacks raises `ExperimentError` naming
  the missing source rather than silently falling back to another measurement.
- Engine spellings live only in each engine's own backend module: vLLM's in
  `inference/_vllm.py` (`PARAMETER_TARGETS`, `VALUE_MAPS`,
  `_CHECKPOINT_QUANTIZATION_METHODS`, `_DEFAULT_IMAGES`, `_METRICS_PATH`,
  `_PREFILL_RATE_SERIES`, `_DECODE_RATE_SERIES`), and llama.cpp's in
  `inference/_llamacpp.py` (`PARAMETER_TARGETS`, `VALUE_MAPS`,
  `_GGUF_FILE_TYPE_LABELS`, `_DEFAULT_IMAGES`, `_TIMINGS_KEY`,
  `_PREFILL_TIMING_KEYS`, `_DECODE_TIMING_KEYS`); the search space uses
  engine-neutral enum values.
- The metrics seam is engine-neutral: the host client knows only the standard
  Prometheus exposition format and a spec-declared `ServerSpec.metrics_path`,
  which it scrapes before and after every measured run and stores as parsed
  name-to-value maps under `raw_json["metrics"]` (`{"before": ..., "after":
  ...}`; absent unless both scrapes succeeded, and a failure never fails the
  run). The backend that owns the engine derives the phase rates from those
  snapshots (vLLM) or from its preserved raw payloads (llama.cpp) in
  `parse_result`; the engine-neutral types carry only the optional
  `RunMetrics.prefill_tokens_per_second` / `decode_tokens_per_second` fields
  and their `ExperimentResult` means.
- A persisted study records the objective mode that produced its scores and the
  phase-metric schema version they were measured under — `_MODE_ATTRIBUTE`,
  `_METRIC_SCHEMA_ATTRIBUTE`, and `_METRIC_SCHEMA_VERSION` in
  `optimization/_optuna.py`. A resume under a different mode, or into a study
  whose direction count does not match the selected mode, is refused with an
  `OptimizationError`; an unmarked study is treated as legacy combined. The
  CLI's derived study name carries a mode suffix, and bumping
  `_METRIC_SCHEMA_VERSION` is the required response to any change in how either
  backend defines a phase rate.
- The llama.cpp backend keeps `max_model_len` a request length at the engine: it
  forwards the sampled value as `--ctx-size` unchanged and cuts `--parallel`
  to `max_model_len // request_tokens` (at least one), because llama.cpp splits
  `--ctx-size` evenly across its slots and rejects a request longer than a
  slot's share. The request length is `input_len + output_len` for the random
  workload and, for a JSONL dataset, the preflight-measured
  `BenchmarkConfig.max_request_tokens`: the composition root starts the
  engine's server once before the sweep and tokenizes every dataset prompt
  through the spec-declared `ServerSpec.tokenize_path`, so the engine's own
  tokenizer decides the cap. A dataset whose preflight could not run falls back
  to one slot. The cap lives in the backend's plan; the engine-neutral surfaces
  carry none of it.
- A backend owns its model-argument rules through `model_request`: vLLM
  resolves a Hub snapshot directory or a local directory, while llama.cpp
  accepts a local `.gguf` file or a Hub repository with an optional `:QUANT`
  selector and fetches a single file. The composition root only resolves what
  the backend describes, so no engine's model logic appears in `cli/`.
- `InferenceBackend.validate_model` is the model-aware half of backend
  validation: it receives the whole `ExperimentConfig` and the resolved model
  path, returns problems rather than raising, and fails open when the
  checkpoint's metadata cannot be read. When the experiment configures a
  speculative setup it also inspects the resolved draft at
  `LLMConfig.resolved_draft_path` — llama.cpp rejects a non-GGUF draft and
  fails open on an unreadable header — before the image is pulled. Reading the
  checkpoint and the method-to-spelling mapping live in the backend modules —
  vLLM's quantization methods in `inference/_vllm.py`, GGUF metadata in
  `inference/_llamacpp.py` — so the engine-neutral surfaces carry no checkpoint
  vocabulary.
- The search space is **total over the backend's applicable set**: every
  parameter the selected backend reports in `applicable_parameters(llm)` must
  be declared, so a value never silently falls back to an engine default. The
  registry itself stays engine-neutral; a backend that lacks a parameter (for
  example llama.cpp has no `tensor_parallel_size`) simply omits it. `llm` is the
  run's fixed configuration, because applicability can depend on it: the two
  speculative tunables `speculative_mode` and `num_speculative_tokens` are in
  the set exactly when `speculative_method` or `draft_model` is configured, and
  a definition declaring them without that setup is rejected as unsupported.
- A trial is host-driven: the backend plans an engine-neutral `ServerSpec` and
  the orchestrator starts the server container, drives it with the host
  benchmark client, stops it, and hands the measured outcome to
  `parse_result`. Neither the orchestrator nor the client learns an engine's
  vocabulary — health and cache-reset paths come from the spec, and the client
  polls the runtime handle's `is_running` during the health wait so a container
  that died on startup fails the run immediately instead of waiting out the
  timeout.
- The output layer (`output/`) is engine-neutral and side-effect free: it
  imports no inference backend, no engine module, and no container runtime
  implementation (only the engine-neutral runtime types), and its writers are
  pure renderers that return text. `output/types.py` holds
  `OutputFormat`, `RenderContext`, and the `ConfigWriter` protocol with
  `requires_plan` and `render`; `output/_registry.py` maps every format to a
  shared writer instance, so the composition root selects one with `writer_for`
  and never imports a concrete writer. The YAML writer needs no plan and keeps
  the historical config-only document; the quadlet and Compose writers need the
  `ServerSpec` a trial planned, because that is the only place the engine
  command, environment, and host mounts exist. The composition root owns
  planning and the filesystem write, so a planning or write failure exits
  non-zero on both the normal and interrupted finish paths. Executable artifacts
  are planned through the engine's deployment backend, which omits trial-scoped
  scratch resources such as vLLM's temporary compile cache, so every path a file
  names outlives the run and the rendered document is deterministic. Generated
  files carry no provenance header and use `curl` for the healthcheck; the
  Compose pull-policy translation (`newer` → `always`) is the only emitted
  comment. The writers are **parameter-agnostic**: adding an
  `LLMConfig` tunable needs no output change, because the backend encodes it
  into `ServerSpec.container.command` and the writers copy that verbatim, while
  the YAML writer reflects every `LLMConfig` field automatically. A new
  **`ContainerConfig` field**, by contrast, must be considered by every container
  writer (`_quadlet.py`, `_compose.py`): a field that describes the deployed
  container must be rendered by both, while a trial-scoped field (`timeout`, the
  blocking-run limit) is deliberately omitted because it has no meaning in a
  long-running service. A deployed field belongs in `_CONTAINER_FIELDS` in
  `_yaml.py` when it also belongs in the curated YAML `container` section.

### Extension recipes

- **New inference engine** — implement `InferenceBackend` (`inference/types.py`):
  `applicable_parameters(llm)`, `validate(space, llm, hardware)`,
  `validate_model`, `model_request`, `image_for`, `plan`, `parse_result`.
  `plan` returns a `ServerSpec`, so the shared host client and orchestrator need
  no engine-specific code; an engine
  with a scrapeable exposition declares `ServerSpec.metrics_path` and derives
  its phase rates from the snapshots in `parse_result`, while one with
  per-response timings derives them from its preserved raw payloads. Register
  the engine token in `_ENGINES` (`cli/__init__.py`) with its context-manager
  backend factory, a deployment factory that plans artifacts without
  trial-scoped resources, and any command-line options it rejects, and export
  the backend from `inference/__init__.py`. The file format and the layers above
  are unchanged, so a second engine is an addition, not a rewrite.
- **New tunable parameter** — see the checklist below; the search space is
  total over a backend's applicable set, so this touches code, every example
  space whose backend forwards it, `docs/search-space.md`, and the tests in one
  change.
- **New container runtime** — implement `run_container`, `start_container`
  (the detached lifecycle), and `pull_image` against the types in
  `runtime/types.py`; keep `_podman.py` engine-free.
- **New config output format** — implement `ConfigWriter` (`output/types.py`)
  with `requires_plan` and a pure `render`, add the member to `OutputFormat`,
  and register the shared instance in `output/_registry.py`. A renderer that
  reproduces the server reads the planned `ServerSpec`; one that describes the
  experiment reads the config and declares `requires_plan = False`. The
  `--output-format` choice follows automatically from the enum; document its
  conventional suffix in `docs/usage.md` and cover it with formatter tests.

### Checklist: adding a new tunable parameter

A tunable parameter is declared in the registry, must be declared in every
space whose backend forwards it, and must be forwarded by every backend whose
applicable set includes it. Because the search space is **total over a
backend's applicable set**, adding one is
not additive for a backend that forwards it: every existing definition for that
backend (examples, docs snippets, test fixtures) becomes invalid until it
declares the new parameter, and the loader will report it as a missing tunable.

Code:

- [ ] `inference/types.py` — add a typed, optional (`| None`) field to
  `LLMConfig`. For a categorical parameter, also add an engine-neutral
  `StrEnum` in the same module; never use an engine's own spelling here.
- [ ] `search_space/registry.py` — add an entry to `TUNABLE_PARAMETERS` mapping
  the YAML name to the field name. The loader reads the expected kind from the
  field annotation, so there is no second kind to keep in sync. Insertion order
  is canonical (the parser emits dimensions in it), so append at the end unless
  the canonical order is deliberately being changed. If the parameter should be
  applicable only under a configured setup, add it to the gated set
  (`SPECULATIVE_PARAMETERS`, whose members are `speculative_mode` and
  `num_speculative_tokens`) so both backends include it from
  `applicable_parameters(llm)` exactly then, and `validate` rejects it otherwise.
- [ ] `inference/_vllm.py` — forward the field via `PARAMETER_TARGETS`
  (`FlatFlag()` for a `--flag value`, `JsonGroup(flag=..., key=...)` for a JSON
  group). For a categorical field, add a vLLM spelling for **every** enum member
  to `VALUE_MAPS`; a member left out is rejected as unsupported. If the
  parameter is platform-specific, also record which targets each member is
  supported on (see `_ATTENTION_BACKEND_TARGETS`), so `validate` rejects a value
  the resolved target cannot run. A parameter the engine bounds by the host's
  device count (like `tensor_parallel_size`) needs the equivalent check against
  `DetectedHardware.device_count`.
- [ ] `inference/_llamacpp.py` — forward the field too when llama.cpp exposes
  it, and add a spelling for every enum member llama.cpp serves to its
  `VALUE_MAPS`. A parameter llama.cpp lacks (like `dtype` or
  `tensor_parallel_size`) stays out of `PARAMETER_TARGETS`, which is what keeps
  it out of the backend's applicable set.
- [ ] `search_space/loader.py` — if the parameter is an integer with a `>= 1`
  floor, add it to `_POSITIVE_PARAMETERS`.
- [ ] `inference/_vllm.py`/`inference/_llamacpp.py` — if the engine imposes a
  cross-parameter rule, add it to `_ENGINE_BOUNDS` (or the equivalent
  engine-specific validation).
- [ ] `cli/arguments.py` + `cli/config.py` — only if the parameter should also
  be settable outside a sweep (like `seed` / `gpu_memory_utilization`, which are
  CLI flags and non-tunables). A pure tuning dimension needs no CLI change.

Docs and examples:

- [ ] `docs/search-space.md` — add a row to the *Tunable parameters* table,
  fix the per-engine parameter counts in the surrounding prose ("vLLM forwards
  all seven parameters below", "llama.cpp forwards all but ..."), and add the
  new dimension to the example YAML snippets so they stay complete. A gated
  dimension is documented in `docs/speculative-decoding.md` as well, and its
  applicability rule is noted in `docs/architecture.md`.
- [ ] Every example space whose backend forwards the parameter — vLLM's
  `examples/vllm/space-full.yaml`, `space-all-fixed.yaml`,
  `space-speculative-defaults.yaml`, and `space-speculative-full.yaml`, and
  llama.cpp's corresponding four under `examples/llamacpp/` — declares it, over
  its full range or pinned to one value respectively.
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
  translates and unsupported values are rejected. Add the same coverage to
  `tests/inference/test_llamacpp_arguments.py` for a parameter llama.cpp
  forwards.
- [ ] Any other test that constructs a full `LLMConfig`, a complete
  `ExperimentSpace`, or asserts the canonical dimension order.
- [ ] Run `uv run ruff format .`, `uv run ruff check .`, `uv run ty check`, and `uv run pytest`.

### Data flow

`cli.main` loads and engine-agnostically validates the space against the
selected backend's `applicable_parameters(base.llm)`, asks the backend to
`validate` it against the same LLM configuration and the resolved hardware,
builds the base configuration, asks the backend's `model_request` what the main
model and — when configured — the draft model refer to, resolves or downloads
both through the shared progress helper, records them on
`BenchmarkConfig.resolved_path` and `LLMConfig.resolved_draft_path`, asks the
backend to `validate_model` each, warns when speculation runs on the synthetic
workload without a dataset, pulls the image, tokenizes a JSONL dataset once
through the backend's declared tokenizer endpoint so llama.cpp can cap its
slots, then calls `optimization.optimize`. Each trial: Optuna samples a point → `inference`
plans a `ServerSpec` → `runtime` starts the server container detached and
resolves its host port → the host benchmark client waits for health, takes the
warmup and measured runs, scraping the spec's metrics path around each measured
run when one is declared, and aggregates them → `runtime` stops the container →
the backend parses the outcome into a result and enriches it with the phase
rates its engine reported → the selected objective maps the result to the trial
value → the reporter callback runs. After the sweep — or on an interrupt — the
finish path prints the summary and, when `--output-config` is set, renders the
recommended configuration through the format's writer: YAML from the config
plus the resolved target and image alone, while `quadlet` and `compose` first
re-plan the recommended config through the engine's deployment backend — which
omits trial-scoped scratch resources such as vLLM's compile cache — so the
artifact reproduces the server the trial ran without naming a path the run
deletes. `cli/__init__.py` maps outcomes onto exit codes.

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
- Generated container artifacts have a hermetic conformance layer in the default
  suite (`tests/output/test_artifact_conformance.py`): it rejects any quadlet
  `[Container]` key outside Podman's documented allowlist, checks the JSON-array
  entrypoint and health-command form, and checks the Compose structure and
  device classification. The allowlist lives in the tests, so the writer layer
  stays tool-neutral; a key or value-format change must update it deliberately.
- Integration tests carry `@pytest.mark.integration` and skip when Podman is
  unavailable. The artifact ones (`tests/output/test_artifact_integration.py`)
  run rendered files through `docker compose config`, `podman-compose config`,
  and the Podman quadlet generator (`QUADLET_UNIT_DIRS` + `--dryrun`, optionally
  `systemd-analyze verify`); each locates its binary through `tests/tooling.py`
  and skips when it is missing, so a partial install stays runnable.
