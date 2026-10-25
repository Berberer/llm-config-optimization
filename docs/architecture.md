# Architecture and tech stack

Part of the [README](../README.md) documentation.

## Tech stack

| Layer                | Technology                                                                 |
| -------------------- | -------------------------------------------------------------------------- |
| Language / packaging | Python **3.14**, `src/` layout, [uv](https://docs.astral.sh/uv/) with the `uv_build` backend |
| Search / optimization | [Optuna](https://optuna.org/) TPE sampler, maximizing prefill and decode throughput as a Pareto front, their weighted-harmonic rate, or the combined token throughput; SQLite or in-memory studies |
| Inference engine     | **vLLM** and **llama.cpp**, each started once per trial (`vllm serve`, `llama-server`) and measured from the host by a shared HTTP client; the layer above them is engine-neutral |
| Benchmark client     | **`httpx2`** async HTTP client, driving the server's OpenAI-compatible API from the host with an injectable transport and clock |
| Container runtime    | **Podman** via the [`podman`](https://pypi.org/project/podman/) Python bindings (CDI for CUDA, device nodes for ROCm, none for CPU) |
| Config format        | **PyYAML** (`yaml.safe_load`) for the search-space and output-config files, and for rendering the Compose artifact |
| Model fetching       | [`huggingface_hub`](https://pypi.org/project/huggingface-hub/) snapshot download for vLLM, single-file GGUF download for llama.cpp |
| Terminal UI          | [`plotext`](https://pypi.org/project/plotext/) for the live best-so-far graph or the prefill-versus-decode scatter with the Pareto front highlighted |
| Testing              | **pytest** (unit tests use fakes; `integration`-marked tests need a Podman daemon and are deselected by default) |
| Linting / formatting | **Ruff** (line length 100, `py314` target)                                  |
| Type checking        | **ty** (checks the library sources in `src/`; `tests/` is excluded)        |
| Dev environment      | Nix flake (`nix develop`) providing Python and uv; run `uv sync` to install the Python tooling (ruff, ty, pytest) |

## Architecture

The codebase is layered so engine details stay at the edges:

- `search_space/`: Engine-neutral definition types, the tunable registry, the
  YAML loader, and validation.
- `inference/`: The engine-neutral experiment surface (the `InferenceBackend`
  `Protocol` and the `ServerSpec` a backend plans) plus the `VLLMBackend` and
  `LlamaCppBackend`, the host-side benchmark client, and hardware detection. A
  backend reports the tunables applicable to the run's fixed `LLMConfig`
  (`applicable_parameters(llm)`), validates a space and that configuration
  (`validate(space, llm, hardware)`), validates a resolved checkpoint, plans a
  `ServerSpec`, and parses results; the client waits for the server's health —
  failing the run as soon as the container exits during that wait — takes the
  warmup and measured runs, snapshots the spec's declared metrics path around
  each one, and aggregates the metrics. A second engine is another
  implementation of the protocol.
- `runtime/`: Backend-agnostic container types, the detached start/stop
  lifecycle a host-driven run uses, and the Podman client (plus SELinux
  auto-detection).
- `optimization/`: The engine-neutral objective model (`ObjectiveMode`,
  `ObjectiveWeights`, and the scoring functions) and the Optuna adapter that
  maps spec kinds to suggestions, scores each trial, drives a study, and guards
  its recorded mode.
- `output/`: Engine-neutral winning-configuration writers and their registry.
  `OutputFormat` names the artifact styles (`yaml`, `quadlet`, `compose`),
  `RenderContext` carries the recommended experiment with the resolved target
  and image plus an optional planned `ServerSpec`, and `ConfigWriter` is the
  pure-renderer protocol (`requires_plan`, `render`). The YAML writer renders
  the historical three-domain experiment document from the config alone; the
  quadlet and Compose writers reproduce the planned server — image, command,
  environment, mounts, devices, resources, SELinux options, and a `curl`
  healthcheck. The package imports no backend, no engine module, and no
  container runtime implementation, so engine vocabulary cannot leak into it.
- `cli/`: The composition root: argument parsing, base-config construction,
  model and draft resolution through the backend's `model_request` hook, the
  log and TUI reporters, the summary, output-writer selection (including the
  planning an executable format needs), and the exit-code mapping.

Because the definition and experiment surfaces import no engine, the same
search space and validation workflow can target a different inference backend
later without changing the file format. The search space is total over the
selected backend's applicable set **for the run's fixed configuration**, so the
engine — not the file format — decides which parameters a definition must
declare. A parameter can be applicable only under a configured setup: the two
speculative tunables (`speculative_mode` and `num_speculative_tokens`) are
required exactly when the run fixes a speculative method or draft model, and
rejected otherwise. Each backend keeps its vocabulary — flags, value spellings,
checkpoint handling, metric series, and default images — in its own module, and
the host client only fetches and parses the Prometheus exposition a spec's
`metrics_path` names.

Model resolution also follows the backend's rules rather than the CLI's. The
composition root asks `model_request` what the main model argument refers to,
resolves or downloads it, records the result on `BenchmarkConfig.resolved_path`,
and does the same for the configured draft model, recording it on
`LLMConfig.resolved_draft_path`. Both paths are resolution artifacts that the
backend mounts at plan time and never forwards as an engine argument; a
draft-requiring method without a draft, or a draft passed to the model-free
`ngram` method, is rejected as a usage error before preparation starts. The
resolved draft is validated where the backend can inspect it cheaply — for
example, llama.cpp rejects a non-GGUF draft through its metadata reader before
the image is pulled — and a space that can sample speculation off still resolves
the draft once per run, because resolution does not consult the space.

## Data flow

`cli.main` loads and engine-agnostically validates the space against the
selected backend's `applicable_parameters(base.llm)`, asks the backend to
`validate` it, builds the base configuration, resolves or downloads the main and
configured draft models through the backend's `model_request` rules, validates
the resolved checkpoints, and pulls the image. A JSONL dataset is tokenized once
through the backend's declared tokenizer endpoint before the sweep when the
engine needs a request-length cap. `optimization.optimize` then drives the study:
Optuna samples a point → the backend plans an engine-neutral `ServerSpec` →
`runtime` starts the server container detached and resolves its host port → the
host benchmark client waits for health, takes the warmup and measured runs
(scraping the spec's metrics path around each measured run when one is declared),
and aggregates them → `runtime` stops the container → the backend parses the
outcome and enriches it with the phase rates its engine reported → the selected
objective maps it to the trial value → the reporter callback runs.

After the sweep, or on an interrupt, the finish path prints the summary and, when
`--output-config` is set, selects the writer for `--output-format`. YAML is
rendered from the recommended config plus the resolved target and image alone and
never plans. The executable formats first re-plan the recommended config through
the engine's deployment backend, which omits trial-only resources such as vLLM's
temporary compile cache, then render the planned server and write the text. The
composition root owns the filesystem write, so the output layer stays a pure
renderer, and a planning or write failure exits non-zero on both finish paths.
