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
| Config format        | **PyYAML** (`yaml.safe_load`) for the search-space and output-config files |
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
- `cli/`: The composition root: argument parsing, base-config construction,
  model and draft resolution through the backend's `model_request` hook, the
  log and TUI reporters, the summary/writer, and exit-code mapping.

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
