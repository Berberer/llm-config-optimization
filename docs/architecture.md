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

- `search_space/`: Engine-neutral definition types — including the
  `SpaceAssessment` result a backend returns for a resolved model — the tunable
  registry, the YAML loader, and validation. Each tunable's kind is derived from
  its `LLMConfig` annotation — enum, integer, float, boolean, or an int-or-enum
  union such as `n_gpu_layers` or `n_cpu_moe` — so `interval` constrains only a
  numeric branch and booleans stay strict. Float is a first-class kind: a
  `fixed`, `categorical`, or `interval` value is parsed as a real number (an
  integer is accepted and stored as a float) without letting a boolean satisfy a
  numeric bound, and an interval over a float field admits fractional
  endpoints.
- `inference/`: The engine-neutral experiment surface (the `InferenceBackend`
  `Protocol` and the `ServerSpec` a backend plans) plus the `VLLMBackend` and
  `LlamaCppBackend`, the host-side benchmark client, and hardware detection. A
  backend reports the tunables applicable to the run's fixed `LLMConfig`
  (`applicable_parameters(llm)`), validates a space and that configuration
  (`validate(space, llm, hardware)`), validates a resolved checkpoint
  (`validate_model`), assesses the declared space against it (`assess_space`),
  plans a `ServerSpec`, and parses results; the client waits for the server's
  health — failing the run as soon as the container exits during that wait —
  takes the warmup and measured runs, snapshots the spec's declared metrics
  path around each one, and aggregates the metrics. Categorical fields hold
  engine-neutral `StrEnum`s declared in the same module — `DType`,
  `QuantizationMethod`, `KVCacheDType`, `AttentionBackend`, `GpuLayersMode`,
  `CpuMoeMode`, `All2AllBackend`, `LoadMode`, `FitMode`, `SpecDraftSampling`,
  and the speculative method and mode — while an exact count stays a plain
  integer in a union; each backend maps only the members it serves, so no
  engine's spelling appears here. A second engine is another implementation of
  the protocol.
- `runtime/`: Backend-agnostic container types, the detached start/stop
  lifecycle a host-driven run uses, and the Podman client (plus SELinux
  auto-detection).
- `optimization/`: The engine-neutral objective model (`ObjectiveMode`,
  `ObjectiveWeights`, and the scoring functions) and the Optuna adapter that
  maps spec kinds to suggestions, scores each trial, drives a study, and guards
  its recorded mode. A float interval is sampled with a float suggestion and a
  float choice with a categorical suggestion, while reconstruction dispatches
  on the declared kind, so a resumed study's float is not truncated to an
  integer.
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
declare. A parameter can be applicable only under a configured setup, and the
registry declares two such gates. The two speculative tunables
(`speculative_mode` and `num_speculative_tokens`) are required exactly when the
run fixes a speculative method or draft model and rejected otherwise. The six
draft-only tunables (`spec_draft_n_min`, `spec_draft_p_split`,
`spec_draft_sampling`, `n_gpu_layers_draft`, `n_cpu_moe_draft`, and
`draft_kv_cache_dtype`) are required exactly when the run names a separate
draft model, because methods such as n-gram and the target's own MTP heads do
not use one. Both backends subtract both gated sets from their base applicable
set and add back only the gated parameters they actually forward, intersected
with their own forwarded-parameter table, so gating cannot drift between
engines and vLLM never requires a llama.cpp-only field. Each backend keeps its
vocabulary — flags, value spellings, checkpoint handling, metric series, and
default images — in its own module, and the host client only fetches and parses
the Prometheus exposition a spec's `metrics_path` names.

The fit family is the one deliberate exception to "`None` leaves it to the
engine": the command line pins `--fit off` by default, because llama.cpp's own
default lets the engine rewrite arguments to fit the device and a trial would
then measure something other than the point the search space declared. The
family is fixed for a run and stays out of the tunable registry, since sweeping
a correctness pin could undo the guarantee it exists to provide. `fit` selects
the engine-neutral `FitMode`, while `fit_target` and `fit_ctx` carry an
optional MiB headroom and minimum context; `--fit on` opts back into auto-fit,
`--fit-target` and `--fit-ctx` require it and are rejected as a usage error
otherwise, and a hand-built configuration that leaves fit unset or off
suppresses them rather than forwarding flags the engine would ignore.

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
llama.cpp's `n_gpu_layers` validation follows the same split: a CPU target
rejects `all` and positive exact counts before the run, while a count above the
resolved checkpoint's GGUF block count is caught once the metadata is readable,
failing open otherwise. Its KV-cache validation rejects a space that can pair a
quantized `kv_cache_dtype` with `attention_backend: off`, because llama.cpp
refuses to build a quantized V cache without flash attention; the same rule
applies to `draft_kv_cache_dtype` when the run names a draft model. Its
flag-level validation also rejects values the engine's parser would refuse at
startup, naming the value and the permitted range: `poll` outside `0…100`,
`prio` outside `-1…3`, `spec_draft_p_split` outside `0.0…1.0`, and
`cache_ram` below `-1`. vLLM's space validation likewise rejects an explicit
`float16`/`bfloat16` cache when `dtype` pins exactly one different explicit
precision, since the engine stores a non-quantized KV cache in the model dtype;
`auto`, deferred, and multi-value dtype pairs are reconciled once the
checkpoint is readable.

Checkpoint-aware work is split between two backend hooks. `validate_model`
rejects a configuration the checkpoint cannot serve — a quantization method it
was not produced with, a non-GGUF file, an exact `n_gpu_layers` count above the
GGUF architecture's block count — and returns problems rather than raising.
`assess_space` then turns the declared space into the effective one the sweep
runs: dimensions the resolved model, hardware, or fixed configuration make
meaningless are removed, and a `fixed` or `categorical` dimension can be
narrowed to the declared values the checkpoint can actually run. Every change
is one note, and a mismatch the assessment cannot fix is reported through the
same unsupported-configuration path. A dimension whose every declared value is
removed is such a problem rather than a silent fallback to an engine default,
while a space with no dimensions left runs once at engine defaults.

The MoE rules are one family: vLLM reads `config.json` expert keys and
llama.cpp the GGUF `<arch>.expert_count`, so a confidently dense checkpoint
loses its MoE dimensions. llama.cpp also reads the sliding-window fact from
`<arch>.attention.sliding_window`, so a confidently full-attention checkpoint
loses `swa_full`. On a CPU target llama.cpp loses the offload dimensions that
cannot matter there — `n_cpu_moe`, `n_cpu_ffn`, `kv_offload`, `op_offload`, and
the draft model's `n_cpu_moe_draft` and `n_gpu_layers_draft` — plus
`n_gpu_layers` when every declared value already disables offload; a vLLM space
whose tensor-parallel ceiling is one loses expert parallelism with its
all-to-all transport. Retained layer counts are then bounded by the relevant
checkpoint's block count: `n_gpu_layers`, `n_cpu_moe`, and `n_cpu_ffn` by the
target checkpoint, and `n_gpu_layers_draft` and `n_cpu_moe_draft` by the draft
checkpoint. `n_cpu_ffn` has no `all` mode, so its hint asks for a smaller count
while the others point at `all`.
vLLM's `CheckpointFacts` reader also reports the preferred model dtype —
`dtype`/`torch_dtype`, with a nested `text_config` value winning — and whether
any layer type is `linear_attention`. Those facts drive a kernel rule, not a
supported-dtype allowlist: the declared dtype is the precision `auto` resolves
to, and a linear-attention checkpoint loses `float32` because its kernel
requires a half-precision activation. Explicit float KV-cache values are then
reconciled against the dtypes the narrowed dimension can still select, with
`auto` resolving through the checkpoint's declared dtype; `auto` and FP8 cache
values are always retained. llama.cpp has no equivalent rule — GGUF's
`general.file_type` describes weight quantization, not activation support.

Every rule that reads the checkpoint fails open: unreadable or incomplete
metadata leaves the affected dimensions as declared, so the engine stays the
final authority for a checkpoint the reader cannot see. The rules that decide
from the declared space and the detected hardware alone — vLLM's
tensor-parallel/expert-parallel dependency and llama.cpp's CPU-target pruning —
still apply, because they do not depend on checkpoint metadata. Detection and
rules live in the backend modules that own the engine's format and vocabulary;
the CLI only prints the notes and runs the effective space.

## Data flow

`cli.main` loads and engine-agnostically validates the space against the
selected backend's `applicable_parameters(base.llm)`, asks the backend to
`validate` it, builds the base configuration, resolves or downloads the main and
configured draft models through the backend's `model_request` rules, validates
the resolved checkpoints, assesses the declared space against them and the
detected hardware, and pulls the image. The assessment replaces the declared
space with the effective one and prints one note per dropped dimension or
narrowed value; a model-aware problem exits through the
unsupported-configuration path before the pull, and an effective space with no
dimensions runs exactly one trial at engine defaults. A JSONL dataset is
tokenized once through the backend's declared tokenizer endpoint before the
sweep when the engine needs a request-length cap.
`optimization.optimize` then drives the study: Optuna samples a point → the
backend plans an engine-neutral `ServerSpec` → `runtime` starts the server
container detached and resolves its host port → the host benchmark client waits
for health, takes the warmup and measured runs
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
