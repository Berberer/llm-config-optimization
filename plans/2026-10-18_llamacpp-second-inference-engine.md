# llama.cpp as a Second Inference Engine

## Problem Statement

The optimizer only knows how to benchmark vLLM. Everything below the search
space assumes it: a trial runs one container that starts `vllm serve` and drives
it with `vllm bench serve`; the CLI hard-codes `awq/gptq/fp8` as quantization
choices and a `--gpu-memory-utilization` knob; the search space demands all
seven tunables; and the model argument is resolved to a Hugging Face snapshot
directory. A user who wants to tune a llama.cpp serving configuration cannot do
so with this tool at all.

Worse, the current measurement path cannot simply be reused. `vllm bench serve`
lives inside the vLLM image, so a llama.cpp trial has no equivalent client: the
official `llama-server` image contains only the server binary. Any solution that
keeps measuring from inside an engine image must build and publish a custom
image per engine, and the two engines' workloads drift apart.

The search space compounds the problem. It is deliberately total — every tunable
must be declared — and the vLLM backend rejects any declared parameter it does
not forward. llama.cpp has no `--dtype` (GGUF carries its own quant) and no
`tensor_parallel_size`, so a llama.cpp space can neither omit those parameters
(the loader rejects the omission) nor declare them (the backend rejects the
declaration). A single YAML file cannot target both engines today.

Finally, llama.cpp is GGUF-only. Its quantization is a property of the file, not
a server flag, so the existing `--quantization` semantics do not apply, and its
checkpoint is a single file rather than a snapshot directory.

## Solution

Add llama.cpp as a second inference engine, selected by a new mandatory third
positional argument, and move all benchmarking to the host so that both engines
share one measurement client.

The user invokes the tool as `MODEL SPACE ENGINE`, where `ENGINE` is `vllm` or
`llamacpp`. Everything else about the command line is unchanged. A llama.cpp
sweep tunes the parameters that llama.cpp actually exposes
(`max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `kv_cache_dtype`,
`attention_backend`) and leaves `dtype` and `tensor_parallel_size` out, because
the selected backend declares which tunables apply.

Measurement moves out of the container. A trial now starts the engine's server
in a container with a published port, waits for its health endpoint, and drives
it from the host with one shared, engine-neutral HTTP client. The container
contains only the server; no engine image needs to carry a benchmark client, and
both engines are measured by identical logic (random prompts by default, or a
JSONL dataset supplied on the command line). The cache-reset and health
endpoints an engine exposes are described by the backend, so the orchestrator
stays engine-free.

Engine-specific CLI flags stay honest. `--quantization` and
`--gpu-memory-utilization` remain meaningful for vLLM and are rejected as usage
errors for llama.cpp, whose GGUF quant is read from the checkpoint and whose
VRAM is sized by different flags. A llama.cpp checkpoint is resolved from a
local `.gguf` file or a Hugging Face GGUF repository with an optional `:QUANT`
selector, validated by reading its metadata before any image is pulled.

Because this refactor rewrites how every trial runs, it is delivered in two
stages: first the host-driven core with vLLM migrated onto it (behavior
preserved), then the llama.cpp backend on top.

## User Stories

1. As an LLM-optimization user, I want to choose the inference engine on the
   command line, so that I can optimize either vLLM or llama.cpp with one tool.
2. As an LLM-optimization user, I want the engine to be a required positional
   argument, so that I cannot run the wrong engine by omitting it.
3. As an LLM-optimization user, I want an invalid engine name rejected as a
   usage error before any expensive work, so that a typo fails fast.
4. As an LLM-optimization user, I want the model and space to stay the first two
   positionals, so that my existing command lines change as little as possible.
5. As an LLM-optimization user, I want both engines to accept the same
   search-space file format, so that I do not learn a second definition
   language.
6. As an LLM-optimization user, I want a space to omit parameters the selected
   engine does not support, so that I do not declare meaningless values.
7. As an LLM-optimization user, I want the loader to report a missing parameter
   the selected engine requires, so that I fix the definition before the sweep.
8. As an LLM-optimization user, I want the backend to reject a declared value it
   cannot forward, so that a configuration is never silently ignored.
9. As a llama.cpp user, I want to sweep `max_model_len`, `max_num_seqs`,
   `max_num_batched_tokens`, `kv_cache_dtype`, and `attention_backend`, so that
   every declared parameter maps to a real `llama-server` flag.
10. As a llama.cpp user, I want `dtype` and `tensor_parallel_size` to be
    inapplicable, so that I am not asked to set parameters llama.cpp lacks.
11. As a llama.cpp user, I want `max_model_len` to configure the context size,
    `max_num_seqs` the parallel slots, `max_num_batched_tokens` the batch size,
    `kv_cache_dtype` the KV cache type, and `attention_backend` flash attention,
    so that a sweep changes the server the way its name promises.
12. As a llama.cpp user, I want unsupported enum values (for example an FP8 KV
    cache or an xFormers attention backend) rejected up front, so that a sweep
    does not fail halfway through.
13. As a llama.cpp user, I want to measure through the OpenAI-compatible server,
    so that the score reflects real serving throughput rather than an offline
    engine path.
14. As an LLM-optimization user, I want both engines measured by the same
    host-side client, so that their numbers are comparable.
15. As an LLM-optimization user, I want the container to run only the server, so
    that the benchmark client does not depend on tooling baked into an image.
16. As an LLM-optimization user, I want the server's port published to the host,
    so that the host client can reach it.
17. As an LLM-optimization user running parallel trials, I want each trial's
    server to receive its own host port, so that trials never collide.
18. As an LLM-optimization user, I want the client to wait for server health
    before measuring, so that startup time is not counted as throughput.
19. As an LLM-optimization user, I want a server that never becomes healthy to
    fail with a clear message, so that I can diagnose it.
20. As an LLM-optimization user, I want requests fired concurrently by default,
    so that batching and parallel-slot settings are actually exercised.
21. As an LLM-optimization user, I want warmup runs before the measured runs, so
    that first-run effects do not reach the score.
22. As an LLM-optimization user, I want a distinct seed per measured run, so
    that each repetition draws a fresh prompt set.
23. As an LLM-optimization user, I want the engine's cache-reset hook applied
    between runs when it exists, so that a repetition is not flattered by cached
    prompts.
24. As an LLM-optimization user, I want a trial's score to be the mean of its
    measured runs, so that transient noise is reduced.
25. As an LLM-optimization user, I want a strict-majority success rule, so that
    one failed run does not discard an otherwise valid configuration.
26. As an LLM-optimization user, I want the per-run metrics and the failure
    count kept on the result, so that I can inspect the spread behind the mean.
27. As an LLM-optimization user, I want the raw client payloads kept on the
    result, so that details the typed metrics omit remain inspectable.
28. As an LLM-optimization user, I want the server container's logs kept on the
    result, so that a failed trial can be debugged.
29. As an LLM-optimization user, I want random prompts of a configured length,
    so that I can sweep without preparing a dataset.
30. As an LLM-optimization user, I want to supply a custom dataset as JSONL with
    a prompt and an optional output length per line, so that I can measure a
    realistic workload.
31. As an LLM-optimization user, I want `--num-prompts` to cap how many dataset
    lines are used, so that I can control the size of a run.
32. As an LLM-optimization user, I want a clear usage error when I combine a
    non-random `--dataset-name` with `--dataset-path`, so that I never silently
    measure the wrong workload.
33. As a vLLM user, I want `--quantization` to keep its current meaning and
    choices, so that my existing workflows are unchanged.
34. As a llama.cpp user, I want `--quantization` rejected as a usage error, so
    that I do not pass a value that has no effect.
35. As a llama.cpp user, I want the checkpoint's quantization read from the GGUF
    metadata, so that model validation reflects the actual file.
36. As a llama.cpp user, I want a non-GGUF checkpoint rejected with a clear
    message, so that I catch a wrong model before an image is pulled.
37. As a llama.cpp user, I want `--gpu-memory-utilization` rejected, so that a
    vLLM-only knob is never silently ignored.
38. As a llama.cpp user, I want to point `--model` at a local `.gguf` file, so
    that I can use a checkpoint I already have.
39. As a llama.cpp user, I want to point `--model` at a Hugging Face GGUF
    repository, so that the tool downloads the right file for me.
40. As a llama.cpp user, I want to select a quantization in a multi-quant
    repository with `repo:QUANT`, so that I get the file I mean.
41. As a llama.cpp user, I want the tool to choose the only GGUF file when a
    repository holds exactly one, so that the common case needs no selector.
42. As a llama.cpp user, I want the tool to require a selector when a repository
    holds several GGUF files, so that it never guesses.
43. As a llama.cpp user, I want the official `llama-server` image for my target
    by default, so that I do not have to build an image.
44. As an LLM-optimization user, I want to override the image with `--image`, so
    that I can pin a specific build.
45. As an LLM-optimization user, I want the same exit codes for both engines, so
    that my scripts do not branch on the engine.
46. As an existing vLLM user, I want my studies and measurements to keep working
    after the refactor, so that the change does not invalidate my workflow.
47. As a developer, I want the runtime to expose a detached start/stop API, so
    that a host-driven run can manage a long-lived server container.
48. As a developer, I want the backend to describe its server through an
    engine-neutral spec, so that the orchestrator never learns an engine's
    vocabulary.
49. As a developer, I want the search-space layer to stay engine-neutral, so
    that adding an engine never puts engine names into the definition format.
50. As a developer, I want the runtime and HTTP client to be injectable fakes,
    so that unit tests need neither a daemon nor a GPU.

## Implementation Decisions

### Engine selection and CLI

- The command gains a third positional `ENGINE` with `choices=("vllm",
  "llamacpp")`, so `MODEL SPACE ENGINE` is the invocation. argparse rejects an
  unknown engine as a usage error before any work.
- A small engine registry maps the CLI token to a backend factory. The registry
  lives in the composition root; the CLI does not otherwise know engine
  internals.
- `--quantization` and `--gpu-memory-utilization` become engine-scoped. Their
  allowed values/semantics are validated against the selected backend after
  parsing; for `llamacpp` both are usage errors. For `vllm` they keep today's
  behavior and choices.
- No other CLI flags are added or removed. Concurrency stays unbounded (all
  `--num-prompts` requests in flight), matching today's `vllm bench serve`
  defaults. `--dataset-path` gains a defined JSONL contract.

### Host-driven measurement

- The container runs only the engine server. Its port is published to the host,
  and the host client talks to it over HTTP.
- The runtime gains a detached lifecycle: a container can be created and started
  without waiting for exit, returning a handle that exposes the resolved host
  port(s), the container id, log access, and a stop/remove operation. The
  existing blocking run helper stays for callers that want it.
- `ContainerConfig` gains a `ports` field mapping container port specifications
  to host ports, with `None` requesting a runtime-assigned free port. The Podman
  backend forwards this to the Podman `ports` parameter and reads the resolved
  host port back from the started container, so parallel trials never contend
  for a fixed port.
- A new engine-neutral `ServerSpec` describes a server run: the container
  configuration (command, ports, mounts, environment, devices), the health path,
  the engine's cache-reset paths (possibly empty), and the health-wait timeout.
- A new host benchmark client performs the whole measurement: wait for health,
  take warmup runs, take the configured number of measured runs, reset the
  engine caches between runs when the spec declares reset paths, seed each run
  distinctly, and aggregate the successful runs. It returns the per-run metrics,
  the failure count, and the raw payloads.
- The client is asynchronous and uses `httpx`, added as a runtime dependency. It
  uses a bounded connection pool and a semaphore so the in-flight count is the
  configured concurrency (unbounded by default).
- The workload is random prompts by default: `num_prompts` requests of
  `input_len` tokens asking for `output_len` tokens. When `--dataset-path` is
  set, the client reads JSONL, one object per line, with a required `prompt`
  string and an optional `output_len` integer that falls back to `--output-len`;
  `--num-prompts` caps the number of lines used and `--input-len` is ignored.
  Combining a non-`random` `--dataset-name` with `--dataset-path` is a usage
  error.
- Metrics keep today's meaning: `tokens_per_second` is total (prompt plus
  generated) tokens over elapsed wall-clock, `requests_per_second` is completed
  requests over elapsed, `elapsed_time` is the measured duration, and
  `num_prompts` is the completed request count.

### Backend protocol and orchestrator

- `InferenceBackend` is reshaped:
  - `applicable_parameters` reports the tunable names the engine supports.
  - a model-resolution hook returns an engine-neutral request describing what to
    fetch (a local path, or a repository id plus an optional file selector),
    which the composition root turns into a local path using the shared
    download/progress machinery.
  - `plan` returns a `ServerSpec` instead of a bare `ContainerConfig`.
  - `parse_result` receives the already-typed runs, the failure count, the
    `ServerSpec`, and the server `ContainerResult`, and assembles the
    `ExperimentResult`.
  - `validate`, `validate_model`, and `image_for` keep their roles.
- The orchestrator owns the single generic lifecycle: resolve the model, start
  the server container, run the client, stop the container, and parse. It knows
  no engine vocabulary; health and reset paths come from the spec.
- The vLLM backend is migrated onto this path: the generated in-container runner
  and the in-container `vllm bench serve` invocation are removed, and the server
  command becomes `vllm serve` bound to the container's published port with the
  dev-mode environment that enables the cache-reset endpoints. Measurement
  semantics and result fields are unchanged for the user.
- llama.cpp is implemented as a second backend behind the same protocol.

### Search space

- The search-space layer stays engine-neutral: no engine name ever appears in
  the definition format or the registry.
- `load_experiment_space` takes the set of tunables the caller requires
  (defaulting to all seven, so engine-agnostic callers and tests are unchanged)
  and reports exactly those that are missing. The composition root passes the
  selected backend's `applicable_parameters`, so "total" means "total over the
  backend's set".
- The vLLM backend declares all seven tunables; the llama.cpp backend declares
  the five it forwards. A value the backend cannot translate is still rejected
  by `validate`.

### llama.cpp specifics

- The llama.cpp backend maps engine-neutral fields to real `llama-server` flags:
  `max_model_len` to `--ctx-size`, `max_num_seqs` to `--parallel`,
  `max_num_batched_tokens` to `--batch-size`, `kv_cache_dtype` to
  `--cache-type-k`/`--cache-type-v`, and `attention_backend` to `--flash-attn`.
- Value translation is curated. For `kv_cache_dtype`, `float16` maps to `f16`
  and `bfloat16` to `bf16`; `auto` and the FP8 members are unsupported (use
  `default: true` for the engine default). For `attention_backend`,
  `flash_attention` maps to `on`; `flash_infer` and `xformers` are unsupported.
- `dtype` and `tensor_parallel_size` are not applicable and are absent from the
  backend's tunable set.
- Default images are the official llama.cpp server images, one per accelerator
  class (`server`, `server-cuda`, `server-rocm`), and `--image` overrides them.
- Model resolution accepts a local `.gguf` file or a Hugging Face repository id
  with an optional `:QUANT` (or filename) selector. A repository with exactly
  one GGUF file needs no selector; a repository with several requires one. The
  resolved file is mounted into the server and validated by reading its GGUF
  metadata.
- `validate_model` rejects a checkpoint that is not GGUF, reports the detected
  quantization, and rejects `--quantization` (which is also a CLI usage error)
  so the behavior is consistent whichever layer sees it first.
- Engine-specific numeric bounds are added to the backend's validation as
  needed; the engine remains the final authority at load time.

### Sequencing

- Stage 1 lands the host-driven core and migrates vLLM onto it with behavior
  preserved, leaving the build and tests green.
- Stage 2 adds the llama.cpp backend, its GGUF resolution and mappings, the
  engine-scoped CLI rules, and its documentation on top of the same core.

## Testing Decisions

A good test asserts externally observable behavior: the settings a parser
produces, the arguments a backend assembles for a server command, the spec a
backend plans for given hardware, the problems a validator reports, and the
metrics a client derives from injected responses. Tests must not assert on
private helpers, on the shape of generated scripts, or on the number of internal
calls. The runtime and the HTTP client are injected as fakes so unit tests need
neither a Podman daemon nor a GPU; integration tests carry the `integration`
marker and are skipped by default.

Modules to test:

- The argument parser: the third positional, its rejection of unknown engines,
  and that all existing defaults are unchanged.
- The engine registry/composition root: backend selection per token, and the
  engine-scoped rejection of `--quantization` and `--gpu-memory-utilization`.
- The search-space loader: a required-set argument, missing-tunable reporting
  against that set, and the unchanged default behavior.
- The vLLM backend: server command assembly, `ServerSpec` health/reset paths,
  image selection, and result parsing from typed runs.
- The llama.cpp backend: argument assembly for each mapped tunable, value
  translation (including rejection of unsupported members), applicable-set
  declaration, image selection, GGUF resolution (single vs multi-file repos,
  local files, non-GGUF rejection), and metadata validation.
- The host benchmark client: random and JSONL workloads, warmup/seed handling,
  concurrency, cache-reset calls, majority rule, and metric aggregation, driven
  by a fake HTTP transport and a fake clock.
- The runtime: detached start/stop, port forwarding and resolved-port read-back,
  and cleanup on failure, driven by a fake Podman client.

Prior art: the existing argument, search-space loader/registry, vLLM
argument/plan/result, orchestration, and runtime tests already establish these
patterns (direct parser tests, fake Podman clients, injected hardware and
clients, and validator-problem assertions). New tests follow the same style and
extend the existing parametrizations rather than duplicating them.

## Implementation Tasks

Stage 1 — host-driven core and vLLM migration:

- [x] **T1 - Add port publishing and a detached container lifecycle to the runtime**: Extend the container configuration with a `ports` field (container port to host port, `None` for a runtime-assigned port) and add a detached start/stop API returning a running-container handle with resolved ports, log/result access, and removal. Forward ports to Podman and read the assigned host port back. Keep the existing blocking run helper and its tests working; add fake-client tests for the new lifecycle and port read-back.
- [x] **T2 - Add the engine-neutral server spec type**: Introduce a `ServerSpec` that pairs a server container configuration with a health path, cache-reset paths, and a health-wait timeout, exported from the inference package's engine-neutral surface. Add type-level tests.
- [x] **T3 - Add the host-side benchmark client**: Implement the asynchronous client that waits for health, takes warmup and measured runs, resets caches between runs when paths are given, seeds each run, fires requests concurrently, and aggregates metrics. Support the random workload and the JSONL `--dataset-path` contract. Use `httpx` (add the dependency) and make the transport and clock injectable. Test random and JSONL workloads, majority rule, metric aggregation, and reset calls with fakes.
- [x] **T4 - Reshape the backend protocol and migrate vLLM to the host-driven path**: Add `applicable_parameters` and the model-resolution hook, change `plan` to return a `ServerSpec`, and change `parse_result` to consume typed runs plus the server result. Rewrite the orchestrator to own the generic lifecycle. Migrate the vLLM backend: remove the generated in-container runner and in-container benchmark, emit a `vllm serve` server spec with a published port and the cache-reset endpoints, and preserve result semantics. Update the affected inference, vLLM, and orchestration tests so the suite is green.
  - Note: `applicable_parameters` and the model-resolution hook were deliberately left to T6 and T7 (the agreed scope for this task). T4 reshapes `plan`/`parse_result`, rewrites the orchestrator lifecycle, and migrates vLLM; it also drops the now-unused `output_dir` from the backend protocol and adds a `benchmark_client` injection seam threaded through `optimize` and `main`.
- [x] **T5 - Add the ENGINE positional and backend selection to the CLI**: Add the required `ENGINE` positional with `vllm` and `llamacpp` choices, introduce the engine registry mapping the token to a backend factory, and select the backend in the composition root. Only `vllm` is wired in this task. Update parser and CLI tests, including unknown-engine rejection.
  - Note: a registry factory is a context manager, so vLLM's per-run compile cache is owned by the engine entry rather than the CLI, and the startup header now names the selected engine. A CLI test pins that the not-yet-wired `llamacpp` token fails as a usage error before any preparation; T12 replaces that expectation when it registers the engine.
- [x] **T6 - Make the search space require a backend-declared applicable set**: Add a required-set argument to the loader (defaulting to all tunables) and have the composition root pass the selected backend's `applicable_parameters`. Have the vLLM backend declare all seven. Update loader, registry, and CLI tests, including missing-tunable messages against a reduced set.
- [x] **T7 - Add the engine-specific model-resolution hook**: Introduce the neutral model-resolution request (local path, or repository id plus optional file selector) on the backend protocol, move the per-engine decision there, and keep the shared download and progress rendering in the composition root. Generalize the download helper to a single file as well as a snapshot. Update the vLLM backend and its tests.
  - Note: the neutral request is `LocalCheckpoint | HubCheckpoint` behind `InferenceBackend.model_request`; `is_local_model` moved out of `cli/config.py` into the vLLM backend. The composition root records the resolved path on a new `BenchmarkConfig.resolved_path` (agreed with the user), so the backend plans from the checkpoint it has while `model` still names the served model; a direct library caller that leaves it unset keeps today's pass-through behaviour. The download helper gained an optional `filename` (`hf_hub_download`) with its own single-line progress rule, and the summary writer now renders a `Path` field as a string.
  - Follow-up fix: mounting the Hub snapshot directory directly left its relative symlinks dangling inside the container, so vLLM reported `Invalid repository ID or local directory specified: '/model'`. A second pass showed the links can hop twice — a Xet-backed download stores the real weights in the cache's shared `hub/blobs` store, so `model.safetensors` points at the repo's `blobs/` entry which points at `../../blobs/<hash>` — and the engine core then failed to load the weights. The vLLM backend now detects an HF cache snapshot (`<hub>/models--*/snapshots/<rev>` beside a shared `blobs/`), mounts the cache hub root, and points `--model` at `/model/models--*/snapshots/<rev>`, so both hops resolve; a regression test builds the two-hop layout.
- [x] **T8 - Update stage-1 documentation and examples**: Document the host-driven measurement model, the third positional, the engine-neutral `ServerSpec`, and the applicable-set rule in the README; update the architecture notes in `AGENTS.md`; keep the example spaces valid for vLLM.
  - Note: the README documents the required third `ENGINE` positional and names `vllm` as the only engine implemented in stage 1 (the llama.cpp workflow is left to T14). It also documents the JSONL `--dataset-path` contract, which the benchmark-client task implemented but no task had documented. `examples/space-all-fixed.yaml` carried a pre-existing typo (`attention_backend: {default: flash_attention}`) that failed the loader; it is now `fixed: flash_attention`, so both example spaces validate against the vLLM backend.

Stage 2 — llama.cpp backend:

- [x] **T9 - Add llama.cpp argument translation and value maps**: Implement the pure translation of engine-neutral tunables to `llama-server` flags and the curated value maps, with unsupported enum members rejected. Unit-test every mapped flag and every accepted and rejected value.
- [x] **T10 - Add GGUF checkpoint resolution and validation**: Implement resolution from a local `.gguf` path or a Hugging Face repository with an optional `:QUANT`/filename selector, including the single-file and multi-file repository cases and the non-GGUF rejection, plus GGUF metadata reading for quantization reporting. Unit-test with a fake downloader and temporary GGUF fixtures.
  - Note: resolution takes an injectable repository-file lister (defaulting to `huggingface_hub.list_repo_files`) rather than a fake download, because the GGUF file has to be selected before the shared download helper runs; a repository holding several GGUF files and no selector, a selector matching none or several files, and a repository with no GGUF files are all rejected as `ExperimentError`. `read_gguf_metadata` rejects a path that is not a regular file or does not start with the GGUF magic, and fails open (`None`) when the magic is present but the header cannot be parsed; it reports the canonical short tag (`Q4_K_M` etc.) of `general.file_type`. Both helpers live in `inference/_llamacpp.py` for T11 to wire into the backend; nothing is exported yet.
- [x] **T11 - Implement the llama.cpp backend**: Add the backend that declares its five applicable tunables, resolves models, validates the space and checkpoint, plans a `llama-server` `ServerSpec` with the right image, mounts, devices, and port, and assembles results. Unit-test validation problems, planning, and image selection.
  - Note: the backend lives in `inference/_llamacpp.py` beside the T9/T10 helpers and is exported as `LlamaCppBackend`. Planning mounts a resolved GGUF file read-only at `/models/model.gguf`, passes `--alias <model>` to pin the served name (the llama.cpp counterpart of vLLM's `--served-model-name`), publishes container port `8080`, and reports `/health` with empty `reset_paths` because llama.cpp exposes no cache-reset endpoint. A direct caller that leaves `resolved_path` unset still plans a runnable server for a Hub model: the plan emits `--hf-repo`/`--hf-file` and lets `llama-server` fetch the file itself, while the production path mounts the resolved file so the server needs no network. No numeric bounds were added to `validate`: upstream clamps `n_batch` to `n_ctx` and divides the context across `--parallel` slots rather than rejecting, so there is no engine rule to mirror (unlike vLLM's `max_num_batched_tokens >= max_num_seqs`).
- [x] **T12 - Register llama.cpp in the CLI and enforce engine-scoped flags**: Wire `llamacpp` into the engine registry, reject `--quantization` and `--gpu-memory-utilization` for it as usage errors while preserving them for vLLM, and confirm the shared exit-code mapping. Update CLI and composition-root tests.
  - Note: the registry became `_ENGINES: Mapping[str, _EngineEntry]`, pairing each token's context-manager factory with the CLI options that engine rejects; `_rejected_option` raises the usage error before the header, the base config, or any preparation runs, and it consults the entry even when a backend is injected. The T5 test pinning `llamacpp` as unwired was replaced with a registry-selection test (the `LlamaCppBackend` name is monkeypatched with the shared fake), each scoped flag has a usage-error test asserting nothing is planned or pulled, a vLLM test pins that both flags still reach the planned experiment, and the empty-sweep exit-code test is now parametrized over both engine tokens to pin the shared mapping.
- [x] **T13 - Add llama.cpp end-to-end tests and close unit-coverage gaps**: Give llama.cpp the test coverage vLLM has. Integration: add a backend module driving `run_experiment` through Podman against the official `ghcr.io/ggml-org/llama.cpp:server` image with a small GGUF checkpoint (a Hub repository selected with `:QUANT`, or a local `.gguf`), a ROCm module mirroring `test_vllm_rocm_integration.py` that skips without a usable AMD stack, and a CLI integration test driving `main` with `ENGINE=llamacpp`, a space declaring only the five applicable tunables, and no engine-scoped flags. Mirror `test_vllm_integration.py` and `test_cli_integration.py` for module-scoped fixtures, skip conditions (no Podman daemon, unpullable image, uncached and unreachable model), and assertions on the observable result rather than throughput. Unit parity: audit `tests/inference/test_llamacpp_*.py` against the vLLM unit suites and add cases for any behaviour vLLM pins that llama.cpp does not, so the llama.cpp unit suites are at least as broad. Every new integration test carries the `integration` marker and is run with `uv run pytest -m integration`.
  - Note: the integration modules are `tests/inference/test_llamacpp_integration.py` (CPU), `tests/inference/test_llamacpp_rocm_integration.py`, and `tests/cli/test_cli_llamacpp_integration.py`. The checkpoint is `ggml-org/test-model-stories260K:F32` — llama.cpp's tiny test model, 1.2 MB — and the clients pin `vocab_size=512` because the checkpoint's tokenizer is smaller than the shared client's 32000-token default. The backend module passes the downloaded file as `resolved_path` (the production mount path); the CLI module resolves the `:F32` selector through the real download helper and asserts the written YAML, with a space declaring exactly the five applicable tunables and no engine-scoped flags. All four tests ran green on this host, including the ROCm one, so no skip path was exercised and the first run pulled both images (~23 minutes). Unit parity added the two behaviours the vLLM suites pin and the llama.cpp ones did not: the full `--model`/`--alias`/`--host`/`--port` command prefix (plus the Hub-fallback command's alias and binding), and a classification test over every `LLMConfig` field. That classification pins `seed` as deliberately not forwarded: the host client's greedy workload (`temperature: 0.0`) makes an engine-side seed irrelevant, so forwarding it would be a behaviour change rather than a coverage gap.
- [x] **T14 - Add llama.cpp examples and documentation**: Add or update example spaces that declare only the llama.cpp-applicable tunables, and document the llama.cpp workflow, GGUF model argument, engine-scoped flags, and default images in the README and `AGENTS.md`.
  - Note: a llama.cpp pair (`examples/space-llamacpp-full.yaml`, `examples/space-llamacpp-all-fixed.yaml`) was added beside the existing vLLM pair, which was renamed to `examples/space-vllm-full.yaml` and `examples/space-vllm-all-fixed.yaml` so every example names its engine; a single YAML cannot satisfy both engines' applicable sets. The README gained an *Engines* section with the GGUF model argument (local `.gguf`, Hub repository, `:QUANT` selector), the default image per engine and target, the vLLM-only `--quantization`/`--gpu-memory-utilization`/`--llm-seed` behaviour, and the llama.cpp flag mapping, plus per-engine updates to the space format, examples list, validation, example calls, and tech-stack/architecture sections. AGENTS.md's architecture table, engine-spelling, model-resolution, and `validate_model` invariants, the new-engine recipe, and the new-tunable checklist now account for two backends. All four example spaces were loaded and validated against their backends.
- [x] **T15 - Run the full toolchain and fix fallout**: Run `ruff format`, `ruff check`, `ty check`, and `pytest`, and resolve every failure so the library sources are clean and the suite passes.
  - Note: the only fallout was in the integration suite: `tests/optimization/test_optimize_integration.py` never injected the real host benchmark client, so the root `conftest.py` autouse fixture replaced it with `FakeBenchmarkClient` and every trial failed with `ExperimentError: the benchmark produced no successful run`. It now passes `benchmark_client=BenchmarkClient()` like the direct vLLM, CLI, and llama.cpp integration tests. `ruff format`, `ruff check`, `ty check`, and `pytest` (602 passed, 13 deselected) are green; all 13 integration tests have passed — 12 in the full `-m integration` run, and the fixed module on its own (543s).

## Out of Scope

- Building, publishing, or maintaining a custom container image. The solution
  uses the engines' official images.
- Reproducing `vllm bench serve`'s full dataset catalogue. Only the random
  workload and the JSONL `--dataset-path` contract are supported.
- Exposing a concurrency knob on the command line. Concurrency stays unbounded
  by default, matching today's behavior.
- llama.cpp features with no engine-neutral equivalent: `--n-gpu-layers`,
  `--split-mode`/`--tensor-split`, `--ubatch-size`, and speculative decoding are
  not modeled. They remain reachable only through the backend's escape hatch, if
  one is added.
- Adding an engine-neutral representation for "attention off"/"auto" or for FP8
  KV cache so llama.cpp could express them.
- Reworking the optimizer, the reporters, or the summary/writer beyond what the
  protocol change requires.
- Numeric comparability between engines. The client measures the same workload,
  but the engines differ internally.

## Further Notes

- The host-side client must reproduce today's vLLM measurement closely enough
  that existing studies remain meaningful: unbounded concurrency, warmup before
  measured runs, per-run seeds, cache resets between runs, and the same metric
  definitions.
- Whether llama.cpp's OpenAI-compatible `/v1/completions` accepts token-id
  prompts for an exact `input_len` must be verified during implementation; if it
  does not, the client falls back to text prompts and the effective input length
  is tokenizer-dependent. This is the main technical risk. **Verified during
  T11**: upstream's `oaicompat_completion_params_parse` copies `prompt` into the
  native params and the completion route tokenizes it with
  `tokenize_input_prompts`, whose `json_is_array_of_numbers` branch accepts a
  token-id array; `ignore_eos` and `max_tokens` are supported fields, so the
  shared client's payload measures the configured workload and no text-prompt
  fallback is needed.
- A single vLLM trial previously wrote a manifest and per-run JSON files; with
  the host-driven client the runs are available in memory, so file writing is
  optional and should be kept only if it adds debugging value.
- Random host ports are required for `--n-jobs` greater than one, since parallel
  trials run their own server containers.
- `--shm-size` keeps its `8g` default even though it is vLLM-oriented; it is
  harmless for llama.cpp.
- The search-space invariant is preserved in spirit: "total" now means total
  over the selected backend's applicable set, so a value still cannot silently
  fall back to an engine default.
- The HTTP dependency is published as `httpx2` in the pinned environment
  (import `httpx2`), not `httpx`; `huggingface-hub` already depends on it, and
  `httpx` is not installed. The client imports `httpx2` and the dependency is
  declared directly.
- T3's random workload samples token ids from `BenchmarkClient.vocab_size`
  (default 32000), so `input_len` is exact without a tokenizer; a backend that
  knows the real vocabulary can widen or narrow the bound. The llama.cpp
  fallback to text prompts was resolved in T11: the OpenAI-compatible endpoint
  accepts token-id arrays, so no fallback was added.
- The failed-run excerpt keeps the *earliest* exception line rather than the
  last: a vLLM engine-core failure prints the real cause first and a wrapper
  ("Engine core initialization failed. See root cause above") last, so the old
  last-line rule surfaced only the wrapper and dropped the cause. The trailing
  window still carries the wrapper and its traceback.
