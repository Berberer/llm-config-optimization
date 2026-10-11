# vLLM Inference Benchmark Layer

## Problem Statement

The `llm-optimization` project can already run an arbitrary container from a declarative configuration, but it has no way to actually evaluate an LLM. To benchmark a model with vLLM today, a developer must hand-assemble a `podman run` invocation: pick the right vLLM base image for the host's accelerator, map the correct devices, remember the container flags vLLM needs (shared memory, HF cache mounts, SELinux relabeling), construct a `vllm bench throughput` command line, arrange for the benchmark JSON to be written somewhere it can be read back, and then parse the result by hand.

This is tedious and error-prone, and it repeats for every experiment. Worse, the parameters that matter for an evaluation — the model, the serving configuration, the workload — are scattered across shell history and ad-hoc scripts, so results are hard to reproduce and compare. As the project's purpose is to *optimize* LLM serving, the lack of a first-class way to run controlled experiments is the central gap.

The user wants to run experiments with different serving configurations against one model: the container setup and the benchmark workload (including the model) stay fixed across an experiment series, while the LLM configuration changes each run. The tooling should make that loop trivial, choose the right hardware mapping automatically, expose a broad and growing set of LLM parameters, and return a structured throughput measurement.

Crucially, the evaluation surface should not be welded to vLLM. The user may add other inference engines later, and experiment definitions must not be polluted with one engine's runtime specifics.

## Solution

Add an engine-agnostic **experiment layer** above the existing container runtime, with a pluggable **backend** that knows how to drive a specific inference engine. The first backend targets vLLM.

An experiment is described by a single immutable `ExperimentConfig` composed of three clearly separated domains:

- **Container runtime config** — host-facing concerns: hardware target override, image override, memory and shared-memory limits, pull policy, environment, and extra mounts.
- **Benchmark config** — the evaluation workload: the model under test, the dataset/workload shape, and the wall-clock timeout.
- **LLM config** — the serving parameters being varied between experiments, expressed as flat, engine-neutral, typed fields.

Running an experiment means calling `run_experiment(config)`. The layer detects the host hardware, asks the backend to plan a concrete container run (base image, device mapping, mounts, and the `vllm bench throughput` invocation), executes it through the existing runtime, and hands the raw result back to the backend to parse into a typed `ExperimentResult` carrying tokens per second.

Because the backend is a formal seam, the experiment types never mention vLLM. Adding another engine later means writing another backend, not changing how experiments are described. Because the config domains are separated, an experiment series is expressed naturally: hold the container and benchmark domains fixed and vary the LLM domain with `dataclasses.replace`.

The parameter→CLI-flag mapping is deliberately code-only and declarative inside the vLLM backend. Exposing a new parameter is an explicit two-step: add a typed field to the engine-neutral LLM config and add its mapping entry. Values are forwarded to vLLM either as flat CLI flags or as JSON group payloads (`--attention-config '{"backend": …}'`), matching vLLM's own conventions, including snake_case→kebab-case conversion and JSON encoding of nested values.

## User Stories

1. As an LLM-optimization developer, I want to run an LLM benchmark with a single Python call, so that I can measure serving performance without writing shell or Podman plumbing.
2. As an LLM-optimization developer, I want the host's accelerator to be detected automatically, so that I do not have to know which hardware I am on before running an experiment.
3. As an LLM-optimization developer, I want the correct vLLM base image chosen for the detected hardware, so that I do not have to look up image names and tags.
4. As an LLM-optimization developer, I want the correct device mapping applied for the detected hardware, so that GPU access works without hand-writing device flags.
5. As an LLM-optimization developer, I want to override the detected hardware target explicitly, so that I can correct a wrong probe or target a specific accelerator.
6. As an LLM-optimization developer, I want a broken or partial accelerator stack to fall back to CPU rather than misreporting the hardware, so that experiments still run and results are trustworthy.
7. As an LLM-optimization developer, I want to override the base image, so that I can test a newer or custom vLLM build without waiting for a library release.
8. As an LLM-optimization developer, I want to describe an experiment with one immutable configuration object, so that it can be stored, hashed, logged, and iterated over.
9. As an LLM-optimization developer, I want the configuration separated into container, benchmark, and LLM domains, so that the fixed and varied parts of an experiment are obvious.
10. As an LLM-optimization developer, I want the model and workload to live in the benchmark domain, so that a series of experiments against one model shares a single benchmark definition.
11. As an LLM-optimization developer, I want the serving parameters to live in the LLM domain, so that I can vary them per experiment with a simple field replacement.
12. As an LLM-optimization developer, I want the LLM domain to be flat and strictly typed, so that experiments read clearly and cannot silently carry engine-shaped junk.
13. As an LLM-optimization developer, I want a curated starter set of LLM parameters that are known to be supported by the benchmark CLI, so that I can trust that every exposed field actually does something.
14. As an LLM-optimization developer, I want to set the model data type, so that I can evaluate precision/throughput trade-offs.
15. As an LLM-optimization developer, I want to set the weight quantization method, so that I can evaluate quantized serving.
16. As an LLM-optimization developer, I want to set the KV-cache data type, so that I can evaluate KV-cache quantization.
17. As an LLM-optimization developer, I want to set the maximum model length, so that I can control the context window under test.
18. As an LLM-optimization developer, I want to set the maximum number of sequences, so that I can control serving batch size.
19. As an LLM-optimization developer, I want to set the maximum number of batched tokens, so that I can tune scheduling behavior.
20. As an LLM-optimization developer, I want to set the GPU memory utilization, so that I can control the memory footprint of a run.
21. As an LLM-optimization developer, I want the tensor-parallel size to default to the number of detected devices, so that multi-device hosts are used without extra configuration.
22. As an LLM-optimization developer, I want to override the tensor-parallel size, so that I can control sharding explicitly.
23. As an LLM-optimization developer, I want to enable or disable remote-code trust, so that I can run models that need custom code.
24. As an LLM-optimization developer, I want to set a random seed, so that benchmark runs are reproducible.
25. As an LLM-optimization developer, I want to select the attention backend, so that I can compare attention implementations.
26. As an LLM-optimization developer, I want at least one parameter forwarded through a JSON configuration group, so that the JSON-group mechanism is proven end-to-end.
27. As an LLM-optimization developer, I want parameters forwarded to the correct CLI flags automatically, so that I do not have to know the flag names or their grouping.
28. As an LLM-optimization developer, I want nested or list-valued parameters serialized as JSON, so that complex configuration values are passed correctly.
29. As an LLM-optimization developer, I want a raw engine-argument escape hatch on the backend, so that I can pass an unmodeled flag without polluting the experiment config.
30. As an LLM-optimization developer, I want to benchmark a model by its Hugging Face identifier, so that I can evaluate public models without pre-downloading them.
31. As an LLM-optimization developer, I want to benchmark a model from a local path, so that I can evaluate private or locally converted checkpoints.
32. As an LLM-optimization developer, I want a Hugging Face cache mounted into the container, so that repeated runs reuse downloaded weights.
33. As an LLM-optimization developer, I want to pass a Hugging Face token via the environment, so that I can benchmark gated models.
34. As an LLM-optimization developer, I want to choose the workload dataset and prompt shape, so that I can measure throughput under realistic or synthetic load.
35. As an LLM-optimization developer, I want the layer to impose no workload defaults of its own, so that I always know exactly what was measured.
36. As an LLM-optimization developer, I want to set a wall-clock timeout, so that a hung benchmark does not block forever.
37. As an LLM-optimization developer, I want the shared-memory size set to a sensible default, so that vLLM does not crash on an under-sized `/dev/shm`.
38. As an LLM-optimization developer, I want bind mounts relabeled for SELinux by default, so that the layer works on enforcing hosts.
39. As an LLM-optimization developer, I want the benchmark result written to a mounted file and parsed, so that parsing is robust against log noise.
40. As an LLM-optimization developer, I want a typed result with tokens per second, so that I can compare experiments directly.
41. As an LLM-optimization developer, I want the result to include requests per second, elapsed time, and prompt count, so that I can reason about the measurement.
42. As an LLM-optimization developer, I want the result to record the resolved target, image, and devices, so that I can trust and reproduce what ran.
43. As an LLM-optimization developer, I want the raw benchmark JSON and raw container output preserved, so that I can inspect details the typed result omits.
44. As an LLM-optimization developer, I want a single exception type for all benchmark-layer failures, so that I can handle errors uniformly.
45. As an LLM-optimization developer, I want that exception to carry the underlying container error and raw result, so that I can debug failures.
46. As an LLM-optimization developer, I want the experiment API to be engine-agnostic, so that the project can add other inference engines later without reshaping experiment definitions.
47. As an LLM-optimization developer, I want a formal backend interface, so that a new engine is a new implementation rather than a change to the core.
48. As an LLM-optimization developer, I want to select the backend at run time, so that I can target different engines.
49. As a future maintainer, I want all vLLM-specific knowledge confined to one backend module, so that the rest of the codebase is unaffected by vLLM changes.
50. As a future maintainer, I want the parameter→flag mapping to be declarative and in code, so that it is reviewable and typed.
51. As a test author, I want to inject a fake hardware probe, so that detection logic can be tested without specific hardware.
52. As a test author, I want to inject a fake backend, so that orchestration can be tested without an engine.
53. As a test author, I want to reuse the existing fake container client, so that backend planning and parsing can be tested without a Podman daemon.
54. As a CI maintainer, I want integration tests that skip gracefully when suitable hardware or images are unavailable, so that the suite passes everywhere.

## Implementation Decisions

**Package and layering**
- A new `inference` subpackage under the existing `llm_optimization` package, mirroring the `runtime` subpackage's split between shared types and a concrete backend.
- Three conceptual layers: an engine-agnostic experiment layer, an engine-specific backend layer, and the unchanged runtime layer.
- The subpackage contains: a shared types module, a hardware detection module, a vLLM backend module (the only vLLM-aware module), and an `__init__` re-exporting the public API.

**Public API**
- `run_experiment(config, backend=VLLMBackend()) -> ExperimentResult` is the entry point.
- An `InferenceBackend` protocol defines the seam with two operations: `plan(experiment, hardware) -> runtime.ContainerConfig`, which turns an experiment plus detected hardware into a concrete container run, and `parse_result(container_result, output_dir) -> ExperimentResult`, which turns the captured container outcome into a typed benchmark result.
- Hardware detection is performed by the orchestrator and passed into `plan`, so backends do not probe the host themselves and are unit-testable.
- The default backend is vLLM; callers may inject another backend.

**Configuration types**
- `ExperimentConfig` is a frozen dataclass composing three frozen sub-configs: `container`, `benchmark`, and `llm`.
- `ContainerRuntimeConfig` is the engine-neutral container domain: `target` (hardware override, `None` = auto-detect), `image` (override, `None` = backend default), `mem_limit`, `shm_size` (default 8g), `pull_policy`, `env`, and `volumes`.
- `BenchmarkConfig` is the workload domain: `model` (required; a Hugging Face id or a host path), dataset/workload fields, and `timeout` (default `None`). The layer imposes no workload defaults of its own.
- `LLMConfig` is a flat, strictly typed, engine-neutral domain with no free-form passthrough. Its initial fields are: `dtype`, `quantization`, `kv_cache_dtype`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `tensor_parallel_size`, `seed`, and `attention_backend`. Every field was verified to correspond to a real `vllm bench throughput` flag. The four categorical fields hold engine-neutral enums (`DType`, `QuantizationMethod`, `KVCacheDType`, `AttentionBackend`), not engine strings, so an experiment never carries one engine's vocabulary; each backend translates the members it supports to its own spellings and rejects the rest.
- All configuration dataclasses are frozen and hashable so experiments can be stored, compared, and used as cache keys.
- The three domains are separate types rather than one flat config, so the domain split is enforced by the type system.

**Backend mapping and serialization**
- The vLLM backend owns a declarative, code-only mapping from each `LLMConfig` field to a target descriptor. A target is either a flat flag or a JSON group plus key.
- The backend owns a second, parallel `VALUE_MAPS` table translating each engine-neutral enum member to vLLM's spelling (`AttentionBackend.FLASH_ATTENTION` → `FLASH_ATTN`). A member absent from a field's map is unsupported by this backend and raises `ExperimentError` rather than being silently dropped. The enums are curated starter sets, not exhaustive, because `quantization` and `attention_backend` are open value spaces in vLLM; `extra_args` covers the long tail.
- `attention_backend` is intentionally routed through the JSON group form (`--attention-config '{"backend": …}'`) to exercise the JSON path; the equivalent flat flag also exists.
- Serialization rules: snake_case field names become kebab-case flag names; nested dict and list values are JSON-encoded; booleans become `--flag` or `--no-flag` according to vLLM's conventions.
- The mapping lives only in code; there is no external mapping file.
- The backend accepts an optional raw `extra_args` sequence as its only escape hatch for unmodeled engine flags.

**Hardware detection and mapping**
- Detection probes the host for NVIDIA (CUDA), AMD (ROCm), and otherwise falls back to CPU. A vendor probe that finds a broken or partial stack falls back to CPU rather than reporting that vendor.
- Detection also determines the device count, which supplies the default `tensor_parallel_size` (1 on CPU).
- An explicit `target` on the container config overrides detection.
- Default base images use the `latest` tag and are overridable: CUDA → `vllm/vllm-openai`, ROCm → `vllm/vllm-openai-rocm`, CPU → a configurable default (no official CPU image exists).
- Device mapping per target: CUDA → `nvidia.com/gpu=all` (CDI); ROCm → `/dev/kfd` and `/dev/dri`; CPU → none.

**Execution flow**
- The backend's `plan` produces a runtime container configuration: the resolved image, device mapping, shared-memory size, environment, extra mounts, the Hugging Face cache mount, a local-model mount when the model is a host path, and an output-directory mount.
- The bench is invoked as `vllm bench throughput` with the model, the workload arguments, the mapped LLM parameters, and `--output-json` pointing into the mounted output directory.
- Bind mounts default to the SELinux `z` shared relabel option; callers can override per mount.
- The orchestrator runs the container through the existing `runtime.run_container` unchanged.
- After the run, the backend reads and parses the result JSON from the host output directory.

**Result and errors**
- `ExperimentResult` carries typed tokens per second, requests per second, elapsed time, prompt count, the resolved target/image/devices, the raw benchmark JSON, and the raw container result.
- A single `ExperimentError` covers every benchmark-layer failure: container failure (with the underlying container error as the cause), a missing result file, and an unparseable result file. It carries the raw container result when available.
- Sweeps are not built in: a series is a caller-side loop over configurations, with the LLM domain varied via `dataclasses.replace`.

## Testing Decisions

Good tests exercise external behavior and observable contracts, not implementation details. For this feature that means: given a configuration and injected collaborators (a fake hardware probe, a fake backend, or the existing fake container client), what container plan is produced, what command line is assembled, what typed result or exception does the caller observe, and is the failure cause preserved.

- **Hardware detection** is tested by injecting fake probes that report CUDA, ROCm, CPU, broken-stack, and multi-device hosts, asserting the resolved target and device count. No real hardware is required.
- **Mapping and serialization** are tested by constructing LLM configs and asserting the exact argument list produced, including flat flags, the JSON-group form, kebab-case conversion, boolean encoding, and JSON encoding of nested values.
- **Backend planning** is tested by asserting the produced container configuration: resolved image and devices per target, the shared-memory default, the bench command shape, the output-directory mount, the Hugging Face cache mount, and the local-model mount.
- **Result parsing and errors** are tested against well-formed, missing, and malformed result JSON, asserting the typed result and the `ExperimentError` payloads and causes.
- **Orchestration** is tested with a fake backend and the existing fake container client, asserting the call order and that hardware detection output is passed to the backend.
- **Integration tests** are marked `integration` and skip when Podman or suitable hardware/images are unavailable; on the current development host they are expected to skip.
- **Prior art**: the runtime layer's unit tests already establish the pattern of a hand-written fake Podman client driving behavior-level assertions without a daemon, and its integration tests establish the `integration` marker plus graceful skip convention. The new tests follow both.

## Implementation Tasks

[x] **T1 - Add the inference subpackage and hardware target enum**: Create the new `inference` subpackage with its shared types module and package init, and define the engine-neutral hardware target enum (CUDA, ROCm, CPU).
[x] **T2 - Implement hardware detection**: Add the detection probe that resolves the hardware target and device count, including the CPU fallback for broken vendor stacks and support for an explicit override.
[x] **T3 - Define the container runtime config domain**: Add the frozen `ContainerRuntimeConfig` dataclass with target override, image override, memory and shared-memory limits, pull policy, environment, and volumes.
[x] **T4 - Define the benchmark config domain**: Add the frozen `BenchmarkConfig` dataclass with the required model field, workload/dataset fields, and optional timeout, with no layer-imposed workload defaults.
  - Note: the plan left "dataset/workload fields" unnamed. Resolved with the user: `model` (required) plus `dataset_name`, `dataset_path`, `num_prompts`, `input_len`, `output_len`, and `timeout`, all defaulting to `None`.
[x] **T5 - Define the flat LLM config with the starter parameter set**: Add the frozen `LLMConfig` dataclass with the verified starter fields (dtype, quantization, kv_cache_dtype, max_model_len, max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, tensor_parallel_size, seed, attention_backend) and no passthrough.
  - Note: revised with the user after T10/T11 — the four categorical fields are engine-neutral `StrEnum`s (`DType`, `QuantizationMethod`, `KVCacheDType`, `AttentionBackend`) rather than `str`, so experiment definitions carry no engine vocabulary. The enums are curated starter sets; the backend maps members to engine spellings and rejects unsupported ones.
[x] **T6 - Define the experiment config wrapper**: Add the frozen `ExperimentConfig` dataclass composing the container, benchmark, and LLM domains.
[x] **T7 - Define the experiment result type**: Add the frozen `ExperimentResult` dataclass with throughput, requests-per-second, elapsed time, prompt count, resolved target/image/devices, raw JSON, and the raw container result.
  - Note: `tokens_per_second` maps directly to vLLM's JSON key of the same name, which is *total* token throughput (`total_num_tokens / elapsed_time`, i.e. prompt + generated tokens), not generation-only. Kept as-is and documented on the type; vLLM's summary JSON exposes no output-only throughput.
[x] **T8 - Define the experiment error type**: Add `ExperimentError` covering container failure, missing result file, and unparseable result file, carrying the raw result and preserving the cause.
[x] **T9 - Define the inference backend protocol**: Add the `InferenceBackend` protocol with `plan` and `parse_result`, including the injected hardware argument.
  - Note: `output_dir` is an optional parameter on both methods (`Path | None = None`) rather than a required one, resolved with the user: not every backend writes a result file. A backend that does not need one ignores it; a backend that does creates it when the caller supplies `None` and reads from the same directory in `parse_result`. This also gives `plan` the optional directory it needs to mount (T12), which the literal `plan(experiment, hardware)` signature could not supply.
[x] **T10 - Implement the vLLM parameter-to-CLI mapping**: Add the declarative, code-only mapping table from each `LLMConfig` field to a flat flag or JSON group target, with the raw `extra_args` escape hatch.
  - Note: extended with `VALUE_MAPS`, the per-field enum→vLLM-string table that keeps engine spellings out of the engine-neutral config.
[x] **T11 - Implement vLLM argument assembly**: Add the routine that turns an LLM config into CLI arguments using the mapping, applying kebab-case conversion, boolean encoding, and JSON encoding of nested values.
  - Note: no `LLMConfig` field is currently boolean- or container-valued, so the encoding rules live in a generic `encode_arguments(parameters, targets)` primitive with `build_arguments(config, extra_args=...)` as the typed routine. T17 can exercise boolean/nested encoding through the primitive; `extra_args` are appended verbatim by `build_arguments`.
  - Note: value normalisation is recursive and covers enums (plain `Enum` → `.value`, since `str()` on a non-str enum yields `ClassName.MEMBER`), `os.PathLike` → string, any `Mapping` → JSON object, `set`/`frozenset` → string-sorted JSON list, and any non-string `Sequence` → JSON list. This applies both to flat flags and inside a JSON group, so a nested enum/path/set serialises too. `FlatFlag`/`JsonGroup` remain the only target kinds: vLLM's `bench throughput` accepts scalars, booleans, and JSON payloads, with no separate repeatable/`nargs` form the JSON form cannot express.
[x] **T12 - Implement the vLLM container plan**: Add `VLLMBackend.plan` to resolve the base image and device mapping from the target, apply container defaults, mount the Hugging Face cache, the local-model path, and the output directory, and assemble the `vllm bench throughput` command with `--output-json`.
  - Note: the plan assumed no official CPU image exists; vLLM now publishes `vllm/vllm-openai-cpu`, so `VLLMBackend.cpu_image` defaults to `vllm/vllm-openai-cpu:latest` (still overridable).
  - Note: when no output directory is supplied, `plan` falls back to the module constant `DEFAULT_OUTPUT_DIR` under the system temp directory so `parse_result` (T13) can read back the same directory without extra state. The orchestrator (T14) is expected to pass a per-run directory; concurrent direct calls that both fall back to the default share it.
  - Note: follow-up found while doing T21 against a real image — vLLM's official images set `ENTRYPOINT ["vllm", "serve"]`, and a container runtime's `command` only replaces `CMD`, so the planned `vllm bench throughput` was being appended to `vllm serve` and never ran. `runtime.ContainerConfig` gained an optional `entrypoint` (forwarded by the Podman backend), and `VLLMBackend.plan` now sets `entrypoint=("vllm",)` with `command` starting at `bench throughput`. The default images are also registry-qualified (`docker.io/vllm/...`) because podman refuses to resolve a bare short name when it cannot prompt for a registry, as the runtime client runs without a TTY. The T12 plan tests were updated for both changes.
[x] **T13 - Implement the vLLM result parser**: Add `VLLMBackend.parse_result` to read the result JSON from the output directory, build the typed result, and raise `ExperimentError` on missing or malformed output.
  - Note: read `tokens_per_second`, `requests_per_second`, and `elapsed_time` verbatim from the JSON; the prompt count comes from the JSON's `num_requests` key, which vLLM names differently from the typed `num_prompts` field.
  - Note: revised with the user while implementing — `ExperimentResult` must record the resolved target/image/devices (T7), but T9's `parse_result(container_result, output_dir)` carried neither. The protocol signature is now `parse_result(result, plan, output_dir=None)`, where `plan` is the `ContainerConfig` returned by `plan`; the orchestrator (T14) passes it through. `ContainerConfig` has no target field, so the backend recovers the accelerator class by inverting its own `_DEVICE_MAPPINGS` from `plan.devices` (`_target_for`); a mapping the backend never emits raises `ExperimentError`.
  - Note: follow-up after T19 — `_target_for` now takes the captured result and attaches it as `raw_result`, so every `parse_result` failure carries the raw container output uniformly (previously the unknown-device path did not).
[x] **T14 - Implement experiment orchestration**: Add `run_experiment` to detect hardware, call the backend's `plan`, execute through the runtime, and delegate parsing, wrapping container failures in `ExperimentError`.
  - Note: `run_experiment` lives in a new private `_orchestrate` module (the plan's subpackage contents list omitted the orchestrator module; T15 re-exports the entry point). Its signature adds keyword-only `client` (forwarded to `runtime.run_container` so the orchestration tests can inject the existing fake client) and `output_dir`; `backend` is `None`-means-vLLM rather than the literal `VLLMBackend()` default because Ruff's `B008` rejects a call in an argument default. Resolved with the user: when no `output_dir` is supplied the orchestrator creates a per-run temp directory and removes it after parsing (the parsed JSON remains on the result as `raw_json`), while an explicit `output_dir` is used as-is and left in place.
[x] **T15 - Export the public API**: Re-export the experiment types, backend protocol, default backend, and entry point from the inference subpackage.
[x] **T16 - Unit-test hardware detection**: Add tests with injected fake probes covering CUDA, ROCm, CPU, broken stacks, multi-device hosts, and explicit overrides.
[x] **T17 - Unit-test mapping and argument assembly**: Add tests asserting exact argument lists for flat flags, JSON-group form, kebab-case conversion, boolean encoding, nested JSON encoding, raw extra args, engine-neutral enum value translation (including a non-identity mapping such as `FLASH_ATTENTION` → `FLASH_ATTN`), and the `ExperimentError` raised for an unsupported enum member.
[x] **T18 - Unit-test the vLLM container plan**: Add tests asserting the resolved image, devices, shared-memory default, mounts, and bench command for each target and for local versus Hub models.
[x] **T19 - Unit-test result parsing and errors**: Add tests for well-formed, missing, and malformed result JSON, asserting the typed result and the `ExperimentError` payloads and causes.
[x] **T20 - Unit-test orchestration**: Add tests with a fake backend and the existing fake container client asserting call order and hardware injection.
  - Note: reusing the runtime fake client across test packages required extracting it (with `_frame` and the manager doubles) from `tests/runtime/test_runtime.py` into a shared `tests/fakes.py`, importable because a new `tests/conftest.py` puts `tests/` on `sys.path`. `tests/runtime/test_runtime.py` now imports it instead of defining it; no test behaviour changed. Hardware detection is faked by monkeypatching `_orchestrate.detect_hardware`, since `run_experiment` exposes no probe seam; the tests assert the override and the detected hardware reach the backend.
[x] **T21 - Add marked integration tests**: Add `integration`-marked tests that skip when Podman or suitable hardware/images are unavailable.
  - Note: realised as a real CPU end-to-end run rather than a CUDA-only path. `tests/inference/test_vllm_integration.py` forces `HardwareTarget.CPU`, runs the official CPU image against `facebook/opt-125m` with a tiny random workload, and asserts the parsed throughput, prompt count, resolved target/image/devices, and written result file. It skips when Podman is unreachable, the image cannot be pulled, or the model is neither cached nor downloadable. A second test exercises the real vendor probes. On CPU, `--gpu-memory-utilization` defaults to 0.92 of system RAM, so the experiment sets it explicitly (0.5). Verified locally: `uv run pytest -m integration` (4 passed) and `uv run pytest` (88 passed, 4 deselected).
[x] **T22 - Unit-test config immutability and hashability**: Add tests asserting that `ContainerRuntimeConfig`, `BenchmarkConfig`, `LLMConfig`, `ExperimentConfig`, and `ExperimentResult` are frozen and hashable — including that equal configs hash equally regardless of `env` insertion order and that `dataclasses.replace` yields a distinct hash — mirroring `tests/runtime/test_runtime.py::test_config_env_is_immutable_and_config_is_hashable`.
  - Note: raised in review. The new types document hashability and immutability as part of their contract (`src/llm_optimization/inference/types.py`: `ContainerRuntimeConfig`'s read-only `env`, `ExperimentConfig`'s "The config is hashable and its sub-configs are frozen", `ExperimentResult`'s read-only `raw_json`), and the adjacent runtime layer tests the analogous contract, but no inference test asserts it.
  - Deviation: writing the tests showed the review's "verified manually that hashing works" was wrong for `ExperimentResult` — the frozen dataclass's generated `__hash__` tried to hash its `MappingProxyType` `raw_json` and raised `TypeError: unhashable type: 'dict'`. Resolved with the user: added a `__hash__` to `ExperimentResult` (T7) that hashes `raw_json` via `tuple(sorted(...))`, so the documented contract is now true; the other four types already hashed correctly. Tests live in `tests/inference/test_types.py` and assert frozen + hashable for all five types, order-independent hashing for `env`/`raw_json`, and a distinct hash from `dataclasses.replace`.

## Out of Scope

- Detached or long-lived serving containers; port publishing, networking, and health checks.
- Online benchmarking (`vllm bench serve`) against a running server.
- Perplexity evaluation (a nice-to-have; no built-in vLLM perplexity exists, so it would require a separate harness later).
- Sweep or batch orchestration with parameter grids and comparison reports; callers loop themselves.
- A command-line interface or console entrypoint.
- Intel XPU support.
- Building or publishing container images.
- Backends other than vLLM (the seam is prepared, but no second backend is implemented).
- Typed modeling of the full vLLM parameter surface; only the curated starter set is modeled.

## Further Notes

- The vLLM benchmark CLI is a flat argument namespace plus nested config objects. Some config groups have dedicated JSON flags (`--attention-config`, `--kernel-config`, `--compilation-config`, and others) that accept either a JSON string or dotted keys; others are flattened into individual flags (`--model`, `--dtype`, `--block-size`, `--max-num-seqs`, `--tensor-parallel-size`). The backend mapping must therefore distinguish JSON-flagged groups from flattened ones. There is no `--model-config`, `--cache-config`, `--scheduler-config`, `--parallel-config`, or `--load-config` flag.
- vLLM also supports a native `--config <file.yaml>` loader that flattens a nested YAML dict into arguments, converting nested dicts into `--<key> '<json>'` strings. This was considered and rejected in favour of registry-driven argv assembly: it requires a YAML dependency and a temp file while still needing the group-to-flag knowledge, and it is less transparent to test.
- The generic JSON mechanism documented by vLLM (`--json-arg '{"…": …}'` and dotted keys such as `--attention-config.backend`) means any config flag accepts either a JSON payload or dotted keys; the backend uses the JSON payload form.
- `temperature` was considered but is not currently a documented `vllm bench throughput` flag, so it was dropped from the starter set. Only parameters verified to exist as bench CLI flags are exposed; adding an unsupported parameter would produce a silent no-op or a container error.
- The container domain type is named `ContainerRuntimeConfig` specifically to avoid shadowing the existing `runtime.ContainerConfig`, which the new layer consumes internally.
- Default images use the `latest` tag for freshness; this trades reproducibility for currency. Pinning is possible later by adding a version field or relying on the image override. They are registry-qualified (`docker.io/...`) so podman can resolve them without a TTY.
- The current development host has no NVIDIA GPU and no CDI specs, so CUDA runs cannot be validated locally; that path relies on unit tests with injected probes. It does have a working AMD ROCm runtime (`rocminfo` enumerates a `gfx1201` device and `libamdhip64.so.7` is complete), so it resolves to ROCm. `rocm-smi` reports "Device Name: N/A" only because the unversioned `libdrm_amdgpu.so` dev symlink is missing (`libdrm-devel` is not installed); this is a packaging quirk, not a broken GPU stack, and the probe ignores host ROCm tooling for that reason. The CPU path is validated end to end by the T21 integration test.
- The existing runtime layer deliberately excludes networking, detached execution, and port publishing. The one-shot offline benchmark design keeps the new layer entirely within those constraints; the only runtime change was an optional `entrypoint` override on `ContainerConfig` (see T12), needed because the official images ship a `vllm serve` entrypoint.
- The engine-agnostic experiment layer is the key long-term bet: if a second engine is added, it should require only a new backend implementation, with experiment definitions untouched.
- The ROCm probe requires `/dev/kfd` and at least one `/dev/dri/renderD*` node, which is exactly what a container needs: the ROCm userspace is supplied by the image, so host ROCm tooling is not consulted. A KFD node without render nodes is treated as a partial stack and resolves to CPU. The current development host resolves to ROCm with one device.
- The `vllm bench throughput` JSON exposes a single blended `tokens_per_second` (`total_num_tokens / elapsed_time`, i.e. prompt + generated tokens) and no output-only throughput. `ExperimentResult.tokens_per_second` therefore means total throughput and is documented as such. Comparing configs on it mixes prefill and decode, and a workload with a high prompt-to-output ratio can be dominated by prefill; isolating decode would need per-request outputs or stdout parsing, which is deliberately out of scope.
- The engine-neutral categorical enums are curated, not exhaustive. `--dtype` and `--kv-cache-dtype` are closed sets in vLLM, but `--quantization` is a free method string (methods are registered dynamically) and `--attention-backend` accepts any `AttentionBackendEnum` value; both change between vLLM releases. Only widely available members are modelled, and `extra_args` remains the escape hatch, so a new engine value does not require waiting for a library release (it does require the caller to leave the enum field unset).
