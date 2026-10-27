# Extend tunable serving parameters

## Problem Statement

The optimizer currently exposes a narrow slice of each engine's serving knobs. Users who know that prefix caching, a different KV-cache format, a flash-attention mode, vLLM's scheduling caps, or llama.cpp's GPU layer offload materially changes throughput cannot sweep those choices. They must either stop using the optimizer, post-process the winning configuration by hand, or rely on raw engine flags that the experiment document cannot represent. This limits the usefulness of the benchmark for real deployment tuning, especially on hosts where memory, cache behaviour, or GPU offload is the dominant constraint.

## Solution

Add five groups of engine-neutral tunables in isolated, tested phases:

- a boolean `prefix_caching` switch for vLLM and llama.cpp;
- additional llama.cpp KV-cache formats;
- llama.cpp flash-attention `auto` and `off` modes;
- vLLM's `max_num_active_seqs` and `max_num_scheduled_tokens` scheduling caps;
- llama.cpp's `n_gpu_layers`, including `auto`, `all`, and exact counts.

The result is a search space that remains total over each backend's applicable set, with every new value reaching the engine by an explicit mapping. Boolean and int-or-enum parameter kinds are added to the definition system so these values are first-class rather than encoded as engine-specific strings.

## User Stories

1. As an inference operator, I want to sweep prefix caching on and off on vLLM, so that I can measure whether KV reuse helps my workload.
2. As an inference operator, I want to sweep llama.cpp prompt caching on and off, so that I can compare cached and uncached runs on the same benchmark.
3. As an inference operator, I want a boolean tunable to emit both the enabling and the disabling engine flag, so that "off" is a real configuration rather than an omitted default.
4. As a search-space author, I want boolean values validated strictly, so that `0`, `1`, quoted strings, and enum-like spellings cannot silently become booleans.
5. As a search-space author, I want `interval` rejected for boolean parameters, so that invalid definition shapes fail before a sweep starts.
6. As a llama.cpp user, I want to sweep `f32`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1` KV-cache formats, so that I can trade cache precision against memory and throughput.
7. As a vLLM user, I want the new llama.cpp-only cache formats rejected clearly, so that I do not declare a value the selected engine cannot serve.
8. As a llama.cpp user, I want to sweep flash attention between `on`, `off`, and `auto`, so that I can measure the effect of disabling it or letting the engine choose.
9. As a vLLM user, I want llama.cpp-only attention modes rejected clearly, so that cross-engine spaces do not fail silently.
10. As a vLLM user, I want to sweep `max_num_active_seqs`, so that I can reduce the decode batch without shrinking the model runner's capacity.
11. As a vLLM user, I want to sweep `max_num_scheduled_tokens`, so that I can tune the scheduling budget independently of the maximum batched-token budget.
12. As a vLLM user, I want invalid companion combinations rejected before the sweep, so that `max_num_active_seqs > max_num_seqs` or `max_num_scheduled_tokens > max_num_batched_tokens` does not fail midway through a study.
13. As a vLLM user, I want the companion-bound check skipped when either side is deferred, so that I can use engine defaults where I do not know a safe value.
14. As a llama.cpp user, I want to sweep `n_gpu_layers` across `auto`, `all`, and exact layer counts, so that I can compare partial and full GPU offload.
15. As a llama.cpp user, I want a mixed categorical such as `[auto, all, 0, 8, 16]`, so that mode values and exact counts can be compared in one dimension.
16. As a llama.cpp user, I want an interval on `n_gpu_layers` to constrain exact counts only, so that a numeric sweep does not accidentally include `auto` or `all`.
17. As a CPU-only user, I want positive GPU offload on a CPU target rejected before the run, so that I do not waste trials on configurations that cannot work.
18. As a CPU-only user, I want `auto` and zero GPU layers accepted, so that host-agnostic spaces remain usable.
19. As a llama.cpp user, I want an exact layer count above the checkpoint's block count rejected and pointed toward `all`, so that a typo does not silently duplicate full offload.
20. As a llama.cpp user, I want the layer-count check to fail open when GGUF metadata is unreadable, so that custom checkpoints are not blocked.
21. As a user running examples, I want the broad examples to demonstrate the newly supported values, so that I can see how to write a real sweep.
22. As a user running examples, I want the fixed baselines to remain reproducible, so that I can compare a sweep against a stable reference.
23. As a GPU user, I want a dedicated llama.cpp GPU-offload example, so that I can start from a sensible `n_gpu_layers` comparison.
24. As a CPU user, I want the existing broad llama.cpp examples to remain host-agnostic, so that a new GPU-specific knob does not break CPU use of the examples.
25. As a documentation reader, I want the tunable table, engine mappings, parameter counts, and YAML snippets updated together, so that the documented space matches what the loader accepts.
26. As a maintainer, I want each parameter group delivered and verified in its own phase, so that a regression can be attributed to the parameter that introduced it.
27. As a maintainer, I want every phase to pass formatting, linting, type checking, unit tests, and integration tests, so that a phase is not marked complete with hidden breakage.
28. As a maintainer, I want the total-over-applicable-set invariant preserved, so that no parameter silently falls back to an engine default.
29. As a maintainer, I want the new string values reduced to engine-neutral spellings in the experiment document, so that generated YAML never carries one engine's vocabulary.

## Implementation Decisions

### Delivery

- The work is split into five phases, one parameter group each, with a full verification gate after every phase. A phase is complete only when `ruff format`, `ruff check`, `ty check`, unit tests, and the integration suite pass.
- Commit and pull-request handling is out of scope for this plan.
- The existing rule that a space is total over a backend's applicable set is preserved. Existing space files must declare the new tunables; no migration documentation is added at this stage.

### Phase 1: boolean kind and prefix caching

- `LLMConfig` gains `prefix_caching: bool | None`; the registry maps the tunable name `prefix_caching` to that field.
- The registry's parameter-kind abstraction gains a boolean kind. `ParameterValue` gains booleans, and `expected_kind` resolves a `bool | None` annotation to `bool`.
- Boolean definitions are strict: `fixed` and `categorical` accept only real YAML booleans, `interval` is invalid, and booleans are kept distinct from integers in both directions.
- A shared toggle-target type lives in the shared tunables module. It stores the enabling and disabling flag, and argument assembly chooses by the field value: `true` emits the positive flag, `false` the negative flag, `None` neither.
- vLLM forwards `--enable-prefix-caching` and `--no-enable-prefix-caching`; llama.cpp forwards `--cache-prompt` and `--no-cache-prompt`.
- The parameter is applicable to both backends unconditionally.
- All eight existing example spaces declare it: broad and speculative-full spaces use `[true, false]`; all-fixed and speculative-defaults use `fixed: true`.

### Phase 2: KV-cache formats

- `KVCacheDType` gains `float32`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1`.
- The llama.cpp backend maps all seven to `f32`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1`.
- The vLLM backend intentionally provides no mappings for the new members, so vLLM rejects them. vLLM's long tail remains reachable through raw arguments.
- No new field or applicable-set change occurs in this phase.
- llama.cpp broad and speculative examples enumerate the new supported values; the fixed baseline stays on `float16`.

### Phase 3: attention modes

- `AttentionBackend` gains `auto` and `off`.
- llama.cpp maps `flash_attention` to `on`, `off` to `off`, and `auto` to `auto`.
- vLLM rejects `auto` and `off`; it continues to accept `flash_attention` and `flash_infer`.
- No new field or applicable-set change occurs in this phase.
- llama.cpp broad and speculative examples use `[auto, flash_attention, off]`; the fixed baseline pins `flash_attention`.

### Phase 4: vLLM scheduling caps

- `LLMConfig` gains `max_num_active_seqs: int | None` and `max_num_scheduled_tokens: int | None`.
- Both are vLLM-only and are forwarded as flat flags using the existing field-name-to-kebab-case convention.
- Two engine bounds are added: `max_num_seqs >= max_num_active_seqs` and `max_num_batched_tokens >= max_num_scheduled_tokens`.
- Bounds follow the existing policy: validate only when both dimensions expose a numeric range; a deferred companion is left to the engine at runtime.
- The fixed vLLM baseline pins active sequences equal to `max_num_seqs` and scheduled tokens equal to `max_num_batched_tokens`.
- Broad vLLM spaces use `max_num_seqs: 16..256`, `max_num_active_seqs: 1..16`, `max_num_batched_tokens: 4096..8192`, and `max_num_scheduled_tokens: 2048..4096`, so every corner satisfies both bounds.

### Phase 5: llama.cpp GPU layers

- `GpuLayersMode` is a new engine-neutral string enum with `auto` and `all`.
- `LLMConfig` gains `n_gpu_layers: int | GpuLayersMode | None`.
- The registry's kind system gains an int-or-enum union kind. `fixed` and `categorical` accept either branch, including mixed lists; `interval` constrains only the integer branch.
- Exact layer counts are non-negative, so `0` is valid; the field uses a non-negative floor rather than the existing positive-only rule.
- llama.cpp forwards the value through `--n-gpu-layers`; `auto` and `all` have value mappings, integers pass through unchanged. vLLM does not forward the field.
- On a CPU hardware target, `all` and exact counts greater than zero are rejected; `auto` and `0` are accepted.
- GGUF metadata reading is extended to capture the architecture's block/layer count. When readable, an exact count above the block count is rejected and the user is directed to `all`. Unreadable metadata fails open.
- Existing llama.cpp example spaces use `default: true` and remain host-agnostic. The fixed baseline pins `all` and is documented as GPU-only.
- A dedicated GPU-offload example uses `categorical: [auto, 0, all]`, with comments explaining that exact counts are checkpoint-specific and may be added after checking the model's block count.

### Cross-cutting

- No CLI flags are added; all new parameters are pure tuning dimensions set through the space definition.
- No output-writer change is required. The YAML experiment writer reflects new `LLMConfig` fields automatically and reduces new string enums to their engine-neutral values; container writers remain parameter-agnostic.
- The search-space documentation, speculative-decoding documentation, example descriptions, parameter counts, accepted-value tables, engine-mapping tables, and all YAML snippets are updated in the same phase as the code they describe.
- `AGENTS.md` is updated when a phase introduces a new invariant or a new engine-neutral enum, notably the boolean and union kinds and `GpuLayersMode`.
- Integration tests run after every phase, not only at the end.

## Testing Decisions

Good tests assert externally observable behaviour: what a definition accepts or rejects, what command arguments a backend produces, what validation problems it reports, and whether every shipped example space parses and validates. Tests should not pin private helper structure or implementation details.

- **Registry and loader**: cover the new boolean kind, strict boolean coercion, rejection of integers and strings for booleans, rejection of intervals for booleans, canonical tunable order, the int-or-enum union kind, mixed categoricals, integer-only intervals, and the non-negative floor for `n_gpu_layers`. Prior art is the existing registry and loader test suites.
- **Backend argument construction**: cover positive and negative toggle flags and `None` omission; all new llama.cpp KV-cache spellings; all new attention-mode spellings; the two new vLLM scheduling flags; and `n_gpu_layers` with `auto`, `all`, and integers. Cover vLLM rejection of llama.cpp-only values. Prior art is the existing vLLM and llama.cpp argument-translation tests.
- **Backend validation**: cover the vLLM companion bounds, including the deferred-companion skip and the permissive-corner failure; CPU-target rejection for `all` and positive counts; acceptance of `auto` and zero; GGUF block-count rejection; and fail-open behaviour. Prior art is the existing vLLM engine-bound and llama.cpp model-aware validation tests.
- **Example spaces**: every shipped example must parse and pass the selected backend's support validation on a representative target. The broad vLLM examples must stay within the new companion bounds. The new GPU-offload example must be validated against a GPU target.
- **Documentation snippets**: the YAML snippets in the search-space and speculative-decoding documentation must remain complete spaces for their engine, including the new tunables.
- **Integration**: run the existing artifact integration suite after every phase. No new artifact structure is expected, but the planned server command changes for the new flags.

## Implementation Tasks

[x] **T1 - Add boolean tunable kind**: Extend the registry kind abstraction, `ParameterValue`, and `expected_kind` for booleans; teach the loader strict boolean coercion, bool/int separation, and interval rejection; add focused loader and registry tests.
[x] **T2 - Add prefix caching end to end**: Add `LLMConfig.prefix_caching`, its registry entry, the shared toggle-target mechanics, vLLM and llama.cpp flag mappings, and update every test fixture and all eight example spaces so the total-space invariant holds.
[x] **T3 - Add backend and validation tests for prefix caching**: Cover positive/negative flag emission and `None` omission on both backends, applicability on both engines, and example-space validity.
[x] **T4 - Document prefix caching**: Update the tunable table, per-engine prose and counts, all search-space YAML snippets, and the speculative-decoding YAML snippets.
[x] **T5 - Run Phase 1 gate**: Run formatting, linting, type checking, the unit suite, and the integration suite; fix any failures before starting Phase 2.
[x] **T6 - Add llama.cpp KV-cache formats**: Extend `KVCacheDType` with the seven new members, map them in llama.cpp, leave vLLM without mappings, update llama.cpp example KV categoricals, and update enum-order and translation/rejection tests.
[x] **T7 - Document KV-cache expansion**: Update the accepted-value table, llama.cpp mapping table, counts, and YAML snippets.
[x] **T8 - Run Phase 2 gate**: Run formatting, linting, type checking, the unit suite, and the integration suite; fix any failures before starting Phase 3.
[x] **T9 - Add attention auto/off**: Extend `AttentionBackend` with `auto` and `off`, map all three llama.cpp modes, leave vLLM rejecting the new members, update llama.cpp example attention categoricals, and update enum and translation/rejection tests.
  - Note: the example categoricals spell the disabling mode as `"off"` (quoted), because PyYAML parses bare `off` as boolean `false`. Documentation snippets must quote it too.
[x] **T10 - Document attention modes**: Update the accepted-value table, llama.cpp mapping table, and YAML snippets.
[x] **T11 - Run Phase 3 gate**: Run formatting, linting, type checking, the unit suite, and the integration suite; fix any failures before starting Phase 4.
[x] **T12 - Add vLLM scheduling caps**: Add `max_num_active_seqs` and `max_num_scheduled_tokens` to `LLMConfig`, the registry, and vLLM forwarding; add both engine bounds; update all vLLM test fixtures and example spaces; add argument and bound tests.
[x] **T13 - Document vLLM scheduling caps**: Update the tunable table, bound explanation, example ranges, and all vLLM YAML snippets.
[x] **T14 - Run Phase 4 gate**: Run formatting, linting, type checking, the unit suite, and the integration suite; fix any failures before starting Phase 5.
[x] **T15 - Add int-or-enum union kind**: Extend the registry kind abstraction and loader for union fields, mixed fixed/categorical values, integer-only intervals, and a non-negative numeric floor; add focused kind and loader tests without adding a new tunable.
  - Note: `optimization/_optuna.py`'s kind dispatch was extended to handle `UnionKind` so the type check stays clean and a future union tunable samples correctly; no tunable was added.
[x] **T16 - Add GPU layers end to end**: Add `GpuLayersMode`, `LLMConfig.n_gpu_layers`, its registry entry, llama.cpp value mappings and forwarding, and update every llama.cpp test fixture and example space so the total-space invariant holds.
  - Note: vLLM's `applicable_parameters` now derives its set from its own `PARAMETER_TARGETS` instead of the whole registry, because `n_gpu_layers` is llama.cpp-only and must not make vLLM spaces require it.
[x] **T17 - Add GPU-target validation for GPU layers**: Reject `all` and positive exact counts on a CPU target while accepting `auto` and zero; add validation tests.
[x] **T18 - Add GGUF block-count validation**: Extend GGUF metadata reading to capture the block count, reject exact layer counts above it, direct the user to `all`, fail open on unreadable metadata, and add model-aware validation tests.
[x] **T19 - Add the dedicated GPU-offload example**: Add a llama.cpp example sweeping `[auto, 0, all]` with explanatory comments, add it to the example documentation and example count, and validate it against a GPU target.
[x] **T20 - Document GPU layers and union kinds**: Update the tunable table, llama.cpp mapping table, accepted values, example prose, YAML snippets, and the relevant architecture/invariant notes.
[x] **T21 - Run Phase 5 gate**: Run formatting, linting, type checking, the unit suite, and the integration suite; fix any failures.
  - Note: the first integration run exposed two llama.cpp integration fixtures that still omitted `n_gpu_layers` after T16 (`tests/cli/test_cli_llamacpp_integration.py`, `tests/optimization/test_optimize_llamacpp_integration.py`); both now declare it as `0` for the CPU target. Two vLLM integration failures (a container DNS error and a sweep with no completed trial) did not recur on rerun and are environmental/flaky, not code regressions.

## Out of Scope

- Commit, branch, or pull-request handling.
- Migration documentation for existing user space files.
- `gpu_memory_utilization`, `block_size`, `enforce_eager`, cudagraph sizes, CPU offload, and vLLM's long-tail KV-cache formats.
- llama.cpp multi-GPU topology knobs (`split-mode`, `tensor-split`, `main-gpu`), separate logical/physical batch sizes, `cache-reuse`, and `kv-unified`.
- Deep speculative-decoding internals such as minimum draft length, draft sampling probabilities, or n-gram lookup sizes.
- New CLI flags for any added tunable.
- Changes to container output formats; the writers remain parameter-agnostic.
- Space-format versioning.

## Further Notes

- Adding a tunable is intentionally breaking for existing spaces: the loader will report a missing tunable until the definition declares it. This is accepted and follows the existing total-over-applicable-set invariant.
- The union kind is the riskiest change because it touches the central definition model. It is introduced in its own task before `n_gpu_layers` exists.
- GGUF block-count metadata is architecture-prefixed. The reader should locate the block-count key rather than hard-code one architecture; unreadable or unusual metadata must fail open.
- T1's boolean-kind tests register a synthetic boolean tunable, because the real `prefix_caching` field and registry entry arrive with T2; the loader tests still drive the public parser end to end.
- The Phase 4 example ranges are chosen so the existing conservative “every corner must satisfy the bound” check passes; they may look narrower than the old `max_num_seqs` and `max_num_batched_tokens` ranges.
- The integration suite requires a running Podman daemon. If it is unavailable, record that clearly when reporting a phase gate rather than silently skipping it.
