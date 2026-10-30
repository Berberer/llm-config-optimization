# Add Mixture-of-Experts tuning parameters and model-aware dtype assessment

## Problem Statement

The optimizer can tune dense-model serving parameters, but it exposes none of the knobs each engine provides specifically for Mixture-of-Experts models. Users running MoE checkpoints cannot sweep vLLM's expert parallelism or its all-to-all transport, and cannot sweep llama.cpp's CPU offload of MoE expert weights. They must either stop using the optimizer or reach for raw engine flags that the experiment document cannot represent.

A second problem is wasted search effort. MoE-only parameters have no effect on a dense checkpoint, and several existing host-dependent parameters have no effect on a CPU target or when a companion parameter makes them impossible. Because the search space is validated before the checkpoint is resolved, the optimizer cannot know at definition time that some dimensions are irrelevant to the selected model. It currently spends trials sampling those dimensions and producing identical measurements.

A third problem is model-specific dtype support. A space can declare a `dtype` or `kv_cache_dtype` value that the resolved checkpoint cannot run even though the engine supports the value in general. For example, Qwen3.5's linear-attention layers reject `float32`, and vLLM stores a non-quantized KV cache in the resolved model dtype. The optimizer currently learns these facts only when a trial fails at server startup, and TPE can then concentrate on the failing value for the rest of the sweep.

## Solution

Expose three new engine-neutral tunables:

- `expert_parallel`, a boolean for vLLM's expert parallelism;
- `all2all_backend`, a categorical for vLLM's MoE all-to-all transport, restricted to the portable `allgather_reducescatter` and `naive` choices;
- `n_cpu_moe`, an exact layer count or `all` for llama.cpp's CPU offload of MoE expert weights.

Each backend's applicable set includes the MoE tunables it forwards, so every space for that backend declares them and the total-over-applicable-set invariant remains intact. Dense spaces use `default: true`.

After the checkpoint resolves, a new model-aware space assessment detects whether the model is actually a Mixture-of-Experts model and removes MoE dimensions that cannot affect it. The same assessment drops other dimensions that cannot affect the resolved model, hardware, or fixed configuration, and reports model-aware problems such as an `n_cpu_moe` count above the checkpoint's layer count. Pruning is logged, and a space that becomes empty runs exactly one trial at engine defaults instead of repeating identical configurations.

The assessment also narrows model-dependent categorical values. For vLLM it can remove `dtype` values that the checkpoint's kernels reject and reconcile an explicit float `kv_cache_dtype` with the resolved model dtype. This requires extending the assessment beyond whole-dimension pruning to value-level narrowing for categorical dimensions; interval bounds are not narrowed.

## User Stories

1. As a vLLM user running an MoE model, I want to sweep expert parallelism on and off, so that I can measure whether replacing tensor-parallel MoE sharding with expert parallelism improves throughput.
2. As a vLLM user, I want to choose between vLLM's portable all-to-all transports, so that I can compare the engine default against a simple alternative without installing extra communication libraries.
3. As a vLLM user, I want `expert_parallel` to emit both the enabling and disabling flags, so that "off" is a real configuration rather than an omitted default.
4. As a vLLM user, I want `all2all_backend` to reach the engine as a categorical tunable, so that a sweep can compare transports without editing engine commands by hand.
5. As a llama.cpp user, I want to keep the MoE weights of the first N layers on the CPU, so that I can leave attention and KV cache on the GPU while trading expert compute against VRAM.
6. As a llama.cpp user, I want to express "keep all MoE weights on the CPU" without knowing the checkpoint's layer count, so that a space stays model-agnostic.
7. As a llama.cpp user, I want mixed categoricals such as `[0, 8, 16, all]`, so that exact offload counts and the full-offload mode can be compared in one dimension.
8. As a llama.cpp user, I want an interval on `n_cpu_moe` to constrain exact counts only, so that a numeric sweep does not accidentally include the `all` mode.
9. As a llama.cpp user, I want an exact `n_cpu_moe` count above the checkpoint's layer count rejected and pointed toward `all`, so that a typo does not silently duplicate full offload.
10. As a search-space author, I want `n_cpu_moe` to accept zero, so that disabling MoE CPU offload is expressible and the non-negative floor is explicit.
11. As a search-space author, I want MoE tunables to use engine-neutral values, so that generated YAML never carries one engine's spelling.
12. As a search-space author, I want `all2all_backend` to accept only the two portable values, so that a sweep does not fail on communication libraries the default image does not contain.
13. As a vLLM user, I want llama.cpp's `n_cpu_moe` rejected as unsupported, so that a cross-engine space fails clearly.
14. As a llama.cpp user, I want vLLM-only MoE parameters rejected as unsupported if declared, so that a cross-engine space fails clearly.
15. As a user running a dense model, I want MoE dimensions to remain valid in the definition but have no effect, so that the same space file can be used with dense and MoE checkpoints.
16. As a user running a dense model, I want the optimizer to detect that the checkpoint has no experts and drop the MoE dimensions, so that no trial is wasted sampling parameters the model ignores.
17. As a user running an MoE model, I want the optimizer to detect the experts automatically, so that I do not have to tell it whether the checkpoint is MoE.
18. As a user, I want the detection to read vLLM's checkpoint configuration and llama.cpp's GGUF metadata directly, so that no extra CLI flag is needed.
19. As a user with a custom or unusual checkpoint, I want unreadable metadata to fail open, so that the optimizer keeps the declared dimensions instead of silently assuming the model is dense.
20. As a user, I want every dropped dimension printed with a reason, so that I can see which parts of my space were intentionally skipped.
21. As a user whose entire space is irrelevant to the selected model, I want exactly one trial at engine defaults, so that the run produces a baseline instead of repeating identical configurations.
22. As a CPU-only llama.cpp user, I want `n_cpu_moe` dropped on a CPU target, so that the sweep does not sample a flag that cannot change anything when all weights are already on the CPU.
23. As a CPU-only llama.cpp user, I want a space whose `n_gpu_layers` values are already CPU-valid (`auto` or `0`) to have that dimension dropped, so that the run does not repeat identical no-offload trials.
24. As a CPU-only llama.cpp user, I want `all` or positive `n_gpu_layers` counts still rejected before the sweep, so that a GPU-oriented space does not silently change meaning on a CPU host.
25. As a vLLM user, I want `all2all_backend` dropped when expert parallelism can never be enabled, so that the sweep does not vary a transport that is never used.
26. As a vLLM user, I want `expert_parallel` dropped when no tensor-parallel size in the space can exceed one, so that expert parallelism is not sampled where it would be a no-op.
27. As a vLLM user, I want expert parallelism and the all-to-all backend retained when the space can produce a tensor-parallel size above one and can enable EP, so that the meaningful combinations remain searchable.
28. As a vLLM user, I want the all-to-all choice retained only when expert parallelism is actually reachable, so that the search space reflects the dependency between the two parameters.
29. As a user, I want model-aware space problems reported before the image is pulled, so that a doomed configuration does not wait on a large download.
30. As a user, I want model-aware space problems to use the existing unsupported-configuration exit path, so that failures remain scriptable and consistent.
31. As a user, I want the assessment to run after checkpoint validation and before the sweep, so that a bad base configuration still fails for its own reason first.
32. As a user, I want the benchmark workload, objective, and container settings to stay unchanged, so that the new feature only affects which serving dimensions are sampled.
33. As a user, I want the YAML experiment output to include the new parameters automatically, so that a winning MoE configuration is reproducible without manual edits.
34. As a user running the examples, I want every existing vLLM and llama.cpp space updated, so that the examples continue to load under the total-space rule.
35. As a user running the examples, I want a dedicated vLLM MoE example, so that I can start from a sensible expert-parallelism comparison.
36. As a user running the examples, I want a dedicated llama.cpp MoE example, so that I can start from a sensible expert CPU-offload comparison.
37. As a documentation reader, I want the tunable table, per-engine parameter counts, accepted values, mapping tables, and YAML snippets updated together, so that the documentation matches what the loader accepts.
38. As a documentation reader, I want the detection and pruning behaviour explained, so that I understand why a dimension I wrote may not appear in a run.
39. As a maintainer, I want the new engine-neutral enums to live in the shared inference types, so that engine spellings stay in their own backend modules.
40. As a maintainer, I want model-aware space assessment to be a backend hook, so that future no-op rules do not accumulate in the CLI.
41. As a maintainer, I want each new tunable covered by registry, loader, argument-translation, detection, assessment, and example tests, so that the total-space invariant and engine mappings cannot drift.
42. As a maintainer, I want the existing `validate_model` contract preserved, so that checkpoint-format and quantization validation keep their current responsibilities.
43. As a maintainer, I want no new CLI flags, so that MoE tuning remains a search-space concern like the other tunables.
44. As a maintainer, I want the final suite to pass formatting, linting, type checking, unit tests, and integration tests, so that the feature is not merged with hidden breakage.
45. As a vLLM user running a checkpoint with linear-attention layers, I want `float32` removed from the effective `dtype` space when the checkpoint's kernel rejects it, so that the sweep does not waste trials on a value that always fails at startup.
46. As a vLLM user, I want the assessment to read the checkpoint's model dtype and layer types from `config.json`, so that dtype narrowing needs no extra CLI flag.
47. As a user, I want a checkpoint's preferred `dtype` treated as a default rather than an exclusive allowlist, so that other dtypes the engine actually supports remain available.
48. As a vLLM user, I want an explicit float16 or bfloat16 `kv_cache_dtype` reconciled with the resolved model dtype (`auto` included), so that a matching cache remains searchable instead of being rejected wholesale.
49. As a user, I want every removed categorical value printed with its reason, so that I can see exactly what the assessment narrowed.
50. As a user, I want a model-aware problem when a dimension's declared values are all unsupported, rather than silently falling back to an engine default, so that an explicit space is never changed into a different one without notice.
51. As a user with an unreadable or incomplete checkpoint configuration, I want dtype narrowing to fail open, so that the model's declared values are kept and the engine remains the final authority.
52. As a maintainer, I want model-aware dtype rules to stay in the backend that owns the engine's kernels, so that no model-specific vocabulary enters the CLI or the engine-neutral types.
53. As a maintainer, I want value-level pruning covered by tests for removed values, retained values, an emptied dimension, and unreadable metadata, so that the effective space is predictable.

## Implementation Decisions

### New engine-neutral vocabulary

- Add `All2AllBackend`, a string enum with exactly `allgather_reducescatter` and `naive`. It is vLLM-only.
- Add `CpuMoeMode`, a string enum with exactly `all`. It is llama.cpp-only and is the non-numeric branch of the `n_cpu_moe` union.
- Add `expert_parallel: bool | None`, `all2all_backend: All2AllBackend | None`, and `n_cpu_moe: int | CpuMoeMode | None` to `LLMConfig`.
- Export the two new enums from the inference package's public surface.
- Extend `ParameterValue` so a declared value may be an `All2AllBackend` or `CpuMoeMode` member.
- Append the three registry entries in canonical order: `expert_parallel`, `all2all_backend`, `n_cpu_moe`. The registry continues to derive each kind from the `LLMConfig` field annotation.
- Add `n_cpu_moe` to the loader's non-negative floor set, so `0` is valid.
- Add `expert_parallel`, `all2all_backend`, and `n_cpu_moe` to the registry's total set; each backend filters that set through its own forwarding table.

### Backend forwarding

- vLLM forwards `expert_parallel` as a toggle: `true` emits `--enable-expert-parallel`, `false` emits `--no-enable-expert-parallel`, and unset emits neither.
- vLLM forwards `all2all_backend` as the flat flag `--all2all-backend` with its engine-neutral member value; both members map to their vLLM spellings.
- llama.cpp forwards an exact `n_cpu_moe` count as `--n-cpu-moe N` and the `all` mode as `--cpu-moe`. The value map still contains `all` so support validation recognises the mode; argument assembly special-cases the mode to emit the value-less flag.
- vLLM does not forward `n_cpu_moe`; llama.cpp does not forward `expert_parallel` or `all2all_backend`. Each backend rejects a declared dimension it does not forward through the existing per-dimension validation.
- No new CLI flags are added. All three are pure tuning dimensions.

### Model-aware detection

- vLLM treats a checkpoint as MoE when a readable `config.json` contains a positive `num_experts`, `n_routed_experts`, or `num_local_experts`, checking both the top level and a nested `text_config` object.
- A missing, malformed, or non-object configuration is unknown rather than dense, so the assessment keeps the declared MoE dimensions.
- llama.cpp extends its GGUF metadata reader to capture the architecture-prefixed `<arch>.expert_count` value and reports it on the metadata object. A readable header with no expert count, or a count of zero, means dense.
- A GGUF header that cannot be parsed reports unknown metadata, and the assessment keeps the declared MoE dimensions.

### Model-aware dtype detection

- vLLM extends its checkpoint reader to report the model dtype (`torch_dtype`, or a nested `text_config.dtype` when present) and whether any `text_config.layer_types` entry is `linear_attention`. Both facts are optional: a missing or malformed `config.json` reports unknown, not a guessed value.
- The declared model dtype is a preferred/default execution dtype, not a whitelist of every supported `--dtype` value. For example, Qwen3.5 declares `bfloat16`, but `float16` also runs; the model-specific fact that matters is the presence of linear-attention layers and the backend's kernel rule for them.
- vLLM owns the kernel rule: when the checkpoint has linear-attention layers, `float32` is unsupported by `ChunkGatedDeltaRuleFunction`, while `float16` and `bfloat16` remain valid. The fact and the rule stay in `inference/_vllm.py`, never in the CLI or shared types.
- The reader also resolves the dtype that `auto` would select from the checkpoint metadata when that metadata is readable. This lets the assessment reconcile an explicit float KV-cache dtype with `auto` instead of treating every `auto`/float-cache pair as unverifiable.
- llama.cpp has no equivalent activation-dtype allowlist in GGUF metadata: `general.file_type` describes weight quantization, not supported execution dtypes. No new llama.cpp dtype-narrowing rule is added; its existing KV-cache/attention and block-count rules remain unchanged.

### Model-aware space assessment

- Add an engine-neutral `SpaceAssessment` result carrying the effective space, human-readable pruning/narrowing notes, and model-aware problems. The effective space may have whole dimensions removed and categorical values narrowed while preserving each dimension's declared order.
- Add an `assess_space` method to the inference backend protocol. It receives the space, the experiment configuration, the resolved hardware, and the resolved checkpoint path.
- Run the assessment after `validate_model` succeeds and before the image is pulled. Report problems through the existing unsupported-configuration exit path.
- The assessment is backend-owned because the rules depend on engine vocabulary and checkpoint formats. Shared mechanics may live in the shared tunables module, but no engine spelling enters the CLI or the engine-neutral types.
- Value-level pruning is limited to narrowing a `Categorical` or `Fixed` set of values; it never invents a value and never narrows an `Interval`'s bounds. If every declared value of a dimension is unsupported, the assessment reports a model-aware problem rather than silently dropping the dimension or replacing it with `default: true`.
- Prune MoE dimensions when the checkpoint is confidently dense: `expert_parallel` and `all2all_backend` for vLLM, `n_cpu_moe` for llama.cpp. An unknown checkpoint keeps them.
- For vLLM, prune `all2all_backend` when `expert_parallel` can never be enabled, and prune both `expert_parallel` and `all2all_backend` when no declared tensor-parallel size can exceed one. A deferred tensor-parallel dimension is resolved through the detected device count for this decision.
- For vLLM, narrow `dtype` when the checkpoint's linear-attention layers reject `float32`: remove `DType.FLOAT32` from a `Fixed`/`Categorical` dimension and log the checkpoint/kernel reason. A readable config that declares no linear-attention layers leaves `float32` in place; unreadable metadata fails open and leaves it in place too.
- For vLLM, reconcile explicit float `kv_cache_dtype` values after dtype narrowing. A `FLOAT16` or `BFLOAT16` cache is retained only when every remaining dtype possibility resolves to the matching model dtype; `auto` resolves through readable checkpoint metadata, while an explicit dtype resolves directly. Incompatible cache values are removed, and `auto` plus the FP8 formats are always retained. When the metadata cannot resolve `auto`, the existing conservative pre-model rule remains in force.
- For llama.cpp, prune `n_cpu_moe` on the CPU target.
- For llama.cpp, prune `n_gpu_layers` on the CPU target when its declared values are already CPU-valid; the existing pre-model validation still rejects `all` and positive counts on CPU.
- After pruning, apply model-aware bounds to the retained dimensions: reject exact `n_cpu_moe` and exact `n_gpu_layers` values above the GGUF architecture's block count and point the user at `all`. Intervals are rejected when their maximum exceeds the block count. `all`, `auto`, and deferred dimensions are not bounded. Unreadable or absent block-count metadata fails open.
- Preserve canonical dimension order in the effective space and preserve declared value order within a narrowed dimension.
- The CLI prints one line per dropped dimension and one line per narrowed value or rule, naming the reason, and does not silently ignore pruning.
- If the effective space has no dimensions, the CLI logs that fact and runs exactly one trial at engine defaults rather than repeating identical trials under the requested trial count. A dimension emptied by value-level pruning is a problem, not an empty-space case.

### Compatibility and documentation

- The total-over-applicable-set rule is unchanged. vLLM spaces must declare `expert_parallel` and `all2all_backend`; llama.cpp spaces must declare `n_cpu_moe`. Dense spaces use `default: true`.
- Existing example spaces and test fixtures are updated in the same change that adds each backend's tunables.
- Add a dedicated vLLM MoE example that fixes tensor parallelism above one, fixes expert parallelism on, and compares the two portable all-to-all choices.
- Add a dedicated llama.cpp MoE example that compares `0`, a few exact layer counts, and `all`, with a note that exact counts are checkpoint-specific.
- Keep the vLLM examples portable: do not rely on model-specific dtype support for an example to load, and prefer `auto`/FP8 cache choices when `dtype` is swept. Model-aware narrowing is the mechanism that handles checkpoint-specific values, not example-specific omissions alone.
- Update the search-space documentation's tunable table, engine-specific counts, accepted values, engine mapping tables, and YAML snippets. Add a section explaining MoE detection, dtype narrowing, and value-level pruning.
- Update the architecture documentation's data-flow description for checkpoint-aware space assessment, including the distinction between dropped dimensions and narrowed categorical values.
- Update the project agent instructions whenever an invariant changes, notably the new enums, the assessment hook, the value-level pruning rule, the checkpoint dtype facts, and the model-aware pruning rules.
- No output-writer code changes are required: the YAML writer reflects new `LLMConfig` fields automatically, and the container writers remain parameter-agnostic. Output fixtures are updated only where they enumerate fields.

## Testing Decisions

Good tests assert externally observable behaviour: what the loader accepts, what arguments a backend produces, what detection reports, which dimensions an assessment drops or preserves, which categorical values it narrows, what problems it reports, and whether shipped examples remain valid. Tests should not pin private helper structure.

- **Registry and loader**: extend canonical tunable order, kind resolution, enum coercion, union handling, and the non-negative floor for `n_cpu_moe`. Prior art is the existing registry and loader suites.
- **Backend argument translation**: cover the vLLM positive and negative expert-parallel flags and unset omission; both all-to-all spellings; llama.cpp exact `n_cpu_moe` values and the `all` mode's value-less `--cpu-moe` flag; and rejection of cross-engine dimensions. Prior art is the existing vLLM and llama.cpp argument suites.
- **Checkpoint detection**: use temporary vLLM `config.json` files and the existing synthetic GGUF writer to cover top-level and nested expert keys, dense checkpoints, zero counts, missing keys, and unreadable metadata. Prior art is the existing model-aware validation and GGUF metadata suites.
- **Checkpoint dtype facts**: use temporary vLLM `config.json` files to cover top-level and nested `torch_dtype`/`text_config.dtype`, linear-attention and full-attention `layer_types`, missing dtype facts, and malformed or unreadable configurations. Assert that a preferred dtype is reported as a fact but never treated as the complete allowlist.
- **Space assessment**: cover dense-checkpoint pruning and fail-open behaviour on both backends; CPU-target pruning for `n_cpu_moe` and CPU-valid `n_gpu_layers`; the vLLM expert-parallel/tensor-parallel dependency rules, including deferred tensor parallelism; retained dimensions when EP and TP can both be meaningful; and block-count problems for exact counts and intervals while modes and deferrals pass. Prior art is the existing backend validation suites.
- **Dtype and cache value narrowing**: cover removal of `float32` for a linear-attention checkpoint, retention for a full-attention or unreadable checkpoint, and preservation of `float16`/`bfloat16`; cover explicit float cache reconciliation with a pinned dtype, `auto` resolved from metadata, mixed dtype/cache choices, FP8 exemptions, an emptied dimension reporting a problem, and fail-open behaviour when the model dtype is unknown. Assert that declared value order is preserved and that every removed value is named in the notes. Prior art is the existing vLLM engine-bound and model-aware validation suites.
- **CLI preparation**: cover the assessment problem path mapping to the unsupported exit code, the printed pruning and narrowing notes, the empty-space case forcing exactly one trial, and the emptied-dimension problem path. Use the recording fake backend. Prior art is the existing CLI preparation tests.
- **Examples**: every shipped example must parse and pass its backend's support validation on a representative target. The two new MoE examples must load for their own backend, and the broad vLLM examples must remain portable across dense, full-attention, and linear-attention checkpoints. Prior art is the existing example-space suite.
- **Output**: confirm the YAML experiment document carries the new fields with engine-neutral values. Prior art is the existing formatter suite.
- **Final gate**: run formatting, linting, type checking, the default unit suite, and the integration suite. Integration requires a running Podman daemon; if unavailable, record that clearly rather than silently skipping it.

## Implementation Tasks

[x] **T1 - Add engine-neutral MoE enums**: Add `All2AllBackend` with `allgather_reducescatter` and `naive`, add `CpuMoeMode` with `all`, document their engine-neutral purpose, export them from the inference package, and add focused enum/value tests without touching the registry.
[x] **T2 - Add vLLM MoE tunables end to end**: Add `expert_parallel` and `all2all_backend` to `LLMConfig`, the registry, `ParameterValue`, and vLLM forwarding; update all affected registry, loader, backend-applicability, output, and example fixtures; add the two dimensions to every vLLM example; and add vLLM argument tests for the toggle and both all-to-all values.
[x] **T3 - Add llama.cpp MoE tunable end to end**: Add `n_cpu_moe` to `LLMConfig`, the registry, `ParameterValue`, the loader's non-negative set, and llama.cpp forwarding; update all affected registry, loader, backend-applicability, output, and example fixtures; add the dimension to every llama.cpp example; and add argument tests for exact counts and the `all` mode.
[x] **T4 - Add checkpoint MoE detection**: Add the vLLM configuration reader for top-level and nested expert counts, extend GGUF metadata with the architecture-prefixed expert count, and cover dense, MoE, missing-key, zero-count, and unreadable-metadata cases in both detection suites.
[x] **T5 - Add the space assessment hook**: Add the engine-neutral `SpaceAssessment` result and the `assess_space` backend protocol member; implement the hook in both backends with the MoE-on-dense-checkpoint pruning and fail-open rule; update the recording fake backend and the protocol-conformance test; and add assessment tests for dense and unknown checkpoints.
[x] **T6 - Add dependency, CPU, and bound rules**: Extend both backends' assessment with the vLLM expert-parallel/all-to-all dependency rules, llama.cpp CPU-target pruning for `n_cpu_moe` and CPU-valid `n_gpu_layers`, and GGUF block-count problems for exact and interval `n_cpu_moe`/`n_gpu_layers` values; add labelled notes and focused tests for every rule.
[x] **T7 - Integrate assessment into the sweep preparation**: Call the backend assessment after model validation, report problems through the unsupported-configuration path, print pruning notes, replace the space with the effective space, and force exactly one trial when nothing remains; add CLI preparation tests for pruning output, problem mapping, and the empty-space case.
[x] **T8 - Add dedicated MoE example spaces**: Add one focused vLLM space that fixes tensor parallelism above one, enables expert parallelism, and sweeps the portable all-to-all choices, plus one focused llama.cpp space that sweeps zero, exact CPU-expert layer counts, and `all`; register both in the example tests and ensure they parse and validate.
[x] **T9 - Document MoE tuning and pruning**: Update the search-space documentation's tunable table, per-engine counts, accepted values, engine mappings, YAML snippets, and MoE example descriptions; add a detection/pruning section; update the architecture data-flow description and the project agent instructions for the new enums, assessment hook, and pruning invariants. *(Also refreshed the engine-table counts in `docs/usage.md` and the complete vLLM space snippet in `docs/speculative-decoding.md` so no documentation contradicts the total-space rule; dtype value-narrowing documentation is left to T12.)*
[x] **T10 - Add vLLM checkpoint dtype facts**: Extend the vLLM checkpoint reader to report the preferred model dtype and whether the checkpoint declares linear-attention layers, covering top-level and nested dtype keys, missing facts, and unreadable metadata; keep the facts optional and fail open, and add focused detection tests. *(The layer-type fact also reads a top-level `layer_types` list, not only the nested `text_config` one: Qwen3.5 nests it, but hybrid checkpoints such as Qwen3-Next declare the same layout at the top level. `CheckpointFacts` names the dtype `model_dtype` and reports `None` for any spelling `DType` does not model, including `auto`.)*
[x] **T11 - Add value-level pruning and dtype rules**: Extend `SpaceAssessment` and the assessment mechanics so a categorical dimension can be narrowed while preserving declared order; prune `float32` when a linear-attention checkpoint's kernel rejects it; reconcile explicit float KV-cache values with pinned or metadata-resolved `auto` dtypes; report a model-aware problem when a dimension's values are all removed; and add focused assessment tests for each rule. *(Also relaxed the pre-model float-cache check in `validate` so `auto`/deferred/multi-value dtype pairs reach the assessment instead of being rejected before the checkpoint is readable; an unresolved model dtype fails open and keeps the declared cache values, leaving the engine as final authority.)*
[x] **T12 - Document dtype-aware assessment**: Update the search-space and architecture documentation for checkpoint dtype facts, value-level narrowing, preferred-versus-supported dtype semantics, and the emptied-dimension problem; update the project agent instructions for the new assessment invariant and keep the shipped vLLM examples portable. *(Also documented the T11 relaxation of the pre-model float-cache check and added a shipped-example portability test that assesses every vLLM example against dense, full-attention, and linear-attention checkpoints; no example values changed.)*
[x] **T13 - Run the final verification gate**: Run formatting, linting, type checking, the default unit suite, and the integration suite; fix any failures before considering the feature complete. *(Run on 2026-10-28: `ruff format --check`, `ruff check`, and `ty check` are clean, the default suite passes 1230 tests, and the integration suite passes 18 tests with 2 environment skips (`podman-compose` is not installed, and this host's `docker` CLI is Podman's emulation rather than real Docker). Two integration fixtures needed completing for the gate: the vLLM CLI space gained `expert_parallel`/`all2all_backend`, and the llama.cpp CLI test's post-sweep assertions now expect `None` for `n_gpu_layers`/`n_cpu_moe`, because the assessment drops both offload dimensions on a dense checkpoint served by a CPU target. That test also asserts the pruning notes are printed.)*

## Out of Scope

- vLLM's MoE kernel backend selection (`moe_backend`), expert-parallel load balancing (`enable_eplb` and its JSON configuration), DeepEP, PPLX, FlashInfer NVLink, MoRI, and other transports that require extra packages or specific accelerator generations.
- llama.cpp's `--moe-cache-mib`, `--n-cpu-ffn`, arbitrary `--override-kv` metadata overrides, and changing the number of active experts.
- Data-parallel or multi-node expert-parallel deployment settings.
- Value-level pruning of interval bounds; intervals keep their declared bounds and are rejected when a model-aware bound is exceeded. Categorical value-level pruning is in scope for the dtype/KV-cache rules described above.
- A generic per-architecture dtype-support table or a promise that every kernel limitation is detected. Detection is limited to readable checkpoint facts plus backend-owned kernel rules; unknown models keep their declared values and the engine remains the final authority.
- Reordering the preparation flow so the model is downloaded before the space is loaded.
- New CLI flags for any MoE parameter or dtype rule.
- Space-format versioning or migration tooling for existing user space files.
- Changes to benchmark workloads, objectives, container runtimes, or output formats.

## Further Notes

- vLLM's expert-parallel size is derived from the tensor-parallel and data-parallel sizes. Only tensor parallelism is modelled here, so the assessment treats a tensor-parallel size above one as the condition under which expert parallelism can matter.
- The two portable all-to-all choices are intentionally conservative. Faster transports are reachable through the raw-argument escape hatch, but they are not advertised as first-class tunables because the default image may not contain their dependencies.
- llama.cpp's `--n-cpu-moe N` and `--cpu-moe` are not the same flag with a numeric spelling: `--cpu-moe` takes no value. The special-case in argument assembly must be covered by a test so the mode cannot regress into `--n-cpu-moe all`.
- A readable checkpoint with no expert metadata is treated as dense; an unreadable checkpoint is treated as unknown and keeps the MoE dimensions. This asymmetry is deliberate: confident dense detection avoids wasted trials, while unknown metadata leaves the engine as the final authority.
- Pruning runs after `validate_model` so checkpoint format and base-configuration errors still surface first, and before the image pull so a doomed space fails cheaply.
- A usable GGUF checkpoint declares `general.architecture`, so the architecture-prefixed expert count resolves. A header that carries several architectures' expert counts without declaring one reports no expert count and is therefore treated as dense by the assessment; such a file cannot name its own architecture and cannot be served as-is, so the asymmetry is harmless. This is recorded here so a future ambiguity handling change stays deliberate.
- Adding a tunable remains intentionally breaking for existing space files under the total-over-applicable-set rule. This change updates every shipped example and test fixture, but it does not add migration tooling.
- The dedicated llama.cpp MoE example should use counts that are plausible for current MoE checkpoints and should explain that exact counts are checkpoint-specific; a model with fewer blocks will be rejected by the model-aware bound rather than silently clamped.
- A checkpoint's declared `dtype`/`torch_dtype` is a preferred execution dtype, not an allowlist. Qwen3.5 declares `bfloat16`, but `float16` also runs; only `float32` is removed because its linear-attention kernel explicitly rejects it. The assessment must not infer a complete supported set from the preferred dtype.
- Value-level dtype narrowing is deliberately conservative: it removes a value only when the backend has a concrete rule for the checkpoint fact it read. Unknown architectures, unreadable metadata, and absent layer-type facts leave the dimension unchanged.
- A narrowed dimension that loses every value is a model-aware problem, not a reason to silently use `default: true`; the user's explicit space must not be changed into a different one without an error.
- A deferred `expert_parallel` dimension (`default: true`) is treated as "never enables expert parallelism" for the all-to-all rule, because vLLM forwards no flag in that case and its own default is off. This mirrors the deferred-tensor-parallel rule, which resolves `default: true` through the detected device count, rather than treating either deferral as unknown metadata. A tensor-parallel dimension missing from a partial space is treated as an undeterminable ceiling, so both dependency rules stay inactive.
- The block-count rules are applied only to the assessment's retained dimensions; the existing `validate_model` check of an exact `n_gpu_layers` in the fixed configuration is unchanged and still runs first.
- The dedicated vLLM MoE example pins `tensor_parallel_size: 2`, so the example test validates it against a two-device CUDA target (`_EXAMPLE_HARDWARE` in `tests/search_space/test_examples.py`); every other example keeps the one-device target. The example tree is globbed, so a new file registers automatically, and the two focused tests pin each MoE example's defining shape against silent drift.
- The final integration suite requires Podman. If the daemon is unavailable, report that explicitly as part of the gate instead of claiming a passing integration run.
- `tests/cli/test_cli_integration.py`'s `SPACE` fixture omitted `expert_parallel` and `all2all_backend`, so the loader rejected it under the total-space rule. That fixture was missed when T2 updated the other spaces and is deselected from the default suite, so the omission only surfaced in the T13 integration run; T13 completed it, along with the llama.cpp fixture's post-sweep assertions.
