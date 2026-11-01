# Extend llama.cpp tuning parameters

## Problem Statement

The optimizer exposes only a narrow slice of llama.cpp's serving knobs. A user
tuning a dense model on a memory-constrained host cannot vary the physical
micro-batch independently from the logical batch, cannot trade VRAM for speed
by keeping the dense FFN weights or the KV cache in system RAM, cannot choose a
model loading mode, and cannot set the CPU thread counts that dominate
single-turn latency. A user benchmarking agent-style turns cannot reach the
host prompt cache, context-checkpoint, slot-reuse, or continuous-batching
settings that shape cached time-to-first-token and multi-turn reuse. A user
tuning speculative decoding cannot reach the draft model's depth, split
probability, sampling strategy, offload placement, or cache format. Finally, a
user cannot adjust several micro-level knobs (tensor-operation offload, weight
repacking, polling, process priority, host-buffer bypass, full-size sliding
window cache), and the engine's automatic “fit to device memory” behaviour can
silently rewrite arguments a trial declared, so a trial may not measure the
configuration the search space described.

The result is that the optimizer cannot answer many of the questions a real
llama.cpp deployment depends on, and some sweeps measure something other than
what they declare.

## Solution

Extend the engine-neutral `LLMConfig` and the tunable registry with the missing
llama.cpp serving parameters, grouped into the following capability areas:

- **Context and workload:** decouple the logical `--batch-size` from the
  physical `--ubatch-size` so prefill batching can be tuned independently.
- **Placement and memory:** add dense FFN CPU offload, KV-cache offload,
  tensor-operation offload, host-buffer bypass, weight repacking, the full-size
  SWA cache, and the model loading mode.
- **Cache and scheduling:** add the host prompt-cache size, per-slot context
  checkpoints and their minimum spacing, unified KV, continuous batching, and
  idle-slot caching.
- **CPU and process:** add generation and prompt-processing thread counts,
  polling level, and process priority.
- **Speculative decoding:** add the draft minimum token count, split
  probability, draft sampling strategy, draft GPU-layer and CPU-expert offload,
  and the draft KV-cache format — all applicable only when a draft model is
  configured.
- **Correctness pin:** add the `--fit` family (`--fit`, `--fit-target`,
  `--fit-ctx`) as fixed-only settings, with the command-line default pinned to
  `off`, so a trial measures the arguments the space declared. Users can still
  opt into engine auto-fit explicitly.

Most parameters become sweep dimensions. Only the `fit` family is fixed-only.
The new fields are organized into domain groups in `LLMConfig` and in the
canonical registry order, so related knobs sit together.

The backend gains full model-aware assessment for the new knobs: dense FFN and
draft offload counts are bounded by the relevant checkpoint's block count,
device-offload dimensions are pruned on CPU-only targets, `--swa-full` is
pruned on a confidently non-sliding-window checkpoint, and a quantized draft KV
cache is rejected together with flash attention `off`. Unreadable metadata
fails open, leaving the engine as the final authority.

The change is delivered in two stages. First, shared mechanics land: float
tunable support, optional-sided toggles, new engine-neutral enums, and
draft-model gating. Then the fields, model-aware rules, examples, docs, and
tests land together.

## User Stories

1. As a llama.cpp user tuning a prefill-bound model, I want to vary
   `--ubatch-size` independently from `--batch-size`, so that I can find the
   micro-batch size that maximizes prefill throughput without being forced to
   also change the logical batch size.
2. As a llama.cpp user whose model barely fits in VRAM, I want to offload the
   dense FFN weights of the first N layers to system RAM, so that I can trade
   some speed for enough headroom to run at all.
3. As a llama.cpp user with a long context, I want to keep the KV cache in
   system RAM, so that I can use a larger context or a higher cache precision
   than VRAM would otherwise allow.
4. As a llama.cpp user on a host that swaps under load, I want the model
   loading mode (`auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`) to be a
   sweep dimension, so that I can measure the throughput and memory stability
   trade-off on my own machine.
5. As a llama.cpp user running on an 8-core CPU, I want to sweep the generation
   thread count, so that the optimizer finds the setting where additional
   threads stop helping.
6. As a llama.cpp user whose prompt processing and generation want different
   thread counts, I want to sweep `--threads-batch` separately from
   `--threads`, so that each phase can use its best setting.
7. As a llama.cpp user, I want `--fit off` to be the command-line default, so
   that a trial measures exactly the arguments the search space declared rather
   than an engine-adjusted approximation.
8. As a llama.cpp user who deliberately wants engine auto-fit, I want to pass
   `--fit on` with an optional target margin and minimum context, so that I can
   still use the engine's memory-fitting behaviour when I choose it.
9. As a llama.cpp user, I want `--fit-target` or `--fit-ctx` without
   `--fit on` to be rejected as a usage error, so that a flag that would be
   silently ignored is caught immediately.
10. As a llama.cpp user benchmarking agent turns, I want to bound the host
    prompt-cache size in MiB, so that the cache does not compete with page
    cache and swap on a memory-constrained host.
11. As a llama.cpp user measuring cached time-to-first-token, I want to sweep
    the number of context checkpoints per slot, so that prefix-cached follow-up
    turns can be measured at different checkpoint densities.
12. As a llama.cpp user measuring cached time-to-first-token, I want to sweep
    the minimum spacing between context checkpoints, so that I can trade
    checkpoint overhead against reuse granularity.
13. As a llama.cpp user running multiple slots, I want to choose whether the
    KV buffer is unified across sequences, so that multi-slot memory and reuse
    can be matched to my workload.
14. As a llama.cpp user running multiple requests, I want to toggle continuous
    batching, so that I can measure the difference between dynamic batching
    and slot-at-a-time execution.
15. As a llama.cpp user running multiple slots, I want to toggle idle-slot
    caching, so that I can measure whether keeping idle slots in the prompt
    cache helps my multi-turn workload.
16. As a llama.cpp user tuning speculative decoding with a draft model, I want
    to sweep the minimum number of draft tokens, so that I can find where
    drafting stops paying for itself.
17. As a llama.cpp user tuning speculative decoding with a draft model, I want
    to sweep the draft split probability as a real number, so that the
    optimizer can interpolate rather than guess at discrete values.
18. As a llama.cpp user tuning speculative decoding with a draft model, I want
    to choose greedy or probabilistic draft sampling, so that I can measure
    which strategy accepts better on my workload.
19. As a llama.cpp user with a draft model that competes with the target model
    for VRAM, I want to sweep the draft model's GPU-layer offload, so that I can
    trade draft speed for target-model headroom.
20. As a llama.cpp user with an MoE draft model, I want to sweep how many draft
    expert layers stay on the CPU, so that I can fit the draft beside the target.
21. As a llama.cpp user tuning speculative decoding, I want to choose the draft
    KV-cache format independently from the target's, so that the draft's cache
    can be quantized for memory without touching the target's precision.
22. As a llama.cpp user, I want a draft-only parameter to be rejected when no
    draft model is configured, so that I know immediately that the sweep cannot
    affect anything.
23. As a llama.cpp user on a ROCm host, I want to toggle tensor-operation
    offload, so that I can measure whether host operations on the device help
    prefill.
24. As a llama.cpp user, I want to toggle weight repacking, so that I can
    measure its effect on my CPU or GPU configuration.
25. As a llama.cpp user on a contended CPU, I want to sweep the polling level,
    so that I can trade busy-wait latency against CPU contention.
26. As a llama.cpp user on a contended CPU, I want to sweep process priority,
    so that I can measure the effect of scheduling priority on throughput.
27. As a llama.cpp user debugging buffer behaviour, I want to toggle the host
    buffer, so that I can measure the effect of bypassing it.
28. As a llama.cpp user running a sliding-window model, I want to toggle the
    full-size SWA cache, so that I can measure its memory and speed trade-off.
29. As a llama.cpp user, I want a parameter that cannot affect my checkpoint or
    target to be dropped with a printed note, so that I understand why part of
    my declared space was not swept.
30. As a llama.cpp user, I want an offload count larger than the checkpoint's
    block count to be reported as a model-aware problem before the image is
    pulled, so that a typo does not waste a trial.
31. As a llama.cpp user with a draft model, I want draft offload counts bounded
    by the draft checkpoint's block count, so that the bound reflects the model
    the count actually applies to.
32. As a llama.cpp user, I want a quantized draft KV cache with flash attention
    off to be rejected before the sweep, so that the trial does not fail at
    server startup.
33. As a llama.cpp user, I want `--swa-full` dropped on a checkpoint that
    confidently has no sliding-window attention, so that the sweep does not
    measure a no-op.
34. As a vLLM user, I want the new llama.cpp-only dimensions to stay out of my
    required search space, so that existing vLLM experiments keep working
    unchanged.
35. As a vLLM user, I want the new draft-model gating to affect only the
    backend that forwards those parameters, so that vLLM never requires a
    llama.cpp-only field.
36. As a user authoring a search space, I want the YAML error for a missing new
    tunable to name exactly the parameter I forgot, so that I can update my
    space quickly.
37. As a user authoring a search space, I want float-valued dimensions to be
    accepted for the new probability knob, so that I can declare a real-valued
    range rather than approximating it with integers.
38. As a user reading the examples, I want the complete example to show a
    meaningful range for every new tunable, so that I can copy a working
    starting point.
39. As a user reading a focused example (MoE, GPU offload, speculative), I want
    unrelated new knobs to defer to the engine default, so that the example
    stays focused on its theme.
40. As a user reading the documentation, I want a dedicated llama.cpp tuning
    guide that explains the new knobs and their model-aware rules, so that I
    understand when to sweep each one.
41. As a user comparing engines, I want the documented per-engine parameter
    counts to be accurate after the change, so that I do not assume a parameter
    is unsupported when it is not.
42. As a user resuming a study, I want the new dimensions to behave like every
    other tunable, so that saved trials and reconstructed configurations remain
    consistent.
43. As a maintainer, I want the new fields grouped by domain in the canonical
    order, so that related dimensions stay adjacent and reviewable.
44. As a maintainer, I want the new llama.cpp spellings to stay in the llama.cpp
    backend, so that `LLMConfig` carries no engine vocabulary.
45. As a maintainer, I want the new engine-neutral enums to live in the
    inference types module, so that the config surface remains backend-free.
46. As a maintainer, I want the float tunable path to dispatch on the declared
    kind rather than truncating values, so that a resumed study and a fresh
    sample reconstruct the same configuration.
47. As a maintainer, I want the shared toggle mechanics to express one-sided
    flags, so that backend modules do not accumulate bespoke boolean branches.
48. As a maintainer, I want draft-model gating to be declared once in the
    registry and intersected per backend, so that gating cannot drift between
    engines.
49. As a maintainer, I want the new validation bounds to report the offending
    value and its permitted range, so that a malformed space is easy to fix.
50. As a maintainer, I want the new model-aware rules covered by focused tests,
    so that a later refactor cannot silently drop a pruning rule.

## Implementation Decisions

### Configuration surface

`LLMConfig` gains these fields, organized into the following domain groups.
The registry order follows the same relative order, skipping non-tunables.

1. **Context and workload:** existing `max_model_len`, `max_num_seqs`,
   `max_num_batched_tokens`, `max_num_active_seqs`,
   `max_num_scheduled_tokens`; new `ubatch_size: int | None`.
2. **Precision and attention:** existing `dtype`, `quantization`,
   `kv_cache_dtype`, `attention_backend`.
3. **Placement and memory:** existing `n_gpu_layers`, `n_cpu_moe`; new
   `n_cpu_ffn: int | None`, `kv_offload: bool | None`, `op_offload: bool | None`,
   `host_buffer: bool | None`, `repack: bool | None`, `swa_full: bool | None`,
   `load_mode: LoadMode | None`.
4. **Cache and scheduling:** existing `prefix_caching`; new `cache_ram: int |
   None`, `ctx_checkpoints: int | None`, `checkpoint_min_step: int | None`,
   `kv_unified: bool | None`, `continuous_batching: bool | None`,
   `cache_idle_slots: bool | None`.
5. **Parallelism:** existing `tensor_parallel_size`, `expert_parallel`,
   `all2all_backend`.
6. **Speculative:** existing `speculative_method`, `draft_model`,
   `speculative_mode`, `num_speculative_tokens`; new `spec_draft_n_min: int |
   None`, `spec_draft_p_split: float | None`, `spec_draft_sampling:
   SpecDraftSampling | None`, `n_gpu_layers_draft: int | GpuLayersMode | None`,
   `n_cpu_moe_draft: int | CpuMoeMode | None`, `draft_kv_cache_dtype:
   KVCacheDType | None`.
7. **CPU and process:** new `threads: int | None`, `threads_batch: int | None`,
   `poll: int | None`, `prio: int | None`.
8. **Fit:** new fixed-only `fit: FitMode | None`, `fit_target: int | None`,
   `fit_ctx: int | None`.
9. **Run settings:** existing `seed`, `gpu_memory_utilization`,
   `resolved_draft_path`.

All fields remain optional and `None` continues to mean “leave it to the
engine's default”. The `fit` family is the exception only at the command line:
the CLI defaults `--fit` to `off`, so CLI-built experiments pin it, while a
hand-built configuration that leaves `fit` unset keeps the engine's own default.

### New engine-neutral enums

- `LoadMode` with members `auto`, `none`, `mmap`, `mlock`, `mmap_mlock`, `dio`.
  The llama.cpp backend alone maps `mmap_mlock` to the engine's `mmap+mlock`.
- `FitMode` with members `on`, `off`; the `fit` family is not a tunable and is
  forwarded only as a fixed command-line setting.
- `SpecDraftSampling` with members `greedy`, `probabilistic`.

The enums live in the engine-neutral inference types module. Their llama.cpp
spellings live only in that backend's value maps.

### Batch decoupling

`max_num_batched_tokens` stops emitting both `--batch-size` and
`--ubatch-size`; on llama.cpp it now emits only `--batch-size`. The new
`ubatch_size` field emits only `--ubatch-size`. vLLM's existing use of
`max_num_batched_tokens` is unchanged. The two values are forwarded
independently; where llama.cpp clamps a physical batch above the logical batch
or above the context, the engine remains the authority and the backend adds no
cross-field validation.

### Boolean forwarding

New boolean fields use positive semantics (`kv_offload`, `op_offload`,
`host_buffer`, `repack`, `swa_full`, `kv_unified`, `continuous_batching`,
`cache_idle_slots`). The backend maps each side to the engine's enabling or
disabling flag. Both sides are explicit where the engine offers both. For the
one-sided flags, the shared `Toggle` mechanic is extended so either side may be
absent: `host_buffer=True` emits nothing and `host_buffer=False` emits
`--no-host`; `swa_full=True` emits `--swa-full` and `swa_full=False` emits
nothing.

### Float tunables

Float becomes a first-class tunable kind:

- The parameter-value and atom-kind types admit `float`.
- The loader parses float values for fixed, categorical, and interval specs,
  keeping booleans distinct from numbers.
- The optimizer uses a float suggestion for a float interval, and a categorical
  suggestion for float choices.
- Reconstruction dispatches on the declared kind and returns a float rather
  than truncating it to an integer. Union resolution gains a float branch for
  completeness.

`spec_draft_p_split` is the only float field introduced here. Float floors are
not imposed by the loader; the backend validates the permitted range.

### Draft-model gating

The registry gains a second gated set for draft-model parameters:
`spec_draft_n_min`, `spec_draft_p_split`, `spec_draft_sampling`,
`n_gpu_layers_draft`, `n_cpu_moe_draft`, and `draft_kv_cache_dtype`. A helper
reports whether a draft model is configured. Both backends exclude both gated
sets from their base applicable set and add back only the gated parameters they
actually forward, intersected with their own forwarded-parameter table. The
existing broader speculative gate continues to cover `speculative_mode` and
`num_speculative_tokens`, which also apply to model-free drafting methods.

### llama.cpp forwarding

The llama.cpp backend adds forwardings for every new field:

- `--ubatch-size`, `--n-cpu-ffn`, `--load-mode`, `--cache-ram`,
  `--ctx-checkpoints`, `--checkpoint-min-step`, `--threads`, `--threads-batch`,
  `--poll`, `--prio`, `--fit`, `--fit-target`, `--fit-ctx`,
  `--spec-draft-n-min`, `--spec-draft-p-split`, `--spec-draft-sampling`,
  `--n-gpu-layers-draft`, `--n-cpu-moe-draft`.
- Toggle pairs for `--kv-offload`/`--no-kv-offload`,
  `--op-offload`/`--no-op-offload`, `--repack`/`--no-repack`,
  `--kv-unified`/`--no-kv-unified`,
  `--cont-batching`/`--no-cont-batching`,
  `--cache-idle-slots`/`--no-cache-idle-slots`, and the one-sided `--no-host`
  and `--swa-full`.
- `draft_kv_cache_dtype` emits both `--cache-type-k-draft` and
  `--cache-type-v-draft`.
- The `all` mode of `n_cpu_moe_draft` emits the value-less `--cpu-moe-draft`,
  mirroring the existing main-model special case. An exact count emits
  `--n-cpu-moe-draft N`.

`fit_target` and `fit_ctx` are forwarded only when `fit` is `on`. A hand-built
configuration that sets them with `fit` unset or `off` suppresses them rather
than forwarding arguments the engine would ignore.

### Numeric floors and bounds

The loader's floors are extended as follows:

- Positive (>= 1): `ubatch_size`, `threads`, `threads_batch`.
- Non-negative (>= 0): `n_cpu_ffn`, `n_gpu_layers_draft`, `n_cpu_moe_draft`,
  `ctx_checkpoints`, `checkpoint_min_step`, `spec_draft_n_min`, `poll`.
- No loader floor: `cache_ram`, `prio`, `spec_draft_p_split`, `fit_target`,
  `fit_ctx`.

The llama.cpp validation adds explicit range checks:

- `poll` in 0…100.
- `prio` in -1…3.
- `spec_draft_p_split` in 0.0…1.0.
- `cache_ram` >= -1 (`-1` means no limit, `0` disables the cache).

The CLI validates that `fit_target` and `fit_ctx` are at least 1.

### Model-aware assessment

The GGUF metadata reader gains a sliding-window fact from the checkpoint's
architecture-prefixed attention sliding-window key, failing open when the key
is absent or the header is unreadable. The llama.cpp assessment then adds:

- Drop `swa_full` when the checkpoint is readable and confidently has no
  positive sliding window.
- On a CPU target, drop `n_cpu_ffn`, `kv_offload`, `op_offload`,
  `n_gpu_layers_draft`, and `n_cpu_moe_draft`, in addition to the existing
  `n_cpu_moe` and conditional `n_gpu_layers` pruning.
- Bound an exact `n_cpu_ffn` count and its interval maximum by the main
  checkpoint's block count, with a hint to reduce the count (it has no `all`
  mode).
- When a draft path is resolved and readable, bound exact and interval
  `n_gpu_layers_draft` and `n_cpu_moe_draft` values by the draft checkpoint's
  block count, with the existing `all`-mode hints.
- Reject a quantized `draft_kv_cache_dtype` together with
  `attention_backend: off`, mirroring the main KV-cache rule.

Unreadable metadata continues to fail open.

### CLI

The engine argument group gains `--fit` with choices `on`/`off` and default
`off`, `--fit-target` (integer MiB, optional), and `--fit-ctx` (integer,
optional). Configuration building sets the corresponding non-tunable fields and
rejects `--fit-target` or `--fit-ctx` unless `--fit on` is also given, using the
same usage-error path as the speculative-setup checks. No other new parameter
gets a command-line flag; every other new field is reachable through a search
space, including a `fixed` or `default` dimension.

### Registry and examples

The canonical registry order is reorganized into the domain groups above,
including existing dimensions. Every llama.cpp example space declares every
applicable new tunable:

- The complete example declares a meaningful range or set for every new
  tunable.
- The all-fixed example pins each new tunable to a representative value.
- The focused examples (GPU offload, MoE, speculative defaults, speculative
  full) declare real dimensions only for their theme and `default: true` for
  unrelated new tunables.

vLLM example spaces need no new dimensions because the new fields have no vLLM
forwarding.

### Documentation

A dedicated llama.cpp tuning guide explains the prefill, placement, cache,
CPU/process, speculative, and fit knobs, including when each is applicable and
which model-aware rule may drop it. The search-space reference keeps the
canonical tunable table, adds rows for the new parameters, and updates the
per-engine parameter counts. The usage guide's engine table and the
architecture notes are updated for draft gating and the new model-aware rules.
The speculative-decoding guide documents the draft-only sub-knobs.

## Testing Decisions

Good tests assert externally observable behaviour: the command a configuration
produces, the problems and notes an assessment reports, the dimensions a parsed
space contains, the value a reconstructed configuration carries, and the exit
path a CLI invocation takes. Tests should not assert private helper structure or
internal call order.

Modules and surfaces covered:

- **Search-space registry and loader:** canonical order, resolved kinds
  including `float`, required-parameter errors, float parsing, floors, and
  draft/speculative gating.
- **Optuna adapter:** float interval and float categorical suggestions, and
  reconstruction of a float value from a fresh trial and from a resumed study's
  best parameters.
- **llama.cpp argument assembly:** every new field's flag, both sides of every
  toggle, the one-sided `host_buffer` and `swa_full` cases, enum translation,
  the draft `all` mode, and suppression of `fit_target`/`fit_ctx` unless fit is
  `on`.
- **llama.cpp validation:** unsupported draft-only declarations without a
  draft, the new numeric ranges, the quantized draft cache with flash attention
  off, and applicable-parameter intersection.
- **llama.cpp model-aware assessment:** block-count bounds for `n_cpu_ffn` and
  the draft offload fields, CPU-target pruning, SWA pruning, and fail-open
  behaviour on unreadable metadata.
- **GGUF metadata reading:** the sliding-window fact, including absent and
  malformed-header cases.
- **CLI:** `--fit` default, explicit `on`, and rejection of target/ctx without
  `on`.
- **Examples and output:** every example space loads successfully against its
  backend, and the YAML output writer renders the new fields without bespoke
  writer changes.

Prior art:

- Registry and loader tests that pin canonical order and resolved kinds and use
  a complete-space helper.
- Argument tests that build a config, call the backend's argument assembly, and
  assert the exact flag list.
- Model-aware assessment tests that inject metadata and assert the effective
  space and the reported notes/problems.
- CLI tests that assert the mapped exit code for usage errors.
- Example conformance tests that load each shipped space against the selected
  backend.

## Implementation Tasks

[x] **T1 - Add float and new enum kinds to the engine-neutral config surface**: Extend the parameter-value and atom-kind types with `float`, and add the `LoadMode`, `FitMode`, and `SpecDraftSampling` engine-neutral enums to the inference types module. Keep engine spellings out of this module.

[x] **T2 - Teach the loader float tunables**: Parse fixed, categorical, and interval float values while keeping booleans distinct from numbers; resolve a float kind from the field annotation; and preserve the existing integer floors for integer fields. Add loader tests for float values and rejection of a float value for an integer field.

[x] **T3 - Teach the optimizer float tunables**: Use a float suggestion for a float interval, a categorical suggestion for float choices, and kind-dispatched reconstruction that returns a float instead of truncating. Cover fresh and resumed-study reconstruction in the optimizer tests.

[x] **T4 - Allow one-sided toggles**: Extend the shared toggle mechanic so either the enabling or the disabling flag may be absent, emitting nothing for the absent side. Keep both backends' existing toggles working, and add focused tests for a one-sided enabling flag and a one-sided disabling flag.

[x] **T5 - Gate draft-model parameters per backend**: Add the draft-model gated set and the draft-configured helper to the registry. Change both backends' applicable-parameter derivation to exclude both gated sets from the base set and add back only the gated parameters they forward, intersected with their own forwarded table. Add tests proving a llama.cpp draft parameter is absent without a draft and that vLLM never requires it.

[x] **T6 - Reorganize the canonical config and registry order**: Reorder the existing `LLMConfig` fields and registry entries into the domain groups from this spec without adding new fields. Update order-sensitive tests and any documentation order that describes the canonical sequence.

[x] **T7 - Split logical and physical batch dimensions**: Add `ubatch_size`, narrow `max_num_batched_tokens` to `--batch-size` only on llama.cpp, forward `--ubatch-size`, and add registry, forwarding, loader-floor, and example-order coverage. Update the complete example's batching comments and dimensions.

[x] **T8 - Add placement and memory dimensions**: Add `n_cpu_ffn`, `kv_offload`, `op_offload`, `host_buffer`, `repack`, `swa_full`, and `load_mode`; forward their flags and toggles; add `LoadMode` value translation; register them; and add argument and loader-kind tests.

[x] **T9 - Add cache and scheduling dimensions**: Add `cache_ram`, `ctx_checkpoints`, `checkpoint_min_step`, `kv_unified`, `continuous_batching`, and `cache_idle_slots`; forward their flags and toggles; register them; and add argument, floor, and loader-kind tests.

[x] **T10 - Add CPU and process dimensions**: Add `threads`, `threads_batch`, `poll`, and `prio`; forward their flags; register them; and add argument tests and positive/floored loader coverage for the thread counts.

[x] **T11 - Add speculative draft dimensions**: Add `spec_draft_n_min`, `spec_draft_p_split`, `spec_draft_sampling`, `n_gpu_layers_draft`, `n_cpu_moe_draft`, and `draft_kv_cache_dtype`; forward their flags, including the draft `all` mode and the paired draft cache flags; add `SpecDraftSampling` value translation; register them under the draft gate; and add forwarding and reconstruction tests.

[x] **T12 - Add the fixed-only fit family and CLI flags**: Add `fit`, `fit_target`, and `fit_ctx` to the config and llama.cpp forwarding; add `--fit`, `--fit-target`, and `--fit-ctx` to the engine argument group with `--fit` defaulting to `off`; build the fields in the base configuration; reject target/ctx without `--fit on`; and add CLI tests for the default, explicit on, and the usage error.

[x] **T13 - Add numeric validation ranges**: Validate `poll` 0…100, `prio` -1…3, `spec_draft_p_split` 0.0…1.0, and `cache_ram` >= -1 in llama.cpp validation, reporting the offending value and permitted range. Add validation tests for accepted boundaries and rejected values.

[x] **T14 - Add the SWA metadata fact and pruning**: Extend GGUF metadata reading with the architecture-prefixed attention sliding-window value, failing open on absent or unreadable metadata. Drop `swa_full` on a readable checkpoint with no positive sliding window, and add metadata and assessment tests.

[x] **T15 - Extend CPU-target pruning**: Drop `n_cpu_ffn`, `kv_offload`, `op_offload`, `n_gpu_layers_draft`, and `n_cpu_moe_draft` on a CPU target in addition to the existing rules. Add assessment tests for the effective space and notes on CPU and non-CPU targets.

[x] **T16 - Bound offload counts by the relevant checkpoint**: Extend the block-count checks to `n_cpu_ffn` against the main checkpoint and to `n_gpu_layers_draft` and `n_cpu_moe_draft` against the draft checkpoint, including interval maxima and the appropriate hints. Add assessment tests for exact values, intervals, unreadable metadata, and absent drafts.

[x] **T17 - Extend the quantized draft-cache rule**: Reject a space that can pair a quantized `draft_kv_cache_dtype` with `attention_backend: off`, mirroring the main cache rule. Add validation tests for the rejected and accepted combinations.

[x] **T18 - Update every llama.cpp example space**: Make all six llama.cpp examples declare every applicable new tunable in the new canonical order: full ranges in the complete example, representative pins in the all-fixed example, and `default: true` for unrelated knobs in the focused examples. Add or update example-conformance coverage so every shipped space loads. *(All six spaces were reordered to the canonical registry order. The two speculative spaces now declare all six draft-only dimensions — real ranges in `space-speculative-full.yaml`, `default: true` in `space-speculative-defaults.yaml` — and the conformance suite loads them with a configured draft model, so both now require `--draft-model`. The other four spaces omit the draft dimensions because no draft is configured there; the complete example's comments now cover the CPU/process knobs and point at the speculative spaces for draft tuning. New conformance assertions pin the draft dimensions and their canonical placement.)*

[x] **T19 - Write the llama.cpp tuning guide and update references**: Add the dedicated tuning guide covering the new capability areas, applicability, and model-aware rules. Update the search-space table and per-engine counts, the usage guide's engine table, the architecture notes for gating and assessment, and the speculative-decoding guide for the draft-only knobs.

[x] **T20 - Update the project invariants**: Extend the architecture notes with the new engine-neutral enums, float tunable support, draft gating, the fit correctness pin, and the model-aware rules, and adjust the tunable-addition checklist for the new mechanics. *(T19 had already landed the two-gate applicability rule and the model-aware rules in `docs/architecture.md`; T20 added the engine-neutral enums, float support, and the fit pin there, and extended `AGENTS.md`'s invariants and tunable-addition checklist with the enum, float, toggle, gated-set, fit, and model-aware mechanics.)*

[x] **T21 - Run the full verification suite**: Run formatting, lint, type checking, and the default test suite; fix any fallout across examples, docs, and order-sensitive tests. *(All checks passed: `ruff format` left 121 files unchanged, `ruff check` and `ty check` clean, `pytest` 1469 passed / 20 deselected. The integration suite exposed one fallout item: `tests/cli/test_cli_llamacpp_integration.py` still declared only the pre-T7 tunables, so the total-space rule rejected it with exit code 3. Its inline space now declares all 26 llama.cpp-applicable tunables in canonical order (the non-theme dimensions defer to the engine default, with the offload/SWA ones dropped by the assessment), its stale "eight tunables" docstrings were updated, and its winner-config assertions now pin the declared `ubatch_size` alongside the other fixed dimensions. `pytest -m integration` is then green: 18 passed, 2 skipped. No source, example, or documentation change was needed.)*

## Out of Scope

- Adding new vLLM dimensions. The new parameters are llama.cpp-only; vLLM
  continues to reject them because it does not forward them.
- Modelling n-gram-specific speculative knobs beyond what the existing
  `speculative_method` dimension already selects.
- Modelling batch variants of process settings (`--prio-batch`,
  `--poll-batch`), CPU-affinity masks, NUMA settings, tensor splitting, or
  multi-device draft placement.
- Per-device `--fit-target` lists. The fixed-only setting carries a single MiB
  value that the engine broadcasts across devices.
- Exact example parameter ranges are implementation choices, not part of this
  spec's contract.
- Changing the benchmark workload, metric definitions, phase-rate schema, or
  study provenance.
- Adding new output formats or container-runtime behaviour.

## Further Notes

- The engine's `--fit` default is `on`; pinning it `off` by default is a
  deliberate behaviour change for command-line runs and should be called out in
  the tuning guide.
- `n_cpu_ffn` is applicable on both dense and MoE checkpoints. The engine's help
  text describes it as a dense-model knob and points MoE users at
  `--n-cpu-moe`; both may be declared together, and the sweep reports what each
  actually changes.
- `cache_ram` uses `-1` for “no limit” and `0` for “disabled”, which is why it
  has no non-negative loader floor.
- `checkpoint_min_step` uses `0` for “no minimum”, so it is non-negative rather
  than positive.
- The `fit` family stays out of the tunable registry: sweeping a correctness pin
  would let the optimizer undo the guarantee the pin exists to provide.
- Draft-model gating is keyed on a configured draft model rather than on the
  speculative method, because methods such as n-gram and the target model's own
  MTP heads do not use a separate draft.
- T8 declared the new dimensions in every llama.cpp example space ahead of T18
  so the example-conformance suite stays green: the complete space got ranges,
  the all-fixed space representative pins, and the focused spaces
  `default: true`. T18 still owns the final per-example curation.
- T8 also added `n_cpu_ffn` to the loader's non-negative floor set, because the
  field's `0` disables the offload and the floor is intrinsic to the parameter.
- T14 also made the architecture-value resolver honour a declared architecture
  strictly: a checkpoint that names its architecture uses only that
  architecture's key, so it cannot borrow the sole key of a different one. This
  keeps the sliding-window fact tied to the checkpoint's own architecture and
  brings the block and expert counts in line with their documented contract.
- T17 consults a cache dimension only when the backend forwards it for the run:
  `draft_kv_cache_dtype` triggers the flash-attention conflict only alongside a
  configured draft model. Without one the dimension is not applicable, so the
  space is already reported as not forwarded and a second startup-conflict
  problem would describe a cache the run never builds.
- T19 added the dedicated guide at `docs/llamacpp-tuning.md` and linked it from
  the README documentation index, the search-space reference, the usage engine
  table, and the speculative-decoding guide. `docs/architecture.md` gained the
  two-gate applicability rule and the new model-aware rules; the engine-neutral
  enums, float support, and fit pin remain T20's architecture-note work.
