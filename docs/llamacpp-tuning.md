# llama.cpp tuning guide

Part of the [README](../README.md) documentation.

The llama.cpp backend forwards thirty-four of the forty registered tunables to
`llama-server`, covering context and batching, placement and memory, caching and
scheduling, CPU and process settings, and the speculator's own knobs. The
[search-space reference](search-space.md#tunable-parameters) lists the exact
names, kinds, and engine spellings; this guide explains what each knob actually
changes, when it is applicable, and which model-aware rule may drop or reject
it.

The search space is **total over the backend's applicable set**, so every
parameter llama.cpp forwards for the run's fixed configuration must be
declared. Write `default: true` for a knob you do not want to sweep: it omits
the flag and leaves llama.cpp's own default in place. A parameter the backend
does not forward is rejected (exit code `4`) rather than silently ignored.

## How a dimension reaches the server

A declared value becomes the engine's flag verbatim, so a sweep measures the
configuration it names. Categorical values are translated from the
engine-neutral enums; an engine spelling with no mapping is rejected before the
first trial rather than dropped. The complete example
[`examples/llamacpp/space-full.yaml`](../examples/llamacpp/space-full.yaml)
declares every base knob, deferring the host-dependent offloads to the engine.

Most knobs are sweep dimensions, but one family is deliberately fixed: the
`--fit` correctness pin is set on the command line, never in the search space
(see [The fit correctness pin](#the-fit-correctness-pin)).

## Context and workload

### Logical and physical batches

`max_num_batched_tokens` emits only `--batch-size`, the logical batch the engine
fills per step. `ubatch_size` independently emits `--ubatch-size`, the physical
micro-batch the engine processes at once. Decoupling them lets a prefill sweep
lower the micro-batch — and with it the peak activation footprint — without
shrinking the logical batch. A larger physical batch needs more compute-buffer
memory; llama.cpp clamps a physical batch above the logical one or above the
context rather than rejecting it, so the backend adds no cross-field rule.
vLLM has no `ubatch_size` and rejects the dimension.

### Context and slots

`max_model_len` reaches the engine as `--ctx-size` unchanged, and
`max_num_seqs` as `--parallel`. llama.cpp divides the context evenly across the
slots, so the backend caps the forwarded slot count at
`max_model_len // request_tokens` (at least one). The request length is
`input_len + output_len` for the random workload or the preflight-measured
`BenchmarkConfig.max_request_tokens` for a JSONL dataset. A sampled context
below the request length cannot be served at any slot count, but a context
that is merely large enough to serve one request still collapses concurrency
and makes a trial far slower. Keep the floor at a multiple of the request
length: the complete example uses `18432`, which keeps at least sixteen
1024+128-token slots available.

## Placement and memory

### Layer and expert placement

- `n_gpu_layers` (`--n-gpu-layers`) accepts `auto`, `all`, or an exact count.
  `0` keeps the model entirely on the CPU, and a count above the checkpoint's
  block count is rejected after the model resolves with a pointer to `all`.
- `n_cpu_moe` (`--n-cpu-moe N`, or the value-less `--cpu-moe` for `all`) keeps
  a number of expert layers in system RAM. It is meaningful only on a
  Mixture-of-Experts checkpoint and only when weights would otherwise be on a
  device.
- `n_cpu_ffn` (`--n-cpu-ffn`) keeps the dense feed-forward weights of an exact
  number of layers in system RAM. It applies to dense and MoE checkpoints alike
  — llama.cpp's help text points MoE users at `--n-cpu-moe`, but both may be
  declared — and it has no `all` mode, so an oversized count is reported with
  a hint to reduce it.

### Device offloads and buffer behaviour

These use positive semantics, so `true` selects the enabling flag and `false`
the disabling one where the engine offers both:

- `kv_offload` (`--kv-offload` / `--no-kv-offload`) keeps the KV cache on the
  device when enabled; disabling it leaves the cache in system RAM, trading
  speed for headroom on a long context.
- `op_offload` (`--op-offload` / `--no-op-offload`) runs tensor operations on
  the device.
- `repack` (`--repack` / `--no-repack`) toggles weight repacking.
- `host_buffer` exposes only its disabling side: `false` emits `--no-host`,
  while `true` emits nothing and leaves the engine's buffer in place.
- `swa_full` exposes only its enabling side: `true` emits `--swa-full`, while
  `false` emits nothing. It only changes a checkpoint that uses sliding-window
  attention.
- `load_mode` (`--load-mode`) selects a loading strategy: `auto`, `none`,
  `mmap`, `mlock`, `mmap_mlock`, or `dio`. The engine-neutral `mmap_mlock`
  becomes llama.cpp's `mmap+mlock`.

## Cache and scheduling

These knobs shape cached time-to-first-token and multi-turn reuse:

- `prefix_caching` (`--cache-prompt` / `--no-cache-prompt`) is forwarded in both
  directions, so disabling the prompt cache is a real configuration.
- `cache_ram` (`--cache-ram`) bounds the host prompt cache in MiB. `-1` means
  no limit and `0` disables the cache, which is why it has no non-negative
  loader floor.
- `ctx_checkpoints` (`--ctx-checkpoints`) sizes the per-slot context-checkpoint
  cache, so prefix-cached follow-up turns can be measured at different
  checkpoint densities.
- `checkpoint_min_step` (`--checkpoint-min-step`) sets the minimum token
  spacing between checkpoints; `0` means no minimum, so it is non-negative.
- `kv_unified` (`--kv-unified` / `--no-kv-unified`) shares one KV buffer across
  sequences.
- `continuous_batching` (`--cont-batching` / `--no-cont-batching`) toggles
  dynamic batching against slot-at-a-time execution.
- `cache_idle_slots` (`--cache-idle-slots` / `--no-cache-idle-slots`) keeps
  idle slots in the prompt cache.

## CPU and process

- `threads` (`--threads`) and `threads_batch` (`--threads-batch`) set the
  generation and prompt-processing thread counts. Both must be at least one, so
  a host with an 8-core CPU can sweep where additional threads stop helping.
- `poll` (`--poll`) selects the busy-wait level on a `0…100` scale, trading
  latency against CPU contention.
- `prio` (`--prio`) sets the process priority in `-1…3`, so scheduling priority
  can be measured on a contended host.

## Speculative decoding

The mode and depth are shared with vLLM and are documented in the
[speculative-decoding guide](speculative-decoding.md). llama.cpp adds six
draft-only knobs, applicable exactly when the run configures a `--draft-model`:

- `spec_draft_n_min` (`--spec-draft-n-min`) is the minimum number of draft
  tokens.
- `spec_draft_p_split` (`--spec-draft-p-split`) is the split probability, a real
  number in `0.0…1.0`, so the optimizer can interpolate rather than guess at
  discrete values.
- `spec_draft_sampling` (`--spec-draft-sampling`) chooses `greedy` or
  `probabilistic` drafting.
- `n_gpu_layers_draft` (`--n-gpu-layers-draft`) places the draft's layers, with
  the same `auto`/`all`/exact-count shape as the main model.
- `n_cpu_moe_draft` (`--n-cpu-moe-draft N`, or `--cpu-moe-draft` for `all`)
  keeps draft expert layers in system RAM.
- `draft_kv_cache_dtype` emits both `--cache-type-k-draft` and
  `--cache-type-v-draft`, so the draft cache can be quantized independently of
  the target's.

A draft-only dimension declared without a configured draft model is a backend
problem (exit code `4`), because no value could reach the engine. The dimensions
remain applicable on a trial that samples `speculative_mode: disabled` — the
run still names a draft — but their flags are suppressed with the rest of the
setup.

## The fit correctness pin

llama.cpp's own `--fit` default is `on`, which lets the engine rewrite arguments
to fit the available memory. That can make a trial measure something other than
the point the search space declared, so **the command line pins `--fit off` by
default**. This is a deliberate behaviour change for command-line runs: a sweep
now measures exactly the arguments it names.

The fit family is fixed for a run:

- `--fit on` opts back into engine auto-fit.
- `--fit-target` is the headroom, in MiB, that fit leaves on each device.
- `--fit-ctx` is the minimum context fit preserves while fitting.

`--fit-target` and `--fit-ctx` are rejected as a usage error (exit code `2`)
unless `--fit on` is also given, and each must be at least 1. A hand-built
configuration that sets them with `fit` unset or `off` suppresses them rather
than forwarding flags the engine would ignore. The family is not in the tunable
registry: sweeping a correctness pin would let the optimizer undo the guarantee
it exists to provide.

## Model-aware rules

After `validate_model` and before the image is pulled, the backend assesses the
space against the resolved target checkpoint, the configured draft checkpoint,
and the detected hardware. It may drop a whole dimension, and it reports a
model-aware problem the assessment cannot fix through the same
unsupported-configuration path (exit code `4`).

| Rule | Effect | Why |
| ---- | ------ | --- |
| Confidently dense checkpoint | Drops `n_cpu_moe` | A model with no experts has nothing to keep on the CPU. |
| Confidently full-attention checkpoint | Drops `swa_full` | A full-size sliding-window cache changes nothing without a sliding window. |
| CPU target | Drops `n_cpu_moe`, `n_cpu_ffn`, `kv_offload`, `op_offload`, `n_cpu_moe_draft`, `n_gpu_layers_draft`, and `n_gpu_layers` when every declared value already disables offload | No weight leaves system RAM and no operation runs on a device there. |
| Main checkpoint's block count | Rejects an exact `n_cpu_ffn`, `n_gpu_layers`, or `n_cpu_moe` count, or a matching interval maximum, above it | The count names layers the checkpoint does not have. |
| Draft checkpoint's block count | Rejects an exact `n_gpu_layers_draft` or `n_cpu_moe_draft` count, or a matching interval maximum, above it | The bound reflects the model the count actually applies to. |
| Quantized draft cache with `attention_backend: off` | Rejects the space | llama.cpp refuses to create a quantized V cache unless flash attention is enabled. |

The MoE fact comes from the GGUF header's architecture-prefixed
`<arch>.expert_count`, and the sliding-window fact from
`<arch>.attention.sliding_window`; a readable header without either key is
confidently dense or full-attention. The architecture named by
`general.architecture` is honoured strictly, so a checkpoint cannot borrow
another architecture's key.

Every rule that reads the checkpoint **fails open**: unreadable or incomplete
metadata leaves the affected dimensions as declared, so llama.cpp stays the
final authority for a checkpoint the reader cannot see.

## Flag-level validation ranges

Some values are rejected before the sweep because the engine's parser would
reject them at startup. Each problem names the offending value and the
permitted range.

| Parameter | Accepted range |
| --------- | -------------- |
| `poll` | `0…100` |
| `prio` | `-1…3` |
| `spec_draft_p_split` | `0.0…1.0` |
| `cache_ram` | `>= -1` (`-1` is "no limit") |
| `n_gpu_layers` on a CPU target | `auto` or `0` |

The loader additionally enforces each integer parameter's intrinsic floor:
`ubatch_size`, `threads`, and `threads_batch` must be at least 1, while
`n_cpu_ffn`, `n_gpu_layers_draft`, `n_cpu_moe_draft`, `ctx_checkpoints`,
`checkpoint_min_step`, `poll`, and `spec_draft_n_min` accept 0. `cache_ram`,
`prio`, `spec_draft_p_split`, and the fit family have no loader floor; the
backend's own range checks cover them.

## Example

```console
$ llm-config-optimization /models/Qwen3.5-8B-Q4_K_M.gguf \
    examples/llamacpp/space-full.yaml llamacpp \
    --target cpu \
    --fit off \
    --dataset-path examples/prompts.jsonl \
    --n-trials 50 \
    --output log \
    --output-config best-llamacpp.yaml
```

## Related

- [Search-space YAML format](search-space.md) — the definition format, the
  canonical tunable table, per-engine mappings, and validation.
- [Speculative decoding](speculative-decoding.md) — the shared mode and depth
  dimensions plus the draft-only knobs.
- [Usage reference](usage.md#engines) — engine selection, the command-line
  options, and the benchmark workload.
