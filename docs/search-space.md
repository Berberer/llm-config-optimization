# Search-space YAML format

Part of the [README](../README.md) documentation.

The definition file has exactly one top-level key, `space`, mapping each tunable
parameter name to its **spec**. Nothing else is allowed at the top level, and
every parameter the selected engine forwards must be declared. The sample below
is a non-speculative vLLM space, which declares all ten base tunables:

```yaml
space:
  dtype:
    categorical: [bfloat16, float16]
  kv_cache_dtype:
    categorical: [auto, fp8_e4m3]
  max_model_len:
    fixed: 4096
  max_num_seqs:
    interval:
      min: 16
      max: 256
  max_num_batched_tokens:
    interval:
      min: 4096
      max: 8192
  max_num_active_seqs:
    interval:
      min: 1
      max: 16
  max_num_scheduled_tokens:
    interval:
      min: 2048
      max: 4096
  tensor_parallel_size:
    fixed: 1
  attention_backend:
    default: true
  prefix_caching:
    fixed: true
```

## Example definitions

Nine example definitions live under [`examples/vllm/`](../examples/vllm/) and
[`examples/llamacpp/`](../examples/llamacpp/) and can be used as starting points:

* [`examples/vllm/space-all-fixed.yaml`](../examples/vllm/space-all-fixed.yaml) pins
  all ten vLLM base tunables to a single value. This can be used as a
  reproducible baseline. Its `flash_attention` backend is CUDA-only; use
  `attention_backend: {default: true}` there on ROCm or CPU.
* [`examples/vllm/space-full.yaml`](../examples/vllm/space-full.yaml) declares every
  vLLM base tunable with a broad range, deferring the hardware-dependent
  `tensor_parallel_size` and `attention_backend` to the engine so it runs on any
  host.
* [`examples/vllm/space-speculative-defaults.yaml`](../examples/vllm/space-speculative-defaults.yaml)
  declares the same broad base space with speculation enabled at the engine's
  default depth; pair it with `--speculative-method` and, when the method needs
  one, `--draft-model`.
* [`examples/vllm/space-speculative-full.yaml`](../examples/vllm/space-speculative-full.yaml)
  adds the full speculative sweep: `enabled`/`disabled` against the depths
  `1`, `2`, `4`, and `8`.
* [`examples/llamacpp/space-all-fixed.yaml`](../examples/llamacpp/space-all-fixed.yaml)
  pins the seven llama.cpp base tunables to a single value, for a reproducible
  llama.cpp baseline.
* [`examples/llamacpp/space-full.yaml`](../examples/llamacpp/space-full.yaml)
  declares every llama.cpp base tunable with a broad range, deferring the
  build-dependent `attention_backend` to the engine.
* [`examples/llamacpp/space-gpu-offload.yaml`](../examples/llamacpp/space-gpu-offload.yaml)
  pins the other tunables and sweeps `n_gpu_layers` over `auto`, `0`, and
  `all`, so a GPU host can compare no offload, full offload, and the engine's
  own choice. `all` makes the space GPU-only; the comments explain how to add
  checkpoint-specific exact layer counts.
* [`examples/llamacpp/space-speculative-defaults.yaml`](../examples/llamacpp/space-speculative-defaults.yaml)
  is the llama.cpp broad space with speculation enabled at llama.cpp's default
  draft depth.
* [`examples/llamacpp/space-speculative-full.yaml`](../examples/llamacpp/space-speculative-full.yaml)
  is the llama.cpp speculative sweep over the same mode and depth values.

A shared 12-prompt corpus lives in
[`examples/prompts.jsonl`](../examples/prompts.jsonl). Pass
it with `--dataset-path` when you want real prompts rather than the synthetic
random workload; the speculative example calls below use it.

## Spec kinds

Each dimension carries exactly one spec tag:

| Spec          | Form                              | Meaning                                                                 |
| ------------- | --------------------------------- | ----------------------------------------------------------------------- |
| `fixed`       | `fixed: <value>`                  | Pin the parameter to one value that must not be varied.                 |
| `categorical` | `categorical: [<value>, ...]`     | A non-empty, duplicate-free subset of allowed values, in declared order. For a numeric parameter, give integers to sweep only those values. |
| `interval`    | `interval: {min: <int>, max: <int>}` | An inclusive integer range; the optimizer chooses where to sample. `min` must not exceed `max`. |
| `default`     | `default: true`                   | Explicitly defer to the engine's own default. Must be exactly boolean `true`. |

There is no step or count on an `interval`. When only specific numbers are
valid, use `categorical` with the exact values, e.g.
`max_model_len: {categorical: [1024, 4096, 8192]}`; only those values are
sampled and the optimizer cannot interpolate between them.

## Tunable parameters

The definition must declare every parameter the selected engine forwards — the
search space is **total over that engine's applicable set**, so a value can
never silently fall back to an engine default. Omitting one is an error naming
the missing parameter, and declaring one the engine does not forward is a
backend problem (exit code `4`). vLLM forwards ten of the eleven base
parameters below — every one except the llama.cpp-only `n_gpu_layers`;
llama.cpp forwards all but `dtype`, `tensor_parallel_size`,
`max_num_active_seqs`, and `max_num_scheduled_tokens`, which it omits
because a GGUF file carries its own numerical format, llama.cpp has no
tensor parallelism, and its scheduler has no equivalent caps, leaving the seven
that map to a `llama-server` flag. The two speculative parameters at the end of the table are
**conditional**: a definition must declare them exactly when the run configures
a speculative setup (see
[Speculative decoding](speculative-decoding.md)). Without one they
are not part of the applicable set, so a space with no speculation stays
unchanged. Write `default: true` for using an engine default.

| Parameter                | Kind                                            | Accepted values                                       |
| ------------------------ | ----------------------------------------------- | ----------------------------------------------------- |
| `dtype`                  | enum (fixed, categorical, default)              | `auto`, `float16`, `bfloat16`, `float32`              |
| `kv_cache_dtype`         | enum (fixed, categorical, default)              | `auto`, `float16`, `bfloat16`, `fp8_e4m3`, `fp8_e5m2`; llama.cpp also `float32`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` |
| `attention_backend`      | enum (fixed, categorical, default)              | `flash_attention`, `flash_infer`; llama.cpp also `auto`, `off` |
| `max_model_len`          | integer (interval, fixed, categorical, default) | ≥ 1                                                   |
| `max_num_seqs`           | integer (interval, fixed, categorical, default) | ≥ 1                                                   |
| `max_num_batched_tokens` | integer (interval, fixed, categorical, default) | ≥ 1                                                   |
| `max_num_active_seqs`    | integer (interval, fixed, categorical, default) | ≥ 1; vLLM only                                        |
| `max_num_scheduled_tokens` | integer (interval, fixed, categorical, default) | ≥ 1; vLLM only                                      |
| `tensor_parallel_size`   | integer (interval, fixed, categorical, default) | ≥ 1                                                   |
| `prefix_caching`         | boolean (fixed, categorical, default)           | `true`, `false`                                       |
| `n_gpu_layers`           | int-or-enum (fixed, categorical, interval, default) | `auto`, `all`, or an integer ≥ 0; llama.cpp only      |
| `speculative_mode`       | enum (fixed, categorical, default)              | `enabled`, `disabled`; conditional                    |
| `num_speculative_tokens` | integer (interval, fixed, categorical, default) | ≥ 1; conditional                                      |

`interval` is only valid for the integer parameters and the integer branch of an
int-or-enum union such as `n_gpu_layers`; declaring one for an enum, a boolean,
or a union without an integer branch is an error. `fixed` and `categorical` are
valid for every kind: an enum parameter's categorical values are written using
the engine-neutral spellings above (not the engine's own names), a numeric
parameter's categorical values are the integers themselves, and a boolean
parameter's values are the YAML booleans `true` and `false`. Booleans are
strict: the strings `"true"`/`"false"` and the integers `0`/`1` are rejected
rather than coerced. On a union parameter, an `interval` constrains the integer
values only, so `auto` and `all` are reachable through `categorical` or `fixed`
but never through a numeric range.

`prefix_caching` is applicable to both engines and is forwarded explicitly in
both directions: `true` enables prompt caching and `false` disables it, so
`false` is a real configuration rather than an omitted default. An omitted
value (`default: true`) leaves the engine's own default in place.

vLLM's two scheduling caps narrow the scheduler without shrinking the model
runner's capacity. `max_num_active_seqs` bounds the number of sequences active
in one decode batch, and `max_num_scheduled_tokens` bounds the token budget the
scheduler may fill per step; both are forwarded as the flat flags
`--max-num-active-seqs` and `--max-num-scheduled-tokens`. They also have to
stay within the values they narrow: `max_num_active_seqs` must not exceed
`max_num_seqs`, and `max_num_scheduled_tokens` must not exceed
`max_num_batched_tokens`. Each bound is checked against the most permissive
point the space allows, and is skipped when either side defers to the engine
with `default: true`; see [Validation](#validation). llama.cpp has no
equivalent caps and does not forward either parameter.

`n_gpu_layers` is llama.cpp-only and accepts the mode names `auto` and `all` or
an exact offloaded-layer count, which is a non-negative integer (`0` keeps the
model entirely on the CPU). A mixed categorical such as `[auto, 0, 8, all]`
compares modes and counts in one dimension, while an `interval` constrains only
the integer branch. A CPU target rejects `all` and any positive count while
accepting `auto` and `0`, so a host-agnostic space can still declare it. Once
the checkpoint is resolved, an exact count above the GGUF architecture's block
count is rejected and the user is directed to `all`; unreadable metadata fails
open and leaves the engine as the final authority. vLLM has no equivalent
setting and does not forward the parameter.

For vLLM, `float32` is accepted in general but some model implementations
reject it: Qwen3.5's linear-attention layers, for example, require bfloat16 or
float16. The shipped examples therefore omit `float32` so they stay portable,
and it should be added only for a checkpoint known to support it. Likewise, an
explicit `float16` or `bfloat16` `kv_cache_dtype` must match the resolved model
dtype; `auto` follows the model dtype and the FP8 cache formats may differ. The
backend rejects an explicit float cache unless the `dtype` dimension is pinned
to exactly that value, so a space that sweeps `dtype` should use
`kv_cache_dtype: auto` or the FP8 formats rather than the float cache formats.
The examples follow that rule, and a space that pins a float cache should do so
only for a checkpoint whose dtype is known.

For vLLM, `attention_backend` and `tensor_parallel_size` are the two
host-dependent parameters. `flash_attention` and `flash_infer` are CUDA-only, so
a ROCm or CPU run must declare `attention_backend: {default: true}` and let the
engine select its own backend; a space that can produce a
`tensor_parallel_size` larger than the resolved device count must likewise defer
it with `{default: true}` or pin a value the host can run. Either problem is
rejected up front (exit code `4`). vLLM does not model llama.cpp's `auto` and
`off` flash-attention settings, so a vLLM space declaring either one is
rejected as unsupported.

## Engine-specific spaces

A llama.cpp space without a speculative setup declares exactly the seven base
tunables that map to real `llama-server` flags:

```yaml
space:
  kv_cache_dtype:
    categorical: [float16, bfloat16, float32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1]
  max_model_len:
    interval: {min: 1024, max: 32768}
  max_num_seqs:
    interval: {min: 1, max: 256}
  max_num_batched_tokens:
    interval: {min: 2048, max: 8192}
  attention_backend:
    categorical: [auto, flash_attention]
  prefix_caching:
    fixed: true
  n_gpu_layers:
    default: true
```

llama.cpp requires flash attention for a quantized V cache, so the backend
rejects a space that can sample a quantized `kv_cache_dtype` together with
`attention_backend: off` (exit code `4`) instead of failing those trials at
server startup. Keep the attention dimension to `auto` and `flash_attention`
when the cache can be quantized, or restrict the cache to `float16`,
`bfloat16`, and `float32` when comparing flash attention off. The disabling
mode is written `"off"` in quotes: YAML parses a bare `off` as the boolean
`false`, which the enum kind rejects.

The engine-neutral names translate to llama.cpp as follows:

| Parameter                | `llama-server` flags               | Accepted values                                                                     |
| ------------------------ | ---------------------------------- | ----------------------------------------------------------------------------------- |
| `kv_cache_dtype`         | `--cache-type-k`, `--cache-type-v` | `float16` → `f16`, `bfloat16` → `bf16`, `float32` → `f32`; `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` pass through; `auto` and the FP8 members are unsupported. |
| `max_model_len`          | `--ctx-size`                       | Any integer ≥ 1.                                                                    |
| `max_num_seqs`           | `--parallel`                       | Any integer ≥ 1; capped to what `max_model_len` can serve (see below).              |
| `max_num_batched_tokens` | `--batch-size`, `--ubatch-size`    | Any integer ≥ 1; sets both the logical and the physical batch, which llama.cpp clamps to the context size rather than rejecting. A larger value needs more compute-buffer memory. |
| `attention_backend`      | `--flash-attn`                     | `auto` → `auto`, `flash_attention` → `on`, `off` → `off`; `off` needs a float KV cache; `flash_infer` and `xformers` are unsupported. |
| `prefix_caching`         | `--cache-prompt`, `--no-cache-prompt` | `true` → `--cache-prompt`, `false` → `--no-cache-prompt`; unset → neither.       |
| `n_gpu_layers`           | `--n-gpu-layers`                   | `auto` → `auto`, `all` → `all`; any integer ≥ 0 passes through unchanged.           |

llama.cpp divides `--ctx-size` evenly across the `--parallel` slots, so a
request longer than a slot's share is rejected. The backend therefore keeps
`--ctx-size` exactly at the sampled `max_model_len` and caps the forwarded slot
count at `max_model_len // request_tokens` for the workload, with at least one
slot; a sampled context below the workload's request length cannot be served at
any slot count and its trial fails. For the random workload the request length
is `input_len + output_len`. For a JSONL `--dataset-path` workload, the runner
starts the engine once before the sweep, tokenizes every prompt through the
server's `/tokenize` endpoint, and records the largest
`prompt_tokens + output_len` as `max_request_tokens`, so slots are capped from
the engine's real tokenization rather than from character counts. If that
preflight cannot run, llama.cpp conservatively forwards one slot, which gives
each prompt the whole context.

Only the values above are spellable: a fixed or categorical value with no
mapping — an `auto` KV cache, say — is rejected before the sweep starts rather
than silently dropped, so a declared value always reaches the server. Use
`default: true` to let llama.cpp choose for itself. `dtype` and
`tensor_parallel_size` must not be declared for llama.cpp; the backend reports
them as parameters it does not forward.

vLLM's non-speculative space is the sample at the top of this section: it
declares all ten base tunables, and every value is translated to vLLM's own
spelling. The KV-cache `auto` and FP8 members and the `flash_infer` attention
backend have no llama.cpp mapping, while vLLM has no spelling for `float32` or
the quantized KV-cache formats and for llama.cpp's `auto` and `off` attention
modes; `xformers` is not modelled by either engine.
A run that configures speculative decoding adds the two gated dimensions to
either engine's space; see
[Speculative decoding](speculative-decoding.md).

`seed`, `gpu_memory_utilization`, and `quantization` are intentionally **not**
tunable. `seed` (vLLM only) and `gpu_memory_utilization` (vLLM only) are
reproducibility and fit knobs set on the command line (`--llm-seed`,
`--gpu-memory-utilization`); `quantization` describes how the checkpoint was
produced, so it is a model property rather than a runtime choice and is set with
`--quantization` (vLLM only). The measurement controls — `repetitions`,
`warmup`, `startup_timeout`, and the overall container `timeout` — are
non-tunables too: they configure *how* a point is measured, not which point is
tried, so they are set on the command line (`--repetitions`, `--warmup-runs`,
`--startup-timeout`, `--timeout`) and never enter the search space.

Because it is a checkpoint property, vLLM's `--quantization` is validated
against the resolved checkpoint before the image is pulled: a quantized
checkpoint accepts only the method it was produced with, and an unquantized
checkpoint accepts only a method the engine can apply online (`fp8`). Omit
`--quantization` to let the engine auto-detect the method, which is always
accepted. For llama.cpp the flag is a usage error in the first place: the GGUF
metadata is the checkpoint's quantization, and the tool reads it to report what
the run serves. A checkpoint that is not GGUF is rejected before the image is
pulled.

## Validation

Validation happens in three stages, all before any container runs:

1. **Engine-agnostic**: Performed when the file is loaded. It checks the
   top-level shape, registry membership, that each dimension carries exactly one
   recognized tag, that values match the parameter's kind, that enum values are
   known, that categoricals are non-empty and unique, that intervals are ordered,
   and the universal `≥ 1` bounds. **Every problem found is reported at once**,
   so a broken definition can be fixed in a single pass.
2. **Backend-specific**: The selected backend is asked whether it forwards each
   declared parameter and supports each value. vLLM additionally checks that a
   value is supported on the resolved accelerator class and fits the resolved
   device count, and enforces its cross-parameter rules
   `max_num_batched_tokens >= max_num_seqs`,
   `max_num_seqs >= max_num_active_seqs`, and
   `max_num_batched_tokens >= max_num_scheduled_tokens` (each checked against
   the most permissive point the space allows, and skipped when either side
   defers to the engine). It also rejects an explicit `float16`/`bfloat16`
   cache unless `dtype` is pinned to the matching value, because vLLM stores a
   non-quantized KV cache in the model dtype; FP8 caches are exempt. llama.cpp
   rejects a parameter it does not forward, any enum value it has no spelling
   for, a space that can pair a quantized KV cache with
   `attention_backend: off`, and — on a CPU target — `n_gpu_layers=all` or a
   positive exact count.
3. **Model-aware**: Once the checkpoint is resolved to a local path (a host path,
   the snapshot a Hugging Face download produced, or the single GGUF file
   llama.cpp fetches), the backend inspects it and rejects a configuration it
   cannot serve. vLLM reads the quantization method the checkpoint declares and
   rejects a `--quantization` the checkpoint cannot serve; llama.cpp rejects a
   non-GGUF checkpoint, a configured `--quantization`, and an exact
   `n_gpu_layers` count above the architecture's block count (pointing at
   `all`), naming the quantization the GGUF metadata declares. The check runs
   before the container image is pulled, so a doomed configuration does not
   wait on a multi-gigabyte download. An omitted `--quantization` always
   passes, because the engine auto-detects the method, and a checkpoint whose
   metadata cannot be read — including one without a block count — is likewise
   accepted, so custom checkpoints are never blocked.

Malformed YAML and a non-mapping document are reported through the same error as
a structural problem, so a broken file fails uniformly with exit code `3`; an
unsupported space or model fails with exit code `4`.

## All-default and mixed examples

A vLLM space that pins nothing and lets the engine choose everything:

```yaml
space:
  dtype: {default: true}
  kv_cache_dtype: {default: true}
  max_model_len: {default: true}
  max_num_seqs: {default: true}
  max_num_batched_tokens: {default: true}
  max_num_active_seqs: {default: true}
  max_num_scheduled_tokens: {default: true}
  tensor_parallel_size: {default: true}
  attention_backend: {default: true}
  prefix_caching: {default: true}
```

The llama.cpp equivalent declares only the seven it forwards:

```yaml
space:
  kv_cache_dtype: {default: true}
  max_model_len: {default: true}
  max_num_seqs: {default: true}
  max_num_batched_tokens: {default: true}
  attention_backend: {default: true}
  prefix_caching: {default: true}
  n_gpu_layers: {default: true}
```

A mixed vLLM space exploring sequence length, batch size, and precision:

```yaml
space:
  dtype:
    categorical: [bfloat16, float16]
  kv_cache_dtype:
    categorical: [auto, fp8_e4m3]
  max_model_len:
    fixed: 8192
  max_num_seqs:
    interval: {min: 32, max: 512}
  max_num_batched_tokens:
    interval: {min: 2048, max: 16384}
  max_num_active_seqs:
    interval: {min: 1, max: 32}
  max_num_scheduled_tokens:
    interval: {min: 1024, max: 2048}
  tensor_parallel_size:
    categorical: [1, 2]
  attention_backend:
    default: true
  prefix_caching:
    categorical: [true, false]
```

---
