# Search-space YAML format

Part of the [README](../README.md) documentation.

The definition file has exactly one top-level key, `space`, mapping each tunable
parameter name to its **spec**. Nothing else is allowed at the top level, and
every parameter the selected engine forwards must be declared. The sample below
is a non-speculative vLLM space, which declares all twelve base tunables:

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
  expert_parallel:
    categorical: [true, false]
  all2all_backend:
    categorical: [allgather_reducescatter, naive]
```

## Example definitions

Eleven example definitions live under [`examples/vllm/`](../examples/vllm/) and
[`examples/llamacpp/`](../examples/llamacpp/) and can be used as starting points:

* [`examples/vllm/space-all-fixed.yaml`](../examples/vllm/space-all-fixed.yaml) pins
  all twelve vLLM base tunables to a single value. This can be used as a
  reproducible baseline. Its `flash_attention` backend is CUDA-only; use
  `attention_backend: {default: true}` there on ROCm or CPU.
* [`examples/vllm/space-full.yaml`](../examples/vllm/space-full.yaml) declares every
  vLLM base tunable with a broad range, deferring the hardware-dependent
  `tensor_parallel_size` and `attention_backend` to the engine so it runs on any
  host.
* [`examples/vllm/space-moe.yaml`](../examples/vllm/space-moe.yaml) compares the
  two portable `all2all_backend` transports with `tensor_parallel_size: 2` and
  `expert_parallel: true` pinned, so the experts are actually sharded. It needs
  a host with at least two devices; on a dense checkpoint the model-aware
  assessment drops both MoE dimensions after the checkpoint resolves.
* [`examples/vllm/space-speculative-defaults.yaml`](../examples/vllm/space-speculative-defaults.yaml)
  declares the same broad base space with speculation enabled at the engine's
  default depth; pair it with `--speculative-method` and, when the method needs
  one, `--draft-model`.
* [`examples/vllm/space-speculative-full.yaml`](../examples/vllm/space-speculative-full.yaml)
  adds the full speculative sweep: `enabled`/`disabled` against the depths
  `1`, `2`, `4`, and `8`.
* [`examples/llamacpp/space-all-fixed.yaml`](../examples/llamacpp/space-all-fixed.yaml)
  pins the eight llama.cpp base tunables to a single value, for a reproducible
  llama.cpp baseline.
* [`examples/llamacpp/space-full.yaml`](../examples/llamacpp/space-full.yaml)
  declares every llama.cpp base tunable with a broad range, deferring the
  build-dependent `attention_backend` to the engine.
* [`examples/llamacpp/space-gpu-offload.yaml`](../examples/llamacpp/space-gpu-offload.yaml)
  pins the other tunables and sweeps `n_gpu_layers` over `auto`, `0`, and
  `all`, so a GPU host can compare no offload, full offload, and the engine's
  own choice. `all` makes the space GPU-only; the comments explain how to add
  checkpoint-specific exact layer counts.
* [`examples/llamacpp/space-moe.yaml`](../examples/llamacpp/space-moe.yaml)
  compares expert CPU offload over `0`, two exact layer counts, and `all`, with
  `n_gpu_layers: all` pinned so the CPU placement of the experts is meaningful.
  The exact counts are checkpoint-specific — a count above the GGUF block count
  is rejected after the model resolves — while `all` names full offload without
  knowing the layer count.
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
backend problem (exit code `4`). vLLM forwards twelve of the fourteen base
parameters below — every one except the two llama.cpp offload settings
`n_gpu_layers` and `n_cpu_moe`; llama.cpp forwards all but `dtype`,
`tensor_parallel_size`, `max_num_active_seqs`, `max_num_scheduled_tokens`,
`expert_parallel`, and `all2all_backend`, which it omits because a GGUF file
carries its own numerical format, llama.cpp has no tensor or expert parallelism
and no all-to-all transport, and its scheduler has no equivalent caps, leaving
the eight that map to a `llama-server` flag. The two speculative parameters at
the end of the table are **conditional**: a definition must declare them
exactly when the run configures a speculative setup (see
[Speculative decoding](speculative-decoding.md)). Without one they are not part
of the applicable set, so a space with no speculation stays unchanged. Write
`default: true` for using an engine default.

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
| `expert_parallel`        | boolean (fixed, categorical, default)           | `true`, `false`; vLLM only                            |
| `all2all_backend`        | enum (fixed, categorical, default)              | `allgather_reducescatter`, `naive`; vLLM only         |
| `n_cpu_moe`              | int-or-enum (fixed, categorical, interval, default) | `all`, or an integer ≥ 0; llama.cpp only          |
| `speculative_mode`       | enum (fixed, categorical, default)              | `enabled`, `disabled`; conditional                    |
| `num_speculative_tokens` | integer (interval, fixed, categorical, default) | ≥ 1; conditional                                      |

`interval` is only valid for the integer parameters and the integer branch of an
int-or-enum union such as `n_gpu_layers` or `n_cpu_moe`; declaring one for an
enum, a boolean, or a union without an integer branch is an error. `fixed` and
`categorical` are valid for every kind: an enum parameter's categorical values
are written using the engine-neutral spellings above (not the engine's own
names), a numeric parameter's categorical values are the integers themselves,
and a boolean parameter's values are the YAML booleans `true` and `false`.
Booleans are strict: the strings `"true"`/`"false"` and the integers `0`/`1`
are rejected rather than coerced. On a union parameter, an `interval` constrains
the integer values only, so `auto` and `all` are reachable through
`categorical` or `fixed` but never through a numeric range.

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

`n_cpu_moe` is llama.cpp's MoE counterpart and accepts the mode name `all` or
an exact count of expert layers to keep in system RAM, which is a non-negative
integer (`0` disables the offload). The engine models the two choices as
different flags: an exact count is forwarded as `--n-cpu-moe N`, while `all`
emits the value-less `--cpu-moe`, so the mode never becomes `--n-cpu-moe all`.
A mixed categorical such as `[0, 8, 16, all]` compares counts with full offload,
while an `interval` constrains only the integer branch. Once the checkpoint is
resolved, an exact count or an interval maximum above the GGUF architecture's
block count is rejected and the user is directed to `all`; unreadable metadata
fails open. A CPU target drops the dimension entirely, because no weight ever
leaves system RAM there. vLLM has no equivalent setting and does not forward the
parameter.

`expert_parallel` and `all2all_backend` are vLLM's MoE pair. Expert parallelism
shards the experts across devices and is forwarded in both directions: `true`
emits `--enable-expert-parallel` and `false` emits `--no-enable-expert-parallel`,
so disabling it is a real configuration rather than an omitted default. The
all-to-all backend names the transport that sharding communicates over and is
forwarded flat as `--all2all-backend <value>`; only `allgather_reducescatter` and
`naive` are modelled, because they run in the default image without extra
communication libraries. Faster transports are reachable through `--extra-arg`
but are not first-class tunables. llama.cpp has neither setting and does not
forward either parameter.

The MoE settings can be no-ops, so the [model-aware
assessment](#model-aware-assessment) drops them when they cannot change the
run: vLLM drops both on a checkpoint with no experts and when the space can
never select a tensor-parallel size above one, plus `all2all_backend` alone
when expert parallelism can never be enabled; llama.cpp drops `n_cpu_moe` on a
dense checkpoint and on a CPU target.

For vLLM, `float32` is accepted in general but some model implementations
reject it: Qwen3.5's linear-attention layers, for example, require bfloat16 or
float16. A checkpoint's declared `dtype` is its **preferred** execution dtype,
not an allowlist: Qwen3.5 declares bfloat16, yet float16 also runs. The
[model-aware assessment](#model-aware-assessment) therefore removes `float32`
from the `dtype` dimension when the checkpoint declares linear-attention layers
once the model resolves, and the shipped examples omit `float32` so they stay
portable across checkpoints. Add it only for a checkpoint whose kernels are
known to accept it; the assessment covers the linear-attention case, not every
model.

Likewise, an explicit `float16` or `bfloat16` `kv_cache_dtype` must match the
resolved model dtype; `auto` follows the model dtype and the FP8 cache formats
may differ. Before the checkpoint is read, the backend rejects an explicit
float cache only when `dtype` pins exactly one different explicit precision. An
`auto`, deferred, or multi-value `dtype` is reconciled after the model
resolves: a float cache value survives only when every dtype the narrowed
dimension can still select resolves to that precision. A space that sweeps
`dtype` should therefore use `kv_cache_dtype: auto` or the FP8 formats, as the
examples do, unless the dtype dimension pins one known precision.

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

A llama.cpp space without a speculative setup declares exactly the eight base
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
  n_cpu_moe:
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
| `n_cpu_moe`              | `--n-cpu-moe`, `--cpu-moe`         | `all` → the value-less `--cpu-moe`; any integer ≥ 0 → `--n-cpu-moe N`.              |

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
`default: true` to let llama.cpp choose for itself. `dtype`,
`tensor_parallel_size`, `expert_parallel`, and `all2all_backend` must not be
declared for llama.cpp; the backend reports them as parameters it does not
forward.

vLLM's non-speculative space is the sample at the top of this section: it
declares all twelve base tunables, and every value is translated to vLLM's own
spelling. The KV-cache `auto` and FP8 members and the `flash_infer` attention
backend have no llama.cpp mapping, while vLLM has no spelling for `float32` or
the quantized KV-cache formats and for llama.cpp's `auto` and `off` attention
modes; `xformers` is not modelled by either engine. vLLM's MoE pair has no
llama.cpp counterpart, just as `n_cpu_moe` has no vLLM one.
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
   cache when `dtype` pins exactly one different explicit precision, because
   vLLM stores a non-quantized KV cache in the model dtype; an `auto`,
   deferred, or multi-value `dtype` is reconciled after the checkpoint
   resolves, and FP8 caches are exempt. llama.cpp
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
   accepted, so custom checkpoints are never blocked. The [model-aware
   assessment](#model-aware-assessment) then runs, and the sweep uses the
   effective space it returns.

Malformed YAML and a non-mapping document are reported through the same error as
a structural problem, so a broken file fails uniformly with exit code `3`; an
unsupported space or model fails with exit code `4`.

## Model-aware assessment

A space is written before the checkpoint is resolved, so its author cannot know
which dimensions still change the run. After model validation succeeds and
before the image is pulled, the backend assesses the space against the resolved
checkpoint, the fixed configuration, and the detected hardware, and the sweep
runs the **effective** space it returns. Assessment can change the space in two
ways: a dimension that cannot affect the run is removed, and a `fixed` or
`categorical` dimension can be narrowed to the declared values the checkpoint
can actually run. An `interval` is never narrowed. Every retained dimension
keeps its canonical order, a narrowed dimension keeps its declared value order,
and each removal or narrowing is reported.

### Mixture-of-Experts detection

Each backend decides from the checkpoint itself whether it is a
Mixture-of-Experts model:

* vLLM reads `config.json` and treats the checkpoint as MoE when a positive
  `num_experts`, `n_routed_experts`, or `num_local_experts` appears at the top
  level or in a nested `text_config` object. A readable configuration with none
  of those keys is confidently dense.
* llama.cpp reads the GGUF header's architecture-prefixed `<arch>.expert_count`.
  A readable header without that key, or with a count of zero, is confidently
  dense.

A checkpoint whose metadata cannot be read is **unknown**, not dense: the MoE
dimensions are kept and the engine remains the final authority. The asymmetry
is deliberate — confident dense detection avoids wasted trials, while an
unreadable custom checkpoint must not be treated as dense on a guess.

### Dimensions dropped

On a confidently dense checkpoint, vLLM drops `expert_parallel` and
`all2all_backend`, and llama.cpp drops `n_cpu_moe`; none of them can change a
model with no experts. Beyond that:

* vLLM drops both MoE dimensions when the space can never select a
  tensor-parallel size above one, because vLLM derives its expert-parallel size
  from the tensor-parallel and data-parallel sizes. A deferred
  `tensor_parallel_size: {default: true}` resolves through the detected device
  count for this decision; a space that does not declare the dimension has no
  determinable ceiling, so the rule stays inactive.
* vLLM drops `all2all_backend` when the space can never pass
  `--enable-expert-parallel` — a categorical without `true`, a `false` toggle,
  or a deferred `expert_parallel: {default: true}`, which forwards no flag and
  leaves vLLM's own default (off) in place. The toggle stays samplable, because
  it can still matter to a space that splits the model.
* llama.cpp drops `n_cpu_moe` on a CPU target, where no weight ever leaves
  system RAM, and `n_gpu_layers` when every value it can select already
  disables offload (`auto` or an exact `0`). `all` and positive counts are
  still rejected before the sweep by the CPU-target validation, so a
  GPU-oriented space never silently changes meaning on a CPU host.

### Model dtype and value narrowing

vLLM reads two optional facts from the resolved checkpoint's `config.json`:

* the **preferred execution dtype** — `dtype` or `torch_dtype`, with a nested
  `text_config` value winning over the top level — which is the precision an
  unset or `auto` `--dtype` adopts;
* the **per-layer attention types**, from a `layer_types` list at the top level
  or in `text_config`, whose `linear_attention` entries identify
  linear-attention layers.

The preferred dtype is **not** an allowlist of every precision the checkpoint
can run: a checkpoint that declares `bfloat16` often runs `float16` as well. It
identifies the precision an `auto` dtype resolves to; it does not narrow the
space by itself.

The one backend-owned kernel rule is the linear-attention case: those layers
call `ChunkGatedDeltaRuleFunction`, which requires `float16` or `bfloat16` and
rejects `float32`. A checkpoint that declares linear-attention layers therefore
loses `float32` from `dtype`, keeping whichever of `float16`, `bfloat16`, and
`auto` the space declared. A checkpoint that declares only full-attention
layers, declares no layer types, or cannot be read keeps every value.

After dtype narrowing, explicit `float16` and `bfloat16` **KV-cache values** are
reconciled with the dtypes that remain. vLLM stores a non-quantized KV cache in
the model dtype, so such a cache value is retained only when *every* dtype the
narrowed dimension can still select resolves to that precision. An explicit
dtype resolves directly, `auto` resolves to the checkpoint's declared model
dtype, and a deferred `dtype: {default: true}` means `auto`. The `auto` and FP8
cache values are always retained because they do not have to match the model
dtype. This is the model-aware half of the cache rule: `validate` rejects only
a `dtype` dimension pinned to a different explicit precision before the
checkpoint is read, while `auto`, deferred, and multi-value dtype dimensions
reach this reconciliation.

Both dtype rules fail open. An unreadable or incomplete `config.json`, or one
that declares no layer types, leaves `dtype` alone. An unknown model dtype is
not evidence that a float cache is wrong: when the `dtype` dimension contains
an `auto` or deferred choice the checkpoint cannot resolve, the cache
reconciliation is skipped and the declared cache values stay in place. In both
cases the engine remains the final authority. llama.cpp has no equivalent rule:
a GGUF file's `general.file_type` describes its weight quantization, not which
activation dtypes the engine can execute.

### Layer-count bounds

The retained llama.cpp dimensions are checked against the checkpoint's block
count: an exact `n_cpu_moe` or `n_gpu_layers` count above it, or an interval
whose maximum exceeds it, is a model-aware problem that points at the matching
`all` mode. `all`, `auto`, and deferred dimensions declare no count and are
never bounded, and absent or unreadable block-count metadata fails open so the
engine remains the final authority.

### Reported changes and the empty space

Every dropped dimension and every removed value is one printed line naming the
parameter, the value, and the fact that made it unusable, for example:

```text
dropped space.expert_parallel: the checkpoint declares no experts
dropped space.all2all_backend: the checkpoint declares no experts
dropped space.dtype.categorical[0] (float32): the checkpoint's linear-attention layers reject float32 (ChunkGatedDeltaRuleFunction requires float16 or bfloat16)
dropped space.kv_cache_dtype.categorical[0] (float16): vLLM stores an explicit float KV cache in the model dtype, but the dtype dimension does not resolve only to 'float16'
```

A model-aware problem that pruning cannot fix takes the same
unsupported-configuration path as the validation stages (exit code `4`) and is
reported before the image is pulled. A dimension whose declared values are
**all** removed is such a problem, not an empty space: the dimension is left as
declared so the failure names it, and the run exits with code `4` instead of
silently substituting an engine default. If the assessment instead removes
every *dimension*, the run prints that the effective space is empty and
executes exactly **one** trial at engine defaults, so the run still produces a
baseline instead of repeating one configuration under the requested trial
count.

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
  expert_parallel: {default: true}
  all2all_backend: {default: true}
```

The llama.cpp equivalent declares only the eight it forwards:

```yaml
space:
  kv_cache_dtype: {default: true}
  max_model_len: {default: true}
  max_num_seqs: {default: true}
  max_num_batched_tokens: {default: true}
  attention_backend: {default: true}
  prefix_caching: {default: true}
  n_gpu_layers: {default: true}
  n_cpu_moe: {default: true}
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
  expert_parallel:
    categorical: [true, false]
  all2all_backend:
    categorical: [allgather_reducescatter, naive]
```

---
