# Speculative decoding spaces

Part of the [README](../README.md) documentation.

Speculative decoding lets a small draft model (or the target's own heads) propose
tokens that the target verifies in one pass. The **method** and the optional
**draft model** are fixed for the whole run with `--speculative-method` and
`--draft-model`, while the sweep decides whether speculation is on and how many
tokens the draft proposes:

- `speculative_mode` — `enabled` or `disabled`. A disabled trial starts the
  engine with **no** speculative flags at all, so the draft model's memory is
  genuinely freed and the on/off comparison is honest.
- `num_speculative_tokens` — the draft depth, an integer ≥ 1.

Both dimensions are required exactly when the run configures a speculative
setup, and rejected when it does not. Omitting either with a setup is a missing
parameter error (exit code `3`); declaring either without one is a backend
problem (exit code `4`). A non-speculative space is therefore unchanged.

The method is fixed per run on purpose: methods are checkpoint-specific, so
sweeping them would mostly produce server-start-up failures. Compare setups
with one run each.

## End-to-end example

```console
$ llm-config-optimization Qwen/Qwen3.8-27B examples/vllm/space-speculative-full.yaml vllm \
    --speculative-method eagle3 \
    --draft-model Qwen/Qwen3.8-27B-EAGLE3 \
    --dataset-path examples/prompts.jsonl \
    --study-name qwen3.8-27b-eagle3 \
    --output-config best-eagle3.yaml
```

The matching space declares the twelve base vLLM tunables **and** both gated
speculative dimensions:

```yaml
space:
  dtype:
    categorical: [auto, float16, bfloat16]
  kv_cache_dtype:
    categorical: [auto, fp8_e4m3, fp8_e5m2]
  max_model_len:
    interval: {min: 1024, max: 32768}
  max_num_seqs:
    interval: {min: 16, max: 256}
  max_num_batched_tokens:
    interval: {min: 4096, max: 8192}
  max_num_active_seqs:
    interval: {min: 1, max: 16}
  max_num_scheduled_tokens:
    interval: {min: 2048, max: 4096}
  tensor_parallel_size:
    default: true
  attention_backend:
    default: true
  prefix_caching:
    fixed: true
  expert_parallel:
    categorical: [true, false]
  all2all_backend:
    categorical: [allgather_reducescatter, naive]
  speculative_mode:
    categorical: [enabled, disabled]
  num_speculative_tokens:
    categorical: [1, 2, 4, 8]
```

For llama.cpp the space is the eight base tunables for that engine — including
`n_gpu_layers` and `n_cpu_moe` — plus the same two speculative dimensions, and
`--draft-model` takes a local `.gguf` file or a Hub repository with the optional
`:QUANT` selector, exactly like `MODEL`:

```console
$ llm-config-optimization /models/Qwen3.5-8B-Q4_K_M.gguf examples/llamacpp/space-speculative-full.yaml llamacpp \
    --speculative-method draft_model \
    --draft-model bartowski/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf \
    --dataset-path examples/prompts.jsonl \
    --study-name qwen3.5-8b-draft
```

Both engines also ship a `space-speculative-defaults.yaml` variant that enables
the configured setup at the engine's default draft depth instead of sweeping
it; see the [example definitions](search-space.md#example-definitions).

The six engine-neutral methods map onto each engine's own vocabulary:

| Method        | vLLM `--speculative-config` `method` | llama.cpp `--spec-type` |
| ------------- | ------------------------------------ | ----------------------- |
| `draft_model` | `draft_model`                        | `draft-simple`          |
| `eagle3`      | `eagle3`                             | `draft-eagle3`          |
| `mtp`         | `mtp`                                | `draft-mtp`             |
| `dflash`      | `dflash`                             | `draft-dflash`          |
| `dspark`      | `dspark`                             | `draft-dspark`          |
| `ngram`       | `ngram`                              | `ngram-simple`          |

`mtp` uses the target model's own multi-token-prediction heads and needs no
draft checkpoint; `ngram` is model-free and rejects a draft; the other four
require one. `--draft-model` without `--speculative-method` implies
`draft_model`, so the common case is a single flag. A method that requires a
draft without one, or a draft passed to `ngram`, is a usage error (exit code
`2`).

vLLM forwards the method, draft, and depth in one `--speculative-config` JSON
object (`method`, `model`, `num_speculative_tokens`). llama.cpp forwards
`--spec-type`, the draft as `--spec-draft-model` (or `--spec-draft-hf` for a
repository it leaves to the server), and the depth as `--spec-draft-n-max`. The
draft follows the same source rules as `MODEL`: vLLM accepts a Hugging Face id
or a local directory, while llama.cpp accepts a local `.gguf` file or a Hub
repository with an optional `:QUANT` selector and fetches a single file. It is
resolved and downloaded on the host through the same single progress line as
`MODEL`, mounted into the container, and validated where cheap — a non-GGUF
llama.cpp draft is rejected before the image is pulled. The draft is resolved
(and, for a remote source, downloaded) once per run even when the space can
only sample `disabled`, because the space is not consulted for resolution.

How the dimensions' values behave:

- `speculative_mode: {default: true}` means **enabled**, the configured setup's
  default — not the engine's own no-speculation default, which would contradict
  the setup the run explicitly configured.
- `num_speculative_tokens: {default: true}` omits the depth and lets the engine
  choose: llama.cpp uses its own built-in default, while vLLM infers one where
  the model metadata allows and fails server startup for methods that require
  the value.
- The depth is sampled on every trial but applied only when the mode is enabled.
  A categorical depth axis can therefore produce duplicate disabled points;
  this is expected and harmless to the optimizer.
- The method and draft model are fixed and are written into the winning
  configuration along with the recommended mode, depth, and the resolved draft
  path, so `--output-config` produces a re-runnable speculative configuration.

A study records neither the method nor the draft model, so use a distinct
`--study-name` per speculative setup; see
[Study persistence](usage.md#study-persistence).

> **Warning:** speculative decoding earns its overhead from *accepted* draft
> tokens, and the default random-token workload proposes drafts the target
> almost always rejects. When a setup is configured without `--dataset-path`,
> the runner prints an advisory warning on stderr before the sweep:
>
> ```
> warning: speculative decoding rarely helps on the synthetic random workload; pass `--dataset-path` with real prompts
> ```
>
> Use a real JSONL dataset to measure the benefit. Measuring the worst case might be a
> relevant use-case, which is why this is a warning rather than an error.
