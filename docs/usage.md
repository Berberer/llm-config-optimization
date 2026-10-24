# Usage reference

This guide covers engine selection, the benchmark workload, objectives and phase
metrics, every command-line option, study persistence, exit codes, and complete
example calls. For installation and the command synopsis see the
[README](../README.md#quick-start).

## Engines

The selected engine decides which tunables a space must declare, how `MODEL` is
resolved, which server command runs, and which image is used. Both engines are
driven by the same host-side client, so their workload and metric definitions
are identical, but their defaults and scoped options differ.

| Engine     | Checkpoint                                              | Applicable tunables                                    | Cache reset between runs      | Phase rates from               |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------ | ----------------------------- | ------------------------------ |
| `vllm`     | A Hugging Face id or a local snapshot directory.        | The seven base tunables in [the search-space format](search-space.md), plus the two speculative dimensions when a setup is configured. | vLLM dev-mode endpoints.      | Prometheus `/metrics`.         |
| `llamacpp` | A local `.gguf` file or a Hub repository with `:QUANT`. | The five base tunables llama.cpp forwards, plus the two speculative dimensions when a setup is configured. | None; llama.cpp exposes none. | The `timings` response object. |

The **default images**, used when `--image` is omitted, are:

| Target | vLLM                                     | llama.cpp                                |
| ------ | ---------------------------------------- | ---------------------------------------- |
| `cuda` | `docker.io/vllm/vllm-openai:latest`      | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| `rocm` | `docker.io/vllm/vllm-openai-rocm:latest` | `ghcr.io/ggml-org/llama.cpp:server-rocm` |
| `cpu`  | `docker.io/vllm/vllm-openai-cpu:latest`  | `ghcr.io/ggml-org/llama.cpp:server`      |

**llama.cpp model argument.** `MODEL` is either a host path to a `.gguf` file
or a Hugging Face repository id, optionally followed by `:` and a selector: the
GGUF file's name or a quantization tag. A repository that holds exactly one GGUF
file needs no selector; one that holds several requires it, so the tool never
guesses which quantization you meant. The exact file name always names one
file — `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q6_K.gguf`; anything shorter is
matched as a case-insensitive substring, so `:q4_k_m` finds `...-Q4_K_M.gguf`
and the `.gguf` suffix may be left off. A shorter selector is only unambiguous
while no other file's name contains it, though: in
`unsloth/Qwen3.5-0.8B-GGUF`, `:Q6_K` also matches the dynamic
`...-UD-Q6_K_XL.gguf` and is rejected with the candidates listed, so name the
file when tags overlap. The resolved file is mounted into the server and its
GGUF metadata is read before the image is pulled: a checkpoint that is not GGUF
is rejected, and the quantization it declares is what the run serves.

`--quantization` and `--gpu-memory-utilization` are **vLLM-only**: for
`llamacpp` both are usage errors (exit code `2`) before any preparation runs,
because a GGUF file carries its own quantization and llama.cpp sizes memory with
its own flags. `--llm-seed` is likewise forwarded by vLLM only; the host
client's workload is greedy, so llama.cpp needs no engine-side seed. Everything
else on the command line behaves the same for both engines.

Both engines also support **speculative decoding**. Its method and draft model
are fixed on the command line, while whether it is enabled and how deep it
drafts are two gated search-space dimensions; see
[Speculative decoding](speculative-decoding.md) for the method
mapping and a complete example.

## Objective and phase metrics

Every measured run is summarised by up to three rates, all in tokens per
second:

- **combined** — all prompt and generated tokens over the run's elapsed
  wall-clock time. This is the historical metric; with a large prompt-to-output
  ratio it is dominated by prompt tokens.
- **prefill** — the prompt tokens the engine reports processing over the
  seconds it reports spending on prefill.
- **decode** — the generated tokens over the seconds the engine reports
  spending on generation.

Prefill and decode are **phase-work rates**, not a partition of the run's
wall-clock time: continuous batching and chunked prefill interleave the two
phases, so only the engine can attribute work to one. Each engine is read from
its own source, and neither needs a flag or a changed image:

| Engine     | Phase-rate source |
| ---------- | ----------------- |
| `vllm`     | The Prometheus `/metrics` endpoint, scraped immediately before and after each measured run. Prefill is the delta of `vllm:prompt_tokens_total` over the delta of `vllm:request_prefill_time_seconds_sum`; decode is `vllm:generation_tokens_total` over `vllm:request_decode_time_seconds_sum`. The prefix cache is reset before every run, so prompt tokens served from it are excluded. |
| `llamacpp` | The `timings` object the server attaches to every completion response: `prompt_n` over `prompt_ms` and `predicted_n` over `predicted_ms`, summed over the run's responses. `prompt_n` already excludes tokens served from the cache. |

A scrape that fails, or a response without timings, leaves the phase rate
absent rather than failing the run. Each run's full payload stays on the result
for audit, and a trial's phase rates are the means over the runs that reported
one — a run without a rate is not counted as zero.

`--objective` chooses what the sweep maximizes:

- **`multi`** (the default) submits prefill and decode as two independent
  objectives, so Optuna keeps a **Pareto front** of non-dominated
  configurations. The trade-off decision is therefore made after the sweep
  rather than baked into weights beforehand; the final summary prints the front
  with a recommended row, the per-trial log and the TUI show both rates, and
  `--output-config` writes the recommended point.
- **`weighted`** folds the two rates into the weighted-harmonic effective rate

  ```
  (w_p + w_d) / (w_p / prefill + w_d / decode)
  ```

  where `w_p` and `w_d` are the prefill and decode weights. The weights are a
  **token mix** — counts, not fractions — so any positive scaling of a pair
  gives the same score, and the result stays a tokens-per-second rate
  interpretable as the throughput of a request shaped like the weights. They
  default to the benchmark's `--input-len` and `--output-len`, so
  `--objective weighted` needs no arithmetic while still biasing the score
  toward the deployment's shape; `--prefill-weight` and `--decode-weight`
  override them. A zero weight degenerates the score to the other phase; both
  zero is rejected. For a JSONL `--dataset-path` the workload's real mix is not
  known at launch, so the derived defaults are only a stand-in — pass explicit
  weights to match the dataset.
- **`combined`** keeps the historical single objective: the mean combined
  throughput of the successful runs. It is the escape hatch for an engine that
  reports no phase data, and how studies persisted before objective modes
  existed are interpreted.

The multi and weighted modes require every phase rate they actually score; a
trial whose engine reported neither a `/metrics` snapshot nor response timings
fails with an error naming the missing source instead of silently optimizing
something else. The combined mode tolerates absent phase data. If no token mix
can be derived for multi or weighted, both weights must be given explicitly.
The weights are validated before any preparation runs: both must be finite and
non-negative, and at least one must be positive, or the run exits with a usage
error.

> **Note:** the prefill and decode rates are new measurements. They are
> engine-reported phase-work rates rather than wall-clock partitions, so they
> are **not comparable** to the combined throughput — and because a study
> records the phase-metric schema version its scores were measured under,
> numbers from different versions are not comparable either.

## Options

### Container

| Flag             | Values                                   | Default | Description                                                                 |
| ---------------- | ---------------------------------------- | ------- | --------------------------------------------------------------------------- |
| `--target`       | `cuda`, `rocm`, `cpu`                    | auto    | Accelerator class to target; auto-detected from the host when omitted.       |
| `--image`        | any image reference                      | backend | Container image to run; the backend's image for the resolved target.        |
| `--shm-size`     | size string, e.g. `8g`                   | `8g`    | Size of the container's `/dev/shm`; vLLM needs a large one, while llama.cpp is unaffected. |
| `--mem-limit`    | size string, e.g. `16g`                  | unset   | Container memory limit; unlimited when omitted.                              |
| `--pull-policy`  | `missing`, `always`, `never`, `newer`    | `missing` | When to pull the container image.                                          |

### Benchmark workload

| Flag                | Default      | Description                                                    |
| ------------------- | ------------ | -------------------------------------------------------------- |
| `--dataset-name`    | `random`     | Benchmark dataset to serve prompts from.                       |
| `--dataset-path`    | unset        | Host path to a custom dataset file; the built-in dataset when omitted. |
| `--num-prompts`     | `200`        | Number of prompts to send per benchmark run.                   |
| `--input-len`       | `1024`       | Input prompt length, in tokens.                                |
| `--output-len`      | `128`        | Number of tokens to generate per prompt.                       |
| `--repetitions`     | `3`          | Measured benchmark runs taken per trial.                       |
| `--warmup-runs`     | `1`          | Discarded warmup runs taken before the measured runs.          |
| `--startup-timeout` | `600`        | Seconds to wait for the served model to become healthy; the wait ends early if the container exits. |
| `--timeout`         | `900`        | Overall time limit for a trial's container, covering startup, warmup, and every measured run, in seconds. |

`--dataset-path` reads a JSONL file, one JSON object per line, with a required
string `prompt` and an optional positive integer `output_len` that falls back
to `--output-len`. `--num-prompts` caps how many lines are used and
`--input-len` is ignored, because the dataset carries the prompts. Combining a
non-`random` `--dataset-name` with `--dataset-path` is a usage error, so a named
dataset is never silently ignored. The file is read and validated before the
sweep, so a missing or malformed dataset is a usage error before any space,
model, or image work rather than a failure repeated by every trial. For
llama.cpp the runner also starts the server once before the sweep and uses its
`/tokenize` endpoint to measure the longest dataset request, so `--parallel`
is capped from real token counts; if that preflight fails, llama.cpp falls back
to one slot per trial.

Each trial starts the engine's server once inside a single container and then
drives it from the host with one shared HTTP client — one discarded warmup run
per `--warmup-runs`, then `--repetitions` measured runs. The model is loaded
once, so repeating the benchmark does not repeat the engine's startup cost, and
the container runs only the server, so no engine image needs to carry a
benchmark client. The trial's typed metrics are the **means** of the successful
measured runs — combined throughput, prefill rate, and decode rate — and the
score is what the selected `--objective` makes of them: the `(prefill, decode)`
pair in the default multi mode, the weighted-harmonic rate of the two means in
weighted mode, or the mean combined throughput in combined mode. A trial
succeeds as long as a strict majority of its measured runs succeed, so a single
failed run does not discard an otherwise valid configuration.

Before every run the server's caches are reset through the paths the backend
declares — vLLM's dev-mode endpoints, while llama.cpp exposes none and is left
as it is — and each measured run is seeded with its index `0 … N-1` (warmup
runs are seeded past that range). The reset keeps a run from being flattered by
prompts a previous run left cached, and the per-run seed makes it draw a fresh
prompt set rather than replaying the one that warmed the cache. The per-run
measurements — including each run's phase rates — and the failure count are
kept on the result, so the spread behind the mean stays inspectable.

> **Note:** measurement goes through the server's OpenAI-compatible HTTP API
> rather than an offline engine path (`vllm bench throughput` for vLLM). The
> online numbers include tokenisation and request overhead, so they are expected
> to be somewhat lower and are **not comparable** to numbers recorded by earlier
> versions of this tool.

### Engine

| Flag                         | Default          | Description                                                                 |
| ---------------------------- | ---------------- | --------------------------------------------------------------------------- |
| `--llm-seed`                 | `0`              | Engine seed (vLLM only), so every trial runs under the same prompt randomness. |
| `--quantization`             | engine default   | Weight quantization the checkpoint was produced with (`awq`, `gptq`, `fp8`); checked against the resolved checkpoint. Omit to let the engine auto-detect. |
| `--gpu-memory-utilization`   | engine default   | Fraction of GPU memory the engine may use.                                   |
| `--speculative-method`       | unset            | Speculative-decoding method, named engine-neutrally: `draft_model`, `eagle3`, `mtp`, `dflash`, `dspark`, or `ngram`; fixed for the run. No speculative flags are sent unless a method or draft is configured. |
| `--draft-model`              | unset            | Speculator for the run, in the same source forms as `MODEL`; implies `--speculative-method draft_model` when no method is given. A usage error with `ngram`. |
| `--sampler-seed`             | unset            | Seed for the optimization sampler; a non-deterministic search when omitted.  |

`--quantization` and `--gpu-memory-utilization` are vLLM-only and are rejected
as usage errors for `llamacpp`; see [Engines](#engines).

`draft_model`, `eagle3`, `dflash`, and `dspark` require `--draft-model`; `mtp`
uses the target model's own multi-token-prediction heads and needs none, though
it may take an auxiliary draft; `ngram` is model-free and rejects a draft. A
violation is a usage error (exit code `2`) before any preparation runs. There is
no flag for the mode or the depth: both are search-space tunables (see
[Speculative decoding](speculative-decoding.md)).

### Objective

| Flag               | Default                  | Description                                                        |
| ------------------ | ------------------------ | ------------------------------------------------------------------ |
| `--objective`      | `multi`                  | What the sweep maximizes: `multi` submits prefill and decode as two objectives and reports the Pareto front, `weighted` scores their weighted-harmonic effective rate, and `combined` maximizes the historical total token throughput. |
| `--prefill-weight` | benchmark `--input-len`  | Prompt tokens in the token mix the weighted score and the recommended point are chosen under; see [Objective and phase metrics](#objective-and-phase-metrics). |
| `--decode-weight`  | benchmark `--output-len` | Generated tokens in the same token mix.                            |

The weights are token counts (`1024` and `128` are as valid as `8` and `1`)
and only their ratio matters: they describe the shape of a request, not a
fractional split. The multi and weighted modes need the mix; `combined` ignores
it.

### Budget

| Flag              | Default   | Description                                                                 |
| ----------------- | --------- | --------------------------------------------------------------------------- |
| `--n-trials`      | unbounded | Maximum number of trials to run; unbounded when omitted.                     |
| `--sweep-timeout` | unbounded | Wall-clock time limit for the sweep, in seconds; unbounded when omitted.     |

With neither bound the sweep runs until you press <kbd>Ctrl-C</kbd>; with both it
stops at whichever is reached first. An interrupt is not a failure. Instead the
points found so far are reported: the single best point in a scalar mode — or
the Pareto front and its recommended point in multi mode.

### Study persistence

| Flag            | Default                              | Description                                                              |
| --------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `--storage`     | in-memory                            | Optuna storage URL, such as `sqlite:///study.db`; in-memory when omitted. |
| `--study-name`  | derived from the model and objective | Name of the persisted study (`Qwen/Qwen3.8-27B` → `Qwen_Qwen3.8-27B_multi`). |
| `--n-jobs`      | `1`                                  | Number of trials to run in parallel; requires `--storage` and disjoint devices. |

When `--storage` is set, a study of the same name is **resumed** rather than
recreated, so an interrupted sweep continues where it left off.

The speculative method and draft model are **not** recorded on a study, matching
how the seed and quantization are treated. Give each speculative setup its own
`--study-name` so a resumed study never mixes points measured under different
checkpoints, and keep the study name distinct from a non-speculative run of the
same model.

A persisted study records the `--objective` mode that produced its scores, and
resuming it under a different mode is refused with an error naming the recorded
one — the two sets of scores are not comparable. Resume with
`--objective <recorded>` or choose a different `--study-name`; the derived
default name already carries the mode suffix (`…_multi`, `…_weighted`,
`…_combined`), so the zero-config path cannot collide across modes. A study
persisted before the mode was recorded is treated as legacy **combined**:
`--objective combined` resumes it, while the phase-scoring modes are refused.
A study also records the phase-metric schema version its rates were measured
under, so a future change to the phase definitions stays detectable.

### Output

| Flag              | Values        | Default                | Description                                                     |
| ----------------- | ------------- | ---------------------- | --------------------------------------------------------------- |
| `--output`        | `tui`, `log`  | auto (tty → `tui`)     | Progress output mode.                                            |
| `--output-config` | path          | unset                  | Write the winning configuration to this YAML path.               |
| `--version`       | —             | —                      | Print the version and exit.                                      |
| `-h`, `--help`    | —             | —                      | Print help and exit.                                             |

There is no `auto` value: when `--output` is omitted the mode is chosen
automatically. A terminal gets the TUI graph, a redirected stream gets the log
lines. Asking for `--output tui` without a terminal is a usage error.

Progress and the summary go to **stdout**; warnings and errors go to **stderr**,
so `> run.log` captures a clean record.

Both modes first resolve the model. A Hugging Face id — or, for `llamacpp`, a
GGUF repository — is downloaded with a single progress line — `downloading
<model>: 40%|…| 4/10` — that names the model and shows file progress; the Hub's
own transfer and reconstruction bars are hidden. A warning emitted mid-download,
such as the missing-`HF_TOKEN` advice, is terminated onto its own line rather
than being overwritten by the next bar refresh. A local model path skips the
download entirely.

**TUI mode** draws a plotext graph of the sweep, redrawn at the start of each
trial and again when it finishes. In the default multi mode it is a
prefill-versus-decode scatter of this run's trials with the study's Pareto
front highlighted, and the status line reports the best rate reached in each
phase. The scalar weighted and combined modes keep the historical best-so-far
line with the raw per-trial points. The running frame's status line marks the
trial as in flight, so a long trial is visibly running rather than frozen:

```
trials 3/10  running trial 3  best prefill 1234.56 tps  decode 45.67 tps  avg trial 12.3s  (36.9s)
```

`running trial` appears only while a trial is in flight, and `avg trial` — the
mean seconds per successful trial, with failed trials excluded — only after the
first success, so the remaining budget can be estimated. The graph's axes are
labelled with thousands-separated plain numbers (never scientific notation) and
its x-axis with integer trial numbers, thinned to an even step once a sweep
grows past ten trials.

**Log mode** keeps a single line per finished trial, with no running marker, so
a redirected log stays easy to parse. A multi-mode line names the trial's two
rates, and its best column reports the highest rate reached in each phase:

```
trial 3/10  COMPLETE  prefill 1234.56 tps  decode 45.67 tps  best prefill 1234.56 tps  decode 45.67 tps  (36.9s)
```

A scalar line keeps the historical objective value and appends the measured
phase pair when the engine reported both, so the phase split is visible without
changing the familiar shape:

```
trial 3/10  COMPLETE  42.00 tps  prefill 1234.56 tps  decode 45.67 tps  best 42.00  (36.9s)
```

The sweep's final summary prints the recommended configuration as the same YAML
document `--output-config` writes. In multi mode it also prints the Pareto front
as a table — trial number, both phase rates, the trial's recorded combined
throughput, and the recommended row marked — so the whole trade-off can be
compared at a glance.

With `--n-jobs` greater than one the reporters are driven from several worker
threads; they are not thread-safe, so frames and lines can interleave. Use log
mode with a single job when a clean record matters.

## Exit codes

| Code  | Meaning                                                        |
| ----- | -------------------------------------------------------------- |
| `0`   | Success.                                                       |
| `1`   | Unexpected error.                                              |
| `2`   | Usage error (including an explicit TUI request without a terminal). |
| `3`   | Invalid or unreadable search-space definition.                 |
| `4`   | A configuration the backend does not support — the search space or the resolved model. |
| `5`   | Sweep finished with no completed trial.                        |
| `130` | Interrupted.                                                   |

## Example calls

**Simple sweep, search until interrupted, TUI in a terminal:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm
```

**Bounded sweep, write the winner, log mode for a file:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --n-trials 50 \
    --output log \
    --output-config best.yaml > run.log
```

**Large local checkpoint on an explicit ROCm target with more memory headroom:**

```console
$ llm-config-optimization /models/Qwen3.8-Flash-Next space.yaml vllm \
    --target rocm \
    --mem-limit 400g \
    --shm-size 32g \
    --sweep-timeout 7200
```

**Reproducible, persisted, resumable sweep:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --sampler-seed 42 \
    --llm-seed 0 \
    --storage sqlite:///qwen3.8-27b.db \
    --study-name qwen3.8-27b-tuning \
    --n-trials 100
# run again with the same flags to continue the study
```

**Parallel trials across a multi-GPU host (each trial on a disjoint device):**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --storage sqlite:///study.db \
    --n-jobs 2 \
    --n-trials 40
```

**Quick CPU smoke test with a pinned image and no pull:**

```console
$ llm-config-optimization Qwen/Qwen3.5-0.8B space.yaml vllm \
    --target cpu \
    --image docker.io/vllm/vllm-openai-cpu:latest \
    --pull-policy never \
    --num-prompts 20 \
    --n-trials 3
```

**A representative workload shape:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --dataset-name random \
    --num-prompts 500 \
    --input-len 2048 \
    --output-len 256 \
    --timeout 1800
```

**Prefer generation throughput for a chat-shaped deployment:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --objective weighted \
    --prefill-weight 512 \
    --decode-weight 512
```

**llama.cpp over a local GGUF file, CPU-only and offline:**

```console
$ llm-config-optimization /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    examples/llamacpp/space-full.yaml llamacpp \
    --target cpu \
    --pull-policy never \
    --num-prompts 20 \
    --n-trials 3
```

**llama.cpp with a checkpoint selected from a Hugging Face repository:**

```console
$ llm-config-optimization bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    examples/llamacpp/space-all-fixed.yaml llamacpp \
    --n-trials 10 \
    --output-config best-llamacpp.yaml
```

> **Requirement:** runs execute through Podman, so a running Podman daemon is
> needed. A Hugging Face model id is downloaded to the host cache before the
> first trial; a local path is bind-mounted instead.

Each run also uses a **temporary compile cache** on the host. vLLM rebuilds its
`torch.compile`, Triton, and ROCm kernel caches inside every fresh container, so
without one each trial of a sweep pays several minutes of startup before the
benchmark begins. The cache is mounted into every trial's container, so the
first trial populates it and the rest reuse it, and it is removed when the run
ends (including on interrupt or failure). Nothing is left behind. The cache is
vLLM-only; llama.cpp needs no host directory, so a llama.cpp run places nothing
on the host.

---
