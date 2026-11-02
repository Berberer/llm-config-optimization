# Usage reference

This guide covers engine selection, the benchmark workload, objectives and phase
metrics, the measured memory footprint, every command-line option, study
persistence, exit codes, and complete example calls. For installation and the
command synopsis see the [README](../README.md#quick-start).

## Engines

The selected engine decides which tunables a space must declare, how `MODEL` is
resolved, which server command runs, and which image is used. Both engines are
driven by the same host-side client, so their workload and metric definitions
are identical, but their defaults and scoped options differ.

| Engine     | Checkpoint                                              | Applicable tunables                                    | Cache reset between runs      | Phase rates from               |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------ | ----------------------------- | ------------------------------ |
| `vllm`     | A Hugging Face id or a local snapshot directory.        | The fourteen it forwards: its twelve base tunables in [the search-space format](search-space.md), plus the two speculative dimensions when a setup is configured. | vLLM dev-mode endpoints.      | Prometheus `/metrics`.         |
| `llamacpp` | A local `.gguf` file or a Hub repository with `:QUANT`. | The thirty-four it forwards: its twenty-six base tunables, plus the two speculative dimensions when a setup is configured and the six draft-only dimensions when a draft model is configured. | None; llama.cpp exposes none. | The `timings` response object. |

For what each llama.cpp knob changes, when it applies, and which model-aware
rule may drop it, see the [llama.cpp tuning guide](llamacpp-tuning.md).

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
drafts are two gated search-space dimensions. A configured draft model adds
llama.cpp's six draft-only dimensions, which shape that model's sampling and
placement; see [Speculative decoding](speculative-decoding.md) for the method
mapping and a complete example, and the
[llama.cpp tuning guide](llamacpp-tuning.md#speculative-decoding) for the
draft-only knobs.

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

## Memory footprint

Every trial also reports how much memory its configuration holds, so a
throughput gain can be weighed against the hardware it needs. Measurement is
always on and needs no flag: there is no option to enable, disable, or tune it,
and it never changes a score, a pruner, or the recommended point.

Two peaks are recorded per trial, both ceilings rather than averages:

- the **lifetime peak**, over the server container's whole life — model load,
  the health wait, warmup, and the measured runs — which is the memory a
  configuration needs just to start; and
- the **measured peak**, the maximum over the measured runs' own windows, which
  describes what the configuration holds while actually serving requests.

Keeping both separates a one-off startup allocation from a serving footprint.
Each run's own peak stays on the run, together with the raw samples behind it
and the idle baseline, so a repetition that varied remains inspectable; the
trial reports the lifetime peak and the maximum over its runs.

VRAM is read from `nvidia-smi` on a `cuda` target and from `rocm-smi` on a `rocm`
target. A sample sums the memory of the GPU processes whose cgroup names the
trial's container, so another workload on a shared card does not inflate the
number. When no process can be attributed, the figure falls back to the device's
used memory minus a baseline taken before the container started, clamped at
zero; the run's raw samples record which of the two sources produced it. A
device-level number includes whatever else uses the card, and the recorded
baseline is what makes it interpretable.

RAM is the container's cgroup memory usage, sampled on the same cadence. It
deliberately includes page cache, so an mmap-based llama.cpp server's weights
count as memory the configuration occupies — conservative in the direction a
hardware requirement should err.

Memory is a **best-effort diagnostic**. Sampling runs once a second while the
container is up, so a spike shorter than the cadence can be missed, and each
vendor query is bounded by a ten-second timeout, so a wedged tool can delay a
sample but never block the trial. A host without the vendor tooling, a runtime
that cannot report stats, or an unreadable cgroup leaves the field unmeasured
rather than failing, retrying, or warning; a failed trial records no memory at
all. Sampling stops with the container, including on failure and interrupt, so
nothing is polled after a trial ends, and the tokenizer preflight a JSONL
dataset needs is not sampled. A `cpu` target has no VRAM to report but still
reports RAM.

The final summary reports memory in two places, in both the scalar and
Pareto-front modes. Directly after the failed-trial count, a hardware line gives
the largest footprint **any** evaluated trial recorded — the sweep's worst case —
and the winning-configuration section opens with a `memory` subsection giving the
footprint of the recommended configuration itself:

```
best trial: 7
best throughput: 51.37 tps
trials run: 24
failed trials: 1/24 (4%)
hardware (all trials): 23.8 GiB vram peak (measured 21.2 GiB), 10.4 GiB ram peak (measured 9.7 GiB)

winning configuration:
memory:
  vram: 20.4 GiB lifetime peak (17.1 GiB while serving)
  ram:  6.2 GiB lifetime peak (5.9 GiB while serving)
container:
  …
```

The memory subsection follows the document's own layout — one key per domain at
the left margin, its fields indented once — rather than introducing a second
style inside the block.

The line's figures are the maximums of each field over every successful trial's
recorded result, so its VRAM and RAM may come from different trials; the
subsection reports one point, the configuration printed directly below it, so the
number a user takes away for deployment is never the sweep's worst case. It is
omitted entirely when the recommended trial recorded no measurement at all — a
study persisted before memory was measured — leaving just the configuration.
Both read the recorded projections in the study — the same record the front
table's combined column uses — so neither can contradict the study, and a
resumed study reports the footprint of the trials it already evaluated. A field
that was not measured renders `n/a`, never `0`, so "unmeasured" and "used
nothing" stay distinguishable. Byte sizes use binary units (`GiB`, `MiB`, `KiB`,
then plain bytes) with one decimal place.

Memory is informational metadata, not part of the objective: a configuration
that does not fit is still evaluated and still scored on throughput, and there is
no rejection, warning, or headroom calculation against the available memory. The
live per-trial line, the TUI, the Pareto front table, and the written
configuration artifacts are unchanged. Nor does the addition touch the
phase-metric schema version a study records, so a study persisted before memory
was measured still loads and resumes and simply shows `n/a` until new trials are
measured.

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
| `--num-prompts`     | `50`         | Number of prompts to send per benchmark run.                   |
| `--input-len`       | `1024`       | Input prompt length, in tokens.                                |
| `--output-len`      | `128`        | Number of tokens to generate per prompt.                       |
| `--repetitions`     | `3`          | Measured benchmark runs taken per trial.                       |
| `--warmup-runs`     | `1`          | Discarded warmup runs taken before the measured runs.          |
| `--startup-timeout` | `600`        | Seconds to wait for the served model to become healthy; the wait ends early if the container exits. |
| `--timeout`         | `1500`       | Overall time limit for a trial's container, covering startup, warmup, and every measured run, in seconds. |

The defaults balance a stable measurement against a feasible sweep: 50
prompts over one warmup and three measured runs is enough to rank
configurations for a single model, while `--timeout 1500` leaves room for a
slow configuration without letting a hopeless trial run indefinitely. A space
that deliberately includes slow points (CPU offload, or a small context that
caps `--parallel`) may still need a larger timeout; a smoke test can lower
`--num-prompts` or `--repetitions` further.

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
| `--fit`                      | `off`            | Whether llama.cpp may rewrite arguments to fit the available memory. Pinning it `off` makes a trial measure the declared configuration; pass `on` to opt into engine auto-fit. llama.cpp only; a usage error for vllm. |
| `--fit-target`               | engine default   | Headroom, in MiB, that `--fit` leaves on each device; requires `--fit on` and must be at least 1. llama.cpp only; a usage error for vllm. |
| `--fit-ctx`                  | engine default   | Minimum context `--fit` preserves while fitting; requires `--fit on` and must be at least 1. llama.cpp only; a usage error for vllm. |
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

| Flag              | Values                       | Default                | Description                                                          |
| ----------------- | ---------------------------- | ---------------------- | -------------------------------------------------------------------- |
| `--output`        | `tui`, `log`                 | auto (tty → `tui`)     | Progress output mode.                                                 |
| `--output-config` | path                         | unset                  | Write the winning configuration to this path.                         |
| `--output-format` | `yaml`, `quadlet`, `compose` | `yaml`                 | Artifact to write to `--output-config`; requires `--output-config`.   |
| `--version`       | —                            | —                      | Print the version and exit.                                           |
| `-h`, `--help`    | —                            | —                      | Print help and exit.                                                  |

There is no `auto` value: when `--output` is omitted the mode is chosen
automatically. A terminal gets the TUI graph, a redirected stream gets the log
lines. Asking for `--output tui` without a terminal is a usage error.

`--output-config` writes **one** artifact per invocation, and `--output-format`
selects it. The format defaults to `yaml` when only the path is given, so
existing invocations are unchanged; giving the format without a path is a usage
error (exit code `2`), so a forgotten path never silently writes nothing. The
explicit format is authoritative — the path may carry any suffix (or none).

| Format    | Conventional filename | Contents |
| --------- | --------------------- | -------- |
| `yaml`    | `best.yaml`           | The three-domain experiment document: curated `container` section plus the full `benchmark` and `llm` sections. It is built from the recommended configuration and the resolved target and image, and never invokes backend planning. |
| `quadlet` | `best.container`      | A minimal Podman systemd unit describing the server that actually ran: image, command, entrypoint, environment, mounts, published port, devices, memory and shared-memory limits, pull policy, SELinux options, and a `curl` healthcheck. It carries `[Unit]`, `[Container]`, and `[Service]` with `Restart=on-failure`, and has **no `[Install]`** section, so a benchmark artifact never starts at boot. |
| `compose` | `docker-compose.yml`  | The same server as a modern Compose document: no obsolete `version:` key, one service keyed `server`, the same container data, and the healthcheck. |

The two executable formats **re-plan the recommended configuration through the
selected backend**, so the artifact carries the exact image, command,
entrypoint, environment, host mounts, devices, and resource settings of the
trial rather than values re-derived from the YAML document. That planning uses a
deployment backend without the sweep's trial-only scratch resources, so the
temporary vLLM compile cache and its redirect environment never appear in the
artifact; the generated file names only paths that outlive the run. Planning can
create host cache directories, which is why the YAML format deliberately keeps
its config-only path and only these formats pay that cost. A planned host port
of `None` — a trial asks the runtime for an unused port — is replaced by the
container port, so the endpoint is predictable (`8080/tcp` becomes
`8080:8080/tcp`). Devices are written verbatim, so Podman CDI strings such as
`nvidia.com/gpu=all` and the ROCm `/dev/kfd`/`/dev/dri` nodes survive; older
Docker Compose versions may not understand CDI device strings. Generated files
carry no provenance header or timestamp, so they are deterministic and
diff-friendly. A planning or write failure is reported and exits non-zero
(`1`), including on the interrupted-sweep path, so a missing artifact is never
silent.

The **healthcheck** probes
`curl -fsS http://127.0.0.1:<container-port><health-path>` — the backend's real
health path, so readiness means the engine is serving rather than merely that a
socket is open. Its start period is `--startup-timeout`, the interval is fixed
at 10 seconds, and it retries 3 times; the official vLLM and llama.cpp images
(including their CPU and ROCm variants) all ship `curl`. The quadlet uses the
native `HealthCmd`/`HealthInterval`/`HealthRetries`/`HealthStartPeriod` keys,
and Compose uses `healthcheck`. SELinux options are preserved in both:
`disable` becomes the quadlet's native `SecurityLabelDisable=true`, any other
option is passed through as a `PodmanArgs=--security-opt label=…` argument, and
Compose writes them as `security_opt: label=…`. The only comment a generated
file carries is the Compose pull-policy note: Podman's `newer` is Podman-only,
so Compose gets `always`, a documented approximation that still checks the
registry.

The artifacts are validated against the real consuming tools in the opt-in
integration suite (`uv run pytest -m integration`). That pass was recorded with
Podman 5.8.9 (its quadlet generator, plus the `docker compose` CLI on that host
is Podman's Docker-compatible shim of the same version) and `systemd-analyze`
259.10; `podman-compose` was not installed there, so its provider was not
exercised. Every test skips cleanly when its tool is absent, so a partial install
stays runnable. The CDI limitation is part of that matrix: a portable device list
such as `/dev/kfd` and `/dev/dri` must pass every installed Compose provider,
while a CDI string such as `nvidia.com/gpu=all` passes `podman-compose` and is
expected to be rejected by real Docker Compose. The generated file keeps the
verbatim CDI spelling rather than switching to Docker-specific GPU reservations,
so use Podman (or Docker Compose on a Podman-backed host) for such a
configuration.

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
line with the raw per-trial points. The status line ends with one time group —
`avg trial`, `elapsed`, and the budget's `eta` — after the mode-specific
fields. The running frame's status line marks the trial as in flight, so a long
trial is visibly running rather than frozen:

```
trials 3/10 | failed 0% | running trial 4 | best prefill 1234.56 tps  decode 45.67 tps | avg trial 12s | elapsed 37s | eta 1m 24s
```

`running trial` appears only while a trial is in flight, and `avg trial` — the
mean wall-clock seconds per successful trial, with failed trials excluded — only
after the first success, so the remaining budget can be estimated. Durations
are humanized: `avg trial`, `elapsed`, and `eta` share one at-most-two-unit
format rounded to the nearest second (`0s`, `42s`, `2m 10s`, `1h 5m`), so the
columns read as one group instead of three styles. The graph's axes are
labelled with thousands-separated plain numbers (never scientific notation) and
its x-axis with integer trial numbers, thinned to an even step once a sweep
grows past ten trials.

**Log mode** keeps a single line per finished trial, with no running marker, so
a redirected log stays easy to parse. A multi-mode line names the trial's two
rates, and its best column reports the highest rate reached in each phase:

```
trial 3/10 | COMPLETE | prefill 1234.56 tps  decode 45.67 tps | failed 0% | best prefill 1234.56 tps  decode 45.67 tps | elapsed 37s | eta 1m 24s
```

A scalar line keeps the historical objective value and appends the measured
phase pair when the engine reported both, so the phase split is visible without
changing the familiar shape:

```
trial 3/10 | COMPLETE | 42.00 tps  prefill 1234.56 tps  decode 45.67 tps | failed 0% | best 42.00 | elapsed 37s | eta 1m 24s
```

A failed trial replaces the value and phase columns with the error message the
optimizer recorded, collapsed to one line, so one bad configuration is visible
without ending the sweep:

```
trial 3/10 | FAIL | container exited with code 1 | failed 33% | best 42.00 | elapsed 37s | eta 1m 24s
```

Every finished-trial line — including a failed one — ends with the same
`elapsed` and `eta` segments, so a redirected log records the projection trial
by trial; `eta` is simply absent from any line whose budget cannot project one.

Both live modes join their top-level fields with ` | ` so the fields stay
distinct; inner pairs such as the `prefill …  decode …` rates keep their
spacing. The `failed` segment is the share of trials that failed among those
finished in **this invocation** (not across a resumed study's earlier trials),
and the trial being reported is already included, so a first-trial failure
reads `failed 100%`. The TUI shows the same segment once at least one trial has
finished — it is absent from the frame drawn before the first trial. `failed`
counts only `FAIL` trials; pruned trials and individual failed benchmark runs
are different signals and are not counted. Percentages are whole numbers, and a
nonzero rate that would round to zero is shown as `<1%` rather than `0%`, so a
small but real failure rate is never hidden.

The `avg trial`, `elapsed`, and `eta` segments are the live sweep's time group,
and `elapsed` then `eta` are always the last two segments, so past and remaining
time can be compared directly. `eta` is the budget's projection of how much
longer the sweep will run:

- With only `--sweep-timeout`, it is the time left until the deadline and is
  shown from the first frame, because the deadline is already known.
- With only `--n-trials`, it extrapolates the remaining trials from the observed
  mean duration of successful trials: `(n_trials - finished) × mean`. It is
  absent until the first success gives it a basis, because there is nothing to
  extrapolate from yet.
- With both, the sweep stops at whichever binds first, so the estimate is the
  smaller of the deadline remainder and the trial extrapolation; until the
  first success gives the trial term a basis there is no estimate at all, even
  though the deadline is already known.
- With neither, no estimate is shown.

The mean is exactly the one the `avg trial` column shows, counts only
successful trials, and covers only the trials finished in this invocation, so a
resumed study builds a fresh mean from its own successes rather than counting an
earlier run. Trials running in parallel overlap, so `--n-jobs` greater than one
suppresses the trial-based term — every remaining trial would otherwise be
counted as if it ran alone — and only a deadline can produce an estimate there.
The estimate is clamped at `0s`: Optuna checks the deadline between trials, so
the last trial may run past it, and the TUI's `running trial` marker shows why
the sweep is still alive.

The sweep's final summary always prints the recommended configuration as the
three-domain YAML document, whichever `--output-format` was chosen, so an
executable artifact does not remove the human-readable result. It also reports
how many of the study's trials failed, as `failed trials: n/m (pct)` directly
after `trials run`: the denominator `m` is the same whole-study count, so a
resumed sweep's summary covers every trial the study holds rather than only
this invocation's, unlike the live failure segment. The percentage uses the
same whole-percent format, including its `<1%` floor. A study-wide
`hardware (all trials):` line then follows, giving the largest footprint any
evaluated trial recorded, and the winning-configuration section opens with a
`memory:` subsection giving the recommended configuration's own peak; both are
described under [Memory footprint](#memory-footprint). In multi mode the summary
also prints the Pareto front as a table — trial number, both phase rates, the
trial's recorded combined throughput, and the recommended row marked — so the
whole trade-off can be compared at a glance.

With `--n-jobs` greater than one the reporters are driven from several worker
threads; they are not thread-safe, so frames and lines can interleave. The
trial-based estimate is suppressed in that mode because overlapping trials make
the mean duration useless for extrapolation, so only a `--sweep-timeout` still
produces an `eta`. Use log mode with a single job when a clean record matters.

## Exit codes

| Code  | Meaning                                                        |
| ----- | -------------------------------------------------------------- |
| `0`   | Success.                                                       |
| `1`   | Unexpected error.                                              |
| `2`   | Usage error (including an explicit TUI request without a terminal, or `--output-format` without `--output-config`). |
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

**Write the winner as a Podman quadlet to drop into `~/.config/containers/systemd/`:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --n-trials 50 \
    --output-config qwen.container --output-format quadlet
```

**Write the same winner as a Docker/Podman Compose file:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml vllm \
    --n-trials 50 \
    --output-config docker-compose.yml --output-format compose
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
on the host. Executable artifacts are planned without this cache, because it is
benchmark scratch space rather than part of the served configuration.

---
