# llm-config-optimization

`llm-config-optimization` is a command-line tool that searches an **LLM serving
configuration** for the highest token throughput. You give it a model and a YAML
file describing which serving parameters to sweep and which values each may
take; it runs the model in a Podman container, benchmarks every candidate
configuration through the configured inference engine, and uses
[Optuna](https://optuna.org/) (TPE sampler) to find the best point. Progress is
shown live as either a per-trial log or a terminal graph, and the winning
configuration is printed and optionally written to a YAML file when the sweep
ends.

![Screenshot of the optimization tool TUI showing a plot with number of trials on the X-axis and tokens per second on the y-axis.](./assets/optimization_plot_example.png)

The inference engine is deliberately pluggable: the search-space and experiment
layers are engine-neutral and name no engine, so the same definition and
workflow can target a different backend later. **vLLM is currently the only
backend implemented**.

The search space is deliberately explicit: every tunable parameter must be
declared, so a result is always explainable from the definition alone. Nothing
silently falls back to an engine default unless you write that decision down.

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml --n-trials 50 --output-config best.yaml
```

---

## 1. Usage

### Installation

The tool requires **Python 3.14** and a running **Podman daemon**. The project is
managed with [uv](https://docs.astral.sh/uv/), which creates the virtual
environment and installs the `llm-config-optimization` console script from the repo:

```console
$ git clone <repo-url> llm-config-optimization
$ cd llm-config-optimization
$ uv sync
```

From the repository root the command is then callable in any of these ways:

```console
$ uv run llm-config-optimization --help      # uses the project venv, no activation needed
$ source .venv/bin/activate && llm-config-optimization --help
$ uv run python -m llm_config_optimization --help   # equivalent module entry point
```

To make it available outside the repo, install it as a uv tool:

```console
$ uv tool install .
$ llm-config-optimization --help
```

With [Nix](https://nixos.org/), `nix develop` provides Python and uv in a dev
shell; run `uv sync` inside it once to install the Python tooling.

### Synopsis

```
llm-config-optimization MODEL SPACE [options]
```

Two positional arguments are required:

| Argument  | Description                                                              |
| --------- | ------------------------------------------------------------------------ |
| `MODEL`   | Model to benchmark: a Hugging Face id (`Qwen/Qwen3.8-27B`) or a host path to a local checkpoint. |
| `SPACE`   | Path to the search-space YAML file (see [section 2](#2-search-space-yaml-format)). |

Everything else has a default, so the shortest invocation is just the two
positionals. Run `llm-config-optimization --help` for the same documentation on the
command line.

### Options

#### Container

| Flag             | Values                                   | Default | Description                                                                 |
| ---------------- | ---------------------------------------- | ------- | --------------------------------------------------------------------------- |
| `--target`       | `cuda`, `rocm`, `cpu`                    | auto    | Accelerator class to target; auto-detected from the host when omitted.       |
| `--image`        | any image reference                      | backend | Container image to run; the backend's image for the resolved target.        |
| `--shm-size`     | size string, e.g. `8g`                   | `8g`    | Size of the container's `/dev/shm` (vLLM needs a large one).                 |
| `--mem-limit`    | size string, e.g. `16g`                  | unset   | Container memory limit; unlimited when omitted.                              |
| `--pull-policy`  | `missing`, `always`, `never`, `newer`    | `missing` | When to pull the container image.                                          |

#### Benchmark workload

| Flag                | Default      | Description                                                    |
| ------------------- | ------------ | -------------------------------------------------------------- |
| `--dataset-name`    | `random`     | Benchmark dataset to serve prompts from.                       |
| `--dataset-path`    | unset        | Host path to a custom dataset file; the built-in dataset when omitted. |
| `--num-prompts`     | `200`        | Number of prompts to send per benchmark run.                   |
| `--input-len`       | `1024`       | Input prompt length, in tokens.                                |
| `--output-len`      | `128`        | Number of tokens to generate per prompt.                       |
| `--repetitions`     | `3`          | Measured benchmark runs taken per trial.                       |
| `--warmup-runs`     | `1`          | Discarded warmup runs taken before the measured runs.          |
| `--startup-timeout` | `600`        | Seconds to wait for the served model to become healthy.        |
| `--timeout`         | `900`        | Overall time limit for a trial's container, covering startup, warmup, and every measured run, in seconds. |

Each trial starts `vllm serve` once inside a single container and then drives
that server with `vllm bench serve` — one discarded warmup run per
`--warmup-runs`, then `--repetitions` measured runs. The model is loaded once,
so repeating the benchmark does not repeat vLLM's startup cost. The trial's
score is the **mean** of the successful measured runs' total token throughput,
and a trial succeeds as long as a strict majority of its measured runs succeed,
so a single failed run does not discard an otherwise valid configuration.

Before every run the server's caches are reset through vLLM's dev-mode
endpoints, and each measured run is seeded with its index `0 … N-1` (warmup
runs are seeded past that range). The reset keeps a run from being flattered by
prompts a previous run left cached, and the per-run seed makes it draw a fresh
prompt set rather than replaying the one that warmed the cache. The per-run
measurements and the failure count are kept on the result, so the spread behind
the mean stays inspectable.

> **Note:** measurement goes through the server's HTTP API (`vllm bench
> serve`) rather than the offline engine path (`vllm bench throughput`). The
> online numbers include tokenisation and request overhead, so they are
> expected to be somewhat lower and are **not comparable** to numbers recorded
> by earlier versions of this tool.

#### Engine

| Flag                         | Default          | Description                                                                 |
| ---------------------------- | ---------------- | --------------------------------------------------------------------------- |
| `--llm-seed`                 | `0`              | Engine seed, so every trial runs under the same prompt randomness.           |
| `--quantization`             | engine default   | Weight quantization the checkpoint was produced with (`awq`, `gptq`, `fp8`); checked against the resolved checkpoint. Omit to let the engine auto-detect. |
| `--gpu-memory-utilization`   | engine default   | Fraction of GPU memory the engine may use.                                   |
| `--sampler-seed`             | unset            | Seed for the optimization sampler; a non-deterministic search when omitted.  |

#### Budget

| Flag              | Default   | Description                                                                 |
| ----------------- | --------- | --------------------------------------------------------------------------- |
| `--n-trials`      | unbounded | Maximum number of trials to run; unbounded when omitted.                     |
| `--sweep-timeout` | unbounded | Wall-clock time limit for the sweep, in seconds; unbounded when omitted.     |

With neither bound the sweep runs until you press <kbd>Ctrl-C</kbd>; with both it
stops at whichever is reached first. An interrupt is not a failure. Instead the best
point found so far is reported.

#### Study persistence

| Flag            | Default                    | Description                                                              |
| --------------- | -------------------------- | ------------------------------------------------------------------------ |
| `--storage`     | in-memory                  | Optuna storage URL, such as `sqlite:///study.db`; in-memory when omitted. |
| `--study-name`  | derived from the model     | Name of the persisted study (`Qwen/Qwen3.8-27B` → `Qwen_Qwen3.8-27B`).          |
| `--n-jobs`      | `1`                        | Number of trials to run in parallel; requires `--storage` and disjoint devices. |

When `--storage` is set, a study of the same name is **resumed** rather than
recreated, so an interrupted sweep continues where it left off.

#### Output

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

Both modes first resolve the model. A Hugging Face id is downloaded with a
single progress line — `downloading <model>: 40%|…| 4/10` — that names the model
and shows file progress; the Hub's own transfer and reconstruction bars are
hidden. A warning emitted mid-download, such as the missing-`HF_TOKEN` advice,
is terminated onto its own line rather than being overwritten by the next bar
refresh. A local model path skips the download entirely.

**TUI mode** draws a plotext graph of the sweep, redrawn at the start of each
trial and again when it finishes. The running frame's status line marks the
trial as in flight, so a long trial is visibly running rather than frozen:

```
trials 3/10  running trial 3  best 42.00 tps  avg trial 12.3s  (36.9s)
```

`running trial` appears only while a trial is in flight, and `avg trial` — the
mean seconds per successful trial, with failed trials excluded — only after the
first success, so the remaining budget can be estimated. The graph's y-axis is
labelled with thousands-separated plain numbers (never scientific notation) and
its x-axis with integer trial numbers, thinned to an even step once a sweep
grows past ten trials.

**Log mode** keeps a single line per finished trial, with no running marker, so
a redirected log stays easy to parse:

```
trial 3/10  COMPLETE  42.00 tps  best 42.00  (36.9s)
```

With `--n-jobs` greater than one the reporters are driven from several worker
threads; they are not thread-safe, so frames and lines can interleave. Use log
mode with a single job when a clean record matters.

### Exit codes

| Code  | Meaning                                                        |
| ----- | -------------------------------------------------------------- |
| `0`   | Success.                                                       |
| `1`   | Unexpected error.                                              |
| `2`   | Usage error (including an explicit TUI request without a terminal). |
| `3`   | Invalid or unreadable search-space definition.                 |
| `4`   | A configuration the backend does not support — the search space or the resolved model. |
| `5`   | Sweep finished with no completed trial.                        |
| `130` | Interrupted.                                                   |

### Example calls

**Simple sweep, search until interrupted, TUI in a terminal:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml
```

**Bounded sweep, write the winner, log mode for a file:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml \
    --n-trials 50 \
    --output log \
    --output-config best.yaml > run.log
```

**Large local checkpoint on an explicit ROCm target with more memory headroom:**

```console
$ llm-config-optimization /models/Qwen3.8-Flash-Next space.yaml \
    --target rocm \
    --mem-limit 400g \
    --shm-size 32g \
    --sweep-timeout 7200
```

**Reproducible, persisted, resumable sweep:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml \
    --sampler-seed 42 \
    --llm-seed 0 \
    --storage sqlite:///qwen3.8-27b.db \
    --study-name qwen3.8-27b-tuning \
    --n-trials 100
# run again with the same flags to continue the study
```

**Parallel trials across a multi-GPU host (each trial on a disjoint device):**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml \
    --storage sqlite:///study.db \
    --n-jobs 2 \
    --n-trials 40
```

**Quick CPU smoke test with a pinned image and no pull:**

```console
$ llm-config-optimization Qwen/Qwen3.5-0.8B space.yaml \
    --target cpu \
    --image docker.io/vllm/vllm-openai-cpu:latest \
    --pull-policy never \
    --num-prompts 20 \
    --n-trials 3
```

**A representative workload shape:**

```console
$ llm-config-optimization Qwen/Qwen3.8-27B space.yaml \
    --dataset-name random \
    --num-prompts 500 \
    --input-len 2048 \
    --output-len 256 \
    --timeout 1800
```

> **Requirement:** runs execute through Podman, so a running Podman daemon is
> needed. A Hugging Face model id is downloaded to the host cache before the
> first trial; a local path is bind-mounted instead.

Each run also uses a **temporary compile cache** on the host. vLLM rebuilds its
`torch.compile`, Triton, and ROCm kernel caches inside every fresh container, so
without one each trial of a sweep pays several minutes of startup before the
benchmark begins. The cache is mounted into every trial's container, so the
first trial populates it and the rest reuse it, and it is removed when the run
ends (including on interrupt or failure). Nothing is left behind.

---

## 2. Search-space YAML format

The definition file has exactly one top-level key, `space`, mapping each tunable
parameter name to its **spec**. Nothing else is allowed at the top level, and
every tunable parameter must be declared.

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
      min: 1024
      max: 8192
  tensor_parallel_size:
    fixed: 1
  attention_backend:
    default: true
```

Two example definitions live in [`examples/`](examples/) and can be used
as starting points:

* [`examples/space-all-fixed.yaml`](examples/space-all-fixed.yaml) pins all seven
  tunables to a single value. This can be used as a reproducible baseline.
  Its `flash_attention` backend is CUDA-only; use `attention_backend:
  {default: true}` there on ROCm or CPU.
* [`examples/space-full.yaml`](examples/space-full.yaml) declares every tunable
  with a broad range, deferring the hardware-dependent `tensor_parallel_size`
  and `attention_backend` to the engine so it runs on any host.

### Spec kinds

Each dimension carries exactly one spec tag:

| Spec          | Form                              | Meaning                                                                 |
| ------------- | --------------------------------- | ----------------------------------------------------------------------- |
| `fixed`       | `fixed: <value>`                  | Pin the parameter to one value that must not be varied.                 |
| `categorical` | `categorical: [<value>, ...]`     | A non-empty, duplicate-free subset of allowed values, in declared order. |
| `interval`    | `interval: {min: <int>, max: <int>}` | An inclusive integer range; the optimizer chooses where to sample. `min` must not exceed `max`. |
| `default`     | `default: true`                   | Explicitly defer to the engine's own default. Must be exactly boolean `true`. |

There is no step or count on an `interval`. Instead, sampling is the optimizer's job.
When only specific numbers are valid, use `categorical`.

### Tunable parameters

All seven parameters below are **required** in every definition. Omitting one is
an error naming the missing parameter, so a value can never silently fall back
to an engine default. Write `default: true` for using that default value.

| Parameter                  | Kind        | Accepted values                                                        |
| -------------------------- | ----------- | ---------------------------------------------------------------------- |
| `dtype`                    | categorical | `auto`, `float16`, `bfloat16`, `float32`                               |
| `kv_cache_dtype`           | categorical | `auto`, `float16`, `bfloat16`, `fp8_e4m3`, `fp8_e5m2`                  |
| `attention_backend`        | categorical | `flash_attention`, `flash_infer`                                        |
| `max_model_len`            | integer     | ≥ 1                                                                    |
| `max_num_seqs`             | integer     | ≥ 1                                                                    |
| `max_num_batched_tokens`   | integer     | ≥ 1                                                                    |
| `tensor_parallel_size`     | integer     | ≥ 1                                                                    |

`interval` is only valid for the integer parameters. `fixed` and `categorical`
are valid for both kinds, and categorical values are written using the
engine-neutral spellings above (not vLLM's own names).

`attention_backend` and `tensor_parallel_size` are the two host-dependent
parameters. `flash_attention` and `flash_infer` are CUDA-only, so a ROCm or CPU
run must declare `attention_backend: {default: true}` and let the engine select
its own backend; a space that can produce a `tensor_parallel_size` larger than
the resolved device count must likewise defer it with `{default: true}` or pin
a value the host can run. Either problem is rejected up front (exit code `4`).

`seed`, `gpu_memory_utilization`, and `quantization` are intentionally **not**
tunable. `seed` and `gpu_memory_utilization` are reproducibility and fit knobs
set on the command line (`--llm-seed`, `--gpu-memory-utilization`);
`quantization` describes how the checkpoint was produced, so it is a model
property rather than a runtime choice and is set with `--quantization`. The
measurement controls — `repetitions`, `warmup`, `startup_timeout`, and the
overall container `timeout` — are non-tunables too: they configure *how* a
point is measured, not which point is tried, so they are set on the command
line (`--repetitions`, `--warmup-runs`, `--startup-timeout`, `--timeout`) and
never enter the search space.

Because it is a checkpoint property, `--quantization` is validated against the
resolved checkpoint before the image is pulled: a quantized checkpoint accepts
only the method it was produced with, and an unquantized checkpoint accepts only
a method the engine can apply online (`fp8`). Omit `--quantization` to let the
engine auto-detect the method, which is always accepted.

### Validation

Validation happens in three stages, all before any container runs:

1. **Engine-agnostic**: Performed when the file is loaded. It checks the
   top-level shape, registry membership, that each dimension carries exactly one
   recognized tag, that values match the parameter's kind, that enum values are
   known, that categoricals are non-empty and unique, that intervals are ordered,
   and the universal `≥ 1` bounds. **Every problem found is reported at once**,
   so a broken definition can be fixed in a single pass.
2. **Backend-specific**: `vLLM` is asked whether it forwards each declared
   parameter, supports each value, supports that value on the resolved
   accelerator class, and fits the resolved device count. It also enforces its
   own cross-parameter rule `max_num_batched_tokens >= max_num_seqs` (checked
   against the most permissive point the space allows).
3. **Model-aware**: Once the checkpoint is resolved to a local directory (a host
   path, or the snapshot the Hugging Face download produced), `vLLM` inspects
   the quantization method it declares and rejects a `--quantization` the
   checkpoint cannot serve. The check runs before the container image is pulled,
   so a doomed configuration does not wait on a multi-gigabyte download. An
   omitted `--quantization` always passes, because the engine auto-detects the
   method, and a checkpoint whose metadata cannot be read is likewise accepted,
   so custom checkpoints are never blocked.

Malformed YAML and a non-mapping document are reported through the same error as
a structural problem, so a broken file fails uniformly with exit code `3`; an
unsupported space or model fails with exit code `4`.

### All-default and mixed examples

A space that pins nothing and lets the engine choose everything:

```yaml
space:
  dtype: {default: true}
  kv_cache_dtype: {default: true}
  max_model_len: {default: true}
  max_num_seqs: {default: true}
  max_num_batched_tokens: {default: true}
  tensor_parallel_size: {default: true}
  attention_backend: {default: true}
```

A mixed space exploring sequence length, batch size, and precision:

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
  tensor_parallel_size:
    categorical: [1, 2]
  attention_backend:
    default: true
```

---

## 3. Tech stack

| Layer                | Technology                                                                 |
| -------------------- | -------------------------------------------------------------------------- |
| Language / packaging | Python **3.14**, `src/` layout, [uv](https://docs.astral.sh/uv/) with the `uv_build` backend |
| Search / optimization| [Optuna](https://optuna.org/) TPE sampler, maximizing token throughput; SQLite or in-memory studies |
| Inference engine     | **vLLM** (the only backend implemented so far), served once per trial with `vllm serve` and measured repeatedly by `vllm bench serve`; the layer above it is engine-neutral |
| Container runtime    | **Podman** via the [`podman`](https://pypi.org/project/podman/) Python bindings (CDI for CUDA, device nodes for ROCm, none for CPU) |
| Config format        | **PyYAML** (`yaml.safe_load`) for the search-space and output-config files |
| Model fetching       | [`huggingface_hub`](https://pypi.org/project/huggingface-hub/) snapshot download for Hugging Face ids |
| Terminal UI          | [`plotext`](https://pypi.org/project/plotext/) for the live best-so-far graph |
| Testing              | **pytest** (unit tests use fakes; `integration`-marked tests need a Podman daemon and are deselected by default) |
| Linting / formatting | **Ruff** (line length 100, `py314` target)                                  |
| Type checking        | **ty** (checks the library sources in `src/`; `tests/` is excluded)        |
| Dev environment      | Nix flake (`nix develop`) providing Python and uv; run `uv sync` to install the Python tooling (ruff, ty, pytest) |

### Architecture

The codebase is layered so engine details stay at the edges:

- `search_space/`: Engine-neutral definition types, the tunable registry, the
  YAML loader, and validation.
- `inference/`: The engine-neutral experiment surface (a backend `Protocol`)
  plus the `VLLMBackend` that validates a resolved checkpoint, maps an
  experiment to a container run, and parses results. A second engine would be
  another implementation of that protocol.
- `runtime/`: Backend-agnostic container types and the Podman client (plus
  SELinux auto-detection).
- `optimization/`: The Optuna adapter that maps spec kinds to suggestions and
  drives a study.
- `cli/`: The composition root: argument parsing, base-config construction,
  the log and TUI reporters, the summary/writer, and exit-code mapping.

Because the definition and experiment surfaces import no engine, the same
search space and validation workflow can target a different inference backend
later without changing the file format.
