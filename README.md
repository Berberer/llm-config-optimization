# llm-config-optimization

`llm-config-optimization` is a command-line tool that searches an **LLM serving
configuration** for the best token throughput. You give it a model and a YAML
file describing which serving parameters to sweep and which values each may
take; it runs the model in a Podman container, benchmarks every candidate
configuration through the configured inference engine, and uses
[Optuna](https://optuna.org/) (TPE sampler) to find the best point. Each trial
reports **prefill** and **decode** throughput separately, and by default the
sweep maximizes both at once, producing a Pareto front rather than committing
to a trade-off before the run. Progress is shown live as either a per-trial log
or a terminal graph, and the recommended configuration is printed and
optionally written as a YAML experiment document or as an executable Podman
quadlet or Docker Compose file when the sweep ends.

![Screenshot of the optimization tool TUI in scalar mode, showing a plot with number of trials on the X-axis and tokens per second on the Y-axis; the default multi-objective view is a prefill-versus-decode scatter.](./assets/optimization_plot_example.png)

The inference engine is deliberately pluggable and chosen on the command line:
the search-space and experiment layers are engine-neutral and name no engine, so
the same definition and workflow can target a different backend later. **vLLM
and llama.cpp are implemented today**, and each declares which tunables it
forwards, so a definition names exactly the parameters the selected engine can
serve. A trial starts the engine's server in a container and measures it from
the host through one shared HTTP client, so the container carries only the
server and every engine is measured by identical logic.

The search space is deliberately explicit: every tunable the selected engine
forwards must be declared, so a result is always explainable from the
definition alone. Nothing silently falls back to an engine default unless you
write that decision down.

## Quick start

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

### First sweep

```
llm-config-optimization MODEL SPACE ENGINE [options]
```

| Argument | Description |
| -------- | ----------- |
| `MODEL`  | Model to benchmark: a Hugging Face id (`Qwen/Qwen3.8-27B`) or a host path to a local checkpoint. For `llamacpp`, a `.gguf` file or a GGUF repository with an optional `:QUANT` selector; see [Engines](docs/usage.md#engines). |
| `SPACE`  | Path to the search-space YAML file; see [Search-space YAML format](docs/search-space.md). |
| `ENGINE` | Inference engine to benchmark: `vllm` or `llamacpp`. It selects the backend that checks the search space, plans the server, and interprets the result. |

Everything else has a default, so the shortest invocation is the three
positionals. Run `llm-config-optimization --help` for the same documentation on
the command line. A minimal run using a shipped space looks like:

```console
$ llm-config-optimization Qwen/Qwen3.8-27B examples/vllm/space-full.yaml vllm \
    --n-trials 50 --output-config best.yaml
```

The winning configuration can be written as an experiment document or as a
runnable artifact: add `--output-format quadlet` for a Podman `.container` unit,
or `--output-format compose` for a Docker/Podman Compose file. The executable
formats re-plan the winning configuration through the selected engine, so they
carry the exact image, command, environment, mounts, devices, and healthcheck of
the benchmarked server; trial-only scratch mounts such as the temporary compile
cache are left out. The YAML format is the default and is unchanged.

Runs execute through Podman, so a running Podman daemon is required. A remote
model id is downloaded to the host cache before the first trial; a local path is
bind-mounted instead. More example calls are in the [usage reference](docs/usage.md#example-calls).

## Documentation

Detailed guides are linked here:

| Guide | Contents |
| ----- | -------- |
| [Usage reference](docs/usage.md) | Engine selection, benchmark workload, objectives and phase metrics, every CLI option, winning-configuration output formats, study persistence, exit codes, and complete example calls. |
| [Search-space YAML format](docs/search-space.md) | Definition format, spec kinds, tunable parameters, per-engine mappings, validation, shipped example spaces, and the shared prompt corpus. |
| [llama.cpp tuning guide](docs/llamacpp-tuning.md) | What each llama.cpp knob changes, when it applies, the model-aware rules, the fit correctness pin, and the engine's accepted ranges. |
| [Speculative decoding](docs/speculative-decoding.md) | Methods and draft sources, the gated mode and depth dimensions, engine flag mappings, end-to-end examples, and the random-workload caveat. |
| [Architecture and tech stack](docs/architecture.md) | Layer boundaries, the engine extension surface, and the technology stack. |
