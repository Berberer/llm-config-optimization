# Optimization Runner CLI

## Problem Statement

The optimization adapter can search an `ExperimentSpace` for the best-throughput
serving configuration, but there is no user-facing way to run it. The
`llm-optimization` console script is still a hello-world stub, the base
`ExperimentConfig` (container, benchmark, and non-tunable LLM values) must be
assembled in Python by hand, and a sweep reports nothing until it returns.

That leaves a developer tuning LLM serving parameters writing a throwaway script
per sweep, choosing every fixed value themselves, and staring at a silent
terminal while each expensive benchmark trial runs. There is no progress
feedback, no way to stop a sweep and keep the best point found so far, no way to
resume an interrupted one, and no record of which point won. Long GPU trials
also hide expensive preparation work — downloading a multi-gigabyte model and
pulling a multi-gigabyte image — inside the first trial, where it looks like a
hang rather than progress.

## Solution

Turn the console script into a real command-line tool that composes the existing
layers and drives the optimization adapter end to end. The user names the model
and a search-space YAML file, optionally sets a budget, and watches the sweep
through one of two output modes:

```
llm-optimization MODEL SPACE.yaml [options]
```

Everything else has a sensible default, so the simplest invocation is two
positional arguments. The CLI builds the base `ExperimentConfig` from flags,
announces and performs the preparation steps it controls (loading and validating
the space, detecting hardware, prefetching a Hugging Face model, pulling the
container image, preparing the study), runs the sweep with a live reporter, and
prints a best-configuration summary when the budget is reached or the user
presses Ctrl-C. An optional flag writes the winning configuration to a YAML file
for reuse.

The two output modes are:

- a **log mode** that prints exactly one line per trial result, suitable for
  redirecting to a file; and
- a **TUI mode** that draws a live line graph of the best tokens-per-second
  found so far against trial number, with the raw per-trial points overlaid.

The budget is optional. With neither `--n-trials` nor `--timeout` the sweep runs
until the user interrupts it; either bound stops the sweep, and if both are
given it stops at whichever is reached first. Ctrl-C is not a failure: the CLI
reports the best point found so far and exits with the conventional interrupt
code.

## User Stories

1. As an LLM-optimization developer, I want to run a parameter sweep from the
   command line without writing Python, so that I can tune a model quickly.
2. As an LLM-optimization developer, I want to give the model and the
   search-space file as the only required arguments, so that a sweep needs no
   boilerplate.
3. As an LLM-optimization developer, I want every other setting to have a
   reasonable default, so that I only specify what I want to change.
4. As an LLM-optimization developer, I want to override the accelerator target,
   so that I can run on CUDA, ROCm, or CPU deliberately.
5. As an LLM-optimization developer, I want the accelerator to be auto-detected
   when I do not specify it, so that a default run works on the host's hardware.
6. As an LLM-optimization developer, I want to override the container image, so
   that I can pin a specific vLLM build.
7. As an LLM-optimization developer, I want to set the shared-memory size and
   memory limit, so that a run fits a constrained host.
8. As an LLM-optimization developer, I want to set the image pull policy, so
   that I control when an image is fetched.
9. As an LLM-optimization developer, I want to choose the benchmark dataset, so
   that I can measure against a representative workload.
10. As an LLM-optimization developer, I want to set the prompt count and input
    and output lengths, so that the workload matches my use case.
11. As an LLM-optimization developer, I want to set a per-run benchmark timeout,
    so that a hung trial does not stall the sweep forever.
12. As an LLM-optimization developer, I want to set the engine seed, so that
    every trial runs under the same prompt randomness and differences come from
    the configuration.
13. As an LLM-optimization developer, I want to seed the sampler, so that a
    sweep is reproducible.
14. As an LLM-optimization developer, I want an unseeded sweep to be
    non-deterministic by default, so that repeated runs explore differently
    unless I ask for reproducibility.
15. As an LLM-optimization developer, I want to set the GPU memory utilization,
    so that a run fits the available memory.
16. As an LLM-optimization developer, I want to bound a sweep by a number of
    trials, so that I control how much hardware time it consumes.
17. As an LLM-optimization developer, I want to bound a sweep by wall-clock
    time, so that a sweep stops after a fixed duration.
18. As an LLM-optimization developer, I want to give both bounds at once, so
    that the sweep stops at whichever is reached first.
19. As an LLM-optimization developer, I want to run with no budget, so that a
    sweep continues until I decide to stop it.
20. As an LLM-optimization developer, I want to stop a running sweep with
    Ctrl-C, so that I do not have to wait for the budget.
21. As an LLM-optimization developer, I want Ctrl-C to report the best point
    found so far, so that an interrupted sweep is not wasted.
22. As an LLM-optimization developer, I want an interrupted run to exit with a
    distinct code, so that scripts can tell an interrupt from success.
23. As an LLM-optimization developer, I want a log mode that prints one line per
    trial, so that I can watch progress or redirect it to a file.
24. As an LLM-optimization developer, I want each log line to show the trial
    index, its state, its throughput, the best so far, and the elapsed time, so
    that I can read the sweep at a glance.
25. As an LLM-optimization developer, I want a failed trial to print a line with
    its error rather than abort the sweep, so that one bad configuration does
    not end an expensive run.
26. As an LLM-optimization developer, I want the log line's trial count to show
    the budget when one was set, so that I can see how far along I am.
27. As an LLM-optimization developer, I want a TUI mode that plots progress as a
    line graph, so that I can see the search converge.
28. As an LLM-optimization developer, I want the graph's x-axis to be trial
    number and its y-axis to be tokens per second, so that the plot matches the
    optimization.
29. As an LLM-optimization developer, I want the graph to show the best value
    found so far, so that I can see the incumbent improve.
30. As an LLM-optimization developer, I want the raw per-trial points overlaid,
    so that I can see how noisy individual trials are.
31. As an LLM-optimization developer, I want the graph to be redrawn as each
    trial completes, so that progress is visible during a long sweep.
32. As an LLM-optimization developer, I want a status line under the graph with
    trial count, best value, and elapsed time, so that I have the headline
    numbers without reading the plot.
33. As an LLM-optimization developer, I want to force either output mode, so
    that I am not at the mercy of detection.
34. As an LLM-optimization developer, I want the mode chosen automatically when
    I do not specify one, so that an interactive run plots and a redirected run
    logs.
35. As an LLM-optimization developer, I want an explicit TUI request without a
    terminal to fail rather than emit escape codes into a file, so that I
    notice the mistake.
36. As an LLM-optimization developer, I want progress and the final summary on
    stdout, so that `> run.log` captures them.
37. As an LLM-optimization developer, I want warnings and errors on stderr, so
    that diagnostics do not pollute the progress stream.
38. As an LLM-optimization developer, I want a short startup header naming the
    model, space, target, image, and budget, so that a saved log explains
    itself.
39. As an LLM-optimization developer, I want the header to report preparation
    steps as they happen, so that a long setup does not look like a hang.
40. As an LLM-optimization developer, I want the search space to be validated
    against the backend before the first trial, so that an unsupported space
    fails in seconds instead of after an expensive run.
41. As an LLM-optimization developer, I want every validation problem printed at
    once, so that I can fix the space in one pass.
42. As an LLM-optimization developer, I want a Hugging Face model id to be
    downloaded before the first trial, so that the first trial measures
    inference rather than a download.
43. As an LLM-optimization developer, I want a local model path to be used
    directly without a download, so that offline checkpoints work.
44. As an LLM-optimization developer, I want a Hugging Face token to be read
    from the `HF_TOKEN` environment variable when I download a gated model, so
    that private models can be fetched without the CLI handling the secret.
45. As an LLM-optimization developer, I want the container image pulled before
    the first trial, so that the first trial is not delayed by a pull.
46. As an LLM-optimization developer, I want the winning configuration printed
    when the sweep finishes, so that I can act on the result.
47. As an LLM-optimization developer, I want the winning configuration written
    to a YAML file on request, so that I can reuse or archive it.
48. As an LLM-optimization developer, I want the written configuration to be
    explicit about the resolved target and image, so that it is self-contained
    and re-runnable.
49. As an LLM-optimization developer, I want the written configuration to
    overwrite an existing file silently, so that repeated runs behave like other
    CLI output.
50. As an LLM-optimization developer, I want a persisted study, so that a long
    sweep survives a crash.
51. As an LLM-optimization developer, I want to resume a persisted study by
    name, so that an interrupted sweep continues instead of restarting.
52. As an LLM-optimization developer, I want the study name derived from the
    model when I do not give one, so that resuming needs no extra bookkeeping.
53. As an LLM-optimization developer, I want an in-memory study when I give no
    storage, so that a quick sweep writes no files.
54. As an LLM-optimization developer, I want to run trials in parallel when I
    can isolate devices, so that a multi-GPU host is used efficiently.
55. As an LLM-optimization developer, I want parallel trials off by default, so
    that I do not accidentally distort throughput by contending for one device.
56. As an LLM-optimization developer, I want Optuna's own logging suppressed, so
    that the output is exactly the reporter's.
57. As an LLM-optimization developer, I want distinct exit codes for usage
    errors, bad definitions, unsupported spaces, and empty sweeps, so that
    scripts can react to each.
58. As an LLM-optimization developer, I want a dedicated error when no trial
    completes, so that I can distinguish an empty sweep from a failed run.
59. As an LLM-optimization developer, I want to invoke the tool as a module as
    well as a script, so that it works from any environment.
60. As an LLM-optimization developer, I want to print the version, so that I can
    report which build I ran.
61. As an LLM-optimization developer, I want built-in help that explains the
    invocation, every argument, and its default, so that I can discover the tool
    without reading the source.
62. As a test author, I want the CLI's composition seams injectable, so that I
    can drive it with fakes and no daemon.
63. As a test author, I want the log and summary formats testable as strings, so
    that formatting regressions are caught.
64. As a test author, I want one marked integration test that runs the real CPU
    path, so that the end-to-end wiring is protected without slowing the unit
    suite.
65. As a future maintainer, I want the image-pull step behind the runtime's
    public API, so that the CLI does not import container bindings itself.
66. As a future maintainer, I want the image choice behind a public backend
    method, so that the CLI does not duplicate the default-image mapping.
67. As a future maintainer, I want the progress hook on the optimizer rather
    than in the CLI, so that the study logic is not duplicated.

## Implementation Decisions

**Command shape and inputs**

- The command is `llm-optimization MODEL SPACE.yaml [options]`: two positional
  arguments and no subcommand. `MODEL` is a Hugging Face id or a host path; the
  second argument is the search-space YAML path.
- The fixed experiment setup comes from flags only. No second configuration
  file is introduced; the search-space YAML deliberately holds only the `space`
  document.
- The CLI lives in a new `cli` subpackage with separate modules for argument
  parsing, base-config construction, the log reporter, the TUI reporter, and the
  final summary/writer. The package initializer exposes a single `main`
  entry point returning an integer exit code. The existing top-level `main`
  delegates to it, and a module entry point allows `python -m llm_optimization`.

**Argument surface and defaults**

- Container: `--target` (`cuda` | `rocm` | `cpu`, default auto-detect),
  `--image` (default the backend's image for the resolved target),
  `--shm-size` (default `8g`), `--mem-limit` (default unset), and
  `--pull-policy` (`missing` | `always` | `never` | `newer`, default `missing`).
- Benchmark: `--dataset-name` (default `random`), `--dataset-path` (default
  unset), `--num-prompts` (default `200`), `--input-len` (default `1024`),
  `--output-len` (default `128`), and `--run-timeout` (default `900` seconds,
  the per-run benchmark timeout).
- Non-tunable LLM values: `--llm-seed` (default `0`, so every trial runs under
  the same engine seed), `--gpu-memory-utilization` (default unset, meaning the
  engine's own default), and `--sampler-seed` (default unset, meaning a
  non-deterministic search).
- Budget: `--n-trials` and `--timeout` (optimization wall-clock seconds), both
  optional. With neither, the sweep is unbounded and runs until interrupted;
  with one, that bound stops it; with both, it stops at whichever is reached
  first. Both are passed to the adapter's single optimize call.
- Study: `--storage` (an Optuna storage URL, default unset meaning in-memory),
  `--study-name` (default derived from the model id, for example
  `facebook_opt-125m`), and `--n-jobs` (default `1`).
- Output: `--output` accepts exactly `tui` or `log`; there is deliberately no
  `auto` value. When the flag is absent the mode is chosen automatically: TUI
  when stdout is a terminal, log otherwise. An explicit `tui` without a terminal
  is a usage error. `--output-config` takes a path and defaults to unset.
  `--version` prints the package version.
- Help: `-h`/`--help` is available for the program and every argument and exits
  `0` without loading a space or running a sweep. The program carries a
  description and an epilog of usage examples, and every argument's help string
  states its meaning, its accepted values, and its default. The help output is
  the primary documentation of the invocation.
- The per-run benchmark timeout is named `--run-timeout` to avoid colliding with
  the optimization budget's `--timeout`.

**Preparation sequence**

- Before the sweep, the CLI prints each step it controls as it runs: loading the
  search space, validating it against the backend, detecting the hardware
  target, resolving the model, downloading the model when it is a Hugging Face
  id, pulling the container image, and preparing the study.
- The space is loaded with the existing loader and validated against the
  backend with the existing engine-support validation before any trial. All
  problems are reported at once and the run fails without touching the
  accelerator.
- The hardware target is resolved with the existing detection (honouring
  `--target`), the image is resolved as `--image` or the backend's image for
  that target, and the image is pulled through the runtime's public pull helper
  using `--pull-policy`. The `never` policy skips the pull.
- A model that is an existing host path is used directly and no download
  happens; anything else is treated as a Hugging Face id and downloaded with
  `huggingface_hub`'s snapshot download into the host Hugging Face cache that
  the backend already bind-mounts, so the first trial reads the cached model.
  The snapshot download is called without an explicit token, so
  `huggingface_hub`'s own credential resolution applies: a gated model is
  fetched using the `HF_TOKEN` environment variable or a stored `hf auth login`
  token, and the CLI adds no token flag of its own. A download failure is
  reported and fails the run before any trial.

**Study and optimization**

- The CLI does not create the study itself; it calls the existing `optimize`
  entry point with the resolved space, base config, budget, seeds, storage,
  study name, injected backend and client, and `n_jobs`.
- The adapter gains an optional per-trial callback parameter, forwarded to the
  study's optimize call, so the CLI can report each trial without duplicating
  study logic. The callback receives the study and the completed trial.
- The adapter catches `KeyboardInterrupt`, stops the sweep, and raises a
  dedicated interrupt exception carrying the best-so-far optimization result
  (or the existing empty-sweep error when no trial completed). The CLI catches
  that exception, reports the result, and exits with the interrupt code.
- Optuna's own logging is suppressed at CLI startup so the only output is the
  reporter's.

**Reporters**

- Log mode prints one line per trial to stdout:
  `trial <i>/<N>  <STATE>  <tps> tps  best <best>  (<elapsed>)`, with the
  denominator omitted when no trial budget was given and a failure line
  carrying the trial's error message in place of the throughput. The index is
  the count of trials completed in this invocation; the best-so-far value comes
  from the study, so a resumed study's earlier best is reflected.
- TUI mode prints a preparation message when the sweep starts and then draws a
  line graph with `plotext` after the first trial completes, redrawing it after
  each completed trial: x-axis trial index, y-axis tokens per second, plotting
  both the best-so-far line and the raw per-trial points. Deferring the graph
  keeps the cold start — where the first trial loads the model and compiles
  kernels for minutes — from showing an empty plot. Failed trials advance the
  axis with the best carried forward. A status line under the graph shows trial
  count, best value, and elapsed time. There is no background timer and no
  thread.
- Both modes print a short startup header naming the model, space, target,
  image, and budget, then the preparation steps, and finally a best-config
  summary on completion or interrupt. The summary names the best trial, the
  best tokens per second, the number of trials run, and the full winning
  configuration.
- Progress and the summary go to stdout; warnings and error messages go to
  stderr.

**Winning-configuration file**

- When `--output-config` is given, the winning configuration is written as YAML
  with three domains: a curated container section (`target`, `image`,
  `shm_size`, `mem_limit`, `pull_policy`), the full benchmark section, and the
  full LLM section, using the engine-neutral field and enum spellings and
  explicit `null` for unset values.
- Before writing, the auto-detected target and the backend's resolved image are
  filled in, so the file is explicit and re-runnable. Other values remain as
  reconstructed from the winning trial.
- The file is overwritten silently if it exists. It is written on a normal
  finish and on interrupt when at least one trial completed, and never when no
  trial completed.

**Errors and exit codes**

- Exit codes are distinct: `0` success, `2` usage errors (including an explicit
  TUI request without a terminal), `3` an invalid or unreadable search-space
  definition, `4` a space the backend does not support, `5` a sweep that
  finished with no completed trial, `130` interrupted, and `1` any unexpected
  error.
- A missing or unreadable space file is reported as a definition failure (`3`)
  rather than a usage error.

**Library seams**

- The runtime's public API gains an image-pull helper taking an image, a policy,
  and an optional client, so the only container-binding import stays inside the
  runtime.
- The vLLM backend gains a public image-for-target method, refactoring the
  existing private default-image lookup, so the CLI does not duplicate the
  mapping.
- The optimization layer gains the per-trial callback parameter and the
  dedicated interrupt exception carrying the best-so-far result. The interrupt
  exception is a subtype of the existing optimization error.

**Dependencies**

- `plotext` is added as a runtime dependency for the TUI, and
  `huggingface_hub` as a runtime dependency for the model prefetch. Both are
  declared explicitly and the lockfile is refreshed.

## Testing Decisions

Good tests exercise external behaviour and observable contracts: given an
argument vector and injected fakes, what exit code is returned and what is
written to the output streams; given an experiment configuration, what YAML is
produced; given a sequence of trials, what does the log line look like. Tests
must not assert on private helpers, and unit tests must not require a daemon or
a GPU.

- **Argument parsing and configuration building** are tested directly: defaults
  produce the documented base configuration, explicit flags override them,
  invalid choices are rejected, `--output` rejects `auto`, `--help` exits zero
  and lists the positionals and the documented flags with their defaults, and a
  model that looks like a Hugging Face id is distinguished from an existing host
  path.
- **Reporters** are tested as string producers: the log reporter emits the
  documented line for a successful trial, a failed trial, a bounded sweep, and
  an unbounded sweep; the TUI reporter's rendered frame contains the plotted
  series and the status line; the summary names the best trial and every field
  of the winning configuration.
- **The configuration writer** is tested by writing to a temporary path and
  reading the YAML back: the curated container fields, the full benchmark and
  LLM sections, explicit nulls, and the resolved target and image.
- **The CLI entry point** is tested end to end with the shared fake inference
  backend and fake Podman client and captured output streams. Tests cover a
  successful bounded sweep, a space rejected by validation, an invalid
  definition, a sweep that completes no trial, an interrupted sweep, and an
  explicit TUI request without a terminal, asserting the exit code and the
  captured output.
- **The optimization layer** is tested for the new callback firing once per
  trial and for an interrupt producing the dedicated exception carrying the
  best-so-far result, while an empty interrupted sweep still raises the
  existing empty-sweep error.
- **Integration**: one test marked `integration` runs the real CPU path end to
  end through the CLI with a two-trial budget over a mostly-fixed space and
  asserts the exit code, the log output, and the written configuration. It is
  deselected by default and skips when Podman is unavailable, matching the
  existing integration convention.
- **Prior art**: the inference orchestration tests establish no-daemon backend
  tests with injected fakes; the optimization lifecycle tests establish driving
  the adapter with the shared fake backend and fake Podman client; the
  search-space tests establish engine-neutral spaces; the existing vLLM and
  optimization integration tests establish the `integration` marker and Podman
  skip behaviour.

## Implementation Tasks

[x] **T1 - Add the TUI and model-prefetch dependencies**: Declare `plotext` and
`huggingface_hub` as runtime dependencies and refresh the lockfile so the
reporter and the model prefetch resolve on Python 3.14.

[x] **T2 - Add the public image-pull helper to the runtime**: Add a
`pull_image` function to the runtime's public API that takes an image, a pull
policy defaulting to `missing`, and an optional client, keeping the only
container-binding import inside the runtime module.

[x] **T3 - Add the public image-for-target method to the vLLM backend**:
Refactor the backend's private default-image lookup into a public
`image_for(target)` method and keep the existing planning behaviour using it.

[x] **T4 - Add the per-trial callback and interrupt handling to the optimizer**:
Add an optional callback parameter to the optimization entry point, forwarded to
the study's optimize call, and catch `KeyboardInterrupt` to raise a new
interrupt exception carrying the best-so-far result, falling back to the
existing empty-sweep error when no trial completed.

[x] **T5 - Implement the CLI argument parser**: Add the argument module with the
two positionals, every documented flag, the documented defaults and choices,
`--version`, the rejection of an `auto` value for `--output`, a program
description, per-argument help strings that state each default, and an epilog of
usage examples, with `-h`/`--help` wired to print them and exit zero.

[x] **T6 - Implement base-config construction from arguments**: Add the module
that turns parsed arguments into an `ExperimentConfig`, including the container,
benchmark, and non-tunable LLM values, and that distinguishes a Hugging Face id
from an existing host path.

[x] **T7 - Implement the log-mode reporter**: Add the reporter that renders one
line per trial for successful and failed trials, with and without a trial
budget, and tracks best-so-far and elapsed time.

[x] **T8 - Implement the TUI-mode reporter**: Add the plotext reporter that
prints a preparation message at start and draws the best-so-far line and raw
per-trial points against trial number, with a status line, redrawn after each
completed trial.

[x] **T9 - Implement the final summary and configuration writer**: Add the
module that renders the best-configuration summary and writes the winning
configuration to YAML with the curated container section, the full benchmark and
LLM sections, explicit nulls, and the resolved target and image.

[x] **T10 - Implement the CLI entry point and preparation sequence**: Add the
`main` function that parses arguments, selects the output mode, runs the
preparation steps (load, validate, detect hardware, resolve and prefetch the
model, pull the image, prepare the study), invokes the optimizer with the
injected seams and the chosen reporter, prints the summary, and maps outcomes to
the documented exit codes.

[x] **T11 - Wire the console script and module entry point**: Make the existing
top-level `main` delegate to the CLI entry point and add the module entry point
so `python -m llm_optimization` works.

[x] **T12 - Unit-test argument parsing and configuration building**: Cover the
documented defaults, explicit overrides, invalid choices, the rejected `auto`
value, and the model-path-versus-Hugging-Face-id distinction.

[x] **T13 - Unit-test the reporters and the configuration writer**: Cover the
log line for successful, failed, bounded, and unbounded trials; the TUI frame
contents and status line; the summary fields; and the written YAML's sections,
nulls, and resolved target and image.

[x] **T14 - Unit-test the CLI entry point with fakes**: Drive the entry point
with the shared fake backend and fake Podman client and captured streams,
covering a successful bounded sweep, a validation failure, an invalid
definition, an empty sweep, an interrupt, and an explicit TUI request without a
terminal, asserting exit codes and output.

[x] **T15 - Unit-test the optimizer callback and interrupt path**: Cover the
per-trial callback firing once per trial and an interrupt producing the
dedicated exception with the best-so-far result, and that an interrupted sweep
with no completed trial raises the existing empty-sweep error.

[x] **T16 - Add the marked CLI integration test**: Add an `integration`-marked
test that runs the real CPU path through the CLI with a two-trial budget and
asserts the exit code, log output, and written configuration, skipping when
Podman is unavailable.

## Out of Scope

- A fixed-values experiment YAML as an input; the base experiment setup comes
  from flags only.
- Container environment variables, extra volumes, and explicit SELinux label
  options as flags.
- Device scheduling for `--n-jobs` greater than one; concurrent trials must be
  kept on disjoint devices by the user.
- Pruning, multi-objective search, constraints, and optimizers other than
  Optuna.
- Backends other than vLLM.
- Streaming partial trial results, a mid-trial ticking display, and any
  background timer or thread.
- A validate-only mode or a dry run that prints the resolved container plan.
- Changing the search-space layer, the tunable registry, or the vLLM parameter
  mapping.

## Further Notes

- **Why the setup comes from flags.** The search-space definition deliberately
  holds only the `space` document, and a fixed-values definition was left out of
  scope by the earlier search-space and optimization plans. Keeping the fixed
  setup on the command line means the CLI needs no second file format, and the
  flag defaults make the common case short. The cost is a wide flag surface and
  no reusable, reviewable experiment file; `--output-config` covers the case
  where the winning configuration should be recorded.
- **Why the budget is optional.** The requirement is a budget for users who do
  not want to stop the sweep themselves, which makes the unbounded, Ctrl-C
  driven run the default. Passing both bounds to a single optimize call lets the
  sweep stop at whichever is reached first without extra logic.
- **Why the TUI has no ticker.** Redrawing after each completed trial keeps the
  reporter single-threaded and trivially testable, and the graph is always
  correct. The reporter prints a preparation message at start and draws its
  first frame only when the first trial completes, so the cold start is
explained rather than shown as an empty plot. A long trial leaves the screen
static; a per-second ticker would need a lock around terminal writes and an
injectable clock, and is deliberately deferred.
- **Why the interrupt is an exception rather than a result field.** The adapter
  already signals an empty sweep with an exception, and an interrupt is a
  distinct outcome that should not be silently folded into a normal return. A
  dedicated exception carrying the best-so-far result lets the CLI report and
  exit `130` while library callers who do not catch it still learn the sweep was
  interrupted.
- **The interrupt exception is `OptimizationInterrupted`.** T4 left the name
  open; the implemented type lives in `llm_optimization.optimization.types`, is
  exported from the subpackage, subclasses `OptimizationError`, and carries the
  best-so-far result as the `result` attribute (also reachable as `study`). The
  optimizer entry point takes the per-trial callback as a singular `callback`
  keyword and forwards it to Optuna's `callbacks`.
- **Why the image choice and the pull live behind library seams.** The backend
  owns which image a target uses, and the runtime owns how a container image is
  fetched; routing both through public API keeps the CLI a composition root and
  avoids duplicating the default-image mapping or importing container bindings
  outside the runtime.
- **Why the model is prefetched on the host.** The backend already bind-mounts
  the host Hugging Face cache into the container, so a host-side snapshot
  download writes exactly the cache the first trial reads and the trial then
  measures inference rather than a download. The alternative — prefetching
  inside a throwaway container — avoids a host dependency but is slower and
  depends on the image shipping the Hugging Face CLI.
- **The log reporter is `LogReporter` in `llm_optimization.cli.log`.** It is the
  optimizer's per-trial callback itself (`callback=reporter`), reads best-so-far
  from the study on every line so a resumed study reflects an earlier run's
  best, counts trials finished in the current invocation for the index, and
  takes an injectable `clock` and `stream` so tests drive it as a string
  producer. The concrete line format is
  `trial <i>/<N>  <STATE>  <tps> tps  best <best>  (<elapsed>)`; `<N>` is omitted
  when `n_trials` is `None`, `<best>` is `n/a` before any trial completes, a
  failed trial prints the `error` user attribute in place of `<tps> tps`, and
  elapsed is `{:.1f}` seconds from `start()` (defaulting to construction).
- **Trial numbering on a resumed study.** The log and TUI index trials by the
  count completed in the current invocation, while best-so-far is read from the
  study so an earlier run's best is reflected. A resumed study therefore starts
  its curve at the first new trial rather than replaying history.
- **The written configuration is resolved, not verbatim.** `--output-config`
  fills in the auto-detected target and the backend's resolved image so the file
  stands alone, while leaving other values as the winning trial reconstructed
  them. This trades strict fidelity to the in-memory config for a file that can
  be read and re-used without re-deriving the environment. The printed summary
  shares the same document, so it also names the resolved target and image
  rather than the `None` the reconstructed config leaves behind.
- **The summary and the writer live in `llm_optimization.cli.summary`.** T9
  implements `render_summary(result, *, target, image)` and
  `write_config(path, config, *, target, image)`; both build the same
  three-domain document (a curated container section, the full benchmark and
  LLM sections with explicit nulls, and enum values in their engine-neutral
  spellings) through one private serializer, so the printed and written
  configurations cannot disagree.
- **T10 made `optimize`'s `n_trials` optional.** The plan's budget is optional,
  so an unbounded Ctrl-C-driven sweep must pass `n_trials=None` through to
  Optuna; the adapter's signature changed from a required `int` to
  `int | None = None` (backward compatible for existing callers).
- **T10 added `image_for` to the `InferenceBackend` protocol.** T3 exposed the
  method only on `VLLMBackend`, but the CLI resolves the image through the
  injected backend seam, so the protocol now declares it and the shared
  `FakeBackend` implements it (returning its canned plan image). This keeps the
  CLI a composition root rather than a concrete-backend client.
- **The startup header names the requested target and image.** T10 prints
  `target: auto` / `image: default` when the flags are omitted, and the
  preparation steps then report `resolved target:` and the pulled image, since
  neither value is known until detection and image resolution have run.
- **T10's entry point is `llm_optimization.cli.main`.** It returns an integer
  exit code and takes injectable `backend`, `client`, `stdout`, and `stderr`
  seams; argparse's own `SystemExit` (help, version, usage) is folded into the
  return value so `main` is a pure function of its inputs.
