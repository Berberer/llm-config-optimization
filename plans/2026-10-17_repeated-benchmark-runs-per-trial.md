# Repeated Benchmark Runs per Trial

## Problem Statement

A trial currently produces exactly one benchmark measurement. The optimizer
maximizes that single number, so a transient effect — thermal throttling, a
neighbouring process, a cold scheduler, a one-off slow request — becomes the
trial's score. Nothing distinguishes a configuration that reliably sustains
throughput from one that happened to catch a good moment, and the search's
decisions inherit that noise.

Running more trials instead of more measurements per trial does not fix this
cheaply: each trial launches a fresh vLLM process, and model load, engine
initialisation, and CUDA-graph capture dominate the benchmark's own runtime.
Paying that startup once per measurement is the reason the layer runs a single
`vllm bench throughput` today, so "just run it N times" is unaffordable.

What is missing is a way to take N measurements of one configuration while
loading the model once, so a trial's score reflects the configuration rather
than the moment it was measured.

## Solution

Change a trial from "one container, one offline benchmark" to "one container,
one server, N online benchmark runs". Inside the single container the layer
starts `vllm serve` once, waits for it to become healthy, then runs
`vllm bench serve` a configurable number of times against that server. The
model is loaded once; every repetition is a cheap client pass.

The measurements are made comparable by controlling the two effects that would
otherwise bias repeats:

- the **prefix cache** is reset through vLLM's dev-mode endpoint between runs,
  so a run never benefits from prompts a previous run left cached;
- the **client seed** changes per run, so each repetition draws a fresh prompt
  set rather than replaying the one that warmed the cache.

One **warmup** run is taken first and discarded, so first-run effects do not
reach the score. The measured runs are then averaged into the trial's objective
value. A trial succeeds only when a **strict majority** of its measured runs
succeed, so a single failed run does not discard the trial but a
systematically broken configuration still cannot score.

Because the server is an HTTP service, the measurement becomes online serving
throughput (`vllm bench serve`) rather than offline throughput (`vllm bench
throughput`). That is a deliberate, one-way change: there is a single
measurement path, a single result schema, and no mode switch. Existing studies
and recorded numbers are not numerically comparable to the new ones.

The new controls — `repetitions`, `warmup`, `startup_timeout`, and the overall
container `timeout` — are non-tunables, like `seed` and
`gpu_memory_utilization`. They configure how a point is measured, not which
point is tried, so they are set on the command line and do not enter the search
space or its YAML definitions.

## User Stories

1. As an LLM-optimization developer, I want a trial to take several benchmark
   measurements instead of one, so that the objective reflects the
   configuration rather than a single noisy sample.
2. As an LLM-optimization developer, I want the model loaded once per trial, so
   that repeating the benchmark does not repeat vLLM's startup cost.
3. As an LLM-optimization developer, I want the repetitions to run against a
   single long-lived server, so that the only thing that varies between runs is
   the benchmark itself.
4. As an LLM-optimization developer, I want to choose how many measured runs a
   trial takes, so that I can trade sweep time for stability.
5. As an LLM-optimization developer, I want a sensible default number of runs,
   so that noise reduction is on without me having to configure it.
6. As an LLM-optimization developer, I want an unmeasured warmup run before the
   measured ones, so that first-run effects do not reach the score.
7. As an LLM-optimization developer, I want to control how many warmup runs are
   taken, so that I can discard more on a noisy host or none to save time.
8. As an LLM-optimization developer, I want the prefix cache reset between
   runs, so that a repetition is not flattered by prompts a previous run left
   cached.
9. As an LLM-optimization developer, I want the client seed to change per run,
   so that each repetition samples a fresh prompt set.
10. As an LLM-optimization developer, I want the per-run seeds to be
    deterministic (run index `0 … N-1`), so that a trial is reproducible.
11. As an LLM-optimization developer, I want the measured runs collapsed with
    their mean, so that the objective is a single stable value.
12. As an LLM-optimization developer, I want a failed run to be tolerated as
    long as a strict majority succeed, so that one hiccup does not discard a
    valid configuration.
13. As an LLM-optimization developer, I want a trial whose runs do not reach a
    majority of successes to fail, so that a systematically broken
    configuration cannot score.
14. As an LLM-optimization developer, I want a trial failure to be an ordinary
    `ExperimentError`, so that the optimizer marks it failed and the sweep
    continues.
15. As an LLM-optimization developer, I want the per-run measurements kept on
    the result, so that I can inspect the spread behind an aggregate.
16. As an LLM-optimization developer, I want the number of failed runs kept on
    the result, so that a configuration that only just passed is
    distinguishable from a clean one.
17. As an LLM-optimization developer, I want the aggregate metrics derived from
    the per-run measurements rather than stored separately, so that the mean
    and the runs it summarizes cannot drift apart.
18. As an LLM-optimization developer, I want the raw per-run payloads kept on
    the result, so that details the typed metrics omit remain inspectable.
19. As an LLM-optimization developer, I want the single container result kept on
    the result, so that the server and runner logs survive for debugging.
20. As an LLM-optimization developer, I want the health-check wait bounded
    separately from the overall run, so that a server that never becomes ready
    fails with a clear message rather than a generic timeout.
21. As an LLM-optimization developer, I want the container's timeout to bound
    startup, warmup, and all runs together, so that one hung stage cannot hang
    the trial indefinitely.
22. As an LLM-optimization developer, I want a hung benchmark run killed within
    its own share of the remaining budget, so that the majority rule can
    actually rescue the trial instead of the container timing out.
23. As an LLM-optimization developer, I want the server to run only inside the
    container on loopback, so that no host port is exposed or contended.
24. As an LLM-optimization developer, I want the serving parameters to configure
    the server and the workload parameters to configure the client, so that
    each flag reaches the command that understands it.
25. As an LLM-optimization developer, I want `extra_args` applied to the server
    command, so that unmodeled serving flags still reach the engine.
26. As an LLM-optimization developer, I want the runner script generated as part
    of planning, so that a plan fully describes the run it will execute.
27. As an LLM-optimization developer, I want the runner to fail the container
    when the trial's runs do not reach a majority, so that a bad trial surfaces
    as a failed run rather than a misleading result.
28. As an LLM-optimization developer, I want to set `--repetitions` on the
    command line, so that I control the time/stability trade-off per sweep.
29. As an LLM-optimization developer, I want to set `--warmup-runs` on the
    command line, so that I can discard more or fewer warmup measurements.
30. As an LLM-optimization developer, I want to set `--startup-timeout` on the
    command line, so that a large model's slow start is not mistaken for a
    failure.
31. As an LLM-optimization developer, I want an overall container `--timeout`
    that covers startup, warmup, and every run, so that the run's wall-clock
    cost is bounded by one number.
32. As an LLM-optimization developer, I want the sweep's own wall-clock budget
    to keep a distinct flag, so that the per-trial limit and the sweep limit
    are never confused.
33. As an LLM-optimization developer, I want the new controls validated before
    any container runs, so that a nonsensical value is a usage error rather
    than a failed trial.
34. As an LLM-optimization developer, I want the objective to remain
    `tokens_per_second` (now an online measurement), so that the optimizer needs
    no change.
35. As an LLM-optimization developer, I want the study's recorded projection to
    include the per-run detail and failure count, so that a persisted study
    explains the spread, not just the mean.
36. As an LLM-optimization developer, I want the online measurement to be the
    only path, so that there is no mode switch and no second result schema to
    keep in sync.
37. As an LLM-optimization developer, I want the result type to stay hashable
    and frozen, so that it can still be logged, compared, and used as a cache
    key.
38. As an LLM-optimization developer, I want the container and workload domains
    to stay fixed across a series, so that only the serving parameters vary
    while the measurement method is constant.
39. As a test author, I want to assert the generated runner and the assembled
    serve/bench commands, so that the run's shape is covered without a daemon.
40. As a test author, I want to parse a manifest plus per-run result files into
    an aggregate result, so that the aggregation is covered without an engine.
41. As a test author, I want the aggregate and its per-run detail covered by
    unit tests, so that the mean and the majority rule are pinned.
42. As a test author, I want the existing integration tests updated to the new
    server path, so that the real run stays regression-protected while staying
    deselected by default.
43. As a future maintainer, I want the online-versus-offline change and its
    comparability cost recorded, so that the break is a decision rather than a
    surprise.
44. As a future maintainer, I want the cache-reset requirement (dev mode)
    documented, so that the reason the server runs with it enabled is visible.
45. As a future maintainer, I want the manifest format to be the parser's
    contract, so that the runner and the parser cannot silently disagree about
    which runs succeeded.

## Implementation Decisions

**Measurement model**

- A trial becomes one container that starts `vllm serve` once and runs
  `vllm bench serve` `warmup + repetitions` times against it. Online serving
  throughput replaces offline `bench throughput` entirely; there is no mode
  switch and no fallback path.
- `tokens_per_second` is the mean of the successful measured runs' total token
  throughput. `requests_per_second` and `elapsed_time` are the corresponding
  means. The objective stays `ExperimentResult.tokens_per_second`, so the
  optimizer is unchanged.
- The measurement change is explicitly not comparable to earlier offline
  numbers; this is recorded in the README and in the module docstrings.

**Configuration**

- `BenchmarkConfig` gains three non-tunable fields: `repetitions` (default 3),
  `warmup` (default 1), and `startup_timeout` (default 600 seconds). Its
  existing `timeout` is redefined as the overall container limit covering
  startup, warmup, and all measured runs; it remains optional (unbounded when
  unset at the library level).
- `BenchmarkConfig` validates its own invariants on construction:
  `repetitions >= 1`, `warmup >= 0`, and `startup_timeout > 0`. A violation is
  a `ValueError`. These are not search-space parameters, so the loader and the
  tunable registry are untouched.
- Because these are non-tunables, the search-space "totality" rule, the YAML
  loader, the registry, and the example space files do not change.

**vLLM argument assembly**

- The backend's argument assembly is split by destination. The serving
  parameters (`LLMConfig`) and the backend's `extra_args` build the
  `vllm serve` command; the workload parameters (`BenchmarkConfig`) build the
  `vllm bench serve` command. The existing flat-flag/JSON-group encoding and
  value maps are reused for both.
- The workload length flags keep the existing dataset-dependent routing
  (`--random-input-len`/`--random-output-len` for the random datasets, the
  generic flags otherwise).
- `extra_args` moves to the server command; the client command is fully derived
  from `BenchmarkConfig` and has no escape hatch.

**Runner and plan**

- `plan` writes a generated `runner.sh` into the run's output directory (the
  directory is already created and bind-mounted) and sets the container command
  to run it, overriding the image entrypoint. The runner is self-contained
  POSIX/bash and carries the serve and bench argument lists, the repetition and
  warmup counts, and the deadline arithmetic.
- The runner starts the server on container loopback, redirects its output to a
  log file, and polls the server's health endpoint until it answers or
  `startup_timeout` elapses. A server that never becomes healthy exits non-zero
  with a clear message.
- After health, the runner takes `warmup` runs (discarded), then `repetitions`
  measured runs. Before every run it resets the server's caches; each measured
  run is passed its index as the client seed.
- Each run's deadline is the remaining overall budget divided by the number of
  runs left; a run that exceeds it is killed and counted as a failure. The
  container's overall `timeout` remains the outer guard.
- Each run writes its result to a per-run file in the output directory; the
  warmup writes a separate, non-measured file. When the measured runs finish,
  the runner writes a small manifest recording which runs succeeded and how
  many failed, then shuts the server down. If the successes are not a strict
  majority, the runner exits non-zero, so the container run fails.
- The container environment enables vLLM's dev-mode endpoints, which the cache
  reset requires.

**Result parsing**

- `parse_result` reads the manifest and the per-run result files it references.
  It maps vLLM's online-benchmark keys (`total_token_throughput`,
  `request_throughput`, `duration`, `completed`) onto per-run metrics, builds
  the aggregate, and records the failure count from the manifest. A missing or
  malformed manifest, or a referenced run file that is missing or malformed, is
  an `ExperimentError` carrying the raw container result, matching the existing
  failure contract.
- The parser also defensively enforces the majority rule, so a container that
  somehow exited zero with too few successes still fails the trial.

**Result type**

- A new frozen, slots-based `RunMetrics` holds one measured run's
  `tokens_per_second`, `requests_per_second`, `elapsed_time`, `num_prompts`, and
  its raw JSON payload.
- `ExperimentResult` is refactored to store `runs: tuple[RunMetrics, ...]` (the
  successful runs) and `failed_runs: int`, alongside the existing `target`,
  `image`, `devices`, and `raw_result`. The scalar metrics
  (`tokens_per_second`, `requests_per_second`, `elapsed_time`, `num_prompts`)
  and `raw_json` become derived properties: the means over `runs`, the summed
  measured prompts, and the tuple of per-run payloads. `runs` is the single
  source of truth, so the aggregate cannot drift from what ran.
- `ExperimentResult` requires at least one run, and remains frozen and
  hashable; `RunMetrics` follows the same immutability and hashing pattern the
  result type already uses.
- The optimizer's recorded projection is extended to include the per-run
  throughputs and the failed-run count, keeping the projection
  JSON-serializable.

**CLI**

- The benchmark group gains `--repetitions` (default 3), `--warmup-runs`
  (default 1), and `--startup-timeout` (default 600). The per-run benchmark
  timeout is renamed to `--timeout` and redefined as the overall container
  limit.
- The existing sweep wall-clock budget flag is renamed to `--sweep-timeout`,
  because `--timeout` now names the per-trial container limit. This is a
  breaking rename of both flags; it is called out in the README and in Further
  Notes.
- The new flags are validated before any preparation or container work, so a
  nonsensical value is a usage error rather than a failed trial.
- The CLI's base-config builder maps the new flags onto `BenchmarkConfig`; the
  summary writer already emits every `BenchmarkConfig` field, so it needs no
  structural change.

**Layering**

- The runtime layer is unchanged: it still runs one container to completion.
  The server lifecycle lives entirely inside the container, in the generated
  runner, so the runtime stays engine-free and one-shot.
- The inference types and the backend own all new behavior; the optimizer and
  the CLI change only in the fields they carry.

## Testing Decisions

Good tests assert external behavior and observable contracts: given a
configuration, what plan (image, mounts, command, entrypoint, environment) does
the backend produce; given a manifest and result files, what typed result and
what errors does the parser produce; given a `BenchmarkConfig`, what validation
outcome and what derived aggregates result. Tests should not assert on private
helper internals, and unit tests must not require a daemon or an engine.

- **Configuration validation** is tested directly on `BenchmarkConfig`: the
  defaults are the documented ones, and out-of-range `repetitions`, `warmup`,
  and `startup_timeout` are rejected.
- **Runner generation and command assembly** are tested through the plan, the
  way the existing plan tests work: the returned container config's command
  runs the generated runner, the entrypoint is overridden away from the image's
  serving entrypoint, the dev-mode environment is present, and the generated
  runner text embeds the serve arguments, the bench arguments, and the
  configured repetition and warmup counts. The serve/bench split is also tested
  at the argument-assembly seam, asserting which flag reaches which command and
  that `extra_args` lands on the server command.
- **Result parsing and aggregation** are tested by writing a manifest plus
  per-run result files into a temporary directory and asserting the derived
  means, the per-run tuple, the summed prompts, the failure count, and the
  preserved raw payloads. The existing failure cases are retained: a missing or
  malformed manifest, a missing or malformed run file, and a non-object payload
  each raise `ExperimentError` with the raw result attached.
- **The majority rule** is tested at the parser seam with manifests that meet
  and fail the threshold, asserting the passing case aggregates only the
  successful runs and the failing case raises.
- **The result type** is tested for its derived properties, its rejection of an
  empty run set, and its frozen/hashable behavior.
- **The optimizer projection** is tested with the shared fake backend,
  asserting the recorded user attributes carry the per-run detail and the
  failure count, and that a failed trial still continues the study.
- **The CLI** is tested for the new flags' defaults and parsing, the renamed
  flags, and the validation-to-usage-error mapping, following the existing
  argument and config test patterns.
- **Integration**: the existing `integration`-marked vLLM tests are updated to
  the server path (a short warmup and a small repetition count against the CPU
  image) and remain deselected by default and skipped when Podman is
  unavailable, matching the current convention. The optimize integration smoke
  test is updated only where the result shape changed.
- **Prior art**: the plan and argument tests establish no-daemon command
  assertions with injected hardware; the result tests establish the
  temporary-directory result-file fixtures and the `ExperimentError` payload
  contract; the shared fakes establish the fake backend and fake Podman client
  convention; the existing integration tests establish the marker and skip
  behavior. The new tests follow those patterns and live alongside the modules
  they cover.

## Implementation Tasks

[x] **T1 - Extend `BenchmarkConfig` with the measurement controls**: Add
`repetitions` (default 3), `warmup` (default 1), and `startup_timeout` (default
600) as non-tunable fields, redefine `timeout` as the overall container limit,
and validate the invariants on construction. Add unit tests for the defaults
and the rejected values.

[x] **T2 - Add `RunMetrics` and refactor `ExperimentResult`**: Introduce the
frozen `RunMetrics` type and change `ExperimentResult` to store the successful
`runs` plus `failed_runs`, with the scalar metrics and `raw_json` derived from
`runs`. Update every constructor and consumer (the vLLM parser, the optimizer
projection, the shared fakes, and the affected tests) so the suite passes with
single-run semantics. Add tests for the derived properties, the empty-run
rejection, and hashability.
  - **Note: the optimizer projection needed no change.** The plan listed the
    optimizer projection among T2's consumers, but `_project_result` reads only
    `tokens_per_second`, `requests_per_second`, `elapsed_time`, `target`,
    `image`, and `devices`, and every one of those is now a derived property with
    the same name and type. The projection is therefore unchanged here; T7 owns
    extending it with the per-run throughputs and the failed-run count.

[x] **T3 - Split vLLM argument assembly by destination**: Separate the serving
arguments (from `LLMConfig` and `extra_args`) from the workload arguments (from
`BenchmarkConfig`), reusing the existing encoding, value maps, and
dataset-dependent length routing. Update the argument tests to assert which
flag reaches which command and that `extra_args` lands on the server command.
  - **Note: the split exposes argument builders, not full commands.** The
  task's "which command" wording is implemented as two functions —
  `serve_arguments` (from `LLMConfig` and `extra_args`) and `bench_arguments`
  (from `BenchmarkConfig` and the resolved model) — rather than full `vllm serve`
  and `vllm bench serve` command tuples. `plan` still assembles its existing
  `bench throughput` command from both, byte-for-byte unchanged, so no plan test
  moved; T4 is what embeds the two argument lists in the generated runner and
  repoints the container command at it. Naming the seam after its arguments keeps
  the model prefix and the runner's shell plumbing in one place (T4) instead of
  duplicating them in two command builders.

[x] **T4 - Generate the runner and rework `plan`**: Have `plan` write a
self-contained runner into the output directory that starts the server, waits
on its health endpoint within `startup_timeout`, resets caches, takes the
warmup and measured runs with per-index seeds and adaptive per-run deadlines,
writes the manifest, and enforces the majority before shutting the server down.
Set the container command to run it, override the image entrypoint, and add the
dev-mode environment. Update the plan tests to assert the command, entrypoint,
environment, mounts, and embedded runner content.
  - **Note: the runner is bash, not a bare engine command.** The container
    command is `/output/runner.sh` with the entrypoint overridden to
    `("/bin/bash",)`, and `plan` writes the script into the mounted output
    directory. The server is pinned to container loopback with explicit
    `--host 127.0.0.1 --port 8000`; the client relies on the same vLLM defaults.
    Measured runs are seeded with their index `0 … repetitions-1`; warmup runs
    are seeded past that range (`repetitions + warmup_index`) so a warmup run
    never draws the prompt set a measured run then replays. The runner resets
    all three dev-mode caches (`/reset_prefix_cache`, `/reset_mm_cache`,
    `/reset_encoder_cache`), matching vLLM's own sweep, rather than only the
    prefix cache. The manifest is `{"repetitions", "warmup", "failed_runs",
    "runs"}` with `runs` the successful per-run filenames (`run-<i>.json`), which
    is the contract T5 reads. `parse_result` is intentionally left on the old
    `result.json` schema here, so the `integration`-marked vLLM tests (T9) are
    expected to be red until T5 and T9 land; the deselected default suite is
    green.

[x] **T5 - Rework `parse_result` for the manifest and online schema**: Read the
manifest and the per-run files it references, map the online benchmark keys onto
`RunMetrics`, build the aggregated `ExperimentResult`, record `failed_runs`,
and enforce the majority defensively. Update the result tests to the manifest
and per-run fixtures, including the failure cases and the majority boundary.

[x] **T6 - Expose the controls on the CLI**: Add `--repetitions`,
`--warmup-runs`, and `--startup-timeout`; rename the per-run timeout to
`--timeout`; rename the sweep budget to `--sweep-timeout`; map the new flags
onto `BenchmarkConfig`; and validate them into a usage error before any
preparation work. Update the CLI argument, config, budget-text, and entry-point
tests and the help epilog.
  - **Note: validation maps `BenchmarkConfig`'s `ValueError` to a usage error.**
    `_run` builds the base config before the header and the preparation
    sequence, so an out-of-range `repetitions`, `warmup`, or `startup_timeout`
    returns exit code 2 without loading the space or planning a container;
    the check stays in `BenchmarkConfig` rather than being duplicated in
    argparse. The per-trial `--timeout` default carries over from
    `--run-timeout` as `900` seconds (confirmed with the user); the sweep
    budget is `--sweep-timeout`, still unbounded by default. The CLI
    integration test's flag reference was renamed alongside the flag; its
    server-path and repetition updates remain with T9.

[x] **T7 - Extend the optimizer's recorded projection**: Include the per-run
throughputs and the failed-run count in the JSON-serializable projection, and
update the lifecycle tests to assert the richer record and the unchanged
objective.
  - **Note: the per-run detail is a list of objects.** The projection gains
    `"failed_runs"` and a `"runs"` list holding one object per successful run
    with that run's `tokens_per_second` and `requests_per_second` (confirmed with
    the user), so the recorded spread mirrors the aggregate keys rather than
    being a bare list of throughput floats.

[x] **T8 - Update the documentation**: Update the README's benchmark-workload
and budget option tables for the renamed and new flags, add the new
non-tunables to the note that lists `seed` and `gpu_memory_utilization`, adjust
the example calls, and add a short explanation of repeated measurement, the
cache-reset and per-run-seed rationale, and the online-versus-offline
comparability break.
  - **Note: the tech-stack table's inference-engine row was updated too.** It
    still described the engine as `vllm bench throughput`; it now names the
    one-server-per-trial `vllm serve` plus repeated `vllm bench serve` shape,
    keeping the README's summary of the measurement path accurate.

[x] **T9 - Update the integration tests**: Point the `integration`-marked vLLM
and optimize tests at the server path with a small repetition and warmup count,
and assert the new result shape, keeping them deselected by default and skipped
without Podman.
  - **Note: the CLI integration test was updated too, and the local timeout
    constant renamed.** The CLI test carries the flags and asserts the written
    YAML, so it gained `--repetitions`/`--warmup-runs` and the new
    `benchmark` fields (as T6 deferred). All four touched modules renamed their
    local `_RUN_TIMEOUT_SECONDS` to `_CONTAINER_TIMEOUT_SECONDS`, matching the
    redefinition of `timeout` as the whole-trial limit, and assert against the
    runner's `manifest.json` instead of the retired `result.json`. The counts
    chosen are `repetitions=2`, `warmup=1` — small, but enough to assert a
    repeated `runs` tuple and the summed prompt count.

[x] **T10 - Verify the whole change**: Run the formatter, the linter, and the
full unit suite, and confirm the integration tests are still deselected by
default.
  - **Note: verification result.** `uv run ruff format .` left all 68 files
    unchanged, `uv run ruff check .` reported no problems, and
    `uv run pytest` finished `428 passed, 7 deselected`. The `7 deselected`
    matches the seven `integration`-marked tests, which `addopts =
    "-m 'not integration'"` keeps out of the default run. The integration suite
    itself was executed once under T9 and all seven passed (CPU vLLM end to
    end, CPU optimize sweep, CLI sweep, ROCm, and the two runtime tests).

## Out of Scope

- The offline `vllm bench throughput` path, and any mode switch between offline
  and online measurement. The change is one-way.
- Reading or migrating old single-file (`result.json`) results or persisted
  studies; the result schema changes and old artifacts are not supported.
- Option B (an in-process offline harness that loads the model once and calls
  the engine API directly).
- Making `repetitions` or `warmup` tunable, or letting the search space vary the
  measurement method; the search space and its YAML format are unchanged.
- Variance-aware objectives, confidence intervals, statistical tests, adaptive
  or early-stopping run counts, and Optuna pruning.
- Multi-objective search and any objective other than `tokens_per_second`.
- Changing the runtime layer, adding a long-lived-container primitive, or
  exposing host ports.
- Parallel trials and device scheduling; `n_jobs` remains a passthrough.
- Adding a second inference engine.

## Further Notes

- **CLI timeout naming collision (decided during spec writing).** The CLI
  already used `--timeout` for the sweep's wall-clock budget and
  `--run-timeout` for the per-run benchmark limit. The interview decided the
  per-trial overall limit should be named `--timeout`; making that coherent
  requires renaming the sweep budget to `--sweep-timeout`. This is a breaking
  rename of both flags. The alternative — keeping `--timeout` for the sweep and
  naming the per-trial limit `--trial-timeout` — avoids the double rename but
  contradicts the recorded decision; it is noted here so it can be revisited
  before implementation.

- **Online versus offline measurement.** `vllm bench serve` measures throughput
  through the HTTP API, including tokenisation and request overhead, whereas
  `vllm bench throughput` measured offline engine throughput. The new numbers
  are expected to be somewhat lower and are not comparable to earlier ones. The
  client is pointed at the loopback server so no network is involved beyond the
  container.

- **Why the cache reset needs dev mode.** vLLM's `/reset_prefix_cache` endpoint
  is a development endpoint gated by `VLLM_SERVER_DEV_MODE`. The container
  environment enables it so the runner can reset the cache between runs; this
  is the same mechanism vLLM's own `bench sweep serve` uses. Varying the client
  seed is applied on top, so a run is decorrelated from its predecessor even if
  a cache reset is ineffective.

- **Why the mean (recorded trade-off).** The interview chose the mean over the
  median. The mean uses every measured run, which is conventional, but it is
  less robust to a single slow outlier than the median would be. The per-run
  metrics are kept on the result precisely so the spread is inspectable and a
  future change can adopt a robust statistic without re-running anything.

- **Correlated samples.** The repetitions share a process, allocator, and
  thermal state, so they are not independent samples. The mean is a more stable
  estimate than a single run, but the spread should not be read as the spread
  of independent trials.

- **Why the manifest.** `parse_result` receives the container result, the plan,
  and the output directory — not the `BenchmarkConfig` — so it cannot know how
  many runs were configured. The runner writes a manifest recording the
  successful run files and the failure count, which becomes the parser's
  contract and lets the parser reconstruct `failed_runs` without widening the
  backend protocol.

- **vLLM result keys.** The online benchmark's saved JSON carries
  `total_token_throughput`, `request_throughput`, `duration`, and `completed`
  (plus a `failed` request count per run), which map onto the per-run metrics.
  The previous offline keys (`tokens_per_second`, `requests_per_second`,
  `elapsed_time`, `num_requests`) are no longer read.

- **Adaptive deadline arithmetic.** Each run receives the remaining overall
  budget divided by the runs left, so an early hung run cannot starve the later
  ones and the majority rule remains meaningful. The container's overall
  `timeout` stays the outer guard for the whole trial.

- **Layering rationale.** Keeping the server lifecycle inside the container's
  generated runner means the runtime layer stays a one-shot, engine-free
  container runner and the inference layer stays the only place that knows
  vLLM's commands. A future engine can implement the same
  one-container/one-server/many-runs shape behind the existing backend seam
  without touching the runtime.
