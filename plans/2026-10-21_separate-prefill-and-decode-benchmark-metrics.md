# Separate Prefill and Decode Metrics in Trial Results and Objectives

## Problem Statement

A trial is scored by one number: total token throughput, computed as
(prompt tokens + generated tokens) ÷ elapsed wall-clock. With the default
workload of 200 prompts of 1024 input tokens and 128 output tokens, prompt
tokens outnumber generated tokens roughly eight to one, so the score is
dominated by the prefill side of the model's work. A configuration that
processes prompts quickly can win a trial while generating slowly, and the
tool offers no way to see which phase a configuration actually helped.

The data needed to spot this is mostly already there but thrown away: the
llama.cpp server returns per-request timings (prompt tokens and milliseconds,
generated tokens and milliseconds, cached tokens) in every completion
response, and those payloads are preserved in each run's raw JSON yet never
interpreted. vLLM exposes prompt-token and generation-token counters and
per-request prefill/decode time histograms on its metrics endpoint, which the
host client never reads.

Because the result is a single scalar, the sweep also cannot express a
prefill/decode trade-off: it must guess which phase matters before the run, or
pretend one number captures both.

## Solution

Report the two phases of LLM inference separately for every measured run and
every trial, and let the user optimize for them either as a trade-off or as a
weighted scalar.

A new pair of typed metrics — prefill tokens per second and decode tokens per
second — joins the existing combined throughput on each run and on the trial
aggregate. Each engine is measured from its own most robust server-reported
source:

- **vLLM** is measured through its Prometheus metrics endpoint, which is
  already enabled by default. The host client snapshots the endpoint before and
  after each measured run while the server container is alive, stores the
  standard exposition samples in the run's raw JSON, and the vLLM backend
  derives the two rates from the counter and histogram deltas. No engine flag
  or image changes are needed.
- **llama.cpp** is measured from the `timings` object the server already returns
  in every completion response. The llama.cpp backend derives the two rates from
  those per-response numbers in its result-parsing step. This deliberately
  avoids adding `--metrics` to the planned command, because the default images
  are floating tags and an older overridden image would fail to start on an
  unknown flag.

The host client stays engine-neutral: it only knows how to fetch a
spec-declared metrics path and parse the standard Prometheus exposition format.
Every engine-specific series name and every piece of phase arithmetic lives in
the backend module that owns that engine, matching how parameter spellings are
already confined.

Per-run rates are computed as total phase tokens divided by total phase seconds
(reported by the engine), so a rate is tokens per second of actual phase work
rather than a wall-clock partition — under continuous batching and chunked
prefill the two phases are interleaved, and only the engine can attribute work
to a phase. Trial values are the mean over the successful runs, exactly
mirroring how combined throughput aggregates today. Raw per-request and
snapshot data stays in raw JSON for audit.

The optimizer gains a mode choice:

- **multi** (the new default): a two-objective study maximizing prefill rate and
  decode rate, producing a Pareto front, so the trade-off decision is made after
  the sweep rather than baked into weights beforehand;
- **weighted**: a single scalar using the weighted-harmonic effective rate
  `(w_p + w_d) / (w_p/prefill + w_d/decode)`. The weights are a token mix, so
  the score is a tokens-per-second rate interpretable as "throughput for a
  request shaped like my weights"; they default to the benchmark's
  `input_len`/`output_len`, making `--objective weighted` meaningful without
  flags while still biasing per deployment;
- **combined**: the exact old objective, kept as an escape hatch and as the
  interpretation of legacy studies.

The Pareto front becomes first-class in the optimization result: the final
summary prints it as a table with a recommended row marked, the per-trial log
line shows both rates, and the TUI draws a prefill-versus-decode scatter with
the frontier highlighted. The recommended row is the front point with the
highest weighted-harmonic score under the same weights, and `--output-config`
keeps writing exactly one re-runnable configuration by writing that point.

When a mode needs a rate the engine did not report, the trial fails with a clear
error naming the source rather than silently optimizing something else; the
combined mode tolerates absent phase data as before. Persisted studies record
the objective mode that produced them, refuse a resume whose mode differs, and
default study names gain a mode suffix, so a sweep can never silently append
weighted scores to a combined study.

The work is delivered in two phases: first the metrics end to end with the
objective still combined, then the objective modes, Pareto result, reporting,
CLI, and default flip.

## User Stories

1. As a tuning user, I want each trial to report prefill and decode throughput
   separately, so that I can tell which phase a configuration helps.
2. As a tuning user, I want the combined tokens-per-second metric kept, so that
   existing reports and comparisons keep working.
3. As a tuning user, I want phase rates to be independent of the workload's
   input/output token mix, so that a 1024/128 workload does not inflate the
   score with prompt tokens.
4. As a tuning user, I want phase rates reported per measured run, so that I can
   see the spread across repetitions.
5. As a tuning user, I want the trial's phase rates to be the mean of its
   successful runs, so that the new metrics aggregate exactly like the existing
   throughput.
6. As a tuning user, I want raw engine data preserved on each run, so that I can
   audit a suspicious result later.
7. As a vLLM user, I want phase rates taken from the server's own metrics, so
   that they are not contaminated by client-side queueing under concurrent
   requests.
8. As a vLLM user, I want no new engine flags or image changes, so that my
   existing images keep working.
9. As a llama.cpp user, I want phase rates taken from the response timings the
   server already returns, so that no metrics flag is required and pinned images
   keep working.
10. As a llama.cpp user, I want prompt tokens served from cache excluded from
    the prefill rate, so that the rate measures work actually performed.
11. As a tuning user, I want a missing phase metric to fail the trial with a
    clear message, so that the objective never changes silently.
12. As a tuning user, I want the combined objective to still run when phase data
    is absent, so that reporting works on engines that do not expose it.
13. As a tuning user, I want multi-objective optimization to be the default, so
    that I do not have to commit to a trade-off before the sweep.
14. As a tuning user, I want to select a weighted single objective on the
    command line, so that I can favor one phase deliberately.
15. As a tuning user, I want the weights to act as a token mix and default to
    the benchmark's mix, so that `--objective weighted` works without
    arithmetic.
16. As a tuning user, I want the weighted score to stay in tokens-per-second
    units, so that it is directly interpretable.
17. As a tuning user, I want to opt back into the exact old combined objective,
    so that a familiar workflow remains available.
18. As a tuning user, I want the final result to contain the Pareto front, so
    that I can choose a configuration after seeing the trade-off.
19. As a tuning user, I want a recommended point from the front, so that I get a
    usable configuration without interpreting the whole front.
20. As a tuning user, I want `--output-config` to keep writing one re-runnable
    file, so that my downstream workflow is unchanged.
21. As a tuning user, I want the per-trial log line to show both rates, so that
    I can watch the sweep live.
22. As a TUI user, I want the live plot to show prefill against decode with the
    frontier highlighted, so that the trade-off is visible as it forms.
23. As a TUI user running a scalar mode, I want the existing line plot to keep
    working, so that the familiar view is not lost.
24. As a tuning user, I want the final summary to print the front as a table
    with the recommended row marked, so that I can compare candidates at a
    glance.
25. As a tuning user, I want persisted studies to record which objective
    produced them, so that a resumed run cannot silently mix modes.
26. As a tuning user, I want a clear error when resuming with a different
    objective mode, telling me to pick another study name, so that the fix is
    obvious.
27. As a tuning user, I want the default study name to distinguish objective
    modes, so that the zero-config path cannot collide with an older study.
28. As a user of an old study, I want it treated as combined, so that resuming
    keeps working until I deliberately change modes.
29. As a tuning user, I want weights validated before any expensive work, so
    that a typo fails fast.
30. As a tuning user, I want explicit weights required when the benchmark cannot
    supply a token mix, so that no arbitrary default is silently invented.
31. As a tuning user, I want an interrupted sweep to expose the same result
    shape as a finished one, so that Ctrl-C never discards the front found so
    far.
32. As a maintainer, I want the host client to speak only the standard
    Prometheus exposition format, so that a future engine can adopt the same
    seam.
33. As a maintainer, I want engine spellings confined to backend modules, so
    that the architecture invariants in AGENTS.md keep holding.
34. As a maintainer, I want the change split into a metrics phase and an
    objective/reporting phase, so that each lands testable on its own.
35. As a tuning user, I want the README to explain the phase metrics and
    objective modes, so that I can use them without reading the code.
36. As a maintainer, I want the new metric schema version recorded with a study,
    so that a future metric-definition change is detectable.
37. As a tuning user, I want phase rates for a dataset workload to come from the
    same per-response path, so that custom workloads need no extra
    configuration.

## Implementation Decisions

### Metrics and sources

- `RunMetrics` gains two optional fields, `prefill_tokens_per_second` and
  `decode_tokens_per_second`, defaulting to `None` so existing constructions and
  fixtures remain valid. Its hash and immutability behavior is unchanged.
- `ExperimentResult` gains matching aggregate properties that average the
  present run values and return `None` when no run has one, mirroring the
  existing mean-over-runs semantics of `tokens_per_second`.
- `ServerSpec` gains an optional `metrics_path`. It is engine-neutral data:
  `None` means the engine exposes no scrapeable metrics and the client does
  nothing extra. The vLLM backend declares its path; the llama.cpp backend
  leaves it unset.
- When a spec declares a metrics path, the host benchmark client fetches it
  immediately before and immediately after each measured run while the server
  is alive, parses the standard Prometheus exposition format into a
  name-to-value mapping (labels ignored, single model per server), and stores
  both snapshots in the run's raw JSON under a stable key. Fetch or parse
  failures are best-effort: the key is absent and the run proceeds. The client
  gains no engine vocabulary, only format knowledge.
- The vLLM backend owns its series names and arithmetic: prefill rate from the
  delta of the prompt-token counter divided by the delta of the prefill-time
  histogram sum, decode rate from the generation-token counter divided by the
  decode-time histogram sum. Zero or missing denominators, or a missing
  snapshot, yield `None`.
- The llama.cpp backend owns its arithmetic: per response, it sums `prompt_n`
  tokens and `prompt_ms` milliseconds, and `predicted_n` tokens and
  `predicted_ms` milliseconds, then divides total tokens by total seconds.
  `prompt_n` already excludes tokens served from cache (`cache_n`), so the
  prefill rate measures processed work. Missing timings or a zero millisecond
  denominator yield `None`.
- Backends enrich runs in their result-parsing step; the raw per-request
  payloads and metric snapshots remain in raw JSON.
- These are engine-reported phase work rates, not a wall-clock partition:
  continuous batching and chunked prefill interleave the phases, and the
  documented definition must say so.

### Objective modes

- The optimization layer gains an engine-neutral `ObjectiveMode` with `multi`,
  `weighted`, and `combined`, plus an `ObjectiveWeights` value carrying the
  prefill and decode weights.
- The weighted score is the weighted-harmonic effective rate
  `(w_p + w_d) / (w_p / prefill + w_d / decode)`, which stays in
  tokens-per-second units. A zero weight degenerates to the other phase;
  both weights zero is rejected at validation.
- The combined score is the existing combined `tokens_per_second`.
- Multi mode returns a tuple of the two rates to Optuna and creates the study
  with two maximize directions. Weighted and combined modes remain
  single-objective maximization.
- The library default mode flips to multi together with the CLI, so intermediate
  tasks do not have to update every existing caller at once.
- If a mode needs a rate a result does not carry, the objective raises
  `ExperimentError` with a message naming the missing source (`/metrics` or
  `timings`), which the existing catch path records on a failed trial.

### Result, reporting, and persistence

- `OptimizationResult` carries `solutions` — the Pareto front in multi mode, the
  single winner in scalar modes — and `recommended`, the point `--output-config`
  writes. In multi mode the recommended point is the front member with the
  highest weighted-harmonic score under the configured weights. The existing
  `best_config` and `best_value` remain as aliases of the recommended point so
  existing callers and tests keep working until the reporters are updated.
- The interrupt path builds the same result shape.
- The log reporter prints both rates for a completed trial on its existing
  single line.
- The TUI draws a prefill-versus-decode scatter with the front highlighted in
  multi mode and keeps its best-so-far line plus raw points in scalar modes. The
  status line reports both best rates in multi mode.
- The final summary prints the front as a table (trial number, both rates,
  combined rate, recommended marker) in multi mode and the single winner in
  scalar modes.
- `--output-config` writes the recommended configuration; the single-file
  contract and the resolved target/image substitution are unchanged.
- Studies record an objective-mode user attribute and a metric-schema version
  attribute. On resume, a mismatch raises a clear `OptimizationError` naming the
  recorded mode and instructing a different study name; a missing attribute is
  treated as legacy `combined`.
- The default study name gains a mode suffix.
- The optimizer's per-trial projection is extended with per-run and aggregate
  phase rates and must stay JSON-serializable.

### CLI

- `--objective multi|weighted|combined` selects the mode and defaults to multi.
- `--prefill-weight` and `--decode-weight` are non-negative floats that default
  to the benchmark's `input_len` and `output_len`. When the benchmark cannot
  supply a mix, weighted mode requires both weights explicitly. At least one
  weight must be positive; violations are usage errors before any preparation.
- The help text and README document the phase definitions, the objective modes,
  the weight semantics, and the study-resume behavior.

## Testing Decisions

A good test here asserts typed, external outputs — a parsed run's phase rates, a
trial's objective tuple, a rendered log line or frame, a reconstructed study
result — and not internal call sequences. Engine fakes and injectable transports
keep unit tests free of a daemon and a GPU, matching the existing convention.

- **Inference result types** (`RunMetrics`, `ExperimentResult`): extend the
  existing type tests for the optional fields, hashing, and `None`-aware
  aggregation. Prior art: the inference type tests that pin hash and aggregate
  behavior.
- **Host benchmark client**: extend the fake-transport client tests with
  metrics-endpoint responses before/after a run, a missing endpoint, and an
  unparseable payload; assert the stored raw JSON and that a failure never fails
  the run. Prior art: the benchmark-client tests with `MockTransport` and the
  fake clock.
- **vLLM parsing**: fixture Prometheus expositions with counter and histogram
  deltas produce the expected rates; missing or zero deltas produce `None`.
  Prior art: the vLLM result tests that drive `parse_result` with fabricated
  outcomes.
- **llama.cpp parsing**: fixture completion payloads with and without `timings`
  produce the expected rates and `None`; cached tokens are excluded. Prior art:
  the llama.cpp result tests.
- **Objective scoring**: pure-function tests for the weighted-harmonic formula
  (parity with a phase when the other weight is zero, both-positive ordering,
  `None` handling) and the combined passthrough. Prior art: the optimization
  layer's pure mapping tests.
- **Optuna adapter and lifecycle**: multi mode creates a two-direction study and
  returns tuples; weighted and combined stay scalar; a missing rate fails the
  trial with the expected error; provenance mismatches raise. Prior art: the
  optimization lifecycle tests with their fake experiment runner.
- **Result projection**: the recorded trial attributes include phase rates and
  remain serializable.
- **Reporters**: log and summary tests assert the rendered strings for multi and
  scalar modes, including the recommended marker; the TUI is exercised through
  its string producer. Prior art: the CLI reporter tests.
- **CLI arguments**: parsing, defaults from benchmark lengths, missing-weight
  rejection, zero/negative rejection, and mode default.
- **Integration**: extend the engine-marked integration tests to assert that a
  reported phase rate is positive, and add an optimization-level test with a
  fake backend returning phase rates that exercises multi mode end-to-end and
  the recommended pick. These remain deselected by default.

## Implementation Tasks

### Phase A — measurement and diagnostics

- [x] **T1 - Add phase rate fields to the inference result types**: Add the two
  optional phase-rate fields to `RunMetrics` and the `None`-aware mean aggregate
  properties to `ExperimentResult`, update their docstrings (including the
  existing note about the combined metric's limitation), and extend the type
  tests.
- [x] **T2 - Declare an optional metrics path on the server spec**: Add the
  engine-neutral `metrics_path` field to `ServerSpec` with documentation; no
  behavior change yet, existing plan tests keep passing.
- [x] **T3 - Snapshot declared metrics from the host benchmark client**: When a
  spec declares a metrics path, fetch it before and after each measured run,
  parse the standard Prometheus exposition into a name-to-value mapping, and
  store the snapshots in the run's raw JSON; failures are best-effort. Extend
  the fake-transport client tests for success, absence, and malformed payloads.
- [x] **T4 - Derive vLLM phase rates from its metrics**: Have the vLLM backend
  declare its metrics path in its plan and compute the two rates from the
  counter and histogram-sum deltas in each run's snapshots during result
  parsing, yielding `None` on missing or zero data. Test with fixture
  expositions.
- [x] **T5 - Derive llama.cpp phase rates from response timings**: Have the
  llama.cpp backend compute the two rates from each run's per-response timings
  during result parsing, excluding cached prompt tokens and yielding `None` on
  missing data. Test with fixture completion payloads.
- [x] **T6 - Record phase rates in the trial history**: Extend the optimizer's
  per-trial result projection with per-run and aggregate phase rates, keeping it
  JSON-serializable, and test the projection.

### Phase B — objectives and reporting

- [x] **T7 - Add the objective mode model and scoring functions**: Introduce
  `ObjectiveMode`, `ObjectiveWeights`, and pure scoring functions for the
  weighted-harmonic and combined objectives, with unit tests for the formula and
  edge cases.
- [x] **T8 - Drive the selected objective through the Optuna adapter**: Accept
  mode and weights in the objective builder and sweep entry point, create
  two-direction studies and tuple objectives for multi, keep scalar behavior for
  weighted and combined, and fail a trial with a source-naming error when a
  required rate is missing. Test with the fake experiment runner.
- [x] **T9 - Reshape the optimization result around the Pareto front**: Expose
  `solutions` and `recommended` on `OptimizationResult` (scalar modes yield a
  single solution), compute the recommended point for multi, keep
  `best_config`/`best_value` as aliases, and update the interrupt path. Test the
  front shape, recommended pick, and aliases.
- [x] **T10 - Guard persisted studies by objective mode**: Record the mode and
  metric-schema version as study user attributes, refuse a mismatched resume
  with a clear error, treat an unmarked study as legacy combined, and suffix the
  default study name by mode. Test resume across modes and the legacy case.
- [x] **T11 - Show both rates in the log reporter**: Extend the per-trial line
  to include prefill and decode rates for multi and scalar modes while keeping
  the one-line format and failed-trial handling. Test the rendered strings.
- [x] **T12 - Draw the multi-objective TUI view**: Render a prefill-versus-
  decode scatter with the frontier highlighted in multi mode, keep the existing
  line plot for scalar modes, and report both best rates in the status line.
  Test through the string producer.
- [x] **T13 - Print the Pareto front and write the recommended config**: Render
  the front table with the recommended row marked in multi mode and the single
  winner otherwise, and make `--output-config` write the recommended
  configuration. Test the summary strings and the written document.
- [x] **T14 - Add the objective CLI surface and flip the default**: Add
  `--objective` (default multi), `--prefill-weight`, and `--decode-weight` with
  benchmark-derived defaults and validation, wire them into the sweep, and flip
  the library and CLI defaults to multi. Test parsing, defaults, and rejections.
- [x] **T15 - Document the metrics, objectives, and invariants**: Update the
  README with the phase definitions, the objective modes, weight semantics,
  study-resume behavior, and the note that the new numbers are not comparable to
  older ones; update the AGENTS.md invariants for the metrics seam, backend-owned
  spellings, and mode provenance.
- [x] **T16 - Extend end-to-end coverage**: Assert positive phase rates in the
  engine-marked integration tests and add a fake-backend optimization test that
  exercises multi mode and the recommended pick.

## Out of Scope

- Latency metrics (time to first token, inter-token latency) and SSE streaming
  in the host client.
- Prometheus scraping for llama.cpp or adding `--metrics` to its planned
  command.
- Weighted-sum and other alternative combination formulas beyond the
  weighted-harmonic rate.
- Objectives beyond the two phase rates, such as latency or memory.
- Changing benchmark concurrency or adding a concurrency flag.
- Changes to the search space, engine flags, examples, or the container runtime
  layer.
- Migrating or reinterpreting existing study databases.
- Deprecating the combined objective.

## Further Notes

- The exact vLLM series to read are the prompt-token and generation-token
  counters and the prefill-time and decode-time histogram sums; only their
  names and label handling belong to the vLLM backend. The exact llama.cpp
  fields are the per-response prompt and predicted token counts and
  milliseconds. These names should be re-verified against the pinned engine
  documentation at implementation time.
- Prefix caching is already reset between runs by the vLLM backend, so counter
  deltas for a measured run are dominated by that run and prompt-token counts
  are effectively processed tokens. llama.cpp exposes no reset endpoint; cached
  tokens are excluded from its prefill numerator by `prompt_n` itself.
- Metric snapshots and per-request payloads live only in memory on
  `RunMetrics`; the trial projection stores numbers only, so study storage does
  not grow with exposition text.
- The vLLM metrics endpoint is available because the backend already runs the
  server in dev mode and does not disable log stats; this task adds no flag.
- Multi-objective Optuna studies are natively supported by the TPE sampler; the
  existing `n_jobs` caveat about disjoint devices and storage is unchanged.
- The weighted-harmonic weights are token counts, not fractions; any positive
  scaling of a weight pair produces the same score.
- A future change to the phase-rate definitions must bump the recorded metric
  schema version so old studies are not silently compared with new ones.
- T3's stable snapshot key is ``raw_json["metrics"]`` holding
  ``{"before": {...}, "after": {...}}`` (parsed name-to-value mappings); the key
  is absent unless both snapshots were taken, so T4/T5 read the pair or nothing.
  The before scrape happens after the cache reset and the after scrape after the
  measured run's last response; warmup runs are never scraped.
- T8 leaves multi-objective *result projection* to T9 by design: `optimize(mode=multi)`
  builds the two-direction study and drives tuple objectives, but `_result_from_study`
  raises a deferred `OptimizationError` (with the study attached) instead of picking a
  single winner. T9 replaces that guard with the Pareto-front result; the interrupt
  path goes through the same function.
- T9 requires `weights` for the multi mode as well as the weighted one: the recommended
  front point is the front member with the highest weighted-harmonic score under those
  weights, so no arbitrary pick is invented when they are absent. `make_objective` needs
  no weights for multi (the objective itself returns the tuple); `optimize` does, so
  T14's CLI must reject a multi run whose weights cannot be derived from the benchmark.
  T9 also keeps `best_trial_number` as a third legacy alias, since the current summary
  prints it, and replaces the T8 "deferred multi result" test with one asserting the
  front, the recommended pick, and the aliases.
- T10's provenance guard distinguishes a new or empty study from a legacy one: a study
  with no trials and the direction count of the current mode adopts that mode, which is
  how a new study records its `objective_mode` and `metric_schema_version` attributes at
  all. An unmarked study with trials — or an empty one whose direction count differs,
  since Optuna cannot change a study's directions after creation — is treated as legacy
  combined and a non-combined resume is refused. The schema version is recorded but not
  enforced on resume yet; a future change to the phase-rate definitions should bump
  `_METRIC_SCHEMA_VERSION` in `optimization/_optuna.py` and add the refusal then.
- T11's log line appends the recorded phase pair to a scalar trial's objective value
  only when both rates are present, so an engine that reports no phase data keeps the
  historical line; a multi trial's two objective values are its rates, and its `best`
  column shows the highest prefill and decode rate reached — the two maxima may come
  from different trials, since a multi study has no single best value.
- T12 passes the objective mode into `TuiReporter` at construction instead of inferring
  it from the study, because the argument-less `trial_started` hook draws before any
  study exists. `_make_reporter` therefore takes the mode (still the hard-coded
  `combined` until T14) while `LogReporter` keeps inferring it from the study's
  directions on every line. The multi scatter's raw points are the invocation's own
  trials while the highlighted front is read from the study, mirroring how the scalar
  best line is study-wide and its raw points are local.
- T16 lands its fake-backed multi test as an integration-marked companion in
  `tests/optimization/test_optimize_integration.py`: T9's lifecycle test already
  exercises the same sweep at the unit level, while this one pins the two-direction
  study, the Pareto front, the weighted-harmonic recommended pick, and the legacy
  aliases without a daemon. The real optimization sweeps also assert positive phase
  rates in the recorded study projection.