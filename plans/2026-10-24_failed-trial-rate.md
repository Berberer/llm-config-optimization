# Failed-Trial Rate in Progress Output and Summary

## Problem Statement

While a sweep is running, the user can see each finished trial, its state, and the best objective so far, but cannot see how often trials are failing. A cluster of failures is often the first sign that the search space is malformed, that the selected model or image cannot run the requested configurations, or that the hardware is being pushed past its limits. Today the user must mentally count `FAIL` lines in log mode, or notice the absence of plotted points in TUI mode, to detect that pattern. By the time the final summary arrives, failures are no longer visible at all.

The failure rate is especially important for a resumed study: the user needs to know whether the current run is unhealthy, without earlier runs' failures making the current run look worse than it is.

## Solution

Show a compact failure percentage in both live output modes and in the final summary:

- Every log line gains a `failed <pct>%` segment.
- The TUI status line gains the same segment once at least one trial has finished.
- The final scalar and multi-objective summaries gain a `failed trials: n/m (pct)` line.

The rate counts Optuna `FAIL` trials only. Live modes count trials finished in the current invocation; the summary counts the whole study, including resumed trials. Whole percentages are used, with a nonzero rate that would round to `0%` shown as `<1%`, so a small but real failure rate is never hidden.

Both live modes also switch their top-level segment separator to ` | ` so the added segment and the existing fields remain visually distinct. The log stays one line per finished trial and remains easy to parse.

No warning thresholds, alerts, or new command-line flags are introduced. The displayed rate is the signal; interpreting what counts as "high" remains the user's judgement.

## User Stories

1. As an operator watching a sweep in log mode, I want each trial line to include the failure percentage so that I can see a failing search space developing without counting lines.
2. As an operator watching a sweep in TUI mode, I want the status line to include the failure percentage so that the signal is visible alongside the graph.
3. As an operator diagnosing a bad model or image, I want the failure rate to appear on failed trials as well as successful ones so that a run that fails immediately still shows its health.
4. As an operator running a clean sweep, I want `failed 0%` to be shown explicitly so that the absence of failures is unambiguous rather than inferred from a missing field.
5. As an operator seeing one failure in a large sweep, I want the rate shown as `<1%` rather than `0%` so that a real failure is not erased by rounding.
6. As an operator running a resumed study, I want the live failure rate to cover only trials finished in this invocation so that historical failures do not distort the current run's health.
7. As an operator reading the final summary after a resumed study, I want the summary to report failures across the whole study so that the persisted record is not misleadingly clean.
8. As a user of weighted and combined objectives, I want the failure rate in scalar log lines so that the feature is not limited to the multi-objective mode.
9. As a user of the multi-objective mode, I want the failure rate on multi-objective log lines and in the TUI multi-view status line so that both objective modes have the same diagnostics.
10. As an operator watching the TUI before the first trial finishes, I want the failure segment omitted rather than showing a meaningless `failed 0%` so that the status line does not imply a measured result that does not exist yet.
11. As an operator watching the TUI after a trial fails, I want the failure segment to appear on the next frame so that a failure is visible without waiting for a success.
12. As an operator reading a redirected log, I want the new ` | ` separators and the fixed one-line-per-trial shape so that the log is still easy to read and to split into fields.
13. As an operator reading the TUI status line, I want pipe separators between the trial count, failure rate, running marker, best value, average, and elapsed time so that the fields do not blur together.
14. As an operator whose first trial fails, I want the log line and TUI status line to immediately show `failed 100%`, so that an unusable configuration is obvious before more budget is spent.
15. As an operator whose sweep has a mix of successes and failures, I want the percentage computed against all trials finished in the invocation so that the denominator is the work actually attempted, not only successful work.
16. As an operator whose sweep fails entirely, I want the live failure rate to reach `failed 100%` before the run exits with the empty-sweep error so that the reason is visible in the progress output.
17. As an operator who does not want automated alerting, I want the tool to display the rate without emitting new warnings or failing the sweep based on a threshold so that behaviour stays predictable.
18. As a user of the summary output in scalar mode, I want a `failed trials: n/m (pct)` line next to `trials run` so that the winning configuration is reported with its failure context.
19. As a user of the summary output in multi-objective mode, I want the same `failed trials` line so that the Pareto front is reported with the same failure context.
20. As an operator reading a summary with no failures, I want the `failed trials` line to show `0/m (0%)` so that a clean study is explicitly clean.
21. As a developer maintaining the reporters, I want the percentage formatting and `<1%` rule implemented once so that log mode, TUI mode, and the summary cannot disagree.
22. As a developer extending the reporters, I want failure counting to use the trial state already available in the optimistic callback so that no changes to the optimizer or persisted study are required.
23. As a developer diagnosing the TUI, I want the failure count derived from the reporter's existing failed-trial placeholders so that there is no second source of truth for what has finished.
24. As a reader of the usage documentation, I want the example log and TUI lines updated to the new format and the failure semantics documented so that I know what the number measures.
25. As a test author, I want the log, TUI, and summary reporters tested as string producers so that the new output is pinned without requiring a terminal or a running container.

## Implementation Decisions

- **Failure definition**: A failed trial is one whose Optuna terminal state is `TrialState.FAIL`. `PRUNED` trials are not counted because this project does not prune, and `failed_runs` recorded inside a returnable experiment result is a different, run-level signal that must not be conflated with trial failure.
- **Live scope**: Log and TUI failure counts start at zero for each invocation and cover only trials finished in that invocation. This matches their existing invocation-scoped trial index and `--n-trials` interpretation.
- **Summary scope**: The summary counts `FAIL` trials across every trial in the result's study, including trials from an earlier invocation after a resume. This matches the existing whole-study `trials run` count.
- **Percentage format**: Whole-number percentages with a `%` suffix. A nonzero failure count whose rounded percentage is `0` is rendered as `<1%`. Zero failures render as `0%`. One hundred percent failures render as `100%`.
- **Log placement**: The `failed <pct>%` segment appears after the trial's outcome segment and before the `best` segment. The outcome's error message remains collapsed to one line for failed trials. Phase-rate pairs inside the outcome and inside the best column remain part of their existing semantic segments.
- **TUI placement**: The failure segment appears after the `trials <progress>` segment and before the `running trial` segment when one is present. It is omitted while no trial has finished in this invocation.
- **Separators**: Both live modes use ` | ` between top-level segments. Internal pairs such as `prefill … decode …` keep their existing spacing. The summary continues to use one labelled field per line.
- **Shared formatting**: A small private formatting helper module owns the failure percentage rendering, including the `<1%` rule, and the live segment separator. Log, TUI, and summary all use it so their output cannot drift.
- **Log reporter state**: The log reporter gains an invocation failure counter alongside its completed-trial counter. Rendering a trial includes that trial's failure in the displayed rate without mutating reporter state; the callback path advances both counters after printing.
- **TUI reporter state**: The TUI reporter needs no separate failure counter. Failed trials are already recorded as `None` entries in the scalar and multi series, so the failure count is derived from those series, including the trial currently being rendered.
- **Summary model**: The summary counts failures directly from the result's study. No new field is added to `OptimizationResult` or `OptimizationSolution`.
- **Optimizer and persistence**: No change is made to the Optuna adapter, the trial attributes, the persisted study schema, the objective model, or any backend. The trial state is already available to the existing reporter callback.
- **CLI surface**: No new flags, configuration keys, or output modes. The feature is always on in both live modes and in the summary.
- **Documentation**: The usage guide's TUI and log examples, its description of the status line and log line, and its summary description are updated to the new format and semantics.

## Testing Decisions

A good test pins externally observable output: the rendered log line, the rendered TUI status line, and the rendered summary. Tests should not assert on private counters, helper call counts, or other implementation details. They should cover the documented format and the semantic boundaries: zero failures, mixed outcomes, all failures, the `<1%` floor, invocation-scoped counting, and whole-study summary counting.

- **Log reporter**: Extend the existing string-producer tests. Update every pinned line for the ` | ` format, then add cases for zero failures, a first-trial failure at `100%`, a mixed sweep at a rounded percentage, a large sweep hitting `<1%`, and a resumed-study setup proving earlier failures are excluded from the invocation rate. Cover scalar and multi-objective lines.
- **TUI reporter**: Extend the existing frame/status-line tests. Update pinned status lines for the pipe format, verify the failure segment is absent from the pre-first-trial running frame, appears after the first finished trial, reflects failed trials in both scalar and multi series, and uses the same percentage formatting. Preserve the existing running-marker and average-duration expectations.
- **Summary**: Extend the summary tests to assert the `failed trials: n/m (pct)` line in both scalar and multi-objective summaries, with whole-study counts, zero failures, and the `<1%` floor.
- **Shared formatting helper**: Add focused unit tests for the percentage boundaries and the zero-finished-trials case, so the formatting rule is pinned independently of the three call sites.
- **Prior art**: Follow the existing log and TUI tests, which build frozen study/trial fixtures, inject a fake clock and `StringIO` stream, and assert on rendered strings. Follow the summary tests, which build an `OptimizationResult` with a study and assert on the rendered text.

## Implementation Tasks

[x] **T1 - Add the shared failure-formatting helper**: Create the private CLI formatting helper exposing the failure-percentage text and the live segment separator. Implement the whole-percent `<1%` rule and the zero-finished-trials case. Add focused unit tests for 0 failures, 1 failure in 3, 1 in 200, 200 in 200, and no finished trials.
[x] **T2 - Switch the log line to pipe-delimited segments**: Assemble the existing log line from top-level segments separated by the shared separator, preserving semantic segments such as phase-rate pairs. Update the pinned log expectations and the module's documented line shape. Do not add the failure segment yet.
[x] **T3 - Add the failure percentage to log lines**: Track failures finished in this invocation, include the current trial's `FAIL` state in `render` without advancing state, advance the counter in the callback, and insert the failure segment after the outcome and before `best`. Update the log docstring and add the failure-focused log tests.
[x] **T4 - Switch the TUI status line to pipe-delimited segments**: Assemble the status line from top-level segments separated by the shared separator, preserving the running marker, best text, and average-duration fields. Update the pinned TUI expectations and the module's documented status format. Do not add the failure segment yet.
[x] **T5 - Add the failure percentage to the TUI status line**: Derive the failed-trial count from the scalar values or multi pairs recorded so far, including the trial being rendered, and insert the failure segment after the progress segment. Omit it when no trial has finished, including the pre-first-trial running frame. Update the TUI docstring and add the failure-focused TUI tests.
[x] **T6 - Report failed trials in the final summary**: Count `FAIL` trials across the result's study and add a `failed trials: n/m (pct)` line after `trials run` in both the scalar and multi-objective summaries, using the shared percentage formatting. Update the summary docstring and tests.
[x] **T7 - Update the usage documentation**: Revise the TUI and log example lines and the surrounding prose to the pipe-delimited format, document the failure segment's meaning and invocation scope, and describe the summary's whole-study `failed trials` line.
[x] **T8 - Run the full quality gate**: Run the project formatter, linter, type checker, and unit test suite, and fix any fallout from the format and output changes.

## Out of Scope

- Automatic warnings, alerts, or aborts based on a failure-rate threshold.
- Counting `PRUNED` trials as failures.
- Counting trials with a nonzero `failed_runs` field as failed trials.
- Reporting per-run benchmark failures in the live progress lines.
- Adding a failure rate to persisted study metadata or changing the study schema.
- Machine-readable output formats such as JSON progress streams.
- Changes to the optimizer, objective model, inference backends, runtime, or search-space validation.
- New command-line flags or configuration keys.
- Reworking the summary's existing trial-count or recommended-configuration layout beyond the added failure line.

## Further Notes

- When every trial fails, the sweep raises the existing empty-sweep `OptimizationError` and prints no final summary, so the live `failed 100%` output is the only place the user sees the rate. This is expected; the spec does not change the empty-sweep behaviour.
- The TUI reporter is already documented as not thread-safe under `--n-jobs` greater than one; this feature adds no new concurrency risk because it reads the same per-invocation series the reporter already mutates.
- The log line remains one physical line per finished trial, which keeps redirected logs parseable. Pipe segments replace the previous double-space separation; this is an intentional format change and should be called out in the usage guide.
- The `<1%` rule applies to the summary as well as the live modes, so a study with one failure in a large run does not report a clean `0%` in its final report.
