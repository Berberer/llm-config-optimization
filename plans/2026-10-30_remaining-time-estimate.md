# Remaining-Time Estimate in Progress Output

## Problem Statement

A sweep takes a long time — every trial starts a container, loads the model, warms up, runs the benchmark, and is torn down — and the user has no idea how much longer it will run. The live output already shows the trial count against an `--n-trials` budget, a mean seconds-per-trial figure, and an elapsed clock, but the user must do the arithmetic themselves, and in log mode there is no average at all. When `--sweep-timeout` is set there is a hard deadline that the output never mentions, so the user cannot tell how much of their time budget is left or whether the sweep will end because of the deadline or the trial count.

The gap is worse for an unattended run: the user wants to know whether to wait, whether to come back in ten minutes, or whether the current search is going to consume the whole budget. Knowing the remaining time lets them decide whether to stop the sweep early, adjust the budget, or leave it running.

## Solution

Show a remaining-time estimate, `eta …`, in both live output modes, next to a now-labelled `elapsed …` column, and humanize all duration columns so the time information reads as one group:

- `--sweep-timeout` alone: the estimate is the time left until the deadline, known from the first frame.
- `--n-trials` alone: the estimate extrapolates the remaining trials from the observed mean duration of successful trials.
- Both: the sweep stops at whichever binds first, so the estimate is the smaller of the deadline remainder and the trial extrapolation.
- Neither: no estimate is shown.

The mean is exactly the one the TUI already displays as `avg trial`, counting successful trials finished in this invocation. Because trials running in parallel overlap, `--n-jobs` greater than one suppresses the trial-based term and only a deadline can produce an estimate. The value is clamped at `0s`; Optuna checks the deadline between trials, so the final trial may run past it.

Example lines:

```
TUI:  trials 3/10 | failed 0% | running trial 3 | best 42.00 tps | avg trial 12s | elapsed 37s | eta 2m 10s
log:  trial 3/10 | COMPLETE | 42.00 tps  prefill 1234.56 tps  decode 45.67 tps | failed 0% | best 42.00 | elapsed 37s | eta 2m 10s
```

No new flags are introduced. The startup header, the final summary, the optimizer, the study schema, and the backends are unchanged.

## User Stories

1. As an operator running a sweep with `--n-trials`, I want to see how long the remaining trials will take, so that I can decide whether to keep waiting or reduce the budget.
2. As an operator running a sweep with `--sweep-timeout`, I want to see the time left until the deadline, so that I know when the sweep will stop even before the first trial completes.
3. As an operator who set both budgets, I want the estimate to reflect whichever limit will stop the sweep first, so that the number is not misleadingly optimistic.
4. As an operator watching the TUI, I want the estimate in the status line next to the trial count and best value, so that all live information is in one place.
5. As an operator reading redirected log output, I want the estimate on every finished-trial line, so that a log left running unattended still records the projection.
6. As an operator running an unbounded sweep, I want no estimate rather than a made-up one, so that the output stays honest.
7. As an operator whose first trials have not finished yet, I want the trial-based estimate to stay absent until a successful trial gives it a basis, so that the output does not show a fabricated number.
8. As an operator using only `--sweep-timeout`, I want the estimate from the very first frame, because the deadline is already known.
9. As an operator with a mix of successful and failed trials, I want the estimate to use the same mean duration the TUI reports as `avg trial`, so that I can verify the arithmetic on screen.
10. As an operator whose trials fail, I want the estimate to keep tracking successful-trial duration rather than silently switching bases, so that the meaning of `avg trial` and `eta` stays consistent.
11. As an operator resuming a persisted study, I want the estimate to cover only this invocation's trials and budget, so that earlier runs are not counted against the current `--n-trials`.
12. As an operator resuming a study with only a timeout budget, I want the deadline estimate immediately, even though no trial of this invocation has finished.
13. As an operator running parallel trials with `--n-jobs`, I want the trial-based estimate suppressed rather than wrong by roughly the job count, so that the displayed number stays trustworthy.
14. As an operator running parallel trials with a timeout, I want the deadline estimate to remain available, because it does not depend on trial timing.
15. As an operator whose deadline has passed while the last trial is still running, I want the estimate to read `0s` rather than a negative or invented number, so that the state is clear.
16. As an operator reading the TUI at that point, I want the existing `running trial` marker to explain why the sweep is still alive after `eta 0s`, so that the overrun is not confusing.
17. As an operator reading a duration that is hours long, I want a readable `1h 5m` rather than a raw seconds count, so that the line stays scannable.
18. As an operator reading sub-minute durations, I want `42s` rather than `42.0s`, so that the time columns stay compact.
19. As an operator seeing all three time columns, I want `avg trial`, `elapsed`, and `eta` to use the same format, so that the line reads as one time group instead of three styles.
20. As an operator scanning the log, I want the `elapsed` and `eta` segments adjacent at the end of the line, so that past and remaining time are compared directly.
21. As a user of a forced single-trial run (an empty effective search space), I want the completed line to show `eta 0s` when a budget exists, so that the run's budget status is still truthful.
22. As a user of scalar, weighted, and multi-objective objectives, I want the estimate in every mode, so that the feature is not tied to one objective.
23. As an operator reading a failed log line, I want the time segments to remain present, so that the sweep's progress information is not lost on an error line.
24. As an operator reading a redirected log, I want the one-line-per-finished-trial shape preserved, so that the log stays parseable.
25. As a developer, I want the budget arithmetic implemented once, so that log mode and TUI mode cannot disagree about the deadline, extrapolation, minimum, parallel suppression, or clamp.
26. As a developer, I want the duration formatting implemented once, so that `avg trial`, `elapsed`, and `eta` cannot drift apart.
27. As a developer, I want the log reporter to measure the same start-to-finish trial duration as the TUI, so that both modes report the same mean.
28. As a developer, I want the reporters to keep rendering from injected clocks and streams, so that the feature stays testable without a terminal or real time.
29. As a reader of the usage guide, I want the updated example lines and a clear explanation of how the estimate is derived, so that I know what the number includes and excludes.

## Implementation Decisions

- **Surface**: Both live reporters gain the estimate. The startup header, the final summary, and the written configuration artifacts are unchanged.
- **Budget model**: A new CLI value object, `Budget`, holds the effective trial count, the wall-clock timeout, and whether trials run in parallel. It exposes a single projection method that returns the remaining seconds or `None` when no estimate is possible.
- **Projection rule**: The deadline term is `max(0, timeout - elapsed)`. The trial term is `max(0, n_trials - finished) × mean` over successful trials of this invocation. When only one term exists, it is used; when both exist, their minimum is used. When no term exists, the estimate is absent.
- **Availability**: A timeout-only budget shows the estimate from the first frame. When a trial count participates, the trial term is unknown until at least one successful trial has finished, so the estimate is absent until then — even if a timeout is also set, because the minimum cannot be known.
- **Parallel suppression**: When the budget is marked parallel (`--n-jobs > 1`), the trial term is treated as unknown, so only a timeout can produce an estimate. The progress column still shows `i/N`.
- **Mean basis**: The success-only mean already displayed as `avg trial`, scoped to trials finished in the current invocation. A resume therefore starts with no mean and builds one from its own successes. No study history and no `FrozenTrial.duration` values are read.
- **Log timing**: The log reporter gains an argument-less trial-start hook, and the CLI wires the optimizer's trial-start callback for both reporters. Both modes then average the same start-to-finish duration. The trial-start hook adds no output and does not change the log's one-line-per-finished-trial shape.
- **Deadline tail**: The estimate is clamped at `0s`. Optuna evaluates the deadline between trials, so the last trial may run past it; the documentation states this and the TUI's existing `running trial N` marker explains the still-live sweep.
- **Duration format**: A shared `duration_text` helper renders a duration as at most two units with no zero padding, rounded to the nearest second, and `0s` for zero — `0s`, `42s`, `2m 10s`, `1h`, `1h 5m`, `2d 4h`. It applies to `avg trial`, `elapsed`, and `eta`, replacing the previous one-decimal seconds and the parenthesized elapsed column.
- **Placement and labels**: All duration columns become labelled segments. `elapsed` then `eta` are the last two segments of the line, after the mode-specific fields, in both modes. The log format is `… | elapsed <d> | eta <d>`; the TUI status line is `… | avg trial <d> | elapsed <d> | eta <d>`. No parentheses remain around the elapsed time.
- **Reporter inputs**: The reporters take a single `Budget` instead of a bare trial count; the progress column reads the budget's trial count. The composition root builds the budget from the effective trial count (including the forced single trial for an empty effective space), the sweep timeout, and `--n-jobs`.
- **Rendering purity**: Rendering a trial continues to be side-effect free. The log reporter's current trial duration and success-mean contribution are computed in the render path from the trial-start instant; the callback path records the duration and advances the counters.
- **Documentation**: The log and TUI module docstrings, which spell out the line formats, and the usage guide's example lines and prose are updated to the new segments and to the estimate's semantics, scope, parallel limitation, and deadline overrun.

## Testing Decisions

Good tests pin externally observable output: the rendered log line, the rendered TUI status line, the formatted duration, and the projection number/absence. They should not assert on private counters, clock calls, or other internals. They should cover the documented format and the semantic boundaries: no budget, deadline only, trial count only, both, the minimum, zero remaining trials, clamping, parallel suppression, and the first-success availability rule.

- **Duration formatter**: Add focused unit tests for the boundaries — zero, sub-minute rounding, minutes with seconds, an exact hour, hours with minutes, days, a value that rounds to zero, and the no-zero-padding rule. Prior art: the shared failure-formatting tests.
- **Budget projection**: Add focused unit tests for a deadline-only budget before and after the deadline, a trial-count budget with zero and several remaining trials, the both-budgets minimum, an unknown mean, a parallel budget with and without a timeout, and an unbounded budget.
- **Log reporter**: Extend the existing string-producer tests. Update every pinned line for the labelled, humanized `elapsed` column, then add cases for a timeout-only estimate on the first line, a trial-count estimate after the first success, the both-budgets minimum, absence before the first success, `0s` at the tail, the parallel suppression, and the per-invocation behavior on a resumed study. Cover scalar and multi-objective lines and a failed line.
- **TUI reporter**: Extend the existing frame/status-line tests. Update pinned status lines for the humanized `avg trial` and labelled `elapsed`, then add cases for the estimate in the pre-first-trial running frame when only a timeout is set, in completion frames once a mean exists, the minimum rule, parallel suppression, and `0s` at the deadline. Keep the existing running-marker and graph assertions.
- **CLI composition**: Assert that the reporter receives the effective trial count, the sweep timeout, and the parallel flag from the command-line arguments, and that the trial-start hook is wired in both output modes. Follow the existing CLI/main tests.
- **Prior art**: Follow the existing log and TUI tests, which build real Optuna studies and frozen trials, inject a fake clock and `StringIO` stream, and assert on rendered strings. Follow the shared formatting tests for small pure helpers.

## Implementation Tasks

[x] **T1 - Add the shared duration formatter**: Extend the private CLI formatting helper with `duration_text`, implementing the two-unit, nearest-second, no-zero-padding, `0s` rules. Add boundary tests next to the existing shared-formatting tests.
[x] **T2 - Add the budget projection value object**: Create the private CLI budget module with the frozen `Budget` value object and its projection method implementing the deadline term, the successful-trial extrapolation, their minimum, parallel suppression, and the zero clamp. Add focused unit tests.
[x] **T3 - Give both reporters a budget**: Replace the reporters' bare trial-count argument with a `Budget`, keeping the progress column sourced from the budget's trial count and leaving output unchanged. Build the budget in the CLI composition root from the effective trial count, the sweep timeout, and the parallel flag. Update every reporter construction in the existing tests.
[x] **T4 - Humanize and label the existing time columns**: Render `avg trial` and `elapsed` through `duration_text`, replace the parenthesized elapsed column with an `elapsed <d>` segment, and update the module docstrings and all pinned lines. Do not add the estimate yet.
[x] **T5 - Add the estimate to the log line**: Add the log reporter's trial-start hook and successful-duration tracking, compute the current trial's duration and success-mean contribution in the render path, and append the `eta <d>` segment after `elapsed` when the budget yields one. Wire the optimizer's trial-start callback for both output modes. Update the log docstring and add the log-focused estimate tests.
[x] **T6 - Add the estimate to the TUI status line**: Compute the projection from the status line's completed count and success mean, append `eta <d>` after `elapsed` when it exists, and update the TUI docstring and add the TUI-focused estimate tests, including the timeout-only running frame and parallel suppression.
[x] **T7 - Update the usage documentation**: Revise the TUI and log example lines and the surrounding prose to the labelled, humanized time columns and the `eta` segment, and document the estimate's derivation, minimum rule, success-only invocation-scoped mean, parallel limitation, and deadline overrun.
[x] **T8 - Run the full quality gate**: Run the project formatter, linter, type checker, and unit test suite, and fix any fallout from the format and output changes.

## Out of Scope

- Changing the startup header's `budget:` line or adding an estimate before the sweep starts.
- Adding the estimate to the final summary, the written configuration files, or any machine-readable progress stream.
- New command-line flags, environment variables, or configuration keys for the estimate.
- Optimizing or reordering the search to finish within a budget.
- Persisting duration, ETA, or trial-timing data in the Optuna study or its metadata.
- Changing how `--sweep-timeout`, `--n-trials`, resume semantics, or `--n-jobs` behave.
- Interrupting a running trial when the deadline is reached; the estimate only reports the budget state.
- Reading earlier trials' durations on a resumed study to seed the mean.
- Reworking the TUI graph, its axes, or the log line's existing outcome/best/failure segments beyond the time columns.
- Adding background timers or threads to refresh the estimate between frames.

## Further Notes

- Humanizing `avg trial` and replacing `(36.9s)` with `elapsed 37s` is an intentional change to already-documented output; the usage guide and the pinned reporter tests must be updated together.
- The trial-based estimate slightly overstates the remaining time while a trial is in flight, because the in-flight trial is counted as one of the remaining trials even though part of its duration has already been spent. This is accepted as conservative.
- Optuna's timeout is measured from the start of the sweep call, which is where the reporters' elapsed clock starts, so the deadline term aligns with the sweep's own accounting.
- The log reporter already counts only this invocation's trials, and Optuna treats `--n-trials` as a budget for new trials on a resumed study, so the estimate's per-invocation scope is consistent with both.
- The TUI reporter remains documented as not thread-safe under `--n-jobs` greater than one; suppressing the trial-based estimate removes the estimate's dependence on the racy per-trial timings in that mode.
- T3 kept `Budget | None = None` on both reporters, normalising `None` to an unbounded `Budget()`, so the existing unbounded constructions (and future test call sites) need not pass one. The composition root builds the budget in a small `_sweep_budget` helper, and the optimizer now reads its effective trial count and timeout from that same budget so the reporter and the sweep cannot disagree.
- The success-mean bookkeeping is shared too, not just the budget arithmetic and the duration format: `TrialTimings` in `cli/_budget.py` owns the in-flight start, the recorded successful durations, and the mean, and both reporters delegate to it, so the TUI's `avg trial` and the `eta` input cannot drift apart. Its `start()` also clears the in-flight start, closing an edge where a restart followed by a render without the trial-start hook timed the next trial from the pre-restart instant.
- `duration_text` rounds half up explicitly (`int(seconds + 0.5)`) rather than through `round`, whose banker's rounding would render `2.5` as `2s` and `3.5` as `4s`; the usage prose still reads "rounded to the nearest second".
