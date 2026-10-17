# CLI Reporter UX Improvements

## Problem Statement

A run of `llm-optimization` is hard to read while it is happening, in ways that
misreport state and hide whether the tool is still working:

- During a Hugging Face model download, the Hub's warning about a missing
  `HF_TOKEN` is printed on the same line as the progress bar, and the next bar
  refresh overwrites its first character, so the user sees `arning: Hugging
  face download ...` instead of `Warning: ...`. The warning is still emitted,
  but it looks broken.
- The download progress bar carries a generic description (`Downloading bytes`,
  or `Fetching N files`) that never says which model is being fetched, so a run
  that spends minutes downloading cannot be told apart from one fetching
  something else.
- Neither mode says anything while a trial is running. A benchmark trial can
  take minutes, and with the screen static the user cannot tell a long trial
  from a frozen process.
- The TUI status bar reports the count, the best value, and the elapsed time,
  but not how long a trial typically takes, so there is no way to estimate how
  long the remaining trials will need.
- The TUI y-axis switches to scientific notation for large throughput values
  (`1.2e6`), which is harder to read than a plain formatted number.
- The TUI x-axis (trial number) shows fractional ticks (`1.0`, `1.7`, `2.3`),
  even though trial numbers are always integers.
- When a trial fails, the message shown to the user is nearly useless: the
  engine's captured output is excerpted to its last 2000 characters, which is
  the traceback tail, so the actual root cause — printed earlier, for example
  `AssertionError: ChunkGatedDeltaRuleFunction does not support float32` — is
  cut off. The log reporter then collapses the whole message onto one line, so
  even the traceback frames are flattened. The user cannot tell a configuration
  the engine rejected from a broken environment without re-running the trial by
  hand.

## Solution

Fix the reporter and preparation output so a live run is legible and honest
about what it is doing:

- Take over the download progress display: render one line that names the model
  and shows file progress and percent, and terminate that line before the Hub's
  warning is written so the warning prints intact on its own line.
- In TUI mode, draw a frame at the start of each trial whose status line marks
  the trial as running, so a long trial is visibly in progress. Preparation
  output already precedes the first frame: the CLI prints its last preparation
  message before the reporter starts, and the reporter's `start` prints a
  preparation line rather than an empty plot.
- Add the mean time per successful trial to the TUI status line, so the
  remaining budget can be estimated.
- Format the y-axis tick labels as plain thousands-separated numbers, and use
  integer-only x-axis ticks.
- Make the failure excerpt preserve the engine's root cause instead of only the
  traceback tail, so the reason a trial failed survives both the excerpt and the
  reporter's single-line collapse.

Log mode keeps its current one-line-per-trial format.

## User Stories

1. As an LLM-optimization developer, I want the missing-`HF_TOKEN` warning to
   print intact, so that I can read the advice instead of a truncated word.
2. As an LLM-optimization developer, I want the download progress line to name
   the model being fetched, so that I know what a long download is doing.
3. As an LLM-optimization developer, I want the download line to show file
   progress and a percentage, so that I can judge how far along the download
   is.
4. As an LLM-optimization developer, I want the download progress to stay on a
   single updating line, so that a multi-gigabyte download does not flood my
   terminal with frames.
5. As an LLM-optimization developer, I want the Hub warning and the progress
   line not to overwrite each other, so that both remain readable.
6. As an LLM-optimization developer, I want the download progress to go to the
   same stream as the rest of the CLI progress, so that a redirected run
   captures a consistent record.
7. As an LLM-optimization developer, I want the Hub's internal reconstruction
   and transfer bars hidden, so that only one progress line is shown.
8. As an LLM-optimization developer, I want a local model path to still skip
   the download entirely, so that offline checkpoints are unaffected.
9. As an LLM-optimization developer, I want a failed download to still fail the
   run before any trial, so that the existing error contract is preserved.
10. As an LLM-optimization developer, I want the last preparation message to
    print before the TUI graph, so that the screen does not look like it
    started mid-setup.
11. As an LLM-optimization developer, I want no empty TUI frame before the first
    trial, so that the cold start shows an explanation rather than a blank plot.
12. As an LLM-optimization developer, I want the TUI to mark a trial as running
    when it starts, so that I can tell a long trial from a frozen tool.
13. As an LLM-optimization developer, I want the running marker to name the
    trial number, so that I know which trial is in flight.
14. As an LLM-optimization developer, I want the running marker to coexist with
    the best-so-far value and elapsed time, so that the status bar stays a
    complete summary.
15. As an LLM-optimization developer, I want log mode to keep one line per
    finished trial, so that a redirected log stays easy to parse.
16. As an LLM-optimization developer, I want the TUI status bar to show the
    average time per successful trial, so that I can estimate the remaining
    run time.
17. As an LLM-optimization developer, I want failed trials excluded from that
    average, so that quick out-of-memory failures do not make the estimate
    unrealistically low.
18. As an LLM-optimization developer, I want the average to be omitted until a
    trial succeeds, so that the status bar never shows a meaningless number.
19. As an LLM-optimization developer, I want the y-axis to use plain
    thousands-separated numbers, so that large throughput values are easy to
    read.
20. As an LLM-optimization developer, I want the y-axis to never switch to
    scientific notation, so that the scale is consistent across a sweep.
21. As an LLM-optimization developer, I want the x-axis to show only integer
    trial numbers, so that the labels match the data.
22. As an LLM-optimization developer, I want short sweeps to show every trial
    number, so that a small sweep is easy to follow.
23. As an LLM-optimization developer, I want long sweeps to thin the x-axis to a
    readable set of integer ticks, so that labels never overlap.
24. As an LLM-optimization developer, I want the first and last trial to always
    be ticked, so that the axis endpoints are clear.
25. As an LLM-optimization developer, I want the TUI to keep drawing one frame
    per trial completion, so that the graph still shows the sweep converging.
26. As an LLM-optimization developer, I want the running frame and the
    completion frame to share the same layout, so that the display does not
    jump between two shapes.
27. As a test author, I want the TUI status line and tick labels testable as a
    string, so that I can assert on them without a terminal.
28. As a test author, I want the download progress renderer testable with a
    fake download, so that the model name and warning handling are covered
    without the network.
29. As a test author, I want the preparation ordering testable with captured
    streams, so that "preparation before plot" is pinned.
30. As a test author, I want the new trial-start hook testable with an injected
    callback, so that the wiring is covered without a daemon.
31. As a future maintainer, I want the download progress isolated behind one
    helper, so that Hugging Face progress internals are touched in one place.
32. As a future maintainer, I want the trial-start hook to be optional, so that
    callers that do not report progress need no change.
33. As a future maintainer, I want tick formatting computed by the reporter
    rather than by patching plotext, so that the dependency stays replaceable.
34. As an LLM-optimization developer, I want a failed trial's message to name
    the engine's root cause, so that I can tell a bad configuration from a
    broken environment.
35. As an LLM-optimization developer, I want the root cause to survive even when
    the engine prints a long traceback after it, so that the most useful line is
    not the one truncated away.
36. As an LLM-optimization developer, I want the excerpt to stay bounded, so
    that a chatty engine cannot flood the log.
37. As an LLM-optimization developer, I want the root cause to remain readable
    after the reporter collapses the message to a single line, so that log mode
    keeps its format without losing the reason.
38. As a test author, I want the failure excerpt testable as a string producer
    with synthetic engine output, so that the root-cause selection is covered
    without a container.
39. As a future maintainer, I want the root-cause selection isolated behind one
    helper, so that the heuristic can be tuned in one place.

## Implementation Decisions

**Download progress and warning handling**

- The CLI stops using `snapshot_download`'s default progress rendering. It
  passes a custom `tqdm` subclass (derived from `huggingface_hub.utils.tqdm`)
  that renders exactly one line: the primary bar created by
  `hf_thread_map` — the one constructed without a `name` — is kept and its
  description is rewritten to name the model; the named
  `huggingface_hub.snapshot_download*` transfer and reconstruction bars are
  created disabled so only one line renders.
- The kept line shows file count and percent (`downloading <model>: 40%|…| 4/10`),
  because the file bar has a stable total from the start. Byte progress is not
  used because the transfer bar's total grows as files begin, so it has no
  reliable percentage.
- The progress line is written to the CLI's progress stream, matching the rest
  of the CLI's stdout progress.
- To fix the truncated warning, a temporary logging handler is installed on the
  `huggingface_hub` logger for the duration of the download. It terminates the
  progress line, writes the record on its own line, and restores the previous
  handlers and propagation flag afterwards. This prevents the next progress
  refresh from overwriting the warning's first character.
- The whole download behaviour is isolated behind one helper so Hugging Face
  progress internals are touched in a single place.
- A local model path still bypasses the download; a download failure still
  raises the existing CLI failure and exits before any trial.

**Preparation ordering in TUI mode**

- The CLI prints the final preparation message (`preparing study`) before it
  calls the reporter's `start`. `start` prints a preparation line rather than an
  empty frame, so the cold start is explained and no blank plot is shown, and
  the first graph frame is the running-trial frame the trial-start hook draws.
  The ordering itself needs no further change.

**Running-trial indicator**

- The optimization entry point gains an optional, argument-less
  `on_trial_start` callback. The objective invokes it once at the start of each
  trial, before the experiment runs. `make_objective` and `optimize` both accept
  and forward it, defaulting to `None` so existing callers are unchanged.
- Only the TUI reporter is wired to the hook. The TUI reporter's
  `trial_started` records the trial's start time and draws a frame whose status
  line marks the trial as running. Log mode is deliberately unchanged and keeps
  its one-line-per-finished-trial format.
- The running trial number is derived from the reporter's own finished-trial
  count, so it is numbered within this invocation, matching the existing
  `trials i/N` column.
- With `--n-jobs` greater than one the hook fires from worker threads; the
  reporter is not thread-safe, which is already true of the existing per-trial
  callback. This limitation is documented rather than solved.

**Average time per successful trial**

- The TUI reporter measures each trial's duration from its `trial_started`
  timestamp to its completion callback, and accumulates only trials that
  produced an objective value (successful trials). Failed trials contribute to
  the elapsed time but not to the average.
- The average is rendered in the status line as `avg trial <seconds>s` between
  the best value and the elapsed time, and omitted until the first successful
  trial. Example: `trials 3/10  best 42.00 tps  avg trial 12.3s  (36.9s)`.
- When the reporter is driven without a preceding `trial_started` (as some
  tests do), the trial start falls back to the reporter's own start time so the
  average is still defined.

**Axis formatting in the TUI graph**

- Tick positions and labels are set explicitly on the graph's rulers rather
  than by patching plotext's label generator, so the dependency stays
  replaceable and the two axes can be formatted independently.
- Y-axis ticks are chosen over the plotted value range and labelled with
  thousands separators and minimal decimals, never scientific notation.
- X-axis ticks are integer positions. While the trial numbers fit within the
  display cap, every integer is ticked; beyond the cap an integer step is used
  so the number of ticks stays bounded. The first and last trial numbers are
  always included.

**Failure output preserves the root cause**

- The failure excerpt assembled in the inference orchestration layer keeps the
  engine's root-cause line in addition to the bounded traceback tail. A helper
  selects the last line of the engine's captured output that names an exception
  (the `...Error: ...` / `...Exception: ...` line, which is where vLLM prints
  the cause before its traceback), and the excerpt renders that line first,
  followed by the traceback tail, so the most specific line survives at the
  front of the message.
- The excerpt stays bounded by the existing character limit, so a chatty engine
  cannot flood the log. The limit now budgets the root cause and the tail
  together instead of discarding everything that precedes the tail.
- Because the root cause leads the excerpt, the log reporter's existing
  single-line collapse keeps it near the front of the trial line rather than
  burying it after the traceback frames.
- The selection is isolated behind one helper in the orchestration module, so
  the heuristic can be tuned in one place without touching the reporters.

**Reporter surface**

- `TuiReporter` gains a `trial_started` method alongside its existing `start`
  and per-trial callback. `LogReporter` is unchanged except for documentation.
- The status line format is shared between the running frame and the completion
  frame so the display does not change shape between them.

## Testing Decisions

Good tests exercise external behaviour and observable contracts: given a
reporter's inputs, what string does it render; given an argument vector and
injected collaborators, what does the CLI write and return. Tests must not
assert on private helpers, and unit tests must not require a daemon, a GPU, or
the network.

- **The TUI reporter** is tested as a string producer with an injected clock,
  following the existing pattern in its unit tests. Coverage includes: the
  status line with an average once a trial succeeds; the average being omitted
  before the first success; failed trials not contributing to the average; the
  running frame's status line; integer-only x tick labels; and
  thousands-separated, non-exponential y tick labels. Assertions target the
  status line and the label strings rather than plotext's exact drawing.
- **The download progress helper** is tested with an injected fake download and
  captured streams, so the model name appears in the rendered line and the
  warning handler writes the message on its own line without truncation. The
  test must not perform a real download.
- **The CLI preparation order** is tested end to end with the shared fake
  backend and fake Podman client and captured streams, asserting that the last
  preparation message precedes the first TUI frame.
- **The trial-start hook** is tested in the optimization layer by passing a
  recording callback and asserting it is invoked once per trial, including for
  trials that later fail.
- **The failure excerpt** is tested as a string producer with synthetic engine
  output: an excerpt whose root cause precedes a long traceback keeps the root
  cause, a short message is unchanged, an empty message stays empty, and the
  result stays within the character budget. No container is involved.
- **Prior art**: the reporter unit tests establish the string-producer pattern
  with an injected clock; the CLI entry-point tests establish driving the
  composition root with the shared fakes and captured streams; the optimization
  adapter tests establish asserting callbacks and per-trial behaviour without a
  daemon.
- **Integration**: no new integration test is required. The existing
  end-to-end paths already exercise the reporter wiring and the download helper
  (through the Hub cache) and will catch a regression in either.

## Implementation Tasks

[x] **T1 - Take over the download progress line**: Add a helper that calls
`snapshot_download` with a custom `tqdm` subclass rendering one line naming the
model and showing file count and percent, disabling the named Hugging Face
transfer/reconstruction bars, and writing to the CLI's progress stream.
  - Note: the helper lives in a new public module
    `cli/download.py` as `download_model(model, stream, *, download=snapshot_download)`,
    rather than inside `cli/__init__.py`, so the Hugging Face progress internals
    stay isolated and tests can import it without reaching into a private
    helper. The `cli/` package docstring and the `AGENTS.md` `cli/` file list do
    not yet mention `download.py`; left to T13.

[x] **T2 - Print the Hub warning on its own line**: Install a temporary
`huggingface_hub` logging handler around the download that terminates the
progress line before writing the record and restores the previous handlers and
propagation afterwards.

[x] **T3 - Use the download helper in the CLI prefetch**: Replace the bare
`snapshot_download` call in the model prefetch with the new helper, keeping the
local-path bypass and the existing failure contract unchanged.

[x] **T4 - Keep preparation output before the first TUI frame**: The CLI
already prints the final preparation message before the reporter's `start`,
which now prints a preparation line instead of an empty frame; assert this
ordering and that no frame is drawn until a trial starts.
  - Note: the ordering needed no code change, as the plan anticipated. The
    assertion is the end-to-end `test_tui_mode_prints_preparation_before_the_first_frame`
    in `tests/cli/test_main.py`, which drives a TTY-like captured stream through
    the real composition root with the shared fakes. The reporter-level "no
    frame at start" half was already pinned by
    `test_start_announces_preparation_without_drawing_a_frame` in
    `tests/cli/test_tui.py`. T12's preparation-order assertion is therefore
    already satisfied; only its trial-start-hook half remains.

[x] **T5 - Add the optional trial-start hook to the optimizer**: Add an
argument-less `on_trial_start` callback to `make_objective` and `optimize`,
invoked once at the start of each trial and defaulting to `None`.

[x] **T6 - Add the TUI running-trial frame**: Add `trial_started` to the TUI
reporter, recording the trial start time and drawing a frame whose status line
marks the in-flight trial, and wire the CLI's TUI reporter to the optimizer's
trial-start hook.

[x] **T7 - Show the average time per successful trial**: Accumulate per-trial
durations for successful trials in the TUI reporter and render `avg trial
<seconds>s` in the status line, omitted until the first success.
  - Note: the average is included in the frame that reports the trial, so the
    status line stays consistent with its count and best columns. Because the
    behaviour change broke six existing exact-status assertions, their expected
    strings were updated here, and the average-specific coverage T11 lists —
    omitted before the first success, failed trials excluded, mean over
    successes — was added to `tests/cli/test_tui.py` as part of this task. T11
    therefore only needs the running-frame and axis-label coverage.
  - Revision: user feedback on the first implementation — a bare `avg` did not
    say what was averaged, so the column is now `avg trial <seconds>s` (for
    example `avg trial 12.3s`). This supersedes the `avg <seconds>s` wording in
    the implementation decision above; the reporter, the README, and the
    exact-status tests all use the new label.

[x] **T8 - Format the y-axis as plain numbers**: Compute explicit y tick
positions and label them with thousands separators and minimal decimals, never
scientific notation, via the graph's ruler.
  - Note: positions are round multiples of a *nice* step (1/2/5 × a power of
    ten) spanning the plotted range, so labels stay clean decimals rather than
    arbitrary linspace values. A single-valued range is padded so it still
    yields a sensible axis.

[x] **T9 - Use integer-only x ticks**: Compute explicit integer x tick positions
that show every trial number while they fit and an integer step beyond the cap,
and label them as integers via the graph's ruler.
  - Note: the cap is `_X_TICK_CAP = 10`; past it the step is
    `ceil((count - 1) / 9)`. Ticks are set with `figure.ruler(0, 0).ticks(...)`
    beside the existing y ruler.
  - Revision: user feedback on the first implementation — the last trial is no
    longer force-appended when the step misses it. Appending produced a lone
    tick one trial from its neighbour (e.g. `19 20` for a 20-trial sweep) with a
    visibly different gap, so the step now simply runs from the first trial and
    the last is ticked only when the step lands on it (`1 4 7 10 13 16 19`).
    This supersedes the plan's "first and last always ticked" requirement in
    User Story 24 and the corresponding Implementation Decision.

[x] **T10 - Unit-test the download progress helper**: Cover the model name in
the rendered line, the single-line rendering, and the untruncated warning, with
an injected fake download and captured streams.
  - Note: tests live in `tests/cli/test_download.py`. Five cases: the model name
    replaces the Hub's generic description; only the unnamed file bar renders on
    a single `\r`-updated line while the named transfer bar is disabled; a
    warning emitted mid-download appears intact on its own line; the resolved
    snapshot is returned as a `Path`; and the Hub logger's handlers and
    propagation flag are restored when the download raises. The fakes drive the
    injected `tqdm_class`, so no network is touched.

[x] **T11 - Unit-test the TUI reporter changes**: Cover the status line with and
without an average, failed trials excluded from the average, the running frame,
integer x labels, and non-exponential y labels, using an injected clock.
  - Note: the average-specific coverage (omitted before the first success,
    failed trials excluded, mean over successes) was already added in T7, so
    this task added the running-frame tests
    (`test_the_running_frame_marks_the_trial_in_flight`,
    `test_the_running_frame_keeps_the_summary_columns`) and the axis-label
    tests (`test_a_short_sweep_ticks_every_integer_trial`,
    `test_a_long_sweep_thins_the_x_axis_to_integer_ticks`,
    `test_the_y_axis_labels_are_thousands_separated_and_never_scientific`) to
    `tests/cli/test_tui.py`. The label tests read the tick strings back out of
    the rendered frame rather than calling the private `_x_ticks`/`_y_ticks`
    helpers, per the "no private helpers" testing decision.

[x] **T12 - Unit-test the preparation order and the trial-start hook**: Assert
the last preparation message precedes the first TUI frame, and that the
optimizer invokes `on_trial_start` once per trial including failed trials.
  - Note: the preparation-order half was already satisfied by T4's
    `test_tui_mode_prints_preparation_before_the_first_frame`
    (`tests/cli/test_main.py`) and
    `test_start_announces_preparation_without_drawing_a_frame`
    (`tests/cli/test_tui.py`); this task added the optimizer half as
    `test_the_trial_start_hook_fires_once_per_trial` and
    `test_the_trial_start_hook_fires_for_failed_trials_too` in
    `tests/optimization/test_lifecycle.py`, driving the real orchestration with
    the shared fakes so no daemon or engine is involved.

[x] **T13 - Document the reporter behaviour**: Update the output-mode
documentation and the reporter module documentation to describe the download
line, the running marker, the average, the axis formatting, the log-mode
distinction, and the `--n-jobs` threading caveat.
  - Note: the README's *Output* section gained the download line, the running
    and completion status lines, the average, the axis formatting, the log-mode
    format, and the `--n-jobs` thread-safety caveat. The `TuiReporter` module
    and class docstrings now describe the running frame, the average, and the
    axis labels, and warn that the reporter is not thread-safe under
    `--n-jobs`; `LogReporter`'s docstring records that it deliberately has no
    running marker. As T1's note deferred, `cli/download.py` is now listed in
    the `cli/` package docstring and the `AGENTS.md` `cli/` file list.

[x] **T14 - Preserve the root cause in the failure excerpt**: Add the
root-cause-selecting helper to the inference orchestration module and use it in
the failure excerpt so the engine's exception line survives alongside the
bounded traceback tail.
  - Note: the helper is `_root_cause` in `inference/_orchestrate.py`, selecting
the last line matching `\b\w*(?:Error|Exception):`. `_failure_output` renders
that line first (separated by an ellipsis) only when it falls outside the
trailing window; when the output is short or the root cause is already in the
tail it is returned as before. The whole excerpt now stays at or below
`_FAILURE_OUTPUT_LIMIT` characters, where the old tail-only path returned one
character more.

[x] **T15 - Unit-test the failure excerpt**: Cover a root cause preceding a
long traceback, a short message, an empty message, and the character bound, with
synthetic engine output.
  - Note: the four cases live in `tests/inference/test_orchestrate.py` as
    `test_failure_excerpt_keeps_a_root_cause_before_a_long_traceback`,
    `test_failure_excerpt_leaves_a_short_message_unchanged`,
    `test_failure_excerpt_of_empty_output_stays_empty`, and
    `test_failure_excerpt_stays_bounded_while_keeping_the_root_cause`. They drive
    `run_experiment` with the shared fake Podman client and assert on the
    resulting `ExperimentError` string, per the "no private helpers" testing
    decision, rather than calling `_failure_output` directly; a test-local
    `_engine_output` helper builds the synthetic cause-plus-traceback payload.

[x] **T16 - Run the full quality gate**: Run the formatter, the linter, and the
unit test suite, and fix any fallout from the reporter and optimizer changes.
  - Note: `uv run ruff format .` (67 files left unchanged) and `uv run ruff check .`
    (all checks passed) are clean, and `uv run pytest` is green at 382 passed, 7
    deselected. No fallout needed fixing. The 7 deselected tests are the
    Podman-requiring integration tests, excluded by the project's default
    `addopts`. They were run separately at the user's request —
    `uv run pytest -m integration` — and all 7 passed in 20m49s, including
    `tests/cli/test_cli_integration.py`, which exercises the new download helper
    against the real Hugging Face Hub.

## Out of Scope

- A live-animated spinner or elapsed timer during a trial. The running indicator
  is a static status line; no background thread is introduced.
- A running-trial indicator in log mode. Log mode keeps one line per finished
  trial.
- Polling the container runtime or streaming container logs for progress.
- Changing the graph's series, markers, legend, colours, or the number of frames
  drawn per trial beyond the added running frame.
- Configurable tick density, separators, or number formats (for example a flag
  or YAML option).
- Any change to the search-space, inference-backend, or runtime layers other
  than the optimizer's trial-start hook and the failure-excerpt helper in the
  inference orchestration module. The engine's own logging, the runtime's log
  capture, and how much output is captured are unchanged; only how the captured
  text is excerpted for the user changes.
- Reworking how warnings and errors are routed between stdout and stderr beyond
  the download warning handling.

## Further Notes

- **Why file count rather than bytes.** Hugging Face's byte-transfer bar has no
  total until each file begins downloading, so a percentage against it is
  misleading. The file bar has a stable total from the first frame, giving an
  honest percentage. A model with few large shards therefore advances coarsely,
  which is an accepted trade-off for a correct percentage.
- **Why the warning handler is temporary.** The handler is scoped to the
  download so the rest of the CLI's logging behaviour is untouched. Restoring
  the original handlers and propagation flag keeps the process from accumulating
  handlers across repeated calls.
- **Why the trial-start hook lives in the optimizer.** The reporters only see
  completed trials through the existing callback. The start of a trial is only
  observable from the objective, so the hook is threaded through the optimizer
  rather than inferred by the reporter.
- **Why the running indicator is TUI-only.** Log mode is designed as a
  line-oriented record for redirection; adding a second line per trial would
  halve its signal-to-noise ratio for no benefit to a reader of the final file.
- **Why tick labels are computed in the reporter.** Setting explicit ticks keeps
  the formatting under the project's control and avoids patching plotext's
  private label generator, so a future plotext upgrade is less likely to break
  the display.
- **Interaction with the existing status columns.** The count, best value, and
  elapsed time remain in the same order; the average is inserted before the
  elapsed time and the running marker replaces nothing, so the status line stays
  a single readable summary.
- **Why the root cause is selected by pattern rather than position.** The engine
  prints its cause before the traceback, and the traceback length varies with
  the engine and the failure, so keeping a fixed tail cannot reliably retain the
  cause. Selecting the exception line explicitly is what makes the excerpt
  useful across traceback sizes.
