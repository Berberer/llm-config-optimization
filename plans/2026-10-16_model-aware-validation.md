# Model-Aware Validation

## Problem Statement

The tool validates a sweep in two passes before any trial runs: the search space
is checked engine-agnostically when it is loaded, and the backend is asked
whether it can forward every declared parameter and value on the resolved
accelerator class and device count. Neither pass knows anything about the
*checkpoint* the run will actually load.

That gap became visible when a run was configured with `--quantization awq`
against `Qwen/Qwen3.5-0.8B`, an unquantized checkpoint. Nothing rejected the
combination up front; instead vLLM failed at engine initialisation, after
loading the weights, roughly 70–200 seconds into the first trial — and again on
every subsequent trial that carried the same value. Because quantization is now
a checkpoint property set from the command line rather than a swept dimension,
there is no space-validation pass that could catch it, so the mismatch is only
discovered once expensive hardware time has already been spent.

The user wants the tool to know what the checkpoint is before it runs anything:
if the requested quantization cannot be loaded from the resolved checkpoint, the
run should stop with a clear explanation before the image is pulled and before
the first trial starts.

## Solution

Add a third, model-aware validation pass to the preparation sequence. Once the
model has been resolved to a local directory — either a host path or the
snapshot the Hub download produced — the CLI asks the backend whether the
configured experiment is compatible with that checkpoint. For this first check
the backend reads the checkpoint's declared quantization method and compares it
with the configured quantization, reporting every incompatibility as a
human-readable problem.

A mismatch fails fast with the unsupported-configuration exit code and a message
naming the checkpoint's method and the requested one, so the fix is obvious.
The common case is untouched: when `--quantization` is omitted the engine
auto-detects, and the check accepts the configuration without inspecting
anything. A checkpoint whose configuration cannot be read is likewise accepted,
so custom or unusual checkpoints are never blocked by a check that cannot see
their metadata.

The pass is deliberately a general backend seam rather than a quantization
special case: it receives the full experiment configuration and the resolved
model path, so later model-aware checks (context length against the model's
maximum, dtype support, architecture-specific backends) can be added inside the
same hook without another CLI step or protocol method.

## User Stories

1. As an LLM-optimization developer, I want the tool to check my configured
   quantization against the resolved checkpoint before the sweep, so that I do
   not waste GPU time on a configuration the engine cannot load.
2. As an LLM-optimization developer, I want to be told which quantization method
   the checkpoint actually uses, so that I can set `--quantization` correctly.
3. As an LLM-optimization developer, I want a mismatch to be detected before the
   container image is pulled, so that a doomed run does not wait on a
   multi-gigabyte download.
4. As an LLM-optimization developer, I want to omit `--quantization` and have
   the engine auto-detect, so that the common case needs no change and no extra
   work.
5. As an LLM-optimization developer, I want an unquantized checkpoint to accept
   online quantization (fp8), so that a legitimate dynamic-quantization run is
   not blocked by the check.
6. As an LLM-optimization developer, I want a quantized checkpoint to accept
   only the method it was produced with, so that I cannot accidentally request a
   different one.
7. As an LLM-optimization developer, I want a checkpoint quantized with a method
   the engine does not model to accept only auto-detection, so that I am steered
   away from an explicit value that cannot be right.
8. As an LLM-optimization developer, I want a clear message naming both the
   checkpoint's method and the requested value, so that the correction is
   obvious without reading the engine's logs.
9. As an LLM-optimization developer, I want every compatibility problem reported
   at once, so that I can fix the configuration in a single pass.
10. As an LLM-optimization developer, I want an unsupported configuration to
    fail with the same exit code as an unsupported space, so that scripts need
    one branch for "the backend cannot run this".
11. As an LLM-optimization developer, I want a checkpoint with no readable
    configuration to not block the run, so that custom or unusual checkpoints
    still work and the engine remains the final authority.
12. As an LLM-optimization developer, I want the check to read only the
    already-resolved checkpoint, so that it adds no network round-trip and no
    extra download.
13. As an LLM-optimization developer, I want the check to use the same resolved
    model path the run will mount, so that the validation and the run can never
    disagree about which checkpoint was inspected.
14. As an LLM-optimization developer, I want the check to be announced as a
    preparation step, so that a long setup is still legible in the output.
15. As an LLM-optimization developer, I want a successful check to print nothing
    to stderr, so that a clean run's error stream stays empty.
16. As a test author, I want the compatibility rule testable with hand-written
    checkpoint metadata and no engine or GPU, so that the whole matrix is
    covered in the unit suite.
17. As a test author, I want the CLI preparation step testable with the shared
    fake backend and a temporary checkpoint, so that the wiring is covered
    without a daemon.
18. As a future maintainer, I want the check behind the backend seam, so that a
    second engine can implement its own compatibility rule.
19. As a future maintainer, I want the hook to receive the whole experiment and
    the resolved model path, so that additional model-aware checks are additive
    rather than a restructure.
20. As a future maintainer, I want the checkpoint reading isolated behind one
    helper, so that a different checkpoint layout can be supported later.

## Implementation Decisions

**A new backend seam for model-aware validation**

- The `InferenceBackend` protocol gains a `validate_model` method taking the
  experiment configuration and the resolved local model path, returning a
  sequence of human-readable problem strings. It mirrors the existing
  `validate(space, hardware)` convention: problems are returned rather than
  raised, so the caller aggregates them, and an empty sequence means the
  checkpoint supports the configuration.
- The vLLM backend implements the method; the shared test fake returns no
  problems so the protocol-conformance test keeps passing.
- The hook is the engine seam for every model-dependent check. It receives the
  whole `ExperimentConfig` and the model path rather than only the quantization
  field, so context-length, dtype, and architecture checks can be added later
  without changing the CLI or the protocol.

**How the checkpoint is read**

- The vLLM backend reads the checkpoint's standard `config.json` from the
  resolved local directory and extracts `quantization_config.quant_method`,
  lower-cased. A missing file, malformed JSON, an absent `quantization_config`,
  or a non-string `quant_method` all mean "the checkpoint declares no method".
- Reading is isolated in one helper so a non-standard checkpoint layout can be
  supported later without touching the compatibility rule.

**The compatibility rule**

- An unset `LLMConfig.quantization` — the CLI's default — is always accepted,
  because the engine auto-detects the method from the checkpoint.
- A checkpoint that declares no method accepts only the engine-neutral members
  the engine can apply online (currently fp8).
- A checkpoint that declares a method accepts only the engine-neutral member
  whose set of checkpoint spellings contains it. The mapping is:
  `awq` accepts `awq`, `auto_awq`, and `awq_marlin`; `gptq` accepts `gptq`,
  `auto_gptq`, and `gptq_marlin`; `fp8` accepts `fp8`.
- A declared method no engine-neutral member models (for example
  `compressed-tensors`) accepts no explicit quantization value, so the user must
  use auto-detection.
- The mapping is vLLM-version-specific and lives beside the backend's existing
  value maps, where it is expected to drift with the engine.

**Fail-open on unreadable checkpoints**

- When the checkpoint directory or its configuration cannot be read, the hook
  reports no problems. A false rejection would block legitimate custom
  checkpoints, and the engine still enforces its own rule when the model loads.

**CLI preparation order and the resolved model path**

- The base experiment configuration is built before the model is prefetched,
  because the check needs the configured quantization.
- The CLI's combined prefetch is split so the model step returns the resolved
  local directory (the host path itself, or the snapshot directory the Hub
  download returns) and the image pull remains a separate step.
- The model-aware check runs after the model is resolved and before the image is
  pulled, so a doomed run does not fetch a multi-gigabyte image.
- The returned path is the same directory the backend later mounts, so the check
  and the run inspect the same checkpoint.

**Error handling and exit code**

- The CLI aggregates the hook's problems and renders them with the same
  bulleted format the definition errors use, under an "unsupported
  configuration" headline, then returns exit code `4`.
- Exit code `4`'s documented meaning is broadened from "a space the backend does
  not support" to "a configuration the backend does not support — the search
  space or the resolved model".
- The preparation step prints a "validating model compatibility" line to stdout
  and nothing to stderr on success.

## Testing Decisions

Good tests exercise external behaviour and observable contracts: given a
checkpoint's declared method and a configured quantization, what problems does
the backend return; given an argument vector, a temporary checkpoint, and an
injected backend, what exit code and output does the CLI produce. Tests must not
assert on private helpers, and unit tests must not require a daemon or a GPU.

- **The backend compatibility rule** is tested directly against
  `VLLMBackend.validate_model` with hand-written checkpoint directories. The
  matrix is parameterized over the checkpoint method (none, `awq`, `gptq`,
  `fp8`, an unmodeled method such as `compressed-tensors`, and a malformed or
  absent configuration) crossed with the configured quantization (unset, `awq`,
  `gptq`, `fp8`), asserting only whether a problem is reported and that the
  message names the checkpoint method where relevant.
- **The CLI preparation step** is tested end to end with the shared fake
  backend, the shared fake Podman client, and captured streams, following the
  existing CLI test pattern. A temporary checkpoint directory carries a
  `config.json`; the tests cover an incompatible configuration returning exit
  `4` before the image pull, a compatible configuration proceeding, and an
  omitted `--quantization` short-circuiting the check.
- **The fake backend** gains the new hook so the existing protocol-conformance
  test continues to hold.
- **Prior art**: the engine-support validation tests establish the
  problem-string convention and the aggregation into a definition error; the
  CLI entry-point tests establish driving the composition root with the shared
  fake backend and fake Podman client; the shared fake backend establishes the
  protocol-conformance double.
- **Integration**: no new integration test is required. The existing end-to-end
  paths use an unquantized checkpoint with no `--quantization`, so they already
  exercise the no-op branch of the check.

## Implementation Tasks

[x] **T1 - Add the model-aware validation hook to the backend seam**: Add
`validate_model(config, model)` to the `InferenceBackend` protocol, implement
the checkpoint-quantization check in the vLLM backend (including the checkpoint
reader and the compatibility mapping), and implement the no-op in the shared
fake backend so the protocol-conformance test keeps passing.
  - **Note:** The *How the checkpoint is read* decision grouped a missing
    file and malformed JSON with an absent `quantization_config` as "declares no
    method", which conflicted with *Fail-open on unreadable checkpoints* and
    user story 11. Resolved with the user: a missing `config.json`, malformed
    JSON, or a non-object payload is treated as *unreadable* and reports no
    problems, while a readable config with no `quantization_config` or a
    non-string `quant_method` counts as "declares no method" and accepts only
    online quantization (fp8).

[x] **T2 - Return the resolved model path from the CLI prefetch**: Split the
CLI's combined prefetch so the model step returns the resolved local directory
(the host path or the Hub snapshot directory) and the image pull remains a
separate step, without changing the announced preparation order.
  - **Note:** The local branch returns `Path(model).expanduser().resolve()`
    rather than the literal host path, so the directory handed to the
    model-aware check is exactly the one the backend resolves and mounts. The
    split replaces the combined `_prefetch` with `_prefetch_model` (returning
    `Path`) and `_pull_image` (returning the image name); `_sweep` still calls
    them in the same announced order, and the returned path is consumed by T3.

[x] **T3 - Run the model-aware check in the CLI preparation sequence**: Build
the base configuration before prefetching, call the backend's model validation
after the model is resolved and before the image pull, and map problems to exit
code `4` with the bulleted message.

[x] **T4 - Unit-test the checkpoint-quantization compatibility rule**: Cover
every checkpoint method crossed with every configured value, the unset value,
the unreadable-config fail-open path, and the unmodeled-method path.
  - **Note:** Tests live in a new `tests/inference/test_vllm_validate_model.py`,
    following the one-file-per-backend-method pattern (`test_vllm_plan.py`,
    `test_vllm_result.py`). A table-driven matrix crosses ten checkpoint shapes
    (missing/malformed/non-object config, no quantization config, non-string
    method, unquantized, `awq`, `gptq`, `fp8`, and the unmodeled
    `compressed-tensors`) with the four configured values, plus focused tests
    for the mismatch message, the multi-spelling aliases, case-insensitivity,
    and the unset short-circuit.

[x] **T5 - Unit-test the CLI model-aware preparation step**: Drive the CLI with
the shared fake backend and a temporary checkpoint, covering rejection,
acceptance, and the auto-detect short-circuit.
  - **Note:** Tests live in `tests/cli/test_main.py`, alongside the other
    entry-point tests. A `CheckpointValidatingBackend` (a `FakeBackend` that
    delegates `validate_model` to `VLLMBackend`) is used for the rejection,
    acceptance, and auto-detect cases, because a compatible run must complete a
    trial and the real backend would need a valid benchmark payload; the shared
    fake's canned result avoids that. A `ModelProblemBackend` covers the CLI's
    aggregation of multiple problems into one failure.

[x] **T6 - Document the new check and the broadened exit-code meaning**: Update
the preparation walkthrough, the exit-code description, and the `--quantization`
guidance, and note the new invariant in the contributor documentation.
  - **Note:** The preparation walkthrough is the README *Validation* section
    (now three stages) plus the `cli/__init__.py` module docstring and the
    AGENTS.md *Data flow*; the exit-code meaning was broadened in the README
    table and the CLI docstring; the `--quantization` guidance was updated in
    the README options table, the tunable-parameter prose, and the CLI help
    string; and the new invariant was recorded in AGENTS.md *Invariants* and
    the *New inference engine* recipe.

## Out of Scope

- Model-aware checks other than quantization. The hook is shaped to accommodate
  them, but validating `max_model_len` against the model's maximum context,
  dtype support, or architecture-specific attention requirements is deferred.
- Selecting or inferring the quantization from the checkpoint. The tool only
  validates a user-supplied value; it never changes it.
- Reading checkpoint metadata from anywhere other than the already-resolved
  local snapshot, and supporting non-standard checkpoint layouts.
- Re-validating at trial time. The check runs once during preparation, against
  the base configuration.
- Making quantization a swept dimension again, or otherwise changing the
  search-space layer.
- A dedicated exit code for a model mismatch; the existing unsupported-
  configuration code is reused.

## Further Notes

- **Why the check is fail-open.** A check that cannot read the checkpoint's
  metadata has no evidence of a mismatch, and rejecting on absence would block
  legitimate custom checkpoints. The engine remains the final authority at load
  time, so fail-open costs at most the failure the check would have caught.
- **Why the mapping lives in the backend.** Checkpoint method names and the
  engine's online-capable set are vLLM-version-specific, exactly like the
  existing parameter-target and value maps; keeping them in the backend is what
  lets a second engine define its own rule.
- **Why the check runs before the image pull.** The image can be several
  gigabytes and its pull is one of the most expensive preparation steps, so a
  configuration the checkpoint cannot satisfy should be reported first.
- **Why the whole experiment is passed.** Handing the hook the full
  configuration and the resolved path is what makes the follow-on checks
  additive. Passing only the quantization field would force a protocol change
  for every future model-aware check.
- **Interaction with the existing validation passes.** The engine-support pass
  still runs first, before the model is downloaded, so a malformed space fails
  without a download. The model-aware pass is strictly additive and runs only
  once a checkpoint exists to inspect.
