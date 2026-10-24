# Speculative Decoding Support

## Problem Statement

The runner searches serving configurations for the best token throughput, but it
cannot try speculative decoding. A user who suspects that a draft model, an
EAGLE-3 speculator, or a target model's native multi-token-prediction heads
would speed up their deployment has to leave the tool: run each engine by hand
with its own flags, benchmark both regimes outside the runner, and compare
numbers that the runner knows nothing about. The sweep cannot decide whether
speculation pays off, cannot tune the draft depth, and cannot discover that
under constrained VRAM the draft model's memory cost makes the non-speculative
configuration win.

The draft sources are also not uniform. Some targets carry their speculator
inside the checkpoint (MTP heads), some need a separate Hugging Face repository
(EAGLE-3 and generic draft models), and some need a single local file
(llama.cpp GGUF drafts). Any solution has to cover all of them with one simple
CLI, without leaking each engine's flag vocabulary to the user.

Finally, the default workload is synthetic random token ids, where a draft
model's proposals are almost always rejected. A user enabling speculation
without a real dataset would measure the overhead and none of the benefit, with
nothing in the output to tell them why.

## Solution

A run can now configure speculative decoding on the command line, in
engine-neutral terms, for both vLLM and llama.cpp:

- `--speculative-method` names the method: `draft_model`, `eagle3`, `mtp`,
  `dflash`, `dspark`, or `ngram`. `mtp` uses the target model's own heads, so
  no draft checkpoint is fetched; the others take a draft.
- `--draft-model` points at the speculator with the same source forms the main
  model accepts: a Hugging Face id or a host path, and for llama.cpp a
  repository with a `:QUANT` selector or a local `.gguf` file. `--draft-model`
  alone implies `draft_model`.
- The method and draft model are fixed for the run; the sweep decides two new
  search-space dimensions — `speculative_mode` (`enabled`/`disabled`) and
  `num_speculative_tokens` (the draft depth). Both are required exactly when a
  speculative setup is configured, so users who do not speculate keep their
  existing spaces unchanged.
- Disabled trials start the engine with no speculative flags at all, so the
  draft model's memory is genuinely freed and the on/off comparison is fair.
  The sweep can therefore discover, on constrained hardware, whether
  speculation or more model offload wins.
- The draft model is resolved and downloaded on the host through the same
  backend model rules and progress line as the main model, mounted into the
  container, and validated where cheap (a non-GGUF llama.cpp draft is rejected
  before the image is pulled).
- If speculation is configured without `--dataset-path`, the runner warns on
  stderr that the synthetic random workload will not show a meaningful benefit.

## User Stories

1. As a tuning user, I want to enable speculative decoding with a separate
   draft model from the command line, so that I can measure whether it improves
   throughput for my target.
2. As a tuning user, I want to use a speculator embedded in the target model
   (MTP heads) without downloading a separate checkpoint, so that I can benefit
   from native multi-token prediction.
3. As a tuning user, I want to select EAGLE-3, DFlash, DSpark, and generic
   draft methods by engine-neutral names, so that I do not have to learn each
   engine's spelling.
4. As a tuning user, I want to select N-gram speculation, which needs no draft
   model, so that I can try the cheapest speculative method.
5. As a tuning user, I want to point at a draft with the same source forms as
   the main model, so that I can use a Hugging Face id or a local checkpoint
   without special syntax.
6. As a llama.cpp user, I want to select one draft quantization from a
   repository holding several GGUF files, so that the run fetches exactly the
   file I mean.
7. As a tuning user, I want the draft model downloaded with the same single
   progress line as the main model, so that long downloads stay legible.
8. As a llama.cpp user, I want a non-GGUF draft rejected before any container
   is pulled, so that an impossible configuration does not wait on a large
   image download.
9. As a tuning user on constrained hardware, I want the sweep to decide whether
   speculation is on or off, so that the answer reflects my actual memory
   budget instead of my guess.
10. As a tuning user, I want a disabled trial to start the engine without the
    draft model at all, so that the freed memory makes the comparison honest.
11. As a tuning user, I want the sweep to decide the draft depth, so that the
    optimizer finds the point between accepted tokens and drafting overhead.
12. As a tuning user, I want to pin the mode and the depth to single values, so
    that I can run a reproducible speculative configuration without exploring.
13. As a tuning user, I want `default: true` on `num_speculative_tokens` to
    defer to the engine's own default depth and on `speculative_mode` to mean
    the configured setup's default (enabled), so that I can opt out of deciding
    without deferring to an engine default that contradicts my setup.
14. As a tuning user who does not use speculation, I want my existing search
    spaces to keep working unchanged, so that this feature costs me nothing.
15. As a tuning user, I want a clear error when I declare a speculative
    dimension without configuring a setup, so that my space never silently
    sweeps a value that is never used.
16. As a tuning user, I want a usage error when a method that requires a draft
    is configured without one, so that I find out before the sweep starts.
17. As a tuning user, I want a usage error when I pass a draft to `ngram`,
    which is model-free, so that no argument is silently ignored.
18. As a tuning user, I want `--draft-model` alone to imply the generic draft
    method, so that the common case is a single flag.
19. As a tuning user, I want the run header to name the configured speculative
    setup, so that I can confirm what is being tested.
20. As a tuning user, I want a warning when I speculate against the synthetic
    random workload, so that I do not draw conclusions from numbers that cannot
    show the benefit.
21. As a tuning user, I want the winning configuration to include the method,
    draft source, mode, and depth, so that `--output-config` writes a
    re-runnable speculative configuration.
22. As a tuning user, I want spec-on and spec-off trials in one study and one
    Pareto front, so that the trade-off is reported together instead of across
    two unordered runs.
23. As a tuning user, I want the depth values sampled on disabled trials
    documented as unused, so that I understand the search accounting.
24. As a tuning user, I want the method and draft model fixed per run, so that a
    mismatch cannot burn the trial budget on server-startup failures.
25. As a vLLM user, I want the draft snapshot available inside the container
    from the local cache without duplicated mounts, so that the server loads it
    offline.
26. As a llama.cpp user, I want the draft mounted as a file into the container,
    so that the server needs no network at startup.
27. As a library user, I want the backend protocol to answer applicability and
    validation for a given run configuration, so that I can validate a space
    exactly as the CLI does.
28. As a library user, I want to keep validating engine support with a space and
    hardware only, plus the run's fixed LLM configuration, so that
    configuration-dependent rules have a natural home.
29. As a user, I want engine-neutral names in the search space and the written
    configuration, so that documents never carry one engine's vocabulary.
30. As a user, I want a trial whose speculative configuration cannot start to
    fail individually, so that one bad point does not end the sweep.
31. As a maintainer, I want the phase-rate metric definitions untouched, so that
    studies recorded before this change remain comparable.
32. As a maintainer, I want the tunable checklist and architecture notes
    updated, so that the next parameter follows the same path.
33. As a user, I want the existing exit codes and flag conventions preserved, so
    that scripts and habits keep working.

## Implementation Decisions

### Engine-neutral vocabulary and configuration

- A new `SpeculativeMethod` `StrEnum` lives in the engine-neutral inference
  types with exactly six members, each spellable on both engines:
  `draft_model`, `eagle3`, `mtp`, `dflash`, `dspark`, and `ngram`. Engine-only
  methods, vLLM's per-model MTP aliases, `custom_class`, and
  `extract_hidden_states` are deliberately excluded.
- A new `SpeculativeMode` `StrEnum` has `enabled` and `disabled`. The values are
  spelled that way because PyYAML parses bare `on`/`off` as booleans, which the
  loader's string coercion would reject with a confusing error.
- `LLMConfig` gains five flat, optional fields: `speculative_method`
  (`SpeculativeMethod | None`), `draft_model` (`str | None`), `speculative_mode`
  (`SpeculativeMode | None`), `num_speculative_tokens` (`int | None`), and
  `resolved_draft_path` (`Path | None`). The flat shape matches the documented
  "flat, strictly typed" contract and lets the tunable registry work unchanged:
  a tunable's YAML name maps to a field and its kind is read from the field
  annotation.
- `resolved_draft_path` is a resolution artifact, like `BenchmarkConfig`'s
  `resolved_path`: it is where the composition root put the fetched draft, is
  never forwarded to an engine, and is set on the base configuration before
  backend validation and planning.
- Both enums are exported from the inference package surface alongside the
  other engine-neutral enums.

### Search space and gating

- `TUNABLE_PARAMETERS` gains `speculative_mode` (mapping to the field of the
  same name) and `num_speculative_tokens`, appended in canonical order with
  mode before depth. The loader's positive-value floor gains
  `num_speculative_tokens`.
- Both gated tunables are required exactly when the run configures a setup, and
  rejected when it does not. Non-speculative spaces are unchanged, and an
  engine-agnostic caller using the registry's default required set now also
  declares both.
- `InferenceBackend.applicable_parameters` becomes a method taking the run's
  fixed `LLMConfig` instead of a property. It returns the engine's standard set,
  plus `speculative_mode` and `num_speculative_tokens` iff
  `speculative_method` or `draft_model` is configured.
- `InferenceBackend.validate` takes the same `LLMConfig` (before the optional
  hardware argument). It checks a declared parameter against
  `applicable_parameters(llm)` rather than the raw forwarding table, so a
  declared gated tunable without a setup is reported as unsupported.
- `validate_space` forwards the configuration, and the composition root passes
  the base configuration's LLM domain to both the loader's required set and the
  support check.
- `SpeculativeMode` needs no engine translation, but both backends carry an
  identity value-map entry for it so the generic enum value validation accepts
  the members.

### Semantics of the mode and depth

- A trial with `speculative_mode` disabled emits no speculative flags: vLLM gets
  no `--speculative-config`, llama.cpp gets no `--spec-type`,
  `--spec-draft-model`, or `--spec-draft-n-max`. The draft's memory is genuinely
  freed.
- An unset mode means the setup's default, which is enabled; this is only
  reachable through a `default: true` spec, and the documentation states it
  plainly. This is a deliberate deviation from the `Default` spec kind's generic
  "engine default" wording, because the engine's default — no speculation —
  would contradict the setup the user explicitly configured.
- `num_speculative_tokens: {default: true}` resolves to an unset field, so the
  depth is omitted from the engine's speculative configuration and the engine's
  own default applies: llama.cpp drafts up to its built-in default, while vLLM
  infers a depth where the model metadata allows and fails startup for methods
  that require the value. This is documented with the dimension.
- `num_speculative_tokens` is sampled on every trial but only applied when the
  mode is enabled. This is the axis's defined semantics, not a silent fallback,
  and is documented where the dimension is documented.
- The draft model is resolved and downloaded once per run even when the space
  can sample disabled, because the space is not consulted for resolution. This
  is documented.

### Backend forwarding

- vLLM forwards the group through one `--speculative-config` JSON object with
  `method`, `model`, and `num_speculative_tokens` keys. The six method members
  map to vLLM's own spellings (`draft_model`, `eagle3`, `mtp`, `dflash`,
  `dspark`, `ngram`). Disabling the mode suppresses the whole group.
- llama.cpp maps the six methods to `--spec-type` (`draft-simple`,
  `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`),
  forwards the draft as `--spec-draft-model`, and the depth as
  `--spec-draft-n-max`. `--spec-type` is always emitted explicitly when
  speculation is enabled. Disabling the mode suppresses all three.
- vLLM plan handling rewrites a resolved draft into its in-container path and
  reuses or dedupes the Hugging Face hub-root mount so the draft snapshot's
  symlinks resolve. An unresolved draft string is passed through, matching how
  the main model is handled.
- llama.cpp plan handling mounts a resolved draft file at its fixed container
  draft path and points `--spec-draft-model` at it. An unresolved repository is
  forwarded through the engine's own draft repository argument, mirroring how
  the main model falls back to its repository arguments.
- llama.cpp's model-aware validation also rejects a resolved draft that is not
  GGUF, using the existing GGUF metadata reader, and fails open for a header it
  cannot read, consistent with the main checkpoint.

### CLI behavior

- The engine option group gains `--speculative-method` (choices drawn from the
  six-member enum) and `--draft-model`. There is no depth or mode flag: both are
  tunables and live in the search space.
- `--draft-model` without a method implies `draft_model`. Methods `draft_model`,
  `eagle3`, `dflash`, and `dspark` require a draft model; `mtp` may take an
  auxiliary draft (for assistant-style checkpoints) but needs none; `ngram`
  rejects one. A violation is a usage error, as are a draft-requiring method
  without a draft.
- The base-config builder populates the speculative fields from the flags, and
  the run header prints the configured setup.
- The composition root resolves the draft through the backend's
  model-resolution hook and the shared download helper, records
  `resolved_draft_path` on the base configuration, and maps a resolution
  failure to the unsupported-model exit path, exactly as for the main model.
- When a setup is configured and no dataset path is given, the runner prints an
  advisory warning to stderr before the sweep, e.g. "speculative decoding
  rarely helps on the synthetic random workload; pass `--dataset-path` with
  real prompts". It is a warning, not an error, because measuring the worst case
  can be deliberate.

### Reporting, provenance, and metrics

- The summary and `--output-config` writer emit the new flat fields
  automatically, including the recommended mode and depth; no special-casing is
  needed.
- No study provenance guard is added. A study records neither the speculative
  method nor the draft model, matching how seed and quantization are treated;
  the documentation advises a distinct `--study-name` per speculative setup.
- The phase-rate metric definitions do not change, so the metric schema version
  is not bumped.
- No typed draft-acceptance metric is added.

## Testing Decisions

- Tests assert external behavior: the argument list a backend builds, the
  container plan (command, mounts, environment, devices), the human-readable
  problems a validator returns, the exit code and output the CLI produces, and
  the parsed dimensions a loader yields. Internal helpers are exercised only
  through those surfaces.
- Unit tests must not require a daemon or the network: the Podman client, the
  benchmark client, the download callables, the repository listing, and the
  backends themselves are injected fakes. No new integration tests are added,
  because a compatible target/draft pair is heavy and hardware-dependent;
  manual smoke runs are noted below instead.
- Search-space tests update the registry list, the canonical order, the
  resolved kinds, the positive-value floor, and the complete-space helpers that
  now include the two gated dimensions. Prior art: the existing registry and
  loader tests that iterate `TUNABLE_PARAMETERS`.
- Backend argument tests assert the exact speculative flags and the single
  `--speculative-config` JSON payload, for every method member, and that the
  disabled mode suppresses everything while an unset mode emits it and an unset
  depth is omitted. Prior art: the existing vLLM and llama.cpp argument tests,
  including their JSON-group coverage.
- Backend plan tests assert the draft mount (file-for-llama.cpp,
  snapshot-path-for-vLLM, deduped hub root) and the container paths in the
  command. Prior art: the existing plan tests.
- Backend validation tests assert the aggregated problem strings for a declared
  gated tunable with and without a setup, for each engine. Prior art: the
  existing validate tests.
- Model-aware validation tests assert that a non-GGUF resolved draft is
  rejected and an unreadable header is accepted. Prior art: the existing
  llama.cpp metadata and validate-model tests.
- CLI tests cover the new flags, the coherence usage errors, the implied
  default method, the header line, the two-model prefetch, the warning, and the
  summary output. Prior art: the existing CLI arguments, config, main, and
  summary tests with injected fakes.
- The shared test fakes implement the changed protocol signatures so every
  test suite builds against the same backend contract.

## Implementation Tasks

[x] **T1 - Add the speculative vocabulary and config fields**: Add the
`SpeculativeMethod` and `SpeculativeMode` enums to the engine-neutral inference
types and the five flat optional `LLMConfig` fields
(`speculative_method`, `draft_model`, `speculative_mode`,
`num_speculative_tokens`, `resolved_draft_path`) with docstrings; export both
enums from the inference package; update existing configuration and summary
expectations that now see the new null fields.

  Note: until the mapping tasks land, the vLLM and llama.cpp argument-coverage
  tests classify the five new fields as not yet mapped (with `resolved_draft_path`
  permanently inapplicable), keeping the invariant tests green at this step.
[x] **T2 - Make backend applicability and validation config-aware**: Replace
the `applicable_parameters` property with a method taking the run's `LLMConfig`,
and make `validate` accept the same configuration before the optional hardware
argument; forward the configuration through `validate_space`, the composition
root, and the test fakes; keep behavior unchanged while the registry has not
grown.
[x] **T3 - Register and gate the two speculative tunables**: Add
`speculative_mode` and `num_speculative_tokens` to the tunable registry in
canonical order with mode before depth, and depth to the positive-value floor;
return both from `applicable_parameters` only when a method or draft model is
configured; reject a declared gated tunable when it is not; add the identity
mode value maps in both backends; update registry, loader, and validation tests
and complete-space helpers.
[x] **T4 - Forward the speculative settings in the vLLM backend**: Add the
six-member method value map and JSON-group targets that put method, model, and
depth into one `--speculative-config` object; make a disabled mode suppress the
whole group, an unset mode emit it, and an unset depth omit the depth key;
extend the vLLM argument tests.

  Note: `PARAMETER_TARGETS` gained a third target kind, `Gate`, so
  `speculative_mode` is recognised by validation without being serialised into
  the JSON object; `serve_arguments` reads the mode directly to suppress the
  group. A configured-setup validation test pins that both gated dimensions
  pass vLLM's support check.
[x] **T5 - Forward the speculative settings in the llama.cpp backend**: Add the
six-member method map to `--spec-type`, the draft and depth targets
(`--spec-draft-model`, `--spec-draft-n-max`), and the disabled-mode
suppression, with an unset mode emitting them and an unset depth omitted;
extend the llama.cpp argument tests.

  Note: the `Gate` target kind moved from `_vllm.py` to `_tunables.py` so both
  backends share it instead of the llama.cpp backend importing vLLM's module;
  `_llamacpp.PARAMETER_TARGETS` now maps `speculative_mode` to a `Gate`, and
  `applicable_parameters` excludes the two gated tunables from the base set
  before adding them back for a configured setup.
[x] **T6 - Add the speculative CLI flags and rules**: Add
`--speculative-method` and `--draft-model` to the engine option group; populate
the base configuration; imply `draft_model` from a bare draft flag; raise usage
errors for a draft-requiring method without a draft and for `ngram` paired with
a draft; print the configured setup in the header; cover the parsing, config,
and coherence in the CLI tests.
[x] **T7 - Mount the resolved draft in vLLM plans**: When a resolved draft path
is present, rewrite the speculative config to its in-container path and reuse or
dedupe the Hugging Face hub-root mount; pass an unresolved draft string through
as the main model does; extend the vLLM plan tests.

  Note: a resolved draft that is not inside the main model's cache hub root is
  mounted at a fixed `/draft-model` container path, so a local draft and a local
  target never contend for `/model`. A draft snapshot in the same hub as the
  main model reuses that hub's `/model` mount and is addressed under it.
[x] **T8 - Mount and validate the resolved draft in llama.cpp plans**: Mount a
resolved draft at the fixed container draft path and point `--spec-draft-model`
at it; fall back to the engine's draft repository argument for an unresolved
repository; reject a non-GGUF resolved draft in model-aware validation through
the GGUF metadata reader, failing open on unreadable headers; extend the plan
and validate-model tests.

  Note: the unresolved repository is forwarded as `--spec-draft-hf` with the
  user's original `repo[:QUANT]` string, so no second selector is invented.
  `serve_arguments` gained an optional `draft_repository` keyword that replaces
  the `--spec-draft-model` target at the draft model's canonical position; it is
  suppressed with the other speculative flags by a disabled mode. A local draft
  without a resolved path is mounted like the main model, and the draft gets its
  own fixed `/models/draft.gguf` target so it never collides with
  `/models/model.gguf`.
[x] **T9 - Resolve and download the configured draft in the composition root**:
Ask the backend's model-resolution hook for the draft, download it through the
shared progress helper, and record `resolved_draft_path` on the base
configuration before backend validation and planning; map resolution failures
to the unsupported-model exit path; cover the two-model prefetch in the CLI
tests.
[x] **T10 - Warn about speculation on the synthetic workload**: When a
speculative setup is configured and no dataset path is given, print the
advisory warning on stderr before the sweep; test the warning's presence and
its absence with a dataset path.
[x] **T11 - Document the feature**: Update the README's CLI options,
conditional tunable table, engine method-to-flag mapping, an end-to-end example
with its YAML, the random-workload caveat, the meaning of `default: true` for
both speculative dimensions, and the advice to use a distinct study name per
speculative setup.
[x] **T12 - Update the architecture notes**: Extend the project's architecture
document with the configuration-aware applicable set, the changed protocol
signatures, the draft resolution path, and the two new tunables in the
parameter checklist.
[x] **T13 - Verify the whole change**: Run the formatter, linter, type checker,
and unit test suite; confirm the example spaces remain valid without the
speculative dimensions and that a speculative space requires exactly the two
gated dimensions.

## Out of Scope

- Engine-only speculative methods and their variants: vLLM's `eagle`,
  `ngram_gpu`, `suffix`, `custom_class`, `extract_hidden_states`, and per-model
  MTP aliases; llama.cpp's n-gram cache, map, and mod variants.
- Typed draft-acceptance-rate reporting (vLLM Prometheus counters or llama.cpp
  log statistics) and any metric-schema change.
- Study provenance for the speculative setup, study-name suffixes, and any
  comparison semantics beyond what the existing objective modes provide.
- CLI flags for the mode or the depth; both are search-space tunables only.
- Draft-specific knobs beyond depth, such as minimum draft length, probability
  thresholds, draft tensor parallelism, KV-cache type, or heterogeneous-vocab
  support; these remain reachable only through the library-level extra-argument
  escape hatch.
- New example space files; the shipped examples stay unchanged because
  speculation is opt-in, and the README carries the example instead.

## Further Notes

- The method and draft model are fixed per run on purpose: methods are
  checkpoint-specific (MTP needs target heads, EAGLE-3 needs a draft trained
  for that exact target, DFlash/DSpark need their trained draft), so sweeping
  the method would mostly produce server-startup failures. The sweep's genuine
  questions are on/off and depth.
- The draft is downloaded once before the sweep even when the space can only
  sample disabled; the alternative would require inspecting the space's mode
  values before resolution, which adds special cases for little benefit.
- Because `speculative_mode` is required when a setup is configured, a user who
  only wants the enabled behavior writes `speculative_mode: {fixed: enabled}`
  and a depth spec. A user who only wants off can simply not configure a
  speculative setup.
- Depth is sampled but unused on disabled trials, so a categorical depth axis
  that mixes with a mode axis can produce duplicate disabled points. This is
  harmless to the optimizer and documented.
- The written configuration includes `resolved_draft_path`, mirroring how the
  benchmark section already includes `resolved_path`.
- Suggested manual smoke checks: vLLM with a target that has MTP heads and
  `--speculative-method mtp`; vLLM with an EAGLE-3 draft repository; llama.cpp
  with a small draft GGUF and a real JSONL dataset; and a constrained-memory
  run whose space sweeps `speculative_mode` to confirm the non-speculative
  point can win.
- After the docs task, the user asked for shipped speculative examples:
  `examples/` was reorganised into `vllm/` and `llamacpp/` subdirectories, each
  engine gained `space-speculative-defaults.yaml` and
  `space-speculative-full.yaml`, and a shared
  `examples/prompts.jsonl` corpus was added. This supersedes the
  "New example space files" out-of-scope note.
- The README was subsequently split into `docs/usage.md`,
  `docs/search-space.md`, `docs/speculative-decoding.md`, and
  `docs/architecture.md`, leaving it as a short overview linked to the guides.
