# Experiment Search-Space Definition

## Problem Statement

The inference layer can already run one benchmark experiment from an immutable
`ExperimentConfig`, and a series is expressed by varying `LLMConfig` with
`dataclasses.replace`. But there is no declarative artifact that says *which*
serving parameters to explore and *which values* are allowed. Today that
knowledge lives in ad-hoc Python loops, shell history, or the developer's head:
the model, the container setup, and the chosen parameter values are scattered,
so a sweep is hard to review, reproduce, or hand to someone else.

The more serious problem is timing. Nothing validates the whole parameter set
up front. A typo'd enum member, a value the backend does not support, or a
nonsensical range is only discovered when the backend plans the particular run
that uses it — which may be the fortieth experiment in a series, after
thirty-nine containers have already run. A long, expensive sweep can therefore
fail partway through for a mistake that was visible in the definition from the
start, and the results already collected are hard to interpret.

Finally, if a parameter is simply not mentioned, the inference engine silently
supplies its own default. Those defaults are not known to the user without
consulting engine documentation, so experiments whose results depend on them
are not fully explainable from the definition alone.

## Solution

Add a YAML file format that defines the LLM search space, and a parser that
turns it into an engine-neutral `ExperimentSpace`. The file describes *only*
the search space: the fixed container, benchmark/model, and non-tunable LLM
values live in other (future) definitions and are out of scope.

Each tunable parameter is declared with an explicit, tagged spec:

- `fixed` — pin the parameter to one value;
- `categorical` — a subset of allowed values;
- `interval` — an inclusive `min`/`max` range (the optimizer chooses where to
  sample; the definition does not carry a step);
- `default` — an explicit, recorded decision to defer to the engine's own
  default.

Every tunable parameter must be declared. Omitting one is an error naming the
missing parameters, so a value can never silently fall back to an unknown
engine default; if deferral is wanted, it must be written down as `default`.

Loading the file validates the definition engine-agnostically and reports
*all* problems at once, so a broken definition is fixed in one pass. A second,
explicit step asks the backend whether it supports every declared parameter and
value. Both checks happen before any experiment runs, so an invalid definition
fails once, up front, instead of midway through an expensive series.

## User Stories

1. As an LLM-optimization developer, I want to describe a serving-parameter
   search space in a file, so that a sweep is a reviewable, versionable
   artifact instead of an ad-hoc script.
2. As an LLM-optimization developer, I want the file to hold only the search
   space, so that the container and workload definitions can evolve separately.
3. As an LLM-optimization developer, I want to pin a parameter to a single
   value, so that I can fix a setting that is predetermined and must not be
   varied.
4. As an LLM-optimization developer, I want to list a subset of categorical
   values for a parameter, so that I can compare a few named alternatives.
5. As an LLM-optimization developer, I want to declare an inclusive numeric
   range for a parameter, so that I can express a continuous tuning interval
   without committing to a grid.
6. As an LLM-optimization developer, I want to declare that a parameter is
   deliberately left to the engine default, so that deferral is visible rather
   than silent.
7. As an LLM-optimization developer, I want the file format to use explicit
   tagged spec kinds, so that a dimension's intent is unambiguous and the
   format can grow.
8. As an LLM-optimization developer, I want to write the definition in YAML,
   so that nested specs, lists, and comments are easy to author.
9. As an LLM-optimization developer, I want the parser to produce an
   engine-neutral object, so that the same definition can target a different
   inference engine later.
10. As an LLM-optimization developer, I want the search space to cover only
    parameters that make sense to tune, so that reproducibility knobs and
    hardware-derived values are not accidentally swept.
11. As an LLM-optimization developer, I want `seed` and
    `gpu_memory_utilization` excluded from the search space, so that they are
    managed by the code that drives the sweep, not by the search definition.
12. As an LLM-optimization developer, I want every tunable parameter to be
    declared, so that I always know the full set of values that shaped a run.
13. As an LLM-optimization developer, I want a clear error naming any missing
    tunable parameter, so that I can fix an incomplete definition quickly.
14. As an LLM-optimization developer, I want unknown or non-tunable parameter
    names to be rejected, so that a typo does not silently do nothing.
15. As an LLM-optimization developer, I want values checked against the
    parameter's declared type, so that a string in a numeric field or a float
    in an integer field is caught immediately.
16. As an LLM-optimization developer, I want categorical values checked against
    the engine-neutral enums, so that an invalid enum value is caught before
    any run.
17. As an LLM-optimization developer, I want numeric ranges checked for a valid
    ordering, so that a reversed `min`/`max` is rejected.
18. As an LLM-optimization developer, I want universal bounds enforced, so that
    a zero or negative count/size is rejected.
19. As an LLM-optimization developer, I want a categorical list to be
    non-empty and free of duplicates, so that a degenerate dimension is
    rejected.
20. As an LLM-optimization developer, I want the loader to report every problem
    in one error, so that I can correct the whole definition in one pass.
21. As an LLM-optimization developer, I want that error to expose its problems
    as a list, so that tooling can present them programmatically.
22. As an LLM-optimization developer, I want a second validation step that
    takes a backend, so that engine-specific support is checked before running.
23. As an LLM-optimization developer, I want the backend to reject a parameter
    it does not forward, so that a declared dimension is never silently
    dropped.
24. As an LLM-optimization developer, I want the backend to reject a
    categorical or fixed value it does not support, so that an unsupported
    engine value is caught before the series starts.
25. As an LLM-optimization developer, I want the backend to enforce its own
    numeric bounds, so that engine-specific limits are respected.
26. As an LLM-optimization developer, I want engine-support validation to be
    separate from engine-agnostic validation, so that I can inspect a
    definition without choosing a backend.
27. As an LLM-optimization developer, I want the parsed space to preserve the
    declared order of categorical values, so that I can express a preferred or
    baseline-first ordering.
28. As an LLM-optimization developer, I want the parsed space to be immutable
    and hashable, so that it can be logged, compared, and cached.
29. As an LLM-optimization developer, I want to load from a file path, so that
    definitions live in version control next to the code.
30. As a test author, I want to parse an already-loaded mapping, so that I can
    test definitions without touching the filesystem.
31. As an LLM-optimization developer, I want a file that is not valid YAML, or
    not a mapping, to produce the same error type, so that I handle malformed
    files uniformly.
32. As an LLM-optimization developer, I want unknown top-level keys and a
    missing `space` key to be rejected, so that a mistyped file fails loudly.
33. As a future maintainer, I want the tunable parameter set defined in one
    engine-neutral registry, so that adding or removing a tunable is a single
    edit.
34. As a future maintainer, I want the registry to map a space parameter to an
    `LLMConfig` field, so that the space vocabulary can diverge from the config
    vocabulary without touching the backend.
35. As a future maintainer, I want the backend to keep owning its
    parameter-to-flag mapping, so that engine spellings stay out of the
    engine-neutral layer.
36. As a future maintainer, I want the validator to be a thin, uniform
    aggregator over a backend hook, so that a new backend implements one
    method.
37. As a test author, I want to unit-test the parser and validator with no
    daemon, so that the suite stays fast and hermetic.
38. As a test author, I want to test the vLLM backend's support checks with
    crafted spaces, so that engine-support validation is covered without
    running a container.

## Implementation Decisions

**Format and dependencies**

- The definition file is YAML, parsed with PyYAML via `yaml.safe_load`, added
  as a runtime dependency.
- The file contains a single top-level `space:` mapping and nothing else;
  unknown top-level keys are errors and a missing `space:` key is an error.
- Spec kinds are explicit tagged mappings, one tag per dimension: `fixed`,
  `categorical`, `interval`, or `default`. A spec mapping must contain exactly
  one recognized tag.
- `interval` carries exactly `min` and `max` and is inclusive of `max`. It
  deliberately has no `step` or `count`; choosing sample points is the
  optimizer's job. When only specific numbers are valid, `categorical` is used
  instead.
- `default` is written as `default: true` and must be exactly boolean true.

**Search-space types**

- A new engine-neutral `search_space` subpackage owns the search-space
definition topic; its `types` module holds the parsed types and the definition
error. Executing an experiment stays in the `inference` subpackage.
- `ExperimentSpace` is a frozen dataclass holding `dimensions: tuple[Dimension,
  ...]`.
- `Dimension` is a frozen dataclass holding `name` and a `spec`.
- The spec union is `Fixed | Categorical | Interval | Default`, each a frozen
  dataclass: `Fixed(value)`, `Categorical(values: tuple)`,
  `Interval(minimum, maximum)`, and a fieldless `Default`.
- `Categorical.values` is an ordered tuple, preserving declared order; `Fixed`
  and `Categorical` values are stored as engine-neutral enum members for
  categorical fields and as integers for numeric fields.
- All spec types and `ExperimentSpace` are immutable and hashable.
- Dimensions are emitted in canonical tunable-registry order, independent of
  the file's key order, so the parsed space is deterministic.

**Tunable registry**

- One engine-neutral registry defines which parameters may be swept, mapping a
  space parameter name to its `LLMConfig` field name.
- The initial tunable set is `dtype`, `quantization`, `kv_cache_dtype`,
  `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`,
  `tensor_parallel_size`, and `attention_backend`.
- `seed` and `gpu_memory_utilization` are intentionally not tunable; they are
  managed outside the search space by the code driving a sweep, and the future
  fixed-value definition file.
- The expected kind of each tunable (an engine-neutral enum class, or integer)
  is derived from the `LLMConfig` field's type annotation rather than
  duplicated in the registry, so the registry does not drift from the config.
- The registry is a strict, fixed allow-list: a space parameter that is not in
  it is an unknown parameter, and a declared parameter that is known but not
  tunable is rejected as non-tunable.

**Completeness**

- Every tunable parameter must be declared. A missing one is an error whose
  message names it and states the accepted spec kinds.
- Because completeness is enforced, `ExperimentSpace` is a total description
  over the tunable set; there is no "absent means engine default" case.
  Deferring to the engine is expressed only through the explicit `default`
  spec.

**Parsing and engine-agnostic validation**

- `load_experiment_space(path)` reads the file and delegates to
  `parse_experiment_space(mapping)`; the latter accepts an already-parsed
  mapping so definitions can be validated without the filesystem.
- The loader performs engine-agnostic validation and collects every problem,
  raising a single `ExperimentDefinitionError` whose `.problems` is a tuple of
  message strings and whose message lists them. This includes malformed YAML
  and a non-mapping top level, which are reported through the same error type.
- Engine-agnostic checks are: top-level shape; registry membership; exactly one
  recognized spec tag per dimension; kind/type compatibility (an `interval` is
  valid only for a numeric field; enum fields accept `fixed`/`categorical`);
  value type and enum membership; a non-empty, duplicate-free `categorical`; an
  integer-typed `interval` for an integer field with `min <= max`; and
  universal positivity (`max_model_len`, `max_num_seqs`,
  `max_num_batched_tokens`, and `tensor_parallel_size` values and bounds must
  be at least 1).
- Booleans are rejected where an integer is expected, so a YAML `true`/`false`
  never passes as a count.

**Engine-support validation**

- The `InferenceBackend` protocol gains a `validate(space) -> Sequence[str]`
  method that returns human-readable problem strings. It assumes the space has
  already passed engine-agnostic validation.
- `validate_space(space, backend)` calls `backend.validate(space)`, collects
  the returned problems, and raises `ExperimentDefinitionError` if there are
  any. It returns `None` on success.
- `VLLMBackend.validate` checks, for every non-`Default` dimension, that the
  mapped `LLMConfig` field is actually forwarded by the backend's parameter
  target table, that every fixed/categorical enum member is present in the
  backend's value maps, and that any engine-specific numeric bounds hold.
- `Default` dimensions carry no value to support-check and are accepted as
  long as the parameter itself is known to the backend.
- Validation is intentionally two-step: the engine-neutral loader needs no
  backend, and the engine-support step is explicit and takes the backend the
  caller will run with.

**Error type**

- `ExperimentDefinitionError` is a standalone exception (not a subclass of
  `ExperimentError`), so a bad definition is distinguishable by type from a
  failed run. It exposes `.problems: tuple[str, ...]` and renders a message
  that lists every problem.

**Layering**

- The search-space types module imports the engine-neutral enums and
  `LLMConfig` from the `inference` subpackage; it imports no backend and no
  engine. The loader module imports the types module. The backend imports the
  types module and the registry to translate space parameters to `LLMConfig`
  fields.
- The backend protocol's new method references `ExperimentSpace` only in an
  annotation, so the protocol module can name the space type without importing
  the types module at runtime.

## Testing Decisions

Good tests exercise external behavior and observable contracts: given a YAML
mapping, what `ExperimentSpace` is produced or what problems does
`ExperimentDefinitionError` report; given a space and a backend, which
problems does `validate_space` surface. Tests should not assert on private
helper internals or on the exact wording beyond the parameter path and the
kind of problem.

- **Parsing and engine-agnostic validation** are tested against
  `parse_experiment_space` with in-memory mappings: a complete valid definition
  producing the expected dimensions, each spec kind, and a table of invalid
  definitions asserting that the right problem paths appear. Malformed YAML
  and a non-mapping top level are tested through the file loader.
- **Error aggregation** is tested by a definition with several independent
  mistakes, asserting that all are reported in one `ExperimentDefinitionError`
  and that `.problems` contains them.
- **Completeness** is tested by omitting tunables and asserting they are named
  in the error.
- **Type immutability and hashability** of `ExperimentSpace`, `Dimension`, and
  the spec types are tested directly, mirroring the existing inference-layer
  contract tests.
- **Engine-support validation** is tested with `VLLMBackend.validate` on
  crafted spaces: a parameter the backend does not forward, an unsupported enum
  member, a supported space that returns no problems, and a `Default` dimension
  that is accepted. `validate_space` is tested for raising the aggregated
  error and for returning quietly on success.
- **Prior art**: `tests/inference/test_types.py` establishes the
  immutability/hashability contract tests; `tests/inference/test_vllm_plan.py`
  and `tests/inference/test_vllm_arguments.py` establish backend unit tests
  with no daemon; the runtime and inference fakes in `tests/fakes.py` establish
  the no-daemon testing convention. The new tests need no Podman daemon and
  carry no `integration` marker, and live in `tests/search_space/` alongside the
  subpackage they cover.

## Implementation Tasks

[x] **T1 - Add the PyYAML runtime dependency**: Declare PyYAML as a runtime dependency and refresh the lockfile so the loader can parse YAML.
[x] **T2 - Define the search-space types and error**: Add the engine-neutral search-space types module with the frozen, hashable `ExperimentSpace`, `Dimension`, `Fixed`, `Categorical`, `Interval`, and `Default` types and the standalone `ExperimentDefinitionError` carrying `.problems`.
- Note: realised in the new `search_space` subpackage as `src/llm_optimization/search_space/types.py`, with a docstring-only `search_space/__init__.py` (public re-exports are T9). This moved the module out of the `inference` subpackage, which is scoped to executing benchmarks; the plan's Implementation Decisions and T9 were updated to match. The spec union is exported as `Spec` and the value union as `ParameterValue`. `Categorical.values` and `ExperimentSpace.dimensions` are coerced to tuples in `__post_init__` so a list passed by a caller still yields an immutable, hashable value. `ExperimentDefinitionError.__init__` takes an iterable of problem strings and renders them as a bulleted list. The module imports only the engine-neutral enums from `..inference.types`; the `LLMConfig` import belongs to T3's registry/kind helper.
[x] **T3 - Define the tunable parameter registry**: Add the engine-neutral registry mapping each tunable space parameter to its `LLMConfig` field, plus a helper that resolves a field's expected kind (an engine-neutral enum class or integer) from the `LLMConfig` type annotations.
- Note: realised as `src/llm_optimization/search_space/registry.py` with `TUNABLE_PARAMETERS` (space name to `LLMConfig` field, canonical order) and `expected_kind(field)` reading the single non-`None` member of the field's annotation. The kind union is exported as `ParameterKind`. Not yet re-exported from the subpackage; public exports remain T9.
[x] **T4 - Implement structural parsing of the definition**: Add `parse_experiment_space` and `load_experiment_space` that read the top-level `space:` mapping, reject unknown top-level keys and a missing or non-mapping `space`, and report malformed YAML through `ExperimentDefinitionError`.
- Note: realised as `src/llm_optimization/search_space/loader.py`. Scope is the top-level container only: `load_experiment_space` decodes UTF-8, parses with `yaml.safe_load`, and reports `yaml.YAMLError` as a problem, while `parse_experiment_space` collects every top-level problem (non-mapping document, unknown keys, missing `space`, non-mapping `space`) and raises `ExperimentDefinitionError` once. Dimension/spec parsing inside `space` is deliberately not implemented here and raises `NotImplementedError`; T5 completes it. Not yet re-exported from the subpackage (T9).
[x] **T5 - Implement spec parsing and engine-agnostic validation**: Parse each dimension's tagged spec (`fixed`, `categorical`, `interval`, `default`) and validate registry membership, tag exclusivity, kind/type compatibility, value and enum types, non-empty duplicate-free categoricals, integer intervals with `min <= max`, and universal `>= 1` bounds, collecting every problem.
- Note: realised in `src/llm_optimization/search_space/loader.py`; `parse_experiment_space` now returns a canonical-registry-order `ExperimentSpace`. It collects every problem across all dimensions and raises one `ExperimentDefinitionError`: registry membership (unknown vs known-but-non-tunable, distinguishing `seed`/`gpu_memory_utilization`), exactly one recognized tag per dimension, unknown spec/interval keys, enum-kind compatibility (interval rejected for categorical fields), enum membership and value types with booleans rejected for integers, non-empty/unique/order-preserving categoricals, interval shape with `min <= max`, and `>= 1` positivity for the four integer tunables. Completeness (a missing tunable) is deliberately not enforced, so a partial space parses successfully; that is T6. Not yet re-exported from the subpackage (T9).
[x] **T6 - Enforce tunable completeness**: Require every tunable parameter to be declared and report any missing ones, naming each and the accepted spec kinds.
- Note: realised in `src/llm_optimization/search_space/loader.py`; `_parse_dimensions` now appends a `space.<name>: missing required tunable parameter; declare one of fixed, categorical, interval, default` problem for every `TUNABLE_PARAMETERS` entry absent from the declared `space` mapping. Missing dimensions are reported alongside any other problems in the same single `ExperimentDefinitionError`. A declared-but-invalid dimension is not also reported as missing, since it is already named by its spec problem. The module docstring now states the completeness guarantee.
[x] **T7 - Add the backend validation hook to the protocol**: Add `validate(space) -> Sequence[str]` to the `InferenceBackend` protocol, annotating the space type without a runtime import cycle.
- Note: realised in `src/llm_optimization/inference/types.py`; the protocol gained `validate(space) -> Sequence[str]` between the class docstring and `plan`. The cycle is avoided with a `TYPE_CHECKING`-only import of `ExperimentSpace` from `..search_space.types`, the same convention `_orchestrate.py` already uses for `PodmanClient`; `Sequence` is imported from `collections.abc`. The method returns problem strings and is not implemented here — `VLLMBackend.validate` is T8 and `validate_space` is T9.
[x] **T8 - Implement vLLM engine-support validation**: Implement `VLLMBackend.validate` to check that each non-`Default` dimension maps to a forwarded parameter, that every fixed/categorical enum member is supported by the value maps, and that engine-specific numeric bounds hold.
- Note: realised in `src/llm_optimization/inference/_vllm.py`. `VLLMBackend.validate` returns problem strings (empty when fully supported). Per dimension it checks that the mapped `LLMConfig` field is in `PARAMETER_TARGETS`; for a non-`Default` spec it checks that every enum value has a `VALUE_MAPS` spelling, reporting `space.<name>.fixed` and `space.<name>.categorical[i]` paths. `Default` dimensions skip the value check but still require a forwarded parameter, per the plan. The plan named no concrete engine-specific numeric bound, so the unconditional vLLM rule from `SchedulerConfig.verify_max_model_len` — `max_num_batched_tokens >= max_num_seqs` — is enforced through a declarative `_ENGINE_BOUNDS` pair table: it compares the smallest allowed `max_num_batched_tokens` with the largest allowed `max_num_seqs`, so a space is accepted only when every point it can produce satisfies the bound, and a `Default` on either side skips the check. vLLM's `max_num_batched_tokens >= max_model_len` rule is deliberately not enforced because it applies only when chunked prefill is disabled, which the default vLLM v1 scheduler does not do. Aggregating `validate_space` and the public exports remain T9.
[x] **T9 - Implement the validator and export the public API**: Add `validate_space(space, backend)` that aggregates the backend's problems and raises `ExperimentDefinitionError`, and re-export the new types, error, loader, and validator from the `search_space` subpackage.
- Note: realised as `src/llm_optimization/search_space/validation.py`; `validate_space(space, backend)` is a thin aggregator that raises one `ExperimentDefinitionError` from the backend's problem strings, annotating the backend with `InferenceBackend` under `TYPE_CHECKING` so the engine-neutral layer imports no backend at runtime. The subpackage re-exports the types, error, loader, and validator, and — because T3–T5 deferred their public exports here — the registry's `TUNABLE_PARAMETERS`, `ParameterKind`, and `expected_kind` as well. The re-export also resolves a latent cycle: importing `search_space.registry` on its own previously failed, because the loader (and thus the full `inference` package) is now imported before the registry.
[x] **T10 - Unit-test parsing and engine-agnostic validation**: Add tests for a valid definition, each spec kind, completeness, malformed YAML and non-mapping input, and a table of invalid definitions asserting the reported problem paths and aggregation.
[x] **T11 - Unit-test vLLM engine-support validation**: Add tests for `VLLMBackend.validate` and `validate_space` covering forwarded-parameter coverage, unsupported enum values, accepted `Default` dimensions, a fully supported space, and aggregated failure.
- Note: realised as `tests/search_space/test_validation.py` (16 tests). Two branches the plan expects to exercise — a parameter the backend does not forward and an unsupported enum member — are unreachable through the public vocabulary: all eight tunables are in `PARAMETER_TARGETS` and every member of every curated enum is in `VALUE_MAPS`. The tests inject each condition by monkeypatching `_vllm.PARAMETER_TARGETS` / `_vllm.VALUE_MAPS`, so the rejection paths stay covered as future tunables and enum members are added. The tests also cover the `max_num_batched_tokens >= max_num_seqs` engine bound (violating, satisfying, and skipped-by-`Default` spaces) and a `Default` dimension that is still rejected when its parameter is not forwarded. `validate_space` is tested against a stub backend for aggregation and against `VLLMBackend` for the success path.
[x] **T12 - Unit-test search-space immutability and hashability**: Add tests asserting that `ExperimentSpace`, `Dimension`, and each spec type are frozen and hashable, and that categorical order is preserved.
[x] **T13 - Fix the duplicate test-module basename that breaks collection**: `tests/search_space/test_types.py` shares its basename with the pre-existing `tests/inference/test_types.py`. The test tree is deliberately not a package (no `tests/**/__init__.py`; `tests/conftest.py` relies on pytest's `prepend` import mode to expose `tests/fakes.py`), so pytest imports both files as the top-level module `test_types` and aborts collection with an `import file mismatch`, meaning `uv run pytest` cannot run at all. Rename the new file to a unique basename (for example `test_space_types.py`); this is the only duplicate basename in the tree, and it is not a stale-`__pycache__` problem.
[x] **T14 - Cover the `InferenceBackend` protocol conformance for the new `validate` hook**: T7 added `validate(space) -> Sequence[str]` to the `InferenceBackend` protocol, but the backend stand-ins in `tests/inference/test_orchestrate.py` (`FakeBackend`, `DirectoryAwareBackend`) still implement only `plan`/`parse_result`, so they no longer structurally satisfy the protocol they stand in for. Add a `validate` stub to those stand-ins and add a test that every backend double passed to `run_experiment` conforms to `InferenceBackend` (for example by making the protocol `@runtime_checkable` and asserting `isinstance`), so the protocol and its test doubles cannot drift again.
- Note: `InferenceBackend` is now `@runtime_checkable` (`src/llm_optimization/inference/types.py`). `FakeBackend` gained a `validate(space) -> Sequence[str]` stub that records the call and returns `()`; `DirectoryAwareBackend` inherits it. A parametrized `test_backend_doubles_conform_to_the_inference_protocol` asserts `isinstance(backend, InferenceBackend)` for both doubles. This is the approach the task sanctioned; it guards against a missing hook, and checking signatures is left to a future static type checker rather than a hand-rolled assertion.

## Out of Scope

- Defining the container, benchmark/model, or non-tunable LLM values in this
  file; those live in other definitions (the fixed-value definitions are
  future work).
- Any optimizer, sampler, or enumeration of concrete experiments from the
  space; choosing points (including steps and counts) is out of scope.
- Running a sweep or a series of experiments, and any comparison or reporting.
- A command-line interface or console entrypoint.
- A fixed-value definition file for `seed`, `gpu_memory_utilization`, and other
  non-tunable parameters.
- Extending the tunable registry beyond the initial eight parameters, or making
  it user-extensible from the file.
- Float-typed tunables and float intervals; every numeric tunable in the first
  version is an integer.
- An `all` shorthand that expands a dimension to its full domain.
- Writing or round-tripping YAML, comment preservation, and schema versioning.
- Backends other than vLLM.

## Further Notes

- The file deliberately holds only the search space so that the search space,
  the fixed experiment setup, and the non-tunable values can evolve on
  independent schedules. The cost is that a definition is not self-contained;
  a run needs at least one other definition.
- The `default` spec keeps the completeness guarantee honest: a value is never
  absent by accident, but the user can still choose to defer to the engine. The
  resulting value remains unknown from the definition alone, which is the
  trade-off the spec makes visible.
- Excluding `seed` and `gpu_memory_utilization` reflects that they are not
  tuning dimensions: a seed is a reproducibility input, and GPU memory
  utilization is often derived from the host or fixed per run.
- Because all numeric tunables are integers, `Interval` is exercised only with
  integers in the first version; the type is kept numeric so float fields can
  be added without a format change.
- The registry decouples the file's parameter vocabulary from `LLMConfig` field
  names, so the file's names can be renamed without touching the backend, and
  the backend keeps owning engine spellings through its existing parameter
  target and value-map tables.
- The two-step validation mirrors the existing `backend=None`-means-vLLM
  convention inverted: the loader stays pure, and the engine-support step
  requires an explicit backend so it is obvious which engine a definition was
  checked against.
- The tunable registry is the single place to change what may be swept; adding
  a parameter there plus its existing backend mapping is enough for it to
  become sweepable.
- **Resolved (T13):** the new test file previously broke the suite.
  `tests/search_space/test_types.py` duplicated the basename of
  `tests/inference/test_types.py`, and because the test tree is intentionally
  not a package, pytest's `prepend` import mode could not tell the two modules
  apart; `uv run pytest` failed during collection before a single test ran.
  Renaming the new file to `tests/search_space/test_space_types.py` fixed it;
  the full unit suite passes.
- **Resolved (T14):** the new protocol method is now reflected in the test
  doubles. `FakeBackend` and `DirectoryAwareBackend` in
  `tests/inference/test_orchestrate.py` implement `validate`, and a
  parametrized conformance test asserts both satisfy the `@runtime_checkable`
  `InferenceBackend` protocol, so the protocol and its doubles cannot drift
  again. This is the approach the task sanctioned.
- **Known limitation (T14):** `runtime_checkable` checks method presence, not
  signatures, so the conformance test catches a missing hook but not a
  mismatched one. Closing that gap is the job of a static type checker, which
  the project does not currently run; it is deliberately out of scope here.
