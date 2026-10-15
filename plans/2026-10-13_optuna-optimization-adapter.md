# Optuna Optimization Adapter

## Problem Statement

The search-space layer can describe *which* serving parameters a sweep may vary
and *which* values each may take, and the inference layer can run one benchmark
experiment from an immutable `ExperimentConfig` and return a typed
`ExperimentResult`. But nothing chooses the next point to try. A sweep is still
a hand-written loop around `run_experiment`, so the developer has to decide
every value themselves, keep the results in their head or in scratch files, and
manually track which configuration was best.

This makes tuning LLM serving parameters slow and unrepeatable. There is no
record of the trials that were run, so an interrupted sweep cannot be resumed
and its history cannot be reviewed. There is no principled way to search an
`Interval` — the space deliberately carries no step or grid, because choosing
sample points is the optimizer's job — yet no optimizer is wired up to do it. A
developer who wants to explore eight interacting parameters is reduced to
guessing or to writing throwaway search code for every sweep.

The framework to close this loop — Optuna — has not been evaluated against the
project's engine-neutral search-space semantics. It is not known whether each
spec kind maps faithfully onto Optuna's trial API, whether studies can be
persisted and resumed, whether trials can run concurrently against a GPU, and
whether the dependency footprint is acceptable on Python 3.14. Without that
evaluation, adopting Optuna would be a guess.

## Solution

Add an optimization adapter that turns an engine-neutral `ExperimentSpace` plus
a base `ExperimentConfig` into an Optuna study, runs the benchmark for each
sampled point, records every trial's outcome, and returns the best
configuration found. The developer supplies only the search space, the fixed
container/workload setup, and a trial budget; Optuna chooses the points.

The adapter lives in a new engine-neutral `optimization` subpackage, mirroring
the existing neutral-types-plus-backend split: the result and error types import
no Optuna, and a backend-specific module owns the Optuna study and trial logic.
The public entry point is a single `optimize` function.

Each dimension maps faithfully onto Optuna:

- `Fixed` becomes a one-choice categorical so its value is recorded in
  `trial.params` and the trial stays self-describing;
- `Categorical` becomes `suggest_categorical`;
- `Interval` becomes `suggest_int` with the default step of one;
- `Default` produces no suggestion and leaves the field unset, so deferral to
  the engine is genuinely absent from the trial.

Sampled categorical values are converted back to engine-neutral enum members
before an `LLMConfig` is built, because the backend only translates enum members
to engine spellings and some enum values (notably attention backends) differ
from their engine spelling.

The objective is fixed to maximizing `tokens_per_second` for this first version.
A trial that fails records its error on the trial and re-raises, so Optuna marks
it failed and the study continues instead of aborting an expensive sweep. Every
successful trial records a serializable projection of its result, so the study
explains not only which point won but what actually ran. When no trial
completes, the adapter raises a dedicated optimization error rather than leaking
an Optuna-shaped exception.

Studies are created by the adapter with Optuna's default TPE sampler, seeded on
request. Passing a storage URL persists the study and resumes it when a study of
the same name already exists. Trials run sequentially by default, with `n_jobs`
exposed as a passthrough; the caller is responsible for keeping concurrent
trials on disjoint devices.

The work also evaluates Optuna against explicit pass/fail criteria — spec-mapping
fidelity, Python 3.14 installation, persistence and resumption, `n_jobs`
usability, dependency footprint, and a real small GPU sweep that yields a usable
best configuration — and records a go/no-go verdict.

## User Stories

1. As an LLM-optimization developer, I want to hand a search space and a fixed
   experiment setup to an optimizer, so that I no longer choose every parameter
   value by hand.
2. As an LLM-optimization developer, I want the optimizer to choose points from
   an `Interval` for me, so that continuous numeric parameters are explored
   without me inventing a grid.
3. As an LLM-optimization developer, I want a `Fixed` dimension to be honoured
   exactly, so that a predetermined parameter is never varied.
4. As an LLM-optimization developer, I want a `Categorical` dimension to be
   sampled from its declared values, so that named alternatives are compared.
5. As an LLM-optimization developer, I want a `Default` dimension to leave the
   field unset, so that deferral to the engine is preserved through the sweep.
6. As an LLM-optimization developer, I want a fixed dimension's value to appear
   in the trial record, so that each trial is fully explainable from the study
   alone.
7. As an LLM-optimization developer, I want sampled values converted back to
   engine-neutral enum members, so that the backend forwards the correct engine
   spelling.
8. As an LLM-optimization developer, I want the objective to be throughput
   (`tokens_per_second`), maximized, so that the best configuration is the
   fastest one.
9. As an LLM-optimization developer, I want to supply a base `ExperimentConfig`,
   so that the container and workload setup stay fixed while the serving
   parameters vary.
10. As an LLM-optimization developer, I want the non-tunable LLM values (`seed`,
    `gpu_memory_utilization`) inherited from the base config, so that they are
    managed by the sweep driver rather than the search space.
11. As an LLM-optimization developer, I want the adapter to build a complete
    `LLMConfig` for each trial, so that every run is fully specified.
12. As an LLM-optimization developer, I want to set the number of trials, so
    that I control how much hardware time a sweep consumes.
13. As an LLM-optimization developer, I want to set a wall-clock timeout, so
    that a sweep stops after a bounded time.
14. As an LLM-optimization developer, I want to seed the sampler, so that a
    sweep is reproducible.
15. As an LLM-optimization developer, I want an unseeded sweep to be
    non-deterministic by default, so that repeated runs explore differently
    unless I ask for reproducibility.
16. As an LLM-optimization developer, I want the study created for me, so that I
    do not have to learn Optuna's study-construction API to run a sweep.
17. As an LLM-optimization developer, I want to pass a storage URL, so that a
    study is persisted to disk.
18. As an LLM-optimization developer, I want a persisted study to resume by
    name, so that an interrupted sweep continues instead of restarting.
19. As an LLM-optimization developer, I want an in-memory study when I pass no
    storage, so that a quick sweep needs no files.
20. As an LLM-optimization developer, I want to run trials in parallel when I
    can isolate devices, so that a multi-GPU host is used efficiently.
21. As an LLM-optimization developer, I want parallel trials to be off by
    default, so that I do not accidentally distort throughput measurements by
    contending for one GPU.
22. As an LLM-optimization developer, I want a failed trial to be marked failed
    and the sweep to continue, so that one out-of-memory configuration does not
    end an expensive run.
23. As an LLM-optimization developer, I want a failed trial's error message
    recorded, so that I can see why a configuration was rejected.
24. As an LLM-optimization developer, I want each successful trial to record a
    serializable projection of its result (metrics, target, image, devices), so
    that the study records what actually ran.
25. As an LLM-optimization developer, I want a typed optimization result
    carrying the best configuration, so that I can re-run the winner directly.
26. As an LLM-optimization developer, I want the typed result to carry the best
    objective value and trial number, so that I can report the outcome.
27. As an LLM-optimization developer, I want the raw study exposed on the
    result, so that I can inspect the full trial history myself.
28. As an LLM-optimization developer, I want a dedicated optimization error when
    no trial completes, so that I can distinguish an empty study from a failed
    run.
29. As an LLM-optimization developer, I want the no-completed-trial error to
    carry the study, so that I can inspect why every trial failed.
30. As an LLM-optimization developer, I want the adapter to leave space
    validation to me, so that the validation step stays explicit and the adapter
    stays thin.
31. As an LLM-optimization developer, I want to inject the inference backend and
    the container client, so that I can test the adapter without a daemon or a
    GPU.
32. As a test author, I want the adapter to be unit-testable with fakes, so that
    the mapping and study logic are covered without Podman.
33. As a test author, I want one integration smoke test that runs a real
    two-trial sweep, so that the real path is regression-protected without
    slowing the unit suite.
34. As an LLM-optimization developer, I want a documented manual spike for
    resumption, parallelism, and a fuller sweep, so that the exploratory
    evaluation is repeatable.
35. As an LLM-optimization developer, I want Optuna evaluated against explicit
    criteria, so that adopting it is a recorded decision rather than a guess.
36. As a future maintainer, I want the optimization layer split into
    engine-neutral types and an Optuna-specific module, so that another
    optimizer can be added behind the same result type.
37. As a future maintainer, I want the Optuna dependency declared explicitly, so
    that the footprint is visible and installable in one step.
38. As a future maintainer, I want pruning, constraints, and multi-objective
    search deliberately excluded, so that the first version stays small and the
    reasons are recorded.

## Implementation Decisions

**New subpackage and layering**

- A new `optimization` subpackage owns the optimization topic. It follows the
  existing neutral-types-plus-backend split: a types module holds
  `OptimizationResult` and `OptimizationError` and imports no Optuna; an
  Optuna-specific module holds the study and trial logic; the package initializer
  re-exports the public surface.
- The subpackage imports the existing search-space types (`ExperimentSpace`,
  `Spec` variants) and the inference layer (`ExperimentConfig`, `LLMConfig`,
  `ExperimentResult`, `ExperimentError`, `InferenceBackend`, `run_experiment`).
  It does not modify `search_space`, `inference`, or `runtime`.
- The search-space layer's previous exclusion of "any optimizer" is unchanged;
  the optimizer lives in its own subpackage rather than inside `search_space`.

**Dependency**

- Optuna is declared as a runtime dependency, `optuna>=5.0`, added with
  `uv add optuna` so the lockfile is refreshed. It resolves on Python 3.14 and
  brings numpy, SQLAlchemy, Alembic, Mako, and their transitive dependencies.
- No optional extra is introduced; importing the public `optimize` entry point
  always works once the project is installed.

**Public API**

- The entry point is a function, `optimize`, with keyword-only controls after
  the two required positional arguments: the `ExperimentSpace` and the base
  `ExperimentConfig`.
- The controls are: a required trial count; an optional timeout; an optional
  sampler seed; an optional storage URL; an optional study name; an optional
  inference backend; an optional container client; and an `n_jobs` parallelism
  passthrough defaulting to one.
- `optimize` returns an `OptimizationResult` carrying the best configuration,
  the best objective value, the best trial number, the number of trials run, and
  the raw Optuna study.
- The optimization direction is fixed to maximize. The objective is fixed to
  `ExperimentResult.tokens_per_second`; there is no callable objective, metric
  enum, or multi-objective support in this version.

**Space-to-trial mapping**

- A single reconstruction step builds an `LLMConfig` from the space and a set of
  parameter values. It is used both when a trial samples its point and when the
  winning point is rebuilt from the study, so the two cannot diverge.
- `Fixed` is submitted as a one-choice categorical so its value is recorded in
  `trial.params`.
- `Categorical` is submitted as a categorical over the declared values,
  preserving declared order.
- `Interval` is submitted as an integer suggestion over the inclusive bounds,
  with the default step of one; the space carries no step.
- `Default` produces no suggestion and sets the corresponding `LLMConfig` field
  to `None`.
- Categorical choices are passed to Optuna as their string values, because
  studies are persisted and reloaded. The reconstruction step converts a
  sampled string back to the field's engine-neutral enum class, resolved from
  the `LLMConfig` field's own type annotation, so the backend's value maps apply.
- The base config's `container` and `benchmark` are reused unchanged; only the
  `llm` domain is rebuilt, and the base's non-tunable `seed` and
  `gpu_memory_utilization` are inherited.

**Study and trial lifecycle**

- The study is created by the adapter, not the caller, with the default TPE
  sampler seeded by the seed argument, the fixed maximize direction, the given
  storage and study name, and resume-on-existing enabled when a storage URL is
  supplied.
- An in-memory study is used when no storage is supplied.
- The objective runs one `run_experiment` for the trial's rebuilt configuration,
  using the injected backend and container client, and returns the throughput.
- A raised `ExperimentError` is caught, its message recorded on the trial as a
  user attribute, and then re-raised so Optuna marks the trial failed and the
  study continues.
- A successful trial records a JSON-serializable projection of its
  `ExperimentResult` as a user attribute: throughput, request rate, elapsed
  time, hardware target, image, and devices. The full result is not stored
  because it is not JSON-serializable.
- After optimization, the best trial is read from the study and its parameters
  are reconstructed into an `ExperimentConfig`. If no trial completed, an
  `OptimizationError` is raised carrying the study instead of leaking Optuna's
  own exception.
- `n_jobs` is passed through to the study's optimize call. Parallel trials are
  documented as requiring the caller to keep concurrent runs on disjoint
  devices; the adapter performs no device scheduling.
- No pruner is configured. Optuna's pruners require intermediate objective
  reports, which a single end-to-end benchmark run does not produce; wiring one
  would require a partial-result channel the runtime layer does not have.

**Validation**

- `optimize` does not validate the space. The caller is expected to run the
  existing engine-support validation explicitly before starting a sweep, keeping
  the adapter thin and the validation step visible.

**Error type**

- `OptimizationError` is a standalone exception for the optimization layer,
  distinct from `ExperimentError` and `ExperimentDefinitionError`. It can carry
  the study so a failed sweep remains inspectable.

## Testing Decisions

Good tests exercise external behaviour and observable contracts: given a space
and a base config, what `LLMConfig` does a trial produce; given a study and
canned results, what does `optimize` return; given a failing runner, what state
does the study end in. Tests should not assert on private helper internals, and
unit tests must not require a daemon.

- **Space-to-trial mapping** is tested directly: a `Fixed` dimension produces a
  one-choice categorical and appears in the trial parameters; a `Categorical`
  dimension samples only declared values; an `Interval` samples within its
  inclusive bounds; a `Default` dimension leaves the field unset; and sampled
  strings are converted back to the correct enum members, including an
  attention backend whose engine spelling differs from its enum value.
- **Configuration reconstruction** is tested by rebuilding an `ExperimentConfig`
  from a parameter mapping and asserting the container and benchmark are the
  base's, the tunable fields come from the space, and the non-tunable fields are
  inherited.
- **Study and trial lifecycle** is tested with the existing fakes: a fake
  inference backend and the fake Podman client from the test support module, so
  no daemon or GPU is needed. Tests cover the best-config reconstruction, the
  result projection recorded on a successful trial, a failed trial being marked
  failed while the study continues, and `OptimizationError` when no trial
  completes.
- **Public API** is tested for the in-memory default, the storage and study-name
  path (using a temporary SQLite URL), the seed argument, and the `n_jobs`
  passthrough.
- **Integration**: one test marked `integration` runs `optimize` with a
  two-trial budget over a mostly-fixed space against a real backend and asserts
  the returned result and study shape. It is deselected by default and skipped
  when Podman is unavailable, matching the existing integration convention.
- **Manual spike**: resumption from a persisted study, `n_jobs` with
  caller-managed device isolation, and a fuller trial count are exercised by a
  documented manual procedure rather than a committed test, because they are
  slow and hardware-dependent.
- **Prior art**: the inference-layer tests establish no-daemon backend tests
  with injected hardware; the orchestration tests and the shared fakes
  establish the fake backend and fake Podman client convention; the runtime and
  vLLM integration tests establish the `integration` marker and Podman
  skip behaviour. The new tests follow those patterns and live alongside the
  subpackage they cover.

## Implementation Tasks

[x] **T1 - Add Optuna as a runtime dependency**: Add `optuna>=5.0` to the project's runtime dependencies and refresh the lockfile so the adapter's imports resolve on Python 3.14.
    - Note: `flake.nix` also gained a `shellHook` exporting `LD_LIBRARY_PATH` from `pkgs.stdenv.cc.cc.lib`. numpy's manylinux wheel dynamically links `libstdc++.so.6`, which the dev shell did not put on the loader path; without it `import optuna` fails on Python 3.14.

[x] **T2 - Define the optimization result and error types**: Add the engine-neutral optimization types module with the frozen, slots-based `OptimizationResult` (best config, best value, best trial number, trial count, raw study) and the standalone `OptimizationError` that can carry the study. The module imports no Optuna at runtime; the study is typed only where the package initializer or the Optuna module supplies it.
    - Note: the `optimization` package directory was created with a docstring-only `__init__.py` so the subpackage is a regular package and `optimization.types` is importable; T6 adds the public re-exports there. The study field is annotated `optuna.Study` behind a `TYPE_CHECKING` import, matching the existing `ExperimentSpace` annotation pattern in `inference/types.py`; annotations are lazily evaluated on Python 3.14, so the module imports no Optuna at runtime.

[x] **T3 - Implement space-to-trial suggestion and config reconstruction**: Add the mapping from each spec kind to its Optuna suggestion (`Fixed` as a one-choice categorical, `Categorical` as a categorical over declared values, `Interval` as an integer suggestion with the default step, `Default` as no suggestion and an unset field), and the single reconstruction step that turns a parameter mapping into an `LLMConfig`, converting sampled strings back to the field's engine-neutral enum class and inheriting the base config's non-tunable fields.

[x] **T4 - Implement the trial objective**: Add the objective that runs one benchmark for the trial's rebuilt configuration through `run_experiment` with the injected backend and client, returns throughput, records the serializable result projection on a successful trial, and records the error and re-raises on an `ExperimentError`.

[x] **T5 - Implement study creation and the `optimize` entry point**: Add `optimize` with the agreed signature: it creates the study with the default seeded TPE sampler, maximize direction, optional storage and study name with resume-on-existing, runs the trials with the `n_jobs` passthrough, and returns an `OptimizationResult`; when no trial completes it raises `OptimizationError` carrying the study.

[x] **T6 - Export the public optimization API**: Re-export `optimize`, `OptimizationResult`, and `OptimizationError` from the optimization package initializer so callers import them from the subpackage.

[x] **T7 - Unit-test the space-to-trial mapping and reconstruction**: Add tests for each spec kind's suggestion, the recording of a fixed value in the trial parameters, the unset field for a `Default` dimension, enum round-tripping including an attention backend, and reconstruction of an `ExperimentConfig` from a parameter mapping.

[x] **T8 - Unit-test the study lifecycle with fakes**: Add no-daemon tests using the fake inference backend and fake Podman client covering best-config reconstruction, the recorded result projection, a failed trial continuing the study, `OptimizationError` when no trial completes, the in-memory default, the storage and study-name path, the seed argument, and the `n_jobs` passthrough.
    - Note: the fake inference backend was not in the test support module as assumed; it lived in `tests/inference/test_orchestrate.py`. It was promoted to `tests/fakes.py` (and imported back into the orchestration tests) so both suites share one double. The base experiment forces the CPU target, which short-circuits hardware detection without a probe.

[x] **T9 - Add the real-sweep integration smoke test**: Add an `integration`-marked test that runs `optimize` with a two-trial budget over a mostly-fixed space against a real backend and asserts the result and study shape, skipped when Podman is unavailable.
    - Note: the varying dimension is `max_num_seqs` (`Categorical((2, 4))`), not `max_model_len`. Constraining `max_model_len` to a value below vLLM's benchmark prompt+output length trips `run_vllm`'s assertion (`max_model_len` must exceed prompt plus expected output), so `max_model_len` is pinned to the model default of 2048 instead. The test calls `validate_space` first, exercising the documented caller-side validation step.

[x] **T10 - Run the manual spike and record the evaluation verdict**: Execute the documented manual spike (resumption from a persisted study, `n_jobs` with caller-managed device isolation, and a fuller trial count), record the pass/fail result for each evaluation criterion, and write the go/no-go verdict into the plan.
    - Note: the spike ran on the CPU backend, and the GPU criterion was then satisfied with a real ROCm sweep on the RX 9070 XT (gfx1201). The first ROCm attempt failed with an HSA `Memory in use` error; the root cause was SELinux labeling on this enforcing host, not the ROCm version — disabling container labeling makes the GPU usable, and `run_experiment` then returned ≈6607 TPS. Because the runtime layer exposes no SELinux/IPC option, the GPU sweep was run against a temporary podman service configured with `[containers] label = false`; a runtime option (or a documented host requirement) is a follow-up. `n_jobs` was exercised, but disjoint-device isolation was not, since the host has a single GPU. The verdict is GO; the procedure, per-criterion results, and reasoning are recorded under Further Notes.

## Out of Scope

- Any optimizer other than Optuna; alternative frameworks are named only as
  fallbacks if Optuna is rejected.
- Multi-objective optimization, constraints, and pruning; the objective is the
  single value `tokens_per_second`, maximized, and no pruner is configured.
- A caller-supplied objective callable or a metric enum; other metrics can be
  added later if needed.
- Float-typed tunables and float intervals; every numeric tunable remains an
  integer.
- Parallel device scheduling and GPU assignment; `n_jobs` is a passthrough and
  the caller keeps concurrent trials on disjoint devices.
- A command-line interface or console entrypoint.
- A fixed-values YAML definition for the container, benchmark, and non-tunable
  LLM values; the base `ExperimentConfig` is supplied in code.
- Changing the search-space, inference, or runtime layers, including the tunable
  registry and the backend validation hook.
- Streaming partial benchmark results to enable pruning or early abort.

## Further Notes

- **Evaluation criteria and verdict: GO.** Optuna is adopted. Every criterion
  was checked and the result recorded:

  | Criterion | Result | Evidence |
  | --- | --- | --- |
  | All four spec kinds map faithfully | PASS | `tests/optimization/test_mapping.py`; in the manual sweep `trial.params` carried the `Fixed` values, sampled the `Categorical` and `Interval` dimensions, and omitted both `Default` dimensions, whose fields reconstructed as `None`. |
  | Installs and imports on Python 3.14 | PASS | Optuna 5.0.0 on CPython 3.14.7, eleven transitive packages, no SciPy (T1). |
  | A study persisted to SQLite resumes by name | PASS | Manual spike section 1: one trial, then a second call resumed the same study (`study_name='spike'`, trial numbers `[0, 1]`, SQLite file present). |
  | `n_jobs` runs with caller-managed device isolation | PASS (caveat) | Manual spike section 2: `n_jobs=2` completed both trials. The host has one GPU, so disjoint-device isolation was not exercisable; the CPU run showed the parallel trials contending (≈25 TPS each against ≈57 TPS sequential), which is exactly the distortion caller-side device isolation exists to avoid. |
  | Dependency footprint acceptable | PASS | Eleven transitive packages and no SciPy; Optuna is a runtime dependency declared explicitly (T1). |
  | A real small GPU sweep yields a usable best configuration | PASS (caveat) | A 3-trial sweep through `optimize` on the RX 9070 XT (gfx1201) with `docker.io/vllm/vllm-openai-rocm:v0.18.1` completed every trial; best ≈4239 TPS (`max_num_seqs=8`, `max_num_batched_tokens=3875`), and a single `run_experiment` measured ≈6607 TPS. It required the container to run without SELinux labeling (see below); the project's runtime exposes no such option, so the sweep used a temporary podman service. |

  The GPU criterion initially failed for an environment reason unrelated to
  Optuna: on this SELinux-enforcing host the container's HSA runtime aborts with
  `Memory critical error by agent node-0 … Reason: Memory in use` even for a
  one-element allocation, while the host's own ROCm userspace runs unconfined
  and allocates fine. Disabling container labeling (`[containers] label = false`)
  fixes it, so the earlier "ROCm version mismatch" reading was wrong — ROCm 7.0
  in a container and the host's 7.1 userspace both work once the label is off.
  The project's runtime layer has no field for SELinux or IPC options, so
  running a GPU sweep through `optimize` needs either that option or a host
  podman default; the gap is recorded here rather than fixed, since runtime
  changes are out of scope.

- **Manual spike (documented procedure and results).** The three slow,
  hardware-dependent checks are a manual procedure rather than a committed test.
  It was run on 2026-10-15 against the real vLLM CPU backend through Podman
  (`docker.io/vllm/vllm-openai-cpu:latest`, `facebook/opt-125m`, 4 random
  prompts at 32 in / 8 out tokens) from a scratch script in the dev shell with
  `uv run python <script>`:

  ```python
  import tempfile
  from pathlib import Path

  from podman import PodmanClient

  from llm_optimization.inference import (
      BenchmarkConfig,
      ContainerRuntimeConfig,
      DType,
      ExperimentConfig,
      HardwareTarget,
      KVCacheDType,
      LLMConfig,
      VLLMBackend,
  )
  from llm_optimization.optimization import optimize
  from llm_optimization.search_space import (
      Categorical,
      Default,
      Dimension,
      ExperimentSpace,
      Fixed,
      Interval,
      validate_space,
  )

  IMAGE = "docker.io/vllm/vllm-openai-cpu:latest"
  BASE = ExperimentConfig(
      container=ContainerRuntimeConfig(target=HardwareTarget.CPU, image=IMAGE),
      benchmark=BenchmarkConfig(
          model="facebook/opt-125m",
          dataset_name="random",
          num_prompts=4,
          input_len=32,
          output_len=8,
          timeout=900.0,
      ),
      llm=LLMConfig(dtype=DType.FLOAT32, gpu_memory_utilization=0.3, seed=0),
  )
  SPACE = ExperimentSpace(
      dimensions=(
          Dimension("dtype", Fixed(DType.FLOAT32)),
          Dimension("quantization", Default()),
          Dimension("kv_cache_dtype", Fixed(KVCacheDType.AUTO)),
          Dimension("max_model_len", Fixed(2048)),
          Dimension("max_num_seqs", Categorical((2, 4, 8))),
          Dimension("max_num_batched_tokens", Interval(2048, 4096)),
          Dimension("tensor_parallel_size", Fixed(1)),
          Dimension("attention_backend", Default()),
      )
  )

  client = PodmanClient.from_env()
  backend = VLLMBackend(cpu_image=IMAGE)
  validate_space(SPACE, backend)
  with tempfile.TemporaryDirectory() as directory:
      storage = f"sqlite:///{Path(directory) / 'spike.db'}"
      # 1. resumption from a persisted study
      first = optimize(
          SPACE,
          BASE,
          n_trials=1,
          seed=0,
          storage=storage,
          study_name="spike",
          backend=backend,
          client=client,
      )
      second = optimize(
          SPACE,
          BASE,
          n_trials=1,
          seed=0,
          storage=storage,
          study_name="spike",
          backend=backend,
          client=client,
      )
      assert (first.trial_count, second.trial_count) == (1, 2)
      assert second.study.study_name == "spike"
      # 2. n_jobs passthrough (one device here; isolation is the caller's job)
      parallel = optimize(
          SPACE,
          BASE,
          n_trials=2,
          seed=1,
          storage=storage,
          study_name="parallel",
          n_jobs=2,
          backend=backend,
          client=client,
      )
      assert parallel.trial_count == 2
      # 3. fuller sweep
      full = optimize(SPACE, BASE, n_trials=4, seed=7, backend=backend, client=client)
      assert full.trial_count == 4 and full.best_value > 0
  client.close()
  ```

  Observed results (all PASS):

  | Section | Observation | Wall time |
  | --- | --- | --- |
  | Resumption | `first=1`, `second=2`, trial numbers `[0, 1]`, SQLite file present | 401 s |
  | `n_jobs=2` | both trials `COMPLETE`, ≈25 TPS each (contended against ≈57 TPS sequential) | 366 s |
  | Fuller sweep (4 trials) | best ≈57.7 TPS; winner `max_num_seqs=4`, `max_num_batched_tokens=3530`; `Default` fields `None`, `Fixed` and non-tunable fields honoured | 790 s |

  The ROCm sweep used the same shape with `target=HardwareTarget.ROCM`,
  `docker.io/vllm/vllm-openai-rocm:v0.18.1`, and `DType.FLOAT16`, over a space
  varying `max_num_seqs` (`Categorical((4, 8, 16))`) and
  `max_num_batched_tokens` (`Interval(2048, 4096)`), with `n_trials=3`. Because
  the host is SELinux-enforcing and the HSA runtime aborts under the default
  container label, a temporary podman service with labeling off is used:

  ```bash
  printf '[containers]\nlabel = false\n' > /tmp/spike/conf_label.conf
  CONTAINERS_CONF=/tmp/spike/conf_label.conf \
      podman system service --time=0 unix:///tmp/spike/podman.sock &
  # Point PodmanClient at unix:///tmp/spike/podman.sock, then run optimize.
  ```

  Observed: three `COMPLETE` trials, best ≈4239 TPS (`max_num_seqs=8`,
  `max_num_batched_tokens=3875`); the recorded projection carried
  `target='rocm'` and `devices=['/dev/kfd', '/dev/dri']`. A single
  `run_experiment` on the same image measured ≈6607 TPS. The scratch scripts,
  including the PASS/FAIL reporting, the ROCm probe, and the sweep, are not
  committed; the procedure above is the committed record.
- **Resume semantics.** Resuming a study runs the requested number of *new*
  trials on top of the existing ones; the trial count is a budget for the call,
  not a cap on the study. This is documented rather than changed.
- **Why no pruner.** Optuna's pruners decide whether to stop a trial early from
  intermediate objective values reported during the trial. A benchmark run is
  all-or-nothing and the runtime layer exposes no partial-result channel, so
  there is nothing to report and nothing to prune. The stopping controls are the
  trial count and the timeout.
- **Why categorical choices are strings.** Studies are persisted and reloaded,
  and persisted parameter values come back as plain strings. Passing enum
  members would work in memory but would not survive a round trip; passing
  values and converting back at reconstruction time keeps in-memory and resumed
  studies identical.
- **Why the adapter does not validate.** The search-space layer already provides
  an explicit engine-support validation step that needs a backend. Keeping it in
  the caller's hands makes the checked engine visible and keeps the adapter from
  duplicating policy; the cost is that a caller who forgets it discovers an
  unsupported space on the first trial rather than before the sweep.
- **Layering rationale.** Splitting the neutral result and error types from the
  Optuna module means a second optimizer can be added behind the same
  `OptimizationResult` without changing callers, mirroring how the inference
  layer separates engine-neutral types from the vLLM backend.
