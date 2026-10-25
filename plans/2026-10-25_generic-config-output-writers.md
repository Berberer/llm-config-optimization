# Generic Config Output Writers (PRD)

## Problem Statement

The runner can only write the winning configuration to a YAML file. That YAML is a description of an experiment, not something a user can hand straight to a container runtime: it does not contain the engine command, environment, mounts, ports, devices, or resource settings a trial actually ran with. A user who wants to deploy the winning configuration must manually translate the YAML into a Podman quadlet or a Docker Compose file, re-deriving engine flags and host paths by hand. This is slow, error-prone, and can diverge from what was benchmarked.

## Solution

Replace the YAML-only writer with a generic, engine-neutral config output system. The existing `--output-config` path stays, and a new explicit `--output-format` selects the artifact to produce:

- `yaml` — the current three-domain experiment document (default, unchanged behaviour).
- `quadlet` — a Podman `.container` unit describing the server that actually ran.
- `compose` — a Docker/Podman Compose file describing the same server.

For the executable formats, the runner re-plans the winning configuration through the selected inference backend so the generated artifact carries the exact image, command, entrypoint, environment, mounts, published port, devices, resource limits, pull policy, and SELinux options of the trial. The artifact also gets a healthcheck derived from the backend's declared health endpoint. The system lives in a new `output` package that knows nothing about engines or container runtimes beyond the engine-neutral experiment and container types.

A plausible-looking file can still be rejected by the tool that consumes it, so the writers are additionally covered by a hermetic conformance layer in the default test suite and by opt-in integration tests that run the generated artifacts through the real Podman and Compose tooling.

## User Stories

1. As a benchmark operator, I want to keep writing the winning configuration to YAML, so that existing scripts and workflows continue to work unchanged.
2. As a benchmark operator, I want to choose the output format explicitly with `--output-format`, so that I am never surprised by a format guessed from a filename.
3. As a benchmark operator, I want `--output-config` without `--output-format` to default to YAML, so that my current invocations keep working.
4. As a benchmark operator, I want giving `--output-format` without `--output-config` to be a usage error, so that a forgotten path does not silently produce no file.
5. As a benchmark operator, I want one artifact per invocation, so that the command's behaviour and output remain simple and predictable.
6. As a benchmark operator, I want to write whatever output path I choose, regardless of its suffix, so that the explicit format remains the single source of truth.
7. As a benchmark operator, I want a Podman quadlet for the winning configuration, so that I can drop it into my systemd user directory and start the exact server that was benchmarked.
8. As a benchmark operator, I want a Docker Compose file for the winning configuration, so that I can bring the same server up with Compose tooling.
9. As a benchmark operator, I want the executable artifact to use the same container image that ran during the sweep, so that I do not accidentally deploy a different image.
10. As a benchmark operator, I want the engine command and entrypoint reproduced faithfully, so that the deployed server behaves like the benchmarked one.
11. As a benchmark operator, I want the plan's environment variables reproduced, so that engine-side settings such as dev mode are not lost.
12. As a benchmark operator, I want the plan's host mounts reproduced with their exact host paths, so that the artifact runs offline on the same host without re-downloading the model.
13. As a benchmark operator, I want the model snapshot, draft model, and persistent Hugging Face cache mounts included, so that the deployed server can load the same checkpoints as the trial without depending on benchmark scratch space.
14. As a benchmark operator, I want the server port published on the same host port as the container port, so that the artifact is self-contained and the endpoint is predictable.
15. As a GPU user, I want the plan's device mappings written verbatim, so that CUDA and ROCm passthrough survive the translation into either format.
16. As a benchmark operator, I want shared-memory size, memory limit, and pull policy carried into the artifact, so that runtime resource behaviour matches the sweep.
17. As a user on an SELinux host, I want the plan's SELinux options preserved, so that a configuration that needed relabelling disabled still starts.
18. As a benchmark operator, I want lighter-weight Podman SELinux options mapped to the quadlet's native keys, and anything else passed through as Podman arguments, so that no label configuration is silently dropped.
19. As a benchmark operator, I want a healthcheck in the generated artifacts, so that the runtime can report readiness the same way the benchmark client waited for it.
20. As a benchmark operator, I want the healthcheck built from the backend's real health path, so that readiness means the engine is serving, not merely that a socket is open.
21. As a benchmark operator, I want the healthcheck probe to use `curl`, so that it works with the official vLLM and llama.cpp images that both ship `curl`.
22. As a benchmark operator, I want the healthcheck start period to match the configured startup timeout, so that a slow model load is not marked unhealthy while it is still starting.
23. As a benchmark operator, I want a fixed probing interval and retry count, so that generated files stay simple and deterministic.
24. As a benchmark operator, I want the quadlet to be minimal and not start at boot, so that a benchmark artifact does not unexpectedly become a persistent service.
25. As a benchmark operator, I want the quadlet to restart the server on failure, so that a transient engine crash does not require manual intervention during an evaluation.
26. As a Docker Compose user, I want a modern Compose file without the obsolete `version:` key, so that Compose does not emit warnings.
27. As a Docker Compose user, I want the service keyed `server`, so that the file is engine-neutral and easy to rename.
28. As a benchmark operator, I want the Podman-only `newer` pull policy translated to `always` in Compose, so that the generated file is accepted by Compose and still checks the registry.
29. As a benchmark operator, I want a comment explaining that pull-policy translation, so that I understand why the generated value differs from the YAML document.
30. As a benchmark operator, I want generated files to contain no provenance header or timestamp, so that they are deterministic and diff-friendly.
31. As a benchmark operator, I want the YAML artifact to keep exactly the same three-domain content and layout as before, so that existing consumers and documentation stay valid.
32. As a benchmark operator, I want a failed planning step for an executable artifact to fail the command with a non-zero exit, so that I know the file I asked for was not produced.
33. As a benchmark operator, I want YAML output never to invoke backend planning, so that it keeps its current side effects and failure modes.
34. As a library user, I want writers to be pure renderers that return text, so that I can test and reuse them without touching the filesystem.
35. As a maintainer, I want a writer registry keyed by output format, so that adding a fourth format later is a small, isolated change.
36. As a maintainer, I want the output layer to depend only on engine-neutral experiment and container types, so that no engine vocabulary or backend protocol leaks into it.
37. As a maintainer, I want the composition root to own planning and file writing, so that the output layer has no backend dependency and no planning side effects.
38. As a maintainer, I want generated artifacts covered by focused unit tests using hand-built server specs, so that format behaviour is verified without a container daemon.
39. As a benchmark operator reading the help text, I want each format's conventional filename documented, so that I can choose sensible output names even though suffixes are not enforced.
40. As a benchmark operator, I want the summary printed at the end of a sweep to keep showing the winning configuration, so that choosing an executable format does not remove the human-readable result.
41. As a benchmark operator, I want generated quadlets to use only keys Podman's quadlet generator understands, so that the unit actually generates instead of being ignored.
42. As a benchmark operator, I want a multi-option entrypoint encoded the way quadlet requires, so that the deployed server starts the same command the benchmark ran.
43. As a maintainer, I want the writers' output checked against Podman's documented key set in the default test suite, so that a typo like `Device=` instead of `AddDevice=` fails fast without external tools.
44. As a maintainer, I want opt-in integration tests that run generated artifacts through `docker compose config`, `podman-compose config`, and the Podman quadlet generator, so that acceptance is verified by the real tools.
45. As a maintainer, I want the CDI/Compose compatibility matrix encoded in tests, so that the documented Podman-only limitation is executable and cannot silently regress.
46. As a maintainer, I want validation helpers and tests to skip cleanly when a tool is absent, so that the default suite stays hermetic and integration runs work on hosts with partial tooling.

## Implementation Decisions

- Create a new top-level `output` package. It exposes `OutputFormat` (`StrEnum`: `yaml`, `quadlet`, `compose`), a render context carrying the recommended `ExperimentConfig` plus resolved `HardwareTarget` and image, an optional `ServerSpec`, a `ConfigWriter` protocol with `requires_plan` and `render`, and a registry mapping each format to a writer instance.
- Writers are pure: they return a string. The composition root owns truncating/overwriting the target path and never creates parent directories, matching current behaviour.
- The YAML renderer moves out of the CLI summary module into the output package. Its document shape is unchanged: curated `container` section (`target`, `image`, `shm_size`, `mem_limit`, `pull_policy`), full `benchmark` section, full `llm` section; enum values reduced to strings, paths to strings, unset values emitted as explicit `null`, insertion order preserved.
- The CLI summary module keeps `render_summary` and delegates the YAML document/dump work to the registry-selected YAML writer. Both summary and file continue to describe the same winning document, and the composition root never imports a concrete writer.
- Add `--output-format` with choices `yaml`, `quadlet`, `compose`, defaulting to `yaml`. Giving the flag without `--output-config` is a usage error with the CLI's standard usage exit code.
- No filename suffix validation: the explicit format is authoritative. Help text documents the conventional suffixes.
- The finish path receives the engine's deployment-backend factory and the detected hardware and selects the writer through `writer_for`. It renders and prints the summary as before. For YAML it writes from config plus resolved target/image only, since the writer declares `requires_plan = False`. For `quadlet` and `compose` it plans the recommended config with the factory's backend, then renders and writes. The deployment backend omits trial-scoped scratch resources such as vLLM's temporary compile cache, so every path in the artifact outlives the run. A planning or writing failure is reported and exits non-zero, including on the interrupted-sweep path.
- The executable writers render the planned `ServerSpec` faithfully:
  - Image, command, entrypoint (when set), environment, mounts with exact host paths, devices verbatim (Podman CDI strings such as `nvidia.com/gpu=all` and `/dev/kfd`/`/dev/dri`), memory limit, shared-memory size, pull policy, and SELinux options.
  - A planned host port of `None` is replaced by the container port, producing `container:same` publishing in both formats.
- Quadlet output sections: `[Unit]` with a description; `[Container]` with the container keys and healthcheck keys; `[Service]` with `Restart=on-failure`; no `[Install]` section. `Exec` carries the command, `Entrypoint` is emitted only when the plan sets one and is encoded as the JSON array quadlet requires for a multi-option entrypoint, and devices use the `AddDevice=` key.
- Quadlet SELinux rendering: `disable` maps to `SecurityLabelDisable=true`; any other option is emitted as a `PodmanArgs` security-opt argument.
- Compose output is a modern Compose document: no `version:` key, one service keyed `server`, list-form `command`/`entrypoint`, mapping-form environment, volumes, ports, devices, `shm_size`, `mem_limit`, `security_opt`, `pull_policy`, and `healthcheck`.
- Compose pull policy: `newer` is translated to `always` with an adjacent comment; the other values are emitted as-is.
- Healthcheck: probe `curl -fsS http://127.0.0.1:<container-port><health_path>`. Start period equals `ServerSpec.health_timeout`; interval is fixed at 10 seconds; retries are fixed at 3. Quadlet uses the native health keys, Compose uses `healthcheck`.
- The healthcheck uses `curl` because the official vLLM and llama.cpp images (including their CPU/ROCm variants) were checked and all contain `curl`, while none expose `nc`.
- Generated files carry no provenance header or timestamp. The only comment is the Compose pull-policy translation note.
- The `output` package must import no inference backend, no engine module, and no container runtime implementation. It may import the engine-neutral experiment and runtime container types only.
- Compose keeps device strings verbatim. CDI entries (`nvidia.com/gpu=all`) are Podman-targeted; the validation matrix asserts they pass `podman-compose` and records the Docker limitation as an expected failure rather than changing the artifact semantics.
- Artifact correctness is validated in two layers. The default suite owns a hermetic conformance check: a quadlet `[Container]` key allowlist derived from Podman's documented keys, value-format checks (JSON entrypoint array, non-empty health command), and Compose structural plus device-class checks. Real-tool validation lives in `integration`-marked tests that shell out to `docker compose config`, `podman-compose config`, and the Podman quadlet generator with `QUADLET_UNIT_DIRS`; each skips cleanly when its tool is absent. No runtime code invokes a tool, and no new runtime dependency is added.
- Documentation updates: usage guide (flag, formats, examples, one-artifact rule), architecture guide (new layer and data-flow step), project README (mention executable outputs), and the repository's agent guide (package table, invariants, data flow).

## Testing Decisions

- Test external behaviour, not implementation details. For writers, assert on the rendered document (parsed YAML/INI structure or exact text where the format is line-oriented) rather than on private helpers.
- Build `ServerSpec` fixtures by hand so formatter tests need neither a backend nor a container daemon. The existing planning tests are prior art for constructing realistic specs; the existing summary tests are prior art for asserting the YAML document.
- YAML tests re-verify the three-domain document, curated container fields, explicit nulls, enum/path coercion, and ordering after the move. A compatibility assertion guards against accidental output changes.
- Quadlet tests cover section presence, image/command/entrypoint, env quoting, volume syntax, published ports, verbatim devices, resource limits, pull policy, SELinux mapping, healthcheck, restart policy, and the absence of `[Install]`.
- Compose tests cover the single `server` service, absence of `version`, command/entrypoint list form, environment mapping, ports, volumes, devices, limits, SELinux, the `newer`→`always` translation, and healthcheck shape.
- CLI argument tests cover the format default, valid choices, and the usage error for format without path.
- CLI finish tests cover: YAML never calls planning; quadlet and compose call planning once; the correct writer is selected; planning failure and write failure exit non-zero.
- Summary tests continue to cover scalar and multi-objective rendering and now import the moved YAML renderer through the output package.
- Hermetic conformance tests parse the generated quadlet and fail on any `[Container]` key outside the documented allowlist; they also assert the entrypoint JSON shape and the health command form. They would have caught the `Device=`/`AddDevice=` typo and the shell-joined entrypoint.
- Integration tests, marked `integration`, run the generated files through real tooling: `docker compose config --quiet` and `podman-compose config` for Compose, and `QUADLET_UNIT_DIRS=<tmp> <podman-system-generator> --user --dryrun` (optionally followed by `systemd-analyze --user --generators=true verify`) for quadlet. Each skips when its binary is absent.
- The CDI compatibility matrix is explicit: a portable spec must pass both Compose providers; a CDI spec must pass `podman-compose` and is expected to fail Docker validation, so the documented limitation stays executable.
- Tooling tests use only temporary directories and environment isolation, never the developer's real `~/.config/containers/systemd`.
- Run the project's formatter, linter, type checker, and full unit suite as the final gate, and run `uv run pytest -m integration` where the container tooling is available.

## Implementation Tasks

[x] **T1 - Introduce the output package contract**: Create the `output` package with `OutputFormat`, the render context carrying config, resolved target/image, and an optional server spec, the `ConfigWriter` protocol (`requires_plan`, `render`), and an empty/registry-ready public surface. No CLI behaviour changes yet.

[x] **T2 - Move the YAML renderer into the output package**: Relocate the three-domain document construction, curated container fields, enum/path coercion, and block-style dump from the CLI summary module into the output package as the YAML writer. Rewire the summary module to select it through the registry so the printed and written configuration share the registered instance. Keep the rendered document byte-compatible with the current output and keep all summary tests passing.

[x] **T3 - Implement the Podman quadlet writer**: Add a quadlet renderer that consumes a planned server spec and emits the minimal unit with `[Unit]`, `[Container]`, and `[Service]`, including command/entrypoint, environment, volumes, published port resolution, verbatim devices, resource limits, pull policy, SELinux mapping, and the curl-based healthcheck. Add focused formatter tests with hand-built specs.

[x] **T4 - Implement the Docker Compose writer**: Add a Compose renderer that consumes a planned server spec and emits a modern single-service document without `version`, including the same container data, the `newer`→`always` translation with an explanatory comment, and the healthcheck. Add focused formatter tests.

[x] **T5 - Register the writers and expose selection**: Populate the writer registry with the YAML, quadlet, and compose renderers and expose a selection entry point that maps an `OutputFormat` to its writer. Add tests that every format is registered and that writers correctly declare whether they require planning.

[x] **T6 - Add the `--output-format` CLI option**: Add the flag with the three choices, default `yaml`, update help and the module's usage example, and reject `--output-format` without `--output-config` as a usage error. Extend the argument tests accordingly.

[x] **T7 - Wire format selection into the finish path**: Pass the deployment-backend factory and detected hardware to the finish path on both normal and interrupted runs. Keep YAML on its current config-only input. For quadlet and compose, plan the recommended configuration through the deployment backend and render through the writer selected from the registry. Report planning/write failures and exit non-zero. Add CLI-level tests for writer selection, planning calls, and failure exits.

[x] **T8 - Update documentation and the project guide**: Document the new flag and formats in the usage guide with examples, describe the new output layer and data-flow step in the architecture guide, mention executable outputs in the README, and update the agent guide's package table, invariants, and data flow.

[x] **T9 - Run the full verification gate**: Run the formatter, lint, type checker, and unit test suite; fix any regressions introduced by the refactor or new writers.

[x] **T10 - Correct the quadlet keys and value formats**: Emit `AddDevice=` for devices and a JSON array for `Entrypoint`, because quadlet forwards the entrypoint as one `--entrypoint=` argument and only splits a JSON array. Update the exact-text tests. These defects were found while preparing the validation workstream.

[x] **T11 - Add hermetic artifact conformance checks**: Add default-suite tests that parse the rendered quadlet and reject any `[Container]` key outside the documented allowlist, assert the entrypoint JSON shape and health command form, and check the Compose document structure and device classification. The allowlist lives in the tests, not in the writer, so the production layer stays engine- and tool-neutral.

[x] **T12 - Add real-tool integration validation**: Add `integration`-marked tests that validate rendered artifacts with `docker compose config`, `podman-compose config`, and the Podman quadlet generator (`QUADLET_UNIT_DIRS` + `--dryrun`, optionally `systemd-analyze verify`). Locate tools with a shared helper and skip when absent. Encode the CDI matrix: portable specs pass every available provider, CDI specs pass `podman-compose` and are expected to fail Docker validation.

[x] **T13 - Document and run the extended gate**: Document the validated tool versions and the Compose/CDI limitation in the usage guide, note the conformance layer in the agent guide's testing conventions if it needs one, then run the formatter, linter, type checker, unit suite, and the integration validation where tooling is installed.

## Out of Scope

- Additional output formats beyond YAML, Podman quadlet, and Docker Compose (for example raw `podman run` scripts, Kubernetes manifests, or systemd unit installation).
- Writing multiple artifacts in one invocation, per-format output paths, or a `FORMAT:PATH` syntax.
- Inferring the format from the filename or enforcing filename suffixes.
- Installing, enabling, starting, or reloading generated units and compose stacks.
- Adding `[Install]`/boot autostart to generated quadlets.
- Changing optimization, search-space, inference, or runtime behaviour.
- Generating artifacts for non-winning trials or for the whole Pareto front.
- Adding healthcheck tool fallbacks such as `nc`, `wget`, or custom probe commands; `curl` is assumed.
- Emitting detailed provenance metadata (study name, trial number, timestamps) in generated files.
- Changing the Compose writer's device representation to Docker-specific GPU reservations; the validation matrix records the CDI limitation instead.
- Invoking Podman or Compose from production code or at render time.
- Requiring Docker or Podman tooling for the default unit suite.

## Further Notes

- The planned spec is the only place the engine command, environment, and mounts exist; reconstructing them from the YAML document would duplicate engine knowledge outside the backend modules. This is why executable outputs re-plan the winning config through the selected backend.
- Planning is not side-effect free (it may ensure host cache directories exist), which is why YAML deliberately keeps its config-only path and only executable formats pay the planning cost.
- The planned port mapping uses a runtime-assigned host port (`None`) during a trial. Generated artifacts replace that with the container port so the file is self-contained and the endpoint is predictable.
- The verbatim device approach targets Podman and `podman-compose`, which pass CDI strings such as `nvidia.com/gpu=all` through unchanged. Docker Compose versions that do not understand CDI device strings are the documented limitation; the validation matrix records that as an expected failure rather than changing the artifact semantics.
- The `newer` pull policy is Podman-specific; the Compose translation to `always` is a documented approximation, not an exact semantic match.
- The interrupted-sweep path must follow the same output semantics as the normal path, including non-zero exit on a failed executable artifact.
- The `Device=`/`AddDevice=` and shell-joined `Entrypoint` defects are the motivating evidence for the validation layer: exact-text tests pin whatever the writer emits, so they cannot distinguish correct output from plausible output. The conformance layer catches unknown keys and value-format mistakes; the integration layer catches tool-level rejection.
- `Exec` uses systemd command-line quoting via `shlex.join`. Systemd specifiers (`%`) remain an edge case, so a model path containing `%` would need escaping; a conformance check can cover this later.
- Compose's double-quoted scalars and list-form `CMD` healthcheck are valid Compose and need no change. The unresolved risk is the CDI device form, which is why the compatibility matrix is explicit rather than inferred from prose.
- Validation is additive: it changes no writer semantics, no CLI behaviour, and no YAML document.
