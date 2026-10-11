# Podman Container Runner Utility

## Problem Statement

The `llm-optimization` project needs to execute workloads (inference, evaluation, benchmarks) inside isolated containers. Today there is no way to launch a container from code: the package is an empty skeleton, and every experiment would have to hand-roll its own Podman invocation, argument construction, output capture, and cleanup.

Doing this by hand is error-prone. Podman's Python bindings have sharp edges that are easy to get wrong — image pulling only happens automatically when an image is missing, container logs merge stdout and stderr, and the high-level `run()` helper discards stdout when a container exits non-zero. A developer who just wants to run a container from a declarative description should not have to rediscover these behaviors, and every caller should get consistent exit-code, output, timeout, and cleanup semantics.

## Solution

Provide a small, importable library module that runs a Podman container from a declarative configuration object and returns a structured result.

The caller builds a `ContainerConfig` describing the image, command, environment, volume mounts, GPU devices, memory and shared-memory limits, pull policy, and an optional timeout. Calling `run_container(config)` starts the container, blocks until it exits (or the timeout elapses), captures stdout and stderr separately, removes the container, and returns a `ContainerResult` with the exit code and both output streams. Failures — a non-zero exit code, a timeout, or a Podman API error — are surfaced as a single `ContainerRunError` that still carries the partial result and original cause.

The module is deliberately backend-agnostic at the type level: the configuration, result, and error types live in a `runtime` subpackage's shared types module, while the Podman-specific execution lives in a separate backend module. This keeps the door open for a future Docker or Apptainer backend without reshaping the public API.

## User Stories

1. As an LLM-optimization developer, I want to run a container from a single Python call, so that I can launch workloads without writing shell or Podman plumbing.
2. As an LLM-optimization developer, I want to describe the container declaratively with a typed configuration object, so that my call sites are readable and validated.
3. As an LLM-optimization developer, I want to specify the image by name and tag, so that I can choose the exact runtime environment for a workload.
4. As an LLM-optimization developer, I want to specify the command to run inside the container, so that I can execute an experiment entrypoint.
5. As an LLM-optimization developer, I want to omit the command and fall back to the image's default entrypoint, so that pre-baked images work without extra configuration.
6. As an LLM-optimization developer, I want to pass environment variables as a mapping, so that I can configure the workload without rebuilding an image.
7. As an LLM-optimization developer, I want to mount host directories into the container, so that I can supply datasets and models.
8. As an LLM-optimization developer, I want to mount some directories read-only, so that inputs cannot be accidentally modified by a container.
9. As an LLM-optimization developer, I want to pass extra mount options, so that I can handle relabeling and access flags required by the host.
10. As an LLM-optimization developer, I want to pass GPU devices through to the container, so that inference and training can use hardware acceleration.
11. As an LLM-optimization developer, I want to express GPU access as raw device strings, so that I am not locked to a single vendor or CDI scheme.
12. As an LLM-optimization developer, I want to set a memory limit, so that a runaway workload cannot exhaust host RAM.
13. As an LLM-optimization developer, I want to set the shared-memory size, so that frameworks like PyTorch, vLLM, and NCCL that rely on `/dev/shm` work correctly.
14. As an LLM-optimization developer, I want to choose an image pull policy, so that I can force a re-pull of a mutable tag when needed.
15. As an LLM-optimization developer, I want to forbid network pulls in restricted or offline environments, so that runs do not unexpectedly try to reach a registry.
16. As an LLM-optimization developer, I want the call to block until the container finishes, so that I can reason about results sequentially.
17. As an LLM-optimization developer, I want the container's exit code returned to me, so that I can branch on success or failure.
18. As an LLM-optimization developer, I want stdout and stderr captured separately, so that I can parse structured results while keeping diagnostics distinct.
19. As an LLM-optimization developer, I want output available as raw bytes, so that binary or non-UTF-8 output is preserved losslessly.
20. As an LLM-optimization developer, I want convenience accessors that decode output as text, so that the common case requires no manual decoding.
21. As an LLM-optimization developer, I want a non-zero exit to raise a single, well-defined exception, so that I can handle failures uniformly.
22. As an LLM-optimization developer, I want that exception to carry the captured output and exit code, so that I can still inspect what the container produced before it failed.
23. As an LLM-optimization developer, I want Podman API failures (socket down, image pull denied, invalid config) surfaced through the same exception type, so that I only import one error class.
24. As an LLM-optimization developer, I want the original underlying error preserved as the cause, so that I can debug low-level Podman issues.
25. As an LLM-optimization developer, I want to set an optional wall-clock timeout, so that a hung workload does not block forever.
26. As an LLM-optimization developer, I want a timed-out run to be clearly distinguished, so that I can tell a timeout apart from an ordinary failure.
27. As an LLM-optimization developer, I want partial output from a timed-out run, so that I can see how far the workload got.
28. As an LLM-optimization developer, I want the container removed after every run, so that repeated experiments do not leave containers piling up.
29. As an LLM-optimization developer, I want cleanup to happen even when a run fails or times out, so that the host stays tidy.
30. As an LLM-optimization developer, I want the container identifier available on the result, so that I can correlate runs with logs or metrics.
31. As an LLM-optimization developer, I want the connection to Podman discovered automatically from the environment, so that the common local case needs no configuration.
32. As an LLM-optimization developer, I want to inject a pre-configured Podman client, so that I can target a remote host or a non-default socket.
33. As a test author, I want to inject a fake client, so that I can test the runner's behavior without a running Podman daemon.
34. As a test author, I want the runner to close only clients it created itself, so that an injected client stays usable by the caller.
35. As a CI maintainer, I want integration tests that skip gracefully when Podman is unavailable, so that the suite passes on machines without containers.
36. As a future maintainer, I want the config/result/error types separated from the Podman backend, so that adding another container backend does not break the public API.
37. As a future maintainer, I want the runner exposed through a stable subpackage import path, so that internal file layout can change without breaking callers.

## Implementation Decisions

**Package layout**
- A new `runtime` subpackage under the existing `llm_optimization` package.
- The subpackage contains three modules: a shared `types` module, a Podman backend module, and an `__init__` that re-exports the public API.
- Public import path is the `runtime` subpackage; callers import `ContainerConfig`, `Mount`, `ContainerResult`, `ContainerRunError`, and `run_container` from it.

**Backend**
- The backend uses the `podman` Python bindings (podman-py), not a `subprocess` wrapper, per the developer's decision.
- The `podman` dependency is added to the project's runtime dependencies with a lower bound matching the verified server version line (5.8).

**Configuration type**
- `Mount` is a frozen dataclass with `source`, `target`, `read_only` (default `False`), and `options` (tuple of strings, default empty).
- `ContainerConfig` is a frozen dataclass with slots and these fields: `image` (required string), `command` (tuple of strings, default empty = image default), `env` (mapping of string to string, default empty), `volumes` (tuple of `Mount`, default empty), `devices` (tuple of strings, default empty), `mem_limit` (int or string like `"4g"`, default `None`), `shm_size` (int or string like `"2g"`, default `None`), `pull_policy` (one of `"missing"`, `"always"`, `"never"`, `"newer"`, default `"missing"`), and `timeout` (float seconds, default `None`).
- Networking, port publishing, container naming, user, working directory, and entrypoint overrides are intentionally not represented.

**Result type**
- `ContainerResult` is a frozen dataclass with slots containing `exit_code` (int), `stdout` (bytes), `stderr` (bytes), and `container_id` (optional string).
- It exposes `stdout_text` and `stderr_text` properties that decode the respective bytes as UTF-8 with `errors="replace"`.

**Error type**
- `ContainerRunError` is a single exception type used for every failure mode: non-zero exit, timeout, and Podman API errors.
- It carries the partial `ContainerResult` when available and a `timed_out` boolean.
- The originating Podman exception is preserved as the exception's `__cause__`.

**Execution flow**
- `run_container(config, *, client=None)` accepts an optional injected `PodmanClient`. When omitted, it constructs one via `PodmanClient.from_env()` (honoring `DOCKER_HOST` / `CONTAINER_HOST`) and closes it when done; an injected client is never closed.
- The container is created with a non-TTY configuration and without daemon-side auto-removal, so that logs survive until they are read.
- The container is started, then `wait` is called with the configured timeout; `wait` returns the process exit code directly.
- Logs are read after the process exits via the raw logs endpoint and demultiplexed locally into stdout and stderr. A small internal demux helper parses the 8-byte framed stream; the runner does not rely on podman-py's non-exported demux utility.
- The container is removed with force in a `finally` block so cleanup always occurs.
- Because stdout/stderr separation depends on the multiplexed (non-TTY) log stream, TTY mode is not used.

**Config-to-API mapping**
- Environment is passed through as the environment mapping.
- `Mount` entries are translated to podman-py mount specifications (bind type, source, target, read-only flag, options).
- `devices` is passed through as podman-py `devices` (CDI names such as `nvidia.com/gpu=all`, or device-node strings such as `/dev/dri:/dev/dri`). Docker-style `device_requests` is not used because podman-py rejects it as unsupported.
- `mem_limit` and `shm_size` are passed through unchanged; podman-py parses human-readable sizes.

**Pull policy**
- podman-py's create only auto-pulls when an image is missing, so to honor `"always"` and `"newer"` the runner performs an explicit image pull with the configured policy before creating the container. `"missing"` and `"never"` rely on create's behavior.

**Timeout behavior**
- When `wait` raises because the timeout elapsed, the runner fetches the partial logs produced so far, kills the container, removes it, and raises `ContainerRunError` with `timed_out=True` and the partial result.

**Error behavior**
- A non-zero exit code raises `ContainerRunError` with the full result attached.
- A Podman API error during create/start raises `ContainerRunError` with the original error as `__cause__`, after best-effort cleanup.

## Testing Decisions

Good tests here exercise external behavior: given a configuration and a (possibly fake) Podman client, what call does the runner make, what result or exception does the caller observe, and is the container cleaned up. Tests should assert on the observable contract — the returned result, the raised exception type and payload, and the sequence of client interactions — rather than on private helper internals.

- **Unit tests (fake client).** A hand-written fake Podman client records the arguments passed to create/pull and returns canned containers, exit codes, and log payloads. These tests run with no daemon and cover: full config field mapping (env, mounts, devices, memory, shm), stdout/stderr demux, success returns a `ContainerResult`, non-zero exit raises `ContainerRunError` with result attached, timeout produces `timed_out=True` plus partial output and cleanup, pull-policy pre-pull behavior, and that only self-created clients are closed while injected clients are left open.
- **Integration tests (marked).** One or more tests run a tiny image that is already present locally, writing to both stdout and stderr and exiting non-zero, to confirm real exit-code and demux behavior end to end. These are marked and skipped when Podman or its socket is unavailable.
- **Prior art.** The repository currently has no tests; these establish the pattern for the project. The unit-test approach mirrors how podman-py itself is tested (fake/mock clients), and the integration tests follow the common "skip when daemon absent" convention.

## Implementation Tasks

[x] **T1 - Add the podman runtime dependency**: Declare the `podman` Python bindings as a runtime dependency with a lower bound of 5.8, and refresh the lockfile so the environment installs it.
[x] **T2 - Define the shared runtime types**: Add the `runtime` subpackage and its shared types module containing the `Mount`, `ContainerConfig`, and `ContainerResult` frozen dataclasses and the `ContainerRunError` exception, with the documented fields, defaults, and text-decoding properties.
[x] **T3 - Implement the log demux helper**: Add an internal helper that parses the multiplexed container log stream (8-byte frame header, 1-byte stream type, 4-byte big-endian length) into separate stdout and stderr byte strings.
[x] **T4 - Implement the core container lifecycle**: Add the Podman backend module with `run_container` that creates (non-TTY, no auto-remove), starts, waits, reads and demuxes logs, force-removes in a `finally`, and returns a `ContainerResult` for the image and command fields.
[x] **T5 - Map the remaining config fields to the Podman API**: Wire `env`, `volumes`/`Mount`, `devices`, `mem_limit`, and `shm_size` from `ContainerConfig` into the container creation call.
  - Note: `Mount` entries are translated into podman-py's `volumes` mapping (with `extended_mode`) rather than its `mounts` argument, because podman-py 5.8 rebuilds `mounts` entries and silently drops unrecognised options. See Further Notes.
[x] **T6 - Implement error semantics**: Raise `ContainerRunError` on non-zero exit with the result attached, and wrap Podman API errors from create/start into the same exception with the original as `__cause__`, performing best-effort cleanup.
[x] **T7 - Implement timeout handling**: Pass the configured timeout to `wait`, and on expiry capture partial logs, kill and remove the container, and raise `ContainerRunError` with `timed_out=True` and the partial result.
  - Note: the partial `ContainerResult` uses `exit_code=-1` (no exit code exists once the wait timed out). An `APIError` from `wait` is treated as a timeout only when `config.timeout` is set; otherwise it is wrapped as an ordinary `ContainerRunError`.
[x] **T8 - Implement pull-policy handling**: Pre-pull the image with the configured policy for `"always"` and `"newer"`, and rely on create's auto-pull for `"missing"`/`"never"`.
[x] **T9 - Implement client injection and ownership**: Accept an optional injected Podman client; construct one from the environment when absent and close only self-created clients.
[x] **T10 - Export the public API**: Re-export `ContainerConfig`, `Mount`, `ContainerResult`, `ContainerRunError`, and `run_container` from the `runtime` subpackage so callers have a stable import path.
[x] **T11 - Add unit tests with a fake client**: Add the fake Podman client and unit tests covering field mapping, demux, success, non-zero exit, timeout, pull policy, and client ownership/closure.
[x] **T12 - Add marked integration tests**: Add integration tests that run a small locally-available image exercising stdout, stderr, and a non-zero exit, skipped when Podman or its socket is unavailable.
  - Note: the image is `docker.io/library/alpine:latest`, pulled on demand by a module-scoped fixture rather than required to be present locally; the tests skip when the pull fails (e.g. offline). The client fixture pings the daemon and skips the whole module when it is unreachable. See Further Notes.

## Out of Scope

- A command-line interface or console entrypoint; the deliverable is a library module only.
- Networking features: port publishing, custom networks, hostname, DNS, or extra hosts.
- Container naming, user/UID selection, working-directory overrides, and entrypoint overrides.
- Building images, committing containers, or managing volumes/secrets.
- Streaming or live-tailing output while the container runs; the runner blocks and returns captured output.
- Detached execution and returning a live container handle.
- Concurrent or batched execution of multiple containers.
- CPU pinning and other CPU-quota controls.
- Non-Podman backends (Docker, Apptainer); only the type layer is prepared for them.

## Further Notes

- Verified on the target machine: podman-py 5.8.0 installs and imports on Python 3.14.7; `PodmanClient.from_env()` connects to the local rootless socket (server 5.8.7, API 1.44); `create → start → wait` returns the true exit code; the raw logs endpoint plus demux cleanly separates stdout and stderr; and `wait(timeout=...)` raises a catchable `podman.errors.APIError` after the timeout.
- The `ContainerResult` bytes-plus-text design keeps binary output lossless while making the common text case ergonomic.
- `ContainerRunError` intentionally unifies non-zero exits, timeouts, and API failures so callers have exactly one error type to handle; the `timed_out` flag and `__cause__` preserve the distinctions when needed.
- Because stdout/stderr separation depends on non-TTY multiplexed logs, any future opt-in TTY support would necessarily merge the streams.
- `Mount` entries are passed to podman-py's `volumes` mapping with `extended_mode=list(options)` and `mode="ro"`/`"rw"`, not to its `mounts` argument: podman-py 5.8's `mounts` handling rebuilds each entry from the option keys it recognises (`read_only`, `relabel`, `propagation`, `mode`, `size`) and drops an arbitrary `options` list, so `Mount.options` would never reach Podman. Consequence: a `Mount.source` that looks like a named volume (a relative name such as `data`) is treated by podman-py as a named volume; absolute host paths bind as intended.
- The `runtime` subpackage split (shared types vs. backend) is the seam for a future second backend; the Podman module is the only place that imports the `podman` bindings.
- The T12 integration tests use `docker.io/library/alpine:latest`, pulled on demand by a module-scoped fixture rather than required to be present locally, so they run on a machine with a daemon but no pre-pulled image while still skipping cleanly offline. Verified with `uv run pytest -m integration` (2 passed) and with `CONTAINER_HOST` pointed at a missing socket (2 skipped).
