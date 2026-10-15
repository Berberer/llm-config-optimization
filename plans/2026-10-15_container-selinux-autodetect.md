# Automatic SELinux Handling for Container Runs

## Problem Statement

On a host with SELinux enforcing, a container running the vLLM ROCm backend
cannot use the GPU. The HSA runtime aborts with `Memory critical error by agent
node-0 … Reason: Memory in use` the moment it allocates GPU memory, even for a
one-element tensor, because the container process runs under the `container_t`
label. The host's own unconfined ROCm userspace allocates fine, so the
accelerator and driver are healthy and only the container confinement is at
fault.

The container runtime layer exposes no way to pass Podman's `--security-opt`
(specifically `label=disable`), so a ROCm sweep through `optimize` cannot run on
such a host without changing global Podman configuration or hand-managing a
separate Podman service. That is a container-confinement detail far outside the
optimization domain: a developer tuning serving parameters should not have to
reason about SELinux labels to run a GPU benchmark.

## Solution

Make SELinux handling automatic with a sensible default, while keeping an
explicit override.

The runtime detects whether the host is SELinux-enforcing and, when it is and a
run maps devices, disables container labeling for that run by default. A ROCm
sweep on an SELinux-enforcing host therefore works through the normal
`run_experiment` and `optimize` path with no configuration from the caller and
no global host changes. On hosts without SELinux, or for runs that map no
devices, nothing changes. A caller who needs control can still set the SELinux
options explicitly on the container configuration, including opting back into
confinement.

## User Stories

1. As an LLM-optimization developer, I want GPU benchmark containers to work on
   an SELinux-enforcing host without any extra configuration, so that I can
   focus on tuning serving parameters.
2. As an LLM-optimization developer, I want the default to change nothing on
   hosts without SELinux or for runs that map no devices, so that existing
   behavior is preserved.
3. As an LLM-optimization developer, I want to override the SELinux options
   explicitly, so that I can force confinement on, force it off, or set a
   specific label.
4. As an LLM-optimization developer, I want the choice to live in the
   experiment's container configuration, so that a run's intent is recorded and
   the automatic default is applied consistently.
5. As an LLM-optimization developer, I want the vLLM backend to forward the
   setting into the plan it builds, so that the choice survives the inference
   layer.
6. As a security-conscious developer, I want the automatic relaxation limited
   to device-mapping benchmark containers, so that unrelated containers keep
   their confinement.
7. As a future maintainer, I want the host check isolated behind a small,
   injectable probe, so that it can be tested without the real host.
8. As a future maintainer, I want the runtime config to stay immutable and
   hashable with the new field, so that configs remain safe to cache and
   compare.
9. As a test author, I want deterministic unit tests that pin the detection
   result, so that the suite passes on both SELinux and non-SELinux hosts.
10. As a test author, I want a marked integration test that exercises the
    automatic default on a real GPU, so that the zero-config path is protected
    and skipped when unavailable.

## Implementation Decisions

- Add a small SELinux probe to the runtime layer that reports whether the host
  is enforcing by reading the kernel's enforce flag. It is injectable so the
  runtime can be driven with a fixed result in tests; a missing flag means "not
  enforcing".
- Add a `selinux_opts` field, a tuple of strings or `None`, to both the runtime
  container config and the inference container runtime config. `None` (the
  default) means "choose automatically"; an empty tuple means "explicitly no
  label options"; a populated tuple means "use exactly these". Values are
  Podman SELinux label options such as `disable` (not the CLI spelling
  `label=disable`, which Podman's API rejects).
- Both configs stay frozen and slots-based, and the new field joins each custom
  hash so equal configs hash equally and a field-replacing copy hashes
  differently.
- The runtime resolves the automatic case: when the field is `None`, use
  `("disable",)` if the host is SELinux-enforcing and the run maps devices,
  otherwise no options. Explicit values always win. The resolved options are
  forwarded to the container-create call as `security_opt`, which podman-py
  maps to the API's `selinux_opts`.
- The automatic relaxation is scoped to runs that map devices, so CPU
  containers and unrelated containers keep their confinement. Detection is not
  a global host setting and never mutates Podman configuration.
- The vLLM backend copies the container runtime config's `selinux_opts` into
  the container config it returns from its plan, leaving `None` untouched so
  the runtime still applies the automatic default.
- The container config records the intent (`None` for automatic), not the
  resolved value; the effective options are a deterministic function of the
  config and the host. This keeps the runtime host-independent and the
  inference layer free of SELinux knowledge.
- No `ipc_mode` field is added. It was an incidental workaround during the
  investigation, not the fix.

## Testing Decisions

Good tests assert external behavior: what configuration the runtime forwards to
the container backend, and what plan the vLLM backend returns. They do not
assert on private helpers.

- **Probe**: a unit test reads a temporary enforce-flag file and covers
  enforcing, permissive, and absent, without touching the real host.
- **Resolution**: unit tests cover the automatic matrix — enforcing plus
  devices yields `disable`, enforcing without devices yields nothing, and not
  enforcing yields nothing — and that an explicit value overrides the automatic
  choice in both directions.
- **Runtime**: a unit test with the shared fake Podman client pins the detection
  result and asserts the resolved option arrives as the `security_opt` argument
  to container creation. The existing create-mapping test is pinned likewise so
  it stays deterministic on any host.
- **vLLM plan**: a unit test asserts the backend copies `selinux_opts` from the
  experiment's container config into the returned container config, mirroring
  the existing plan-field assertions.
- **Integration**: an `integration`-marked test runs a small ROCm benchmark with
  no explicit SELinux option and asserts the typed result, skipped when Podman,
  the ROCm image, or an AMD GPU is unavailable. On a non-SELinux host the same
  test still passes, exercising the no-op branch.
- **Prior art**: the inference hardware-detection tests for injectable probes;
  the runtime tests' fake Podman client and recorded create calls for the
  no-daemon convention; the vLLM plan tests' direct plan assertions; the
  existing vLLM and optimization integration tests for the `integration` marker
  and skip behavior.

## Implementation Tasks

[x] **T1 - Add the SELinux-enforcing probe to the runtime layer**: Add an injectable helper that reports whether the host is SELinux-enforcing by reading the kernel enforce flag, treating a missing flag as not enforcing.
[x] **T2 - Add the SELinux option field to the runtime container config**: Add a `selinux_opts` field, a tuple of strings or `None` defaulting to `None`, to the runtime container config and include it in the config's hash, keeping the config frozen and hashable.
[x] **T3 - Resolve and forward the automatic default in the runtime**: When `selinux_opts` is `None`, resolve it to `("disable",)` if the host is enforcing and the run maps devices and to no options otherwise; forward the resolved options to the container-create call as `security_opt`, and pin the detection result in the existing create-mapping test so it stays deterministic.
[x] **T4 - Add and forward the field in the inference layer**: Add `selinux_opts` defaulting to `None` to the inference container runtime config with matching hashing, and copy it into the container config returned by the vLLM backend's plan.
[x] **T5 - Unit-test the probe, resolution, and plumbing**: Add no-daemon tests covering the probe's enforcing/permissive/absent cases, the automatic resolution matrix, explicit overrides, runtime forwarding, and vLLM plan forwarding.
[x] **T6 - Add the marked ROCm integration smoke test**: Add an `integration`-marked test that runs a small ROCm benchmark with the automatic default and asserts the typed result, skipping when Podman, the ROCm image, or an AMD GPU is unavailable.

## Out of Scope

- Generalizing to every Podman/Docker security option (seccomp, AppArmor,
  capabilities); only SELinux label options are exposed.
- Changing the confinement of containers that map no devices, or of any
  container outside this runtime's runs.
- Adding an `ipc_mode` field; the investigation showed it was incidental.
- Changing the vLLM parameter-to-CLI mapping, the search-space layer, or the
  optimization adapter.
- Changing host Podman configuration or a machine-wide default.
- A command-line or YAML surface for the option.
- Detecting or handling AppArmor, which uses a different mechanism.

## Further Notes

- **Root cause.** The failure is SELinux, not ROCm. The host's unconfined ROCm
  userspace allocates GPU memory fine, while the container under `container_t`
  aborts with `Memory critical error by agent node-0 … Reason: Memory in use`.
  Disabling labeling fixes it; `--shm-size`, `--ulimit memlock`, and
  `seccomp=unconfined` do not.
- **Value format.** Podman's REST API takes label options without the `label=`
  prefix, so the config field carries `disable`, not `label=disable`.
- **Why device-scoped.** The observed failure needs device access; keying the
  automatic relaxation on a non-empty device mapping keeps CPU and unrelated
  containers confined, and is overridable when a caller knows better.
- **Validation.** A container created through podman-py with
  `security_opt=["disable"]` ran a GPU allocation and exited zero on an AMD
  RX 9070 XT. A three-trial ROCm sweep through `optimize` with labeling
  disabled returned a best configuration of roughly 4239 tokens per second.
- **Relationship to the optimization adapter.** This is a runtime-layer gap
  surfaced by the Optuna evaluation, not an adapter defect; the adapter already
  works once the container can reach the GPU.
- **Probe location.** T1's probe lives in a new backend-agnostic module,
  `runtime/_selinux.py`, as `selinux_enforcing(enforce_flag=...)`; the enforce
  flag path is the injection point. Unit tests for it land with T5.
