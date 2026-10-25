"""Hermetic conformance checks for the executable artifact writers.

The exact-text tests in the writer modules pin whatever the writers emit, so they
cannot distinguish a correct document from a plausible one: a ``Device=`` typo
or a shell-joined entrypoint would still match its own expectation. These tests
instead check the rendered artifacts against the formats' rules — Podman's
documented ``[Container]`` keys, quadlet's JSON-array entrypoint, the health
probe's command form, Compose's service shape, and the device strings' classes.

Everything is derived from hand-built server plans, so the default suite stays
hermetic: no backend, no container daemon, and no external tooling. The
allowlist lives here rather than in the writers, which keeps the production
layer neutral and lets the list track Podman's documentation.
"""

import json
import shlex

import pytest
import yaml

from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
    ServerSpec,
)
from llm_config_optimization.output import ComposeWriter, QuadletWriter, RenderContext
from llm_config_optimization.runtime import ContainerConfig, Mount

_CONTAINER_KEYS = frozenset(
    {
        "AddCapability",
        "AddDevice",
        "AddHost",
        "Annotation",
        "AppArmor",
        "AutoUpdate",
        "CgroupsMode",
        "ContainerName",
        "ContainersConfModule",
        "DNS",
        "DNSOption",
        "DNSSearch",
        "DropCapability",
        "Entrypoint",
        "Environment",
        "EnvironmentFile",
        "EnvironmentHost",
        "Exec",
        "ExposeHostPort",
        "GIDMap",
        "GlobalArgs",
        "Group",
        "GroupAdd",
        "HealthCmd",
        "HealthInterval",
        "HealthLogDestination",
        "HealthMaxLogCount",
        "HealthMaxLogSize",
        "HealthOnFailure",
        "HealthRetries",
        "HealthStartPeriod",
        "HealthStartupCmd",
        "HealthStartupInterval",
        "HealthStartupRetries",
        "HealthStartupSuccess",
        "HealthStartupTimeout",
        "HealthTimeout",
        "HostName",
        "HttpProxy",
        "Image",
        "IP",
        "IP6",
        "Label",
        "LogDriver",
        "LogOpt",
        "Mask",
        "Memory",
        "Mount",
        "Network",
        "NetworkAlias",
        "NoNewPrivileges",
        "Notify",
        "PidsLimit",
        "Pod",
        "PodmanArgs",
        "PublishPort",
        "Pull",
        "ReadOnly",
        "ReadOnlyTmpfs",
        "ReloadCmd",
        "ReloadSignal",
        "Retry",
        "RetryDelay",
        "Rootfs",
        "RunInit",
        "SeccompProfile",
        "Secret",
        "SecurityLabelDisable",
        "SecurityLabelFileType",
        "SecurityLabelLevel",
        "SecurityLabelNested",
        "SecurityLabelType",
        "ShmSize",
        "StartWithPod",
        "StopSignal",
        "StopTimeout",
        "SubGIDMap",
        "SubUIDMap",
        "Sysctl",
        "Timezone",
        "Tmpfs",
        "UIDMap",
        "Ulimit",
        "Unmask",
        "User",
        "UserNS",
        "Volume",
        "WorkingDir",
    }
)
"""The ``[Container]`` keys ``podman-systemd.unit(5)`` documents for Podman 5.8.

A generated unit may use only these keys; anything else is ignored by the
quadlet generator (or, worse, silently turns into a different key), which is how
the historical ``Device=`` defect produced a unit that looked right and did not
run. The list is test data on purpose: the writer must not learn Podman's
vocabulary beyond the keys it emits.
"""


def _config() -> ExperimentConfig:
    """A minimal recommended experiment for the render context."""
    return ExperimentConfig(
        container=ContainerRuntimeConfig(),
        benchmark=BenchmarkConfig(model="meta-llama/Llama-3.1-8B"),
        llm=LLMConfig(),
    )


def _spec(
    *,
    entrypoint: tuple[str, ...] | None = None,
    ports: dict[str, int | None] | None = None,
    selinux_opts: tuple[str, ...] | None = ("disable",),
) -> ServerSpec:
    """A realistic planned server with every container domain populated."""
    return ServerSpec(
        container=ContainerConfig(
            image="vllm/vllm-openai:cu128",
            command=("vllm", "serve", "meta-llama/Llama-3.1-8B", "--port", "8000"),
            entrypoint=entrypoint,
            env={
                "HF_HOME": "/root/.cache/huggingface",
                "VLLM_FLAGS": "--enable-prefix-caching --disable-log",
            },
            volumes=(
                Mount("/models/llama", "/models/llama", read_only=True),
                Mount(
                    "/home/bench/.cache/huggingface",
                    "/root/.cache/huggingface",
                    options=("z",),
                ),
            ),
            devices=("nvidia.com/gpu=all", "/dev/kfd", "/dev/dri"),
            ports=ports if ports is not None else {"8000/tcp": None},
            mem_limit="16g",
            shm_size="8g",
            selinux_opts=selinux_opts,
            pull_policy="newer",
        ),
        health_path="/health",
        health_timeout=600.0,
    )


def _context(spec: ServerSpec) -> RenderContext:
    return RenderContext(
        config=_config(),
        target=HardwareTarget.CUDA,
        image="vllm/vllm-openai:cu128",
        spec=spec,
    )


def _render_quadlet(spec: ServerSpec) -> str:
    return QuadletWriter().render(_context(spec))


def _render_compose(spec: ServerSpec) -> str:
    return ComposeWriter().render(_context(spec))


def _quadlet_sections(unit: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a quadlet into its sections and ``key=value`` entries.

    The format is line-oriented and allows repeated keys, so a plain
    ``configparser`` would merge the entries these checks need to inspect.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    for line in unit.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = sections.setdefault(stripped[1:-1], [])
        elif current is not None and stripped:
            key, separator, value = stripped.partition("=")
            assert separator, f"malformed quadlet entry: {line!r}"
            current.append((key, value))
    return sections


def _quadlet_entries(unit: str, section: str) -> list[tuple[str, str]]:
    return _quadlet_sections(unit)[section]


def _quadlet_value(unit: str, section: str, key: str) -> str:
    values = [value for entry_key, value in _quadlet_entries(unit, section) if entry_key == key]
    assert len(values) == 1, f"expected one {key}= in [{section}], got {values!r}"
    return values[0]


def _compose_service(spec: ServerSpec) -> dict[str, object]:
    document = yaml.safe_load(_render_compose(spec))
    assert list(document) == ["services"]
    return document["services"]["server"]


def _device_class(device: str) -> str:
    """Classify a device the way Compose providers interpret one.

    A CDI string (``vendor=class``) is Podman-targeted, while a host path is
    portable across Podman and Docker. The classification is what the
    compatibility matrix in the integration layer relies on.
    """
    if "=" in device:
        return "cdi"
    if device.startswith("/"):
        return "host-path"
    return "other"


@pytest.mark.parametrize(
    "spec",
    [
        _spec(),
        _spec(entrypoint=("/bin/sh", "-c")),
        _spec(ports={"9000": None, "8000/tcp": 18000}),
        _spec(selinux_opts=("disable", "type:container_file_t")),
        _spec(selinux_opts=None),
    ],
)
def test_the_quadlet_container_section_uses_only_documented_keys(spec: ServerSpec) -> None:
    """An unknown ``[Container]`` key would make the quadlet generator ignore it."""
    unit = _render_quadlet(spec)
    keys = {key for key, _ in _quadlet_entries(unit, "Container")}

    assert keys <= _CONTAINER_KEYS, f"undocumented quadlet keys: {keys - _CONTAINER_KEYS}"
    assert "AddDevice" in keys
    assert "Device" not in keys


def test_the_unit_contains_only_the_sections_a_container_unit_uses() -> None:
    """Extra sections would describe some other unit, not this container."""
    unit = _render_quadlet(_spec())

    assert set(_quadlet_sections(unit)) == {"Unit", "Container", "Service"}


def test_the_quadlet_entrypoint_is_a_json_array() -> None:
    """Quadlet forwards the entrypoint as one argument and only splits JSON."""
    entrypoint = ("/bin/sh", "-c", "exec vllm serve")
    unit = _render_quadlet(_spec(entrypoint=entrypoint))
    value = _quadlet_value(unit, "Container", "Entrypoint")

    assert json.loads(value) == list(entrypoint)
    assert value != shlex.join(entrypoint)


def test_compose_renders_the_same_entrypoint_as_a_list() -> None:
    """Both writers reproduce the plan's entrypoint with the same arguments."""
    entrypoint = ("/bin/sh", "-c", "exec vllm serve")
    service = _compose_service(_spec(entrypoint=entrypoint))

    assert service["entrypoint"] == list(entrypoint)


def test_both_artifacts_carry_the_same_non_empty_curl_probe() -> None:
    """Readiness is a real curl request in both formats, not an empty command."""
    quadlet_command = _quadlet_value(_render_quadlet(_spec()), "Container", "HealthCmd")
    compose_test = _compose_service(_spec())["healthcheck"]["test"]

    assert compose_test[:1] == ["CMD"]
    probe = compose_test[1:]
    assert shlex.split(quadlet_command) == probe
    assert probe == ["curl", "-fsS", "http://127.0.0.1:8000/health"]


def test_the_compose_document_has_the_shape_compose_expects() -> None:
    """A modern single-service file with list/scalar values in the right places."""
    document = yaml.safe_load(_render_compose(_spec()))

    assert list(document) == ["services"]
    assert "version" not in document
    assert list(document["services"]) == ["server"]

    service = document["services"]["server"]
    assert isinstance(service["image"], str)
    for list_key in ("command", "volumes", "ports", "devices", "security_opt"):
        assert isinstance(service[list_key], list), list_key
        assert all(isinstance(item, str) for item in service[list_key]), list_key
    assert isinstance(service["environment"], dict)
    assert isinstance(service["pull_policy"], str)
    assert isinstance(service["healthcheck"]["test"], list)


def test_compose_keeps_the_planned_device_strings_verbatim() -> None:
    """Device translation would drop or rewrite a GPU passthrough."""
    spec = _spec()
    service = _compose_service(spec)

    assert service["devices"] == list(spec.container.devices)


def test_devices_are_classified_as_portable_or_podman_only() -> None:
    """The CDI/host-path distinction is the documented Compose limitation."""
    devices = _compose_service(_spec())["devices"]

    assert {_device_class(device) for device in devices} == {"cdi", "host-path"}
    assert [_device_class(device) for device in devices] == ["cdi", "host-path", "host-path"]
