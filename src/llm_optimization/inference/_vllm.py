"""vLLM backend for the inference experiment layer.

This module is the only place that knows vLLM's benchmark CLI. It maps the
engine-neutral :class:`~llm_optimization.inference.types.LLMConfig` fields onto
``vllm bench throughput`` arguments, plans a container run for an experiment,
and parses the benchmark result back into a typed
:class:`~llm_optimization.inference.types.ExperimentResult`.
"""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from ..runtime.types import ContainerConfig, ContainerResult, Mount
from ..search_space.registry import TUNABLE_PARAMETERS
from ..search_space.types import (
    Categorical,
    Default,
    Dimension,
    ExperimentSpace,
    Fixed,
    Interval,
    ParameterValue,
    Spec,
)
from .types import (
    AttentionBackend,
    DetectedHardware,
    DType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    KVCacheDType,
    LLMConfig,
    QuantizationMethod,
)


@dataclass(frozen=True, slots=True)
class FlatFlag:
    """A parameter forwarded as its own ``--flag value`` argument pair.

    The flag name is not stored: it is derived from the field name by
    snake_case-to-kebab-case conversion when arguments are assembled, which is
    exactly how vLLM names its flattened flags (``kv_cache_dtype`` becomes
    ``--kv-cache-dtype``).
    """


@dataclass(frozen=True, slots=True)
class JsonGroup:
    """A parameter forwarded inside a JSON configuration group.

    ``flag`` is the group's CLI flag (for example ``--attention-config``) and
    ``key`` is the field's name inside that group's JSON payload (for example
    ``backend``). Fields that share a ``flag`` are merged into one JSON object,
    so the group is emitted once per run.
    """

    flag: str
    key: str


ParameterTarget = FlatFlag | JsonGroup


PARAMETER_TARGETS: Mapping[str, ParameterTarget] = MappingProxyType(
    {
        "dtype": FlatFlag(),
        "quantization": FlatFlag(),
        "kv_cache_dtype": FlatFlag(),
        "max_model_len": FlatFlag(),
        "max_num_seqs": FlatFlag(),
        "max_num_batched_tokens": FlatFlag(),
        "gpu_memory_utilization": FlatFlag(),
        "tensor_parallel_size": FlatFlag(),
        "seed": FlatFlag(),
        # Routed through the JSON group form on purpose, to exercise the JSON
        # path end to end; vLLM also exposes an equivalent flat --attention-backend.
        "attention_backend": JsonGroup(flag="--attention-config", key="backend"),
    }
)
"""Where each :class:`LLMConfig` field is sent on the vLLM command line.

The table is deliberately code-only and declarative: every field that reaches
vLLM has exactly one entry, and adding a parameter is an explicit two-step of
adding a typed field to ``LLMConfig`` and adding its entry here. A field absent
from the table is never forwarded.
"""


VALUE_MAPS: Mapping[str, Mapping[object, str]] = MappingProxyType(
    {
        "dtype": {
            DType.AUTO: "auto",
            DType.FLOAT16: "float16",
            DType.BFLOAT16: "bfloat16",
            DType.FLOAT32: "float32",
        },
        "quantization": {
            QuantizationMethod.AWQ: "awq",
            QuantizationMethod.GPTQ: "gptq",
            QuantizationMethod.FP8: "fp8",
            QuantizationMethod.BITS_AND_BYTES: "bitsandbytes",
            QuantizationMethod.GGUF: "gguf",
        },
        "kv_cache_dtype": {
            KVCacheDType.AUTO: "auto",
            KVCacheDType.FLOAT16: "float16",
            KVCacheDType.BFLOAT16: "bfloat16",
            KVCacheDType.FP8_E4M3: "fp8_e4m3",
            KVCacheDType.FP8_E5M2: "fp8_e5m2",
        },
        "attention_backend": {
            AttentionBackend.FLASH_ATTENTION: "FLASH_ATTN",
            AttentionBackend.FLASH_INFER: "FLASHINFER",
            AttentionBackend.XFORMERS: "XFORMERS",
        },
    }
)
"""How vLLM spells each engine-neutral categorical value.

The enums in :mod:`llm_optimization.inference.types` carry no engine
vocabulary; this table is the only place their vLLM spellings live. A member
missing from its field's map is unsupported by this backend and is rejected
rather than silently forwarded, while the long tail of engine-specific values
is reached through :attr:`VLLMBackend.extra_args`.
"""


def _kebab_case(name: str) -> str:
    """Convert a snake_case field name to vLLM's kebab-case flag spelling."""
    return name.replace("_", "-")


def _normalise(value: object) -> object:
    """Recursively reduce a value to something JSON can serialise.

    vLLM takes most complex configuration as a JSON string, so values that are
    not already JSON-native are converted: enums become their ``value``,
    path-likes become strings, mappings become dicts, and sets and sequences
    become lists. Conversion recurses into containers, so an enum or path nested
    inside a JSON group serialises too. Values JSON already understands are
    returned unchanged.

    Sets are ordered by their string form so a run's arguments are deterministic
    even though the set itself is unordered.
    """
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_normalise(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise(item) for item in value]
    return value


def _encode_value(value: object) -> str:
    """Render a flat flag's value as the single token that carries it.

    Containers — including enums, path-likes, and sets once normalised — are
    JSON-encoded, matching vLLM's convention for complex values. Every other
    value is passed through its natural string form.
    """
    normalised = _normalise(value)
    if isinstance(normalised, (dict, list)):
        return json.dumps(normalised)
    return str(normalised)


def _flat_arguments(name: str, value: object) -> list[str]:
    """Encode one flat flag and its value as CLI arguments.

    Booleans follow vLLM's argparse convention: true is the flag itself and
    false is its ``--no-`` counterpart, neither taking a value. Anything else
    becomes ``--flag value`` with nested values JSON-encoded.
    """
    flag = f"--{_kebab_case(name)}"
    if isinstance(value, bool):
        return [flag] if value else [f"--no-{_kebab_case(name)}"]
    return [flag, _encode_value(value)]


def encode_arguments(
    parameters: Mapping[str, object],
    targets: Mapping[str, ParameterTarget] = PARAMETER_TARGETS,
) -> list[str]:
    """Encode parameter values as vLLM CLI arguments according to ``targets``.

    This is the generic serialisation primitive behind :func:`build_arguments`.
    It is exposed so the encoding rules — flat flags, JSON groups, kebab-case
    names, boolean negation, and JSON encoding of nested values — can be
    exercised directly with values the typed config does not yet model.

    Parameters absent from ``targets`` and parameters whose value is ``None``
    are never forwarded. Flat flags become ``--flag value``; JSON-group fields
    sharing a flag are merged into a single ``--flag '<json>'`` argument emitted
    at the group's first position, so the group appears once per run.

    Values are normalised before encoding, so enums, path-likes, mappings, sets,
    and arbitrary sequences serialise correctly both as flat flags and inside a
    JSON group.
    """
    arguments: list[str] = []
    emitted_groups: set[str] = set()
    for name, target in targets.items():
        value = parameters.get(name)
        if value is None:
            continue
        if isinstance(target, FlatFlag):
            arguments.extend(_flat_arguments(name, value))
            continue
        if target.flag in emitted_groups:
            continue
        members = {
            member.key: _normalise(parameters[member_name])
            for member_name, member in targets.items()
            if isinstance(member, JsonGroup)
            and member.flag == target.flag
            and parameters.get(member_name) is not None
        }
        arguments.extend((target.flag, json.dumps(members)))
        emitted_groups.add(target.flag)
    return arguments


def _translate_value(field: str, value: object) -> object:
    """Translate an engine-neutral categorical value into vLLM's spelling.

    Values for fields without a value map, and values that are not enum members,
    pass through unchanged. An enum member absent from its field's map is not
    supported by this backend and raises rather than being silently dropped, so
    a requested configuration is never quietly ignored.
    """
    mapping = VALUE_MAPS.get(field)
    if mapping is None or not isinstance(value, Enum):
        return value
    try:
        return mapping[value]
    except KeyError:
        raise ExperimentError(f"vLLM does not support {field}={value.value!r}") from None


def build_arguments(
    config: LLMConfig,
    *,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Turn an LLM config into the vLLM CLI arguments that carry it.

    Unset fields (``None``) are omitted so the engine keeps its own defaults,
    categorical fields are translated from their engine-neutral enums into
    vLLM's spellings, and any ``extra_args`` are appended verbatim after the
    mapped parameters as the backend's escape hatch for unmodeled flags.
    """
    parameters = {
        field.name: _translate_value(field.name, getattr(config, field.name))
        for field in fields(config)
        if getattr(config, field.name) is not None
    }
    return [*encode_arguments(parameters), *extra_args]


DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "llm-optimization"
"""Where a run writes its result when the caller supplies no output directory.

:meth:`VLLMBackend.plan` and :meth:`VLLMBackend.parse_result` both fall back to
this path, so the directory mounted into the container is the one read back on
the host. The orchestrator may pass an explicit directory instead.
"""

_RESULT_FILENAME = "result.json"
_OUTPUT_MOUNT_PATH = "/output"
_MODEL_MOUNT_PATH = "/model"
_VLLM_ENTRYPOINT = ("vllm",)
"""The command the container starts with, replacing the image's own entrypoint.

vLLM's official images set ``ENTRYPOINT ["vllm", "serve"]``, which is right
for the serving image but wrong for a one-shot benchmark: the runtime's
``command`` only replaces ``CMD``, so the benchmark arguments would be appended
to ``vllm serve``. The backend overrides the entrypoint so the planned
``bench throughput`` command is what actually runs.
"""
_HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
_HF_CACHE_MOUNT_PATH = "/root/.cache/huggingface"
_SELINUX_OPTIONS = ("z",)
"""Bind-mount options applied to the mounts the backend adds itself.

The ``z`` option asks the container runtime to relabel the host path for
SELinux, so runs work on enforcing hosts. Caller-supplied volumes keep whatever
options they were given.
"""

_DEFAULT_IMAGES: Mapping[HardwareTarget, str] = MappingProxyType(
    {
        HardwareTarget.CUDA: "docker.io/vllm/vllm-openai:latest",
        HardwareTarget.ROCM: "docker.io/vllm/vllm-openai-rocm:latest",
    }
)
"""vLLM's official base image per accelerator class.

The names are registry-qualified because podman refuses to resolve a bare
``vllm/...`` short name when it cannot prompt for a registry (for example with
no TTY, as the runtime client runs). CPU has no entry because its image is
configurable on the backend; see :attr:`VLLMBackend.cpu_image`.
"""

_DEVICE_MAPPINGS: Mapping[HardwareTarget, tuple[str, ...]] = MappingProxyType(
    {
        HardwareTarget.CUDA: ("nvidia.com/gpu=all",),
        HardwareTarget.ROCM: ("/dev/kfd", "/dev/dri"),
        HardwareTarget.CPU: (),
    }
)
"""How each accelerator class reaches the container.

CUDA uses the CDI device name so the runtime supplies the driver; ROCm passes
the KFD and DRM nodes the userspace in the image needs; CPU passes nothing.
"""


def _target_for(
    devices: tuple[str, ...], *, raw_result: ContainerResult | None = None
) -> HardwareTarget:
    """Recover the accelerator class a plan's device mapping was chosen for.

    :meth:`VLLMBackend.plan` fills ``devices`` straight from
    :data:`_DEVICE_MAPPINGS`, so a run's target can be read back from the plan
    that produced it. A mapping the backend never emits cannot be attributed to
    a target and raises rather than guessing; ``raw_result`` is attached to that
    error so every :meth:`VLLMBackend.parse_result` failure carries the captured
    container output uniformly.
    """
    for target, mapping in _DEVICE_MAPPINGS.items():
        if mapping == devices:
            return target
    raise ExperimentError(
        f"cannot determine the accelerator class for devices {devices!r}",
        raw_result=raw_result,
    )


_WORKLOAD_TARGETS: Mapping[str, ParameterTarget] = MappingProxyType(
    {
        "model": FlatFlag(),
        "dataset_name": FlatFlag(),
        "dataset_path": FlatFlag(),
        "num_prompts": FlatFlag(),
        "input_len": FlatFlag(),
        "output_len": FlatFlag(),
    }
)
"""Where each :class:`BenchmarkConfig` field is sent on the command line."""


def _ensure_directory(path: Path) -> Path:
    """Create ``path`` if needed and return it, so it can be bind-mounted."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_model(model: str) -> tuple[str, tuple[Mount, ...]]:
    """Resolve the model argument and any mount it needs.

    A model that exists on the host is a local checkpoint: it is bind-mounted
    read-only at a fixed container path and the container is pointed at that
    path. Anything else — a Hugging Face identifier — is passed through
    unchanged and downloaded into the mounted Hugging Face cache.

    Existence is the only test, so a Hugging Face identifier that happens to
    resolve to a local file or directory — for example a repository whose owner
    name matches a directory in the working tree — is treated as a local
    checkpoint and mounted rather than downloaded, and the container then fails
    to find the model it was asked for.
    """
    path = Path(model).expanduser()
    if not path.exists():
        return model, ()
    mount = Mount(
        source=str(path.resolve()),
        target=_MODEL_MOUNT_PATH,
        read_only=True,
        options=_SELINUX_OPTIONS,
    )
    return _MODEL_MOUNT_PATH, (mount,)


_ENGINE_BOUNDS: tuple[tuple[str, str], ...] = (("max_num_batched_tokens", "max_num_seqs"),)
"""Numeric constraints vLLM imposes between two tunable parameters.

Each pair ``(upper, lower)`` records that ``upper`` must be at least ``lower``.
The engine-agnostic ``>= 1`` floor is already enforced by the loader; these are
the extra limits vLLM's own configuration validation adds.
``SchedulerConfig.verify_max_model_len`` rejects
``max_num_batched_tokens < max_num_seqs`` unconditionally, so a space that
allows a violating pair would fail partway through a series. vLLM's other
scheduler rule, ``max_num_batched_tokens >= max_model_len``, applies only when
chunked prefill is disabled, which the default vLLM v1 scheduler does not do,
so it is deliberately not enforced here.
"""


def _declared_values(dimension: Dimension) -> list[tuple[str, ParameterValue]]:
    """Pair each value a dimension declares with the problem path naming it."""
    spec = dimension.spec
    if isinstance(spec, Fixed):
        return [(f"space.{dimension.name}.fixed", spec.value)]
    if isinstance(spec, Categorical):
        return [
            (f"space.{dimension.name}.categorical[{index}]", value)
            for index, value in enumerate(spec.values)
        ]
    return []


def _validate_spec_values(dimension: Dimension, field: str) -> list[str]:
    """Report every declared value vLLM has no spelling for.

    Only enum values are checked: a numeric parameter is forwarded verbatim, so
    it has nothing to look up in :data:`VALUE_MAPS`. A categorical field with no
    value map at all cannot translate any member and rejects every enum value.
    """
    mapping = VALUE_MAPS.get(field)
    problems: list[str] = []
    for path, value in _declared_values(dimension):
        if isinstance(value, Enum) and (mapping is None or value not in mapping):
            problems.append(f"{path}: vLLM does not support {field}={value.value!r}")
    return problems


def _numeric_range(spec: Spec) -> tuple[int, int] | None:
    """Return the smallest and largest integer a spec allows.

    ``None`` when the spec defers to the engine, or when it describes a
    categorical parameter rather than an integer one, so the caller skips the
    engine-bound check instead of comparing incomparable values.
    """
    if isinstance(spec, Fixed):
        values = (spec.value,)
    elif isinstance(spec, Categorical):
        values = spec.values
    elif isinstance(spec, Interval):
        values = (spec.minimum, spec.maximum)
    else:
        return None
    if any(isinstance(value, Enum) for value in values):
        return None
    return min(values), max(values)


def _validate_engine_bounds(space: ExperimentSpace) -> list[str]:
    """Report every engine-specific numeric bound the space could violate.

    A bound is only checked when both parameters are pinned or bounded by a
    value the optimizer will draw from; a ``Default`` parameter is unknown, so
    the check is skipped and the engine enforces the bound at run time instead.
    The comparison uses the most permissive pair the space allows — the
    smallest ``upper`` against the largest ``lower`` — so a space is accepted
    only when every point it can produce satisfies the bound.
    """
    specs = {dimension.name: dimension.spec for dimension in space.dimensions}
    problems: list[str] = []
    for upper, lower in _ENGINE_BOUNDS:
        upper_range = _numeric_range(specs[upper]) if upper in specs else None
        lower_range = _numeric_range(specs[lower]) if lower in specs else None
        if upper_range is None or lower_range is None:
            continue
        if upper_range[0] < lower_range[1]:
            problems.append(
                f"space.{upper}: vLLM requires {upper} >= {lower}, but the space "
                f"allows {upper}={upper_range[0]} with {lower}={lower_range[1]}"
            )
    return problems


@dataclass(frozen=True, slots=True)
class VLLMBackend:
    """The vLLM implementation of the inference backend seam.

    ``extra_args`` is the single escape hatch for vLLM flags the engine-neutral
    config does not model. The sequence is appended verbatim to the assembled
    benchmark command, after the mapped parameters, so a caller can experiment
    without widening :class:`LLMConfig`; values passed this way are the caller's
    responsibility.

    The layer appends its own ``--output-json`` after the mapped parameters and
    after ``extra_args``, so an ``extra_args`` entry that also sets
    ``--output-json`` is overridden by the layer's value and result routing
    stays under the backend's control.

    ``cpu_image`` is the base image used for CPU runs. vLLM publishes an
    official CPU image, but it is configurable so a caller can pin a different
    build without waiting for a library release. Like the CUDA and ROCm
    defaults it is registry-qualified so podman can resolve it without a TTY.
    """

    extra_args: tuple[str, ...] = ()
    cpu_image: str = "docker.io/vllm/vllm-openai-cpu:latest"

    def validate(self, space: ExperimentSpace) -> Sequence[str]:
        """Report every way vLLM cannot support ``space``.

        The space is assumed to have passed engine-agnostic validation. Every
        dimension's parameter must be one the backend forwards, and every fixed
        or categorical enum value must have a vLLM spelling. ``Default``
        dimensions carry no value to check and are accepted once their
        parameter is known to the backend. The engine's own numeric constraints
        between parameters are checked last. Problems are returned as strings
        so the caller can aggregate them; an empty sequence means vLLM supports
        every dimension.
        """
        problems: list[str] = []
        for dimension in space.dimensions:
            field = TUNABLE_PARAMETERS[dimension.name]
            if field not in PARAMETER_TARGETS:
                problems.append(f"space.{dimension.name}: vLLM does not forward {field!r}")
                continue
            if isinstance(dimension.spec, Default):
                continue
            problems.extend(_validate_spec_values(dimension, field))
        problems.extend(_validate_engine_bounds(space))
        return problems

    def plan(
        self,
        experiment: ExperimentConfig,
        hardware: DetectedHardware,
        output_dir: Path | None = None,
    ) -> ContainerConfig:
        """Turn an experiment and the detected hardware into a container run.

        The accelerator class resolves from the container config's override or,
        failing that, the detected hardware, and selects the base image, the
        device mapping, and — when the LLM config leaves it unset — the
        tensor-parallel size. The Hugging Face cache, the model itself when it
        is a host path, and the result directory are bind-mounted, and the
        benchmark runs as ``vllm bench throughput`` with ``--output-json``
        pointing into that mounted directory. The image's own ``vllm serve``
        entrypoint is overridden so the benchmark subcommand is what runs.

        Planning is not side-effect free: the output directory and the host
        Hugging Face cache directory are created on the host when they do not
        exist, because both must exist before they can be bind-mounted. Merely
        planning a run — as the unit tests do — therefore still touches the
        host filesystem.
        """
        container = experiment.container
        target = container.target or hardware.target
        output_dir = _ensure_directory(output_dir or DEFAULT_OUTPUT_DIR)
        _ensure_directory(_HF_CACHE_HOST)
        model, model_mounts = _resolve_model(experiment.benchmark.model)
        llm = experiment.llm
        if llm.tensor_parallel_size is None:
            llm = replace(llm, tensor_parallel_size=hardware.device_count)
        workload = {
            "model": model,
            "dataset_name": experiment.benchmark.dataset_name,
            "dataset_path": experiment.benchmark.dataset_path,
            "num_prompts": experiment.benchmark.num_prompts,
            "input_len": experiment.benchmark.input_len,
            "output_len": experiment.benchmark.output_len,
        }
        command = (
            "bench",
            "throughput",
            *encode_arguments(workload, _WORKLOAD_TARGETS),
            *build_arguments(llm, extra_args=self.extra_args),
            "--output-json",
            f"{_OUTPUT_MOUNT_PATH}/{_RESULT_FILENAME}",
        )
        volumes = (
            *container.volumes,
            Mount(
                source=str(_HF_CACHE_HOST),
                target=_HF_CACHE_MOUNT_PATH,
                options=_SELINUX_OPTIONS,
            ),
            *model_mounts,
            Mount(
                source=str(output_dir),
                target=_OUTPUT_MOUNT_PATH,
                options=_SELINUX_OPTIONS,
            ),
        )
        return ContainerConfig(
            image=container.image or self._default_image(target),
            command=command,
            entrypoint=_VLLM_ENTRYPOINT,
            env=dict(container.env),
            volumes=volumes,
            devices=_DEVICE_MAPPINGS[target],
            mem_limit=container.mem_limit,
            shm_size=container.shm_size,
            selinux_opts=container.selinux_opts,
            pull_policy=container.pull_policy,
            timeout=experiment.benchmark.timeout,
        )

    def _default_image(self, target: HardwareTarget) -> str:
        """Return the base image for ``target`` when none was overridden."""
        if target == HardwareTarget.CPU:
            return self.cpu_image
        return _DEFAULT_IMAGES[target]

    def parse_result(
        self,
        result: ContainerResult,
        plan: ContainerConfig,
        output_dir: Path | None = None,
    ) -> ExperimentResult:
        """Read vLLM's benchmark JSON and turn it into a typed result.

        The result file is read from the same directory :meth:`plan` mounted (or
        the module default when the caller supplies none). The metrics are taken
        verbatim from vLLM's summary payload, with the prompt count read from its
        ``num_requests`` key; the accelerator class, image, and device mapping
        are recovered from ``plan``, so the result records what ran rather than
        what was requested. A missing or unparseable file raises
        :class:`ExperimentError` with the raw container result attached.
        """
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        result_path = output_dir / _RESULT_FILENAME
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExperimentError(
                f"could not read the vLLM benchmark result at {result_path}",
                raw_result=result,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"vLLM benchmark result at {result_path} is not valid JSON",
                raw_result=result,
            ) from exc
        if not isinstance(payload, Mapping):
            raise ExperimentError(
                f"vLLM benchmark result at {result_path} is not a JSON object",
                raw_result=result,
            )
        target = _target_for(plan.devices, raw_result=result)
        try:
            return ExperimentResult(
                tokens_per_second=float(payload["tokens_per_second"]),
                requests_per_second=float(payload["requests_per_second"]),
                elapsed_time=float(payload["elapsed_time"]),
                num_prompts=int(payload["num_requests"]),
                target=target,
                image=plan.image,
                devices=plan.devices,
                raw_json=payload,
                raw_result=result,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(
                f"vLLM benchmark result at {result_path} is missing or malformed: {exc}",
                raw_result=result,
            ) from exc
