"""Unit tests for base experiment and objective construction from parsed arguments.

``build_base_config``'s observable contract is the
:class:`~llm_config_optimization.inference.types.ExperimentConfig` it produces: the
container, benchmark, and non-tunable LLM values come from the flags, every
tunable serving parameter is left unset for the space to fill, and an omitted
target or image stays ``None`` for the backend to resolve. ``resolve_objective``'s
contract is the mode and token-mix weights it resolves: the ``--objective``
choice, explicit ``--prefill-weight``/``--decode-weight`` values, and the
benchmark's own input and output lengths as the fallback, with a missing mix or
an invalid pair rejected before anything expensive runs. These tests parse real
argument vectors and assert on the resulting values, so the parser and the
builders are exercised together without a backend.
"""

from pathlib import Path

import pytest

from llm_config_optimization.cli.arguments import parse_args
from llm_config_optimization.cli.config import build_base_config, resolve_objective
from llm_config_optimization.inference import (
    BenchmarkConfig,
    FitMode,
    HardwareTarget,
    QuantizationMethod,
    SpeculativeMethod,
)
from llm_config_optimization.optimization import ObjectiveMode, ObjectiveWeights


def test_defaults_produce_the_documented_base_config() -> None:
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "vllm"]))

    assert config.container.target is None
    assert config.container.image is None
    assert config.container.shm_size == "8g"
    assert config.container.mem_limit is None
    assert config.container.pull_policy == "missing"
    assert config.benchmark.model == "facebook/opt-125m"
    assert config.benchmark.dataset_name == "random"
    assert config.benchmark.dataset_path is None
    assert config.benchmark.num_prompts == 50
    assert config.benchmark.input_len == 1024
    assert config.benchmark.output_len == 128
    assert config.benchmark.repetitions == 3
    assert config.benchmark.warmup == 1
    assert config.benchmark.startup_timeout == 600
    assert config.benchmark.timeout == 1500
    assert config.llm.seed == 0
    assert config.llm.quantization is None
    assert config.llm.gpu_memory_utilization is None
    assert config.llm.fit is FitMode.OFF
    assert config.llm.fit_target is None
    assert config.llm.fit_ctx is None


def test_tunable_serving_parameters_are_left_unset_for_the_space() -> None:
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "vllm"]))

    assert config.llm.dtype is None
    assert config.llm.kv_cache_dtype is None
    assert config.llm.max_model_len is None
    assert config.llm.max_num_seqs is None
    assert config.llm.max_num_batched_tokens is None
    assert config.llm.max_num_active_seqs is None
    assert config.llm.max_num_scheduled_tokens is None
    assert config.llm.ubatch_size is None
    assert config.llm.tensor_parallel_size is None
    assert config.llm.attention_backend is None
    assert config.llm.speculative_method is None
    assert config.llm.draft_model is None
    assert config.llm.speculative_mode is None
    assert config.llm.num_speculative_tokens is None
    assert config.llm.prefix_caching is None
    assert config.llm.n_gpu_layers is None
    assert config.llm.expert_parallel is None
    assert config.llm.all2all_backend is None
    assert config.llm.n_cpu_moe is None
    assert config.llm.resolved_draft_path is None


def test_explicit_flags_override_the_base_config() -> None:
    config = build_base_config(
        parse_args(
            [
                "facebook/opt-125m",
                "space.yaml",
                "vllm",
                "--target",
                "rocm",
                "--image",
                "vllm/vllm-rocm:latest",
                "--shm-size",
                "16g",
                "--mem-limit",
                "32g",
                "--pull-policy",
                "always",
                "--dataset-name",
                "sharegpt",
                "--dataset-path",
                "/data/sharegpt.json",
                "--num-prompts",
                "64",
                "--input-len",
                "512",
                "--output-len",
                "64",
                "--repetitions",
                "5",
                "--warmup-runs",
                "2",
                "--startup-timeout",
                "300",
                "--timeout",
                "120",
                "--llm-seed",
                "7",
                "--quantization",
                "awq",
                "--gpu-memory-utilization",
                "0.9",
            ]
        )
    )

    assert config.container.target is HardwareTarget.ROCM
    assert config.container.image == "vllm/vllm-rocm:latest"
    assert config.container.shm_size == "16g"
    assert config.container.mem_limit == "32g"
    assert config.container.pull_policy == "always"
    assert config.benchmark.model == "facebook/opt-125m"
    assert config.benchmark.dataset_name == "sharegpt"
    assert config.benchmark.dataset_path == "/data/sharegpt.json"
    assert config.benchmark.num_prompts == 64
    assert config.benchmark.input_len == 512
    assert config.benchmark.output_len == 64
    assert config.benchmark.repetitions == 5
    assert config.benchmark.warmup == 2
    assert config.benchmark.startup_timeout == 300
    assert config.benchmark.timeout == 120
    assert config.llm.seed == 7
    assert config.llm.quantization is QuantizationMethod.AWQ
    assert config.llm.gpu_memory_utilization == 0.9


def test_fit_can_be_enabled_with_its_optional_settings() -> None:
    config = build_base_config(
        parse_args(
            [
                "facebook/opt-125m",
                "space.yaml",
                "llamacpp",
                "--fit",
                "on",
                "--fit-target",
                "1024",
                "--fit-ctx",
                "8192",
            ]
        )
    )

    assert config.llm.fit is FitMode.ON
    assert config.llm.fit_target == 1024
    assert config.llm.fit_ctx == 8192


def test_the_default_fit_pins_the_declared_configuration() -> None:
    """The command line pins fit off so a trial measures what the space declared."""
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "llamacpp"]))

    assert config.llm.fit is FitMode.OFF


@pytest.mark.parametrize("option", ["--fit-target", "--fit-ctx"])
def test_fit_settings_without_fit_on_are_rejected(option: str) -> None:
    with pytest.raises(ValueError, match="require --fit on"):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "llamacpp", option, "1024"])
        )


@pytest.mark.parametrize("option", ["--fit-target", "--fit-ctx"])
def test_non_positive_fit_settings_are_rejected(option: str) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "llamacpp", "--fit", "on", option, "0"])
        )


def test_the_speculative_flags_populate_the_base_config() -> None:
    config = build_base_config(
        parse_args(
            [
                "facebook/opt-125m",
                "space.yaml",
                "vllm",
                "--speculative-method",
                "eagle3",
                "--draft-model",
                "org/draft",
            ]
        )
    )

    assert config.llm.speculative_method is SpeculativeMethod.EAGLE3
    assert config.llm.draft_model == "org/draft"
    assert config.llm.resolved_draft_path is None


def test_a_bare_draft_model_implies_the_generic_draft_method() -> None:
    config = build_base_config(
        parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--draft-model", "org/draft"])
    )

    assert config.llm.speculative_method is SpeculativeMethod.DRAFT_MODEL
    assert config.llm.draft_model == "org/draft"


@pytest.mark.parametrize("method", ["mtp", "ngram"], ids=["mtp", "ngram"])
def test_draft_free_methods_need_no_draft_model(method: str) -> None:
    config = build_base_config(
        parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--speculative-method", method])
    )

    assert config.llm.speculative_method is SpeculativeMethod(method)
    assert config.llm.draft_model is None


@pytest.mark.parametrize(
    "method",
    ["draft_model", "eagle3", "dflash", "dspark"],
)
def test_a_draft_requiring_method_without_a_draft_is_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="requires --draft-model"):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--speculative-method", method])
        )


def test_a_draft_model_with_ngram_is_rejected() -> None:
    """``ngram`` is model-free, so a draft would be silently ignored."""
    with pytest.raises(ValueError, match="ngram"):
        build_base_config(
            parse_args(
                [
                    "facebook/opt-125m",
                    "space.yaml",
                    "vllm",
                    "--speculative-method",
                    "ngram",
                    "--draft-model",
                    "org/draft",
                ]
            )
        )


def test_out_of_range_measurement_controls_are_rejected() -> None:
    """A nonsensical control reaches ``BenchmarkConfig``'s own invariant check."""
    with pytest.raises(ValueError):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--repetitions", "0"])
        )

    with pytest.raises(ValueError):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--warmup-runs", "-1"])
        )

    with pytest.raises(ValueError):
        build_base_config(
            parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--startup-timeout", "0"])
        )


def test_an_existing_host_path_is_a_local_model(tmp_path: Path) -> None:
    """A local checkpoint is carried through unresolved until preparation runs."""
    config = build_base_config(parse_args([str(tmp_path), "space.yaml", "vllm"]))

    assert config.benchmark.model == str(tmp_path)
    assert config.benchmark.resolved_path is None


def test_a_hugging_face_id_is_left_unresolved(tmp_path: Path) -> None:
    config = build_base_config(parse_args(["facebook/opt-125m", "space.yaml", "vllm"]))

    assert config.benchmark.model == "facebook/opt-125m"
    assert config.benchmark.resolved_path is None


def test_a_missing_host_path_is_left_unresolved(tmp_path: Path) -> None:
    config = build_base_config(parse_args([str(tmp_path / "absent"), "space.yaml", "vllm"]))

    assert config.benchmark.resolved_path is None


def test_the_default_objective_is_multi_scored_under_the_benchmark_mix() -> None:
    """No flags resolves to the multi default and the benchmark's token mix."""
    args = parse_args(["facebook/opt-125m", "space.yaml", "vllm"])
    config = build_base_config(args)

    mode, weights = resolve_objective(args, config.benchmark)

    assert mode is ObjectiveMode.MULTI
    assert weights == ObjectiveWeights(prefill=1024, decode=128)


def test_explicit_weights_override_the_benchmark_mix() -> None:
    args = parse_args(
        [
            "facebook/opt-125m",
            "space.yaml",
            "vllm",
            "--objective",
            "weighted",
            "--prefill-weight",
            "512",
            "--decode-weight",
            "64",
        ]
    )
    config = build_base_config(args)

    mode, weights = resolve_objective(args, config.benchmark)

    assert mode is ObjectiveMode.WEIGHTED
    assert weights == ObjectiveWeights(prefill=512, decode=64)


def test_one_explicit_weight_falls_back_to_the_benchmark_for_the_other() -> None:
    """The two weight flags are resolved independently against the workload."""
    args = parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--prefill-weight", "256"])
    config = build_base_config(args)

    mode, weights = resolve_objective(args, config.benchmark)

    assert mode is ObjectiveMode.MULTI
    assert weights == ObjectiveWeights(prefill=256, decode=128)


@pytest.mark.parametrize("mode", ["multi", "weighted"], ids=["multi", "weighted"])
def test_a_phase_scoring_mode_without_a_benchmark_mix_is_rejected(mode: str) -> None:
    """No arbitrary token mix is invented when the benchmark cannot supply one."""
    benchmark = BenchmarkConfig(model="facebook/opt-125m")
    args = parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--objective", mode])

    with pytest.raises(ValueError, match="prefill-weight"):
        resolve_objective(args, benchmark)

    explicit = parse_args(
        [
            "facebook/opt-125m",
            "space.yaml",
            "vllm",
            "--objective",
            mode,
            "--prefill-weight",
            "1024",
            "--decode-weight",
            "128",
        ]
    )

    resolved_mode, weights = resolve_objective(explicit, benchmark)

    assert resolved_mode is ObjectiveMode(mode)
    assert weights == ObjectiveWeights(prefill=1024, decode=128)


def test_combined_tolerates_a_benchmark_without_a_mix() -> None:
    """The historical objective needs no phase weights and accepts their absence."""
    benchmark = BenchmarkConfig(model="facebook/opt-125m")
    args = parse_args(["facebook/opt-125m", "space.yaml", "vllm", "--objective", "combined"])

    mode, weights = resolve_objective(args, benchmark)

    assert mode is ObjectiveMode.COMBINED
    assert weights is None


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--prefill-weight", "0", "--decode-weight", "0"], "zero"),
        (["--prefill-weight", "-1"], "non-negative"),
        (["--prefill-weight", "nan"], "finite"),
    ],
    ids=["both-zero", "negative", "non-finite"],
)
def test_an_invalid_pair_is_rejected_before_any_preparation(flags: list[str], message: str) -> None:
    """A weight outside the model fails here rather than mid-sweep."""
    benchmark = BenchmarkConfig(model="facebook/opt-125m", input_len=1024, output_len=128)
    args = parse_args(["facebook/opt-125m", "space.yaml", "vllm", *flags])

    with pytest.raises(ValueError, match=message):
        resolve_objective(args, benchmark)


def test_combined_still_validates_a_resolvable_pair() -> None:
    """An explicit nonsensical weight is a usage error even in the combined mode."""
    benchmark = BenchmarkConfig(model="facebook/opt-125m", input_len=1024, output_len=128)
    args = parse_args(
        [
            "facebook/opt-125m",
            "space.yaml",
            "vllm",
            "--objective",
            "combined",
            "--prefill-weight",
            "-1",
        ]
    )

    with pytest.raises(ValueError, match="non-negative"):
        resolve_objective(args, benchmark)
