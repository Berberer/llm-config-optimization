"""Unit tests for the command-line argument surface.

``parse_args``'s observable contract is the settings namespace it produces from
an argument vector, plus the exit it performs for help, version, and usage
errors. These tests drive the parser directly, so no space is loaded, no
hardware is probed, and no sweep runs. They assert on the documented defaults
and choices rather than on help wording, so messages can be reworded freely.
"""

from pathlib import Path

import pytest

from llm_config_optimization.cli.arguments import parse_args


def test_the_two_positionals_are_parsed_as_model_and_space_path() -> None:
    args = parse_args(["facebook/opt-125m", "space.yaml"])

    assert args.model == "facebook/opt-125m"
    assert args.space == Path("space.yaml")


def test_defaults_match_the_documented_settings() -> None:
    args = parse_args(["facebook/opt-125m", "space.yaml"])

    assert args.target is None
    assert args.image is None
    assert args.shm_size == "8g"
    assert args.mem_limit is None
    assert args.pull_policy == "missing"
    assert args.dataset_name == "random"
    assert args.dataset_path is None
    assert args.num_prompts == 200
    assert args.input_len == 1024
    assert args.output_len == 128
    assert args.run_timeout == 900
    assert args.llm_seed == 0
    assert args.quantization is None
    assert args.gpu_memory_utilization is None
    assert args.sampler_seed is None
    assert args.n_trials is None
    assert args.timeout is None
    assert args.storage is None
    assert args.study_name is None
    assert args.n_jobs == 1
    assert args.output is None
    assert args.output_config is None


def test_explicit_flags_override_every_default() -> None:
    args = parse_args(
        [
            "facebook/opt-125m",
            "space.yaml",
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
            "--run-timeout",
            "120",
            "--llm-seed",
            "7",
            "--quantization",
            "awq",
            "--gpu-memory-utilization",
            "0.9",
            "--sampler-seed",
            "42",
            "--n-trials",
            "10",
            "--timeout",
            "600",
            "--storage",
            "sqlite:///study.db",
            "--study-name",
            "sweep",
            "--n-jobs",
            "2",
            "--output",
            "log",
            "--output-config",
            "best.yaml",
        ]
    )

    assert args.target == "rocm"
    assert args.image == "vllm/vllm-rocm:latest"
    assert args.shm_size == "16g"
    assert args.mem_limit == "32g"
    assert args.pull_policy == "always"
    assert args.dataset_name == "sharegpt"
    assert args.dataset_path == "/data/sharegpt.json"
    assert args.num_prompts == 64
    assert args.input_len == 512
    assert args.output_len == 64
    assert args.run_timeout == 120
    assert args.llm_seed == 7
    assert args.quantization == "awq"
    assert args.gpu_memory_utilization == 0.9
    assert args.sampler_seed == 42
    assert args.n_trials == 10
    assert args.timeout == 600
    assert args.storage == "sqlite:///study.db"
    assert args.study_name == "sweep"
    assert args.n_jobs == 2
    assert args.output == "log"
    assert args.output_config == Path("best.yaml")


@pytest.mark.parametrize(
    "argv",
    [
        ["model", "space.yaml", "--target", "gpu"],
        ["model", "space.yaml", "--pull-policy", "sometimes"],
        ["model", "space.yaml", "--quantization", "int8"],
        ["model", "space.yaml", "--output", "json"],
        ["model", "space.yaml", "--unknown"],
        ["model"],
        [],
    ],
    ids=[
        "bad-target",
        "bad-pull-policy",
        "bad-quantization",
        "bad-output",
        "unknown-flag",
        "missing-space",
        "empty",
    ],
)
def test_invalid_input_is_a_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(argv)

    assert excinfo.value.code == 2


def test_output_rejects_auto(capsys: pytest.CaptureFixture[str]) -> None:
    """There is no ``auto`` value: omission is how the mode is detected."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["model", "space.yaml", "--output", "auto"])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_exits_zero_and_documents_the_invocation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])

    assert excinfo.value.code == 0
    text = capsys.readouterr().out
    assert "MODEL" in text
    assert "SPACE" in text
    for flag in ("--target", "--n-trials", "--output", "--output-config"):
        assert flag in text, flag
    assert "(default: 8g)" in text
    assert "(default: 1024)" in text
    assert "examples:" in text


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--version"])

    assert excinfo.value.code == 0
    assert "llm-config-optimization" in capsys.readouterr().out
