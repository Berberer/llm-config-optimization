"""Unit tests for the final summary and the winning-configuration writer.

``render_summary`` and ``write_config`` share one document: a curated container
section, the full benchmark and LLM sections with explicit nulls, and the
resolved target and image filled in. These tests build an
:class:`~llm_config_optimization.optimization.types.OptimizationResult` directly and
assert on the summary's headline fields and on the YAML read back from a
temporary file, so the printed and written configurations are checked against
the same structure.
"""

from dataclasses import fields
from pathlib import Path

import optuna
import yaml

from llm_config_optimization.cli.summary import render_summary, write_config
from llm_config_optimization.inference import (
    BenchmarkConfig,
    ContainerRuntimeConfig,
    DType,
    ExperimentConfig,
    HardwareTarget,
    LLMConfig,
)
from llm_config_optimization.optimization import OptimizationResult

TARGET = HardwareTarget.CUDA
IMAGE = "vllm/vllm-openai:latest"


def base_config() -> ExperimentConfig:
    """A winning configuration with an unresolved target and image.

    The optimizer reconstructs exactly this shape — ``target`` and ``image`` are
    ``None`` because hardware detection and the backend's image lookup happen
    outside the config — so the writer is exercised filling them in.
    """
    return ExperimentConfig(
        container=ContainerRuntimeConfig(shm_size="8g", mem_limit="16g", pull_policy="missing"),
        benchmark=BenchmarkConfig(
            model="facebook/opt-125m",
            dataset_name="random",
            dataset_path=None,
            num_prompts=200,
            input_len=1024,
            output_len=128,
            timeout=900,
        ),
        llm=LLMConfig(dtype=DType.FLOAT16, max_model_len=2048, seed=0),
    )


def result() -> OptimizationResult:
    """A finished sweep whose winning configuration is :func:`base_config`."""
    return OptimizationResult(
        best_config=base_config(),
        best_value=42.0,
        best_trial_number=3,
        trial_count=5,
        study=optuna.create_study(direction="maximize"),
    )


def test_the_summary_names_the_best_trial_throughput_and_count() -> None:
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert "best trial: 3" in summary
    assert "best throughput: 42.00 tps" in summary
    assert "trials run: 5" in summary


def test_the_summary_carries_the_winning_configuration() -> None:
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert "winning configuration:" in summary
    assert "dataset_name: random" in summary
    assert "input_len: 1024" in summary
    assert "dtype: float16" in summary
    assert "max_model_len: 2048" in summary
    assert "gpu_memory_utilization: null" in summary


def test_write_config_writes_the_three_domains(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    assert set(yaml.safe_load(path.read_text(encoding="utf-8"))) == {
        "container",
        "benchmark",
        "llm",
    }


def test_the_container_section_is_curated(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    container = yaml.safe_load(path.read_text(encoding="utf-8"))["container"]
    assert set(container) == {"target", "image", "shm_size", "mem_limit", "pull_policy"}
    assert container["shm_size"] == "8g"
    assert container["mem_limit"] == "16g"
    assert container["pull_policy"] == "missing"


def test_the_resolved_target_and_image_are_filled_in(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    container = yaml.safe_load(path.read_text(encoding="utf-8"))["container"]
    assert container["target"] == "cuda"
    assert container["image"] == IMAGE


def test_every_benchmark_and_llm_field_is_written(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(document["benchmark"]) == {field.name for field in fields(BenchmarkConfig)}
    assert set(document["llm"]) == {field.name for field in fields(LLMConfig)}


def test_unset_values_are_written_as_explicit_nulls(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert document["benchmark"]["dataset_path"] is None
    assert document["llm"]["gpu_memory_utilization"] is None
    assert "dataset_path: null" in text
    assert "gpu_memory_utilization: null" in text


def test_enum_values_use_engine_neutral_spellings(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["llm"]["dtype"] == "float16"
    assert document["container"]["target"] == "cuda"


def test_write_config_overwrites_an_existing_file_silently(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"
    path.write_text("stale: true\n", encoding="utf-8")

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "stale" not in document
    assert set(document) == {"container", "benchmark", "llm"}


def test_the_summary_and_the_file_share_the_same_document(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"
    write_config(path, base_config(), target=TARGET, image=IMAGE)
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    printed = summary.split("winning configuration:\n", 1)[1]

    assert yaml.safe_load(printed) == yaml.safe_load(path.read_text(encoding="utf-8"))
