"""Unit tests for the final summary and the winning-configuration writer.

``render_summary`` and ``write_config`` share one document: a curated container
section, the full benchmark and LLM sections with explicit nulls, and the
resolved target and image filled in. These tests build an
:class:`~llm_config_optimization.optimization.types.OptimizationResult` directly and
assert on the summary's headline fields and on the YAML read back from a
temporary file, so the printed and written configurations are checked against
the same structure. A multi-objective study is exercised too: its summary names
the recommended point and prints the Pareto front as a table with the
recommended row marked, and the written document is the recommended point's
configuration either way.
"""

from dataclasses import fields, replace
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
from llm_config_optimization.optimization import OptimizationResult, OptimizationSolution

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
    """A finished sweep whose recommended configuration is :func:`base_config`."""
    solution = OptimizationSolution(
        config=base_config(),
        value=42.0,
        trial_number=3,
    )
    return OptimizationResult(
        solutions=(solution,),
        recommended=solution,
        trial_count=5,
        study=optuna.create_study(direction="maximize"),
    )


_FRONT = (
    # trial number, prefill tps, decode tps, combined tps, max_model_len
    (0, 1000.0, 50.0, 128.0, 4096),
    (1, 500.0, 200.0, 96.0, 2048),
    (2, 200.0, 500.0, 64.0, 1024),
)
"""A three-point Pareto front, each point distinguished by its ``max_model_len``."""


def completed_multi_trial(
    prefill: float,
    decode: float,
    *,
    combined: float | None = None,
) -> optuna.trial.FrozenTrial:
    """A frozen multi trial carrying its phase objectives and recorded combined rate.

    The optimizer records a JSON projection of every successful run under the
    ``result`` user attribute; ``combined`` fills in just the aggregate combined
    throughput the front table reads. Omitting it exercises the trial whose
    record is absent.
    """
    user_attrs: dict[str, object] = {}
    if combined is not None:
        user_attrs["result"] = {"tokens_per_second": combined}
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[prefill, decode],
        params={},
        distributions={},
        user_attrs=user_attrs,
    )


def config_with_max_model_len(value: int) -> ExperimentConfig:
    """A copy of :func:`base_config` whose distinguishing field is ``max_model_len``."""
    config = base_config()
    return replace(config, llm=replace(config.llm, max_model_len=value))


def multi_result() -> OptimizationResult:
    """A finished multi sweep whose three-point front includes the recommended point.

    The recommended point is the middle of the front, and the solutions are
    deliberately stored out of trial order — as Optuna's ``best_trials`` may
    return them — so the table's ordering cannot pass by accident.
    """
    study = optuna.create_study(directions=["maximize", "maximize"])
    for _, prefill, decode, combined, _ in _FRONT:
        study.add_trial(completed_multi_trial(prefill, decode, combined=combined))
    solutions = {
        number: OptimizationSolution(
            config=config_with_max_model_len(max_model_len),
            value=(prefill, decode),
            trial_number=number,
        )
        for number, prefill, decode, _, max_model_len in _FRONT
    }
    return OptimizationResult(
        solutions=(solutions[2], solutions[0], solutions[1]),
        recommended=solutions[1],
        trial_count=4,
        study=study,
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


def test_a_multi_summary_names_the_recommended_point_and_the_trial_count() -> None:
    summary = render_summary(multi_result(), target=TARGET, image=IMAGE)

    assert "recommended trial: 1" in summary
    assert "recommended rates: prefill 500.00 tps  decode 200.00 tps" in summary
    assert "trials run: 4" in summary


def test_a_multi_summary_prints_the_pareto_front_as_a_table() -> None:
    """Rows read in trial order however the study stored them, with the marker column."""
    summary = render_summary(multi_result(), target=TARGET, image=IMAGE)

    table = summary.split("pareto front:\n", 1)[1].split("\n\nwinning configuration:", 1)[0]

    assert table.splitlines() == [
        "trial  prefill tps  decode tps  combined tps  recommended",
        "0      1000.00      50.00       128.00",
        "1      500.00       200.00      96.00         yes",
        "2      200.00       500.00      64.00",
    ]


def test_a_front_trial_without_a_recorded_combined_rate_renders_na() -> None:
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.add_trial(completed_multi_trial(300.0, 60.0))
    solution = OptimizationSolution(
        config=config_with_max_model_len(512), value=(300.0, 60.0), trial_number=0
    )
    result = OptimizationResult(
        solutions=(solution,), recommended=solution, trial_count=1, study=study
    )

    summary = render_summary(result, target=TARGET, image=IMAGE)

    assert "0      300.00       60.00       n/a" in summary


def test_a_multi_summary_carries_the_recommended_configuration() -> None:
    summary = render_summary(multi_result(), target=TARGET, image=IMAGE)

    assert "winning configuration:" in summary
    assert "max_model_len: 2048" in summary
    assert "max_model_len: 4096" not in summary
    assert "max_model_len: 1024" not in summary


def test_a_multi_summary_and_the_file_share_the_recommended_document(tmp_path: Path) -> None:
    path = tmp_path / "best.yaml"
    result = multi_result()

    write_config(path, result.recommended.config, target=TARGET, image=IMAGE)
    summary = render_summary(result, target=TARGET, image=IMAGE)

    printed = summary.split("winning configuration:\n", 1)[1]

    assert yaml.safe_load(printed) == yaml.safe_load(path.read_text(encoding="utf-8"))
    assert yaml.safe_load(printed)["llm"]["max_model_len"] == 2048
