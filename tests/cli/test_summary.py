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

The memory figures are informational metadata read from the study's per-trial
result projections. The hardware line reports the largest peak any evaluated
trial recorded, and the `memory` subsection of the winning configuration reports
the recommended configuration's own footprint, so both are asserted for a fully
measured study, a partially measured one, and a study that recorded no memory at
all — where each missing field renders ``n/a`` rather than a zero.
"""

from dataclasses import fields, replace
from pathlib import Path

import optuna
import pytest
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
from llm_config_optimization.output import OutputFormat

TARGET = HardwareTarget.CUDA
IMAGE = "vllm/vllm-openai:latest"
_GIB = 1024**3


def gib(gigabytes: float) -> int:
    """A byte count the byte-size format renders back as ``gigabytes GiB``."""
    return round(gigabytes * _GIB)


def projection(
    *,
    combined: float = 42.0,
    lifetime_vram: int | None = None,
    lifetime_ram: int | None = None,
    runs: tuple[tuple[int | None, int | None], ...] = (),
) -> dict[str, object]:
    """The result projection the optimizer records on a successful trial.

    Only the keys these tests read are filled in; the real projection carries
    the full aggregate metrics beside them. ``runs`` is one
    ``(peak_vram_bytes, peak_ram_bytes)`` pair per measured run.
    """
    return {
        "tokens_per_second": combined,
        "lifetime_peak_vram_bytes": lifetime_vram,
        "lifetime_peak_ram_bytes": lifetime_ram,
        "runs": [{"peak_vram_bytes": vram, "peak_ram_bytes": ram} for vram, ram in runs],
    }


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
    study = optuna.create_study(direction="maximize")
    for _ in range(5):
        study.add_trial(completed_scalar_trial())
    return OptimizationResult(
        solutions=(solution,),
        recommended=solution,
        trial_count=5,
        study=study,
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
    recorded: dict[str, object] | None = None,
) -> optuna.trial.FrozenTrial:
    """A frozen multi trial carrying its phase objectives and recorded projection.

    The optimizer records a JSON projection of every successful run under the
    ``result`` user attribute; ``recorded`` supplies it, typically built with
    :func:`projection`. Omitting it exercises the trial whose record is absent.
    """
    user_attrs: dict[str, object] = {}
    if recorded is not None:
        user_attrs["result"] = recorded
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


def multi_result(peaks: dict[int, int] | None = None) -> OptimizationResult:
    """A finished multi sweep whose three-point front includes the recommended point.

    The recommended point is the middle of the front, and the solutions are
    deliberately stored out of trial order — as Optuna's ``best_trials`` may
    return them — so the table's ordering cannot pass by accident. ``peaks``
    maps a trial number to the lifetime VRAM its projection recorded; without
    it the trials carry their combined rate alone, as a study recorded before
    memory was measured does.
    """
    study = optuna.create_study(directions=["maximize", "maximize"])
    for number, prefill, decode, combined, _ in _FRONT:
        lifetime = None if peaks is None else peaks[number]
        runs = () if lifetime is None else ((lifetime - gib(1.0), None),)
        study.add_trial(
            completed_multi_trial(
                prefill,
                decode,
                recorded=projection(combined=combined, lifetime_vram=lifetime, runs=runs),
            )
        )
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
        trial_count=3,
        study=study,
    )


def failed_trial() -> optuna.trial.FrozenTrial:
    """A finished trial whose terminal state is ``FAIL``."""
    return optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL)


def completed_scalar_trial(recorded: dict[str, object] | None = None) -> optuna.trial.FrozenTrial:
    """A finished scalar trial with a value the summary's headline can read.

    ``recorded`` fills in the result projection the optimizer writes under the
    ``result`` user attribute; omitting it leaves the trial exactly as a study
    recorded before memory was measured holds it.
    """
    user_attrs: dict[str, object] = {}
    if recorded is not None:
        user_attrs["result"] = recorded
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=1.0,
        distributions={},
        user_attrs=user_attrs,
    )


def recorded_result(recorded: dict[str, object] | None) -> OptimizationResult:
    """A finished scalar sweep whose recommended trial carries ``recorded``."""
    solution = OptimizationSolution(config=base_config(), value=42.0, trial_number=0)
    study = optuna.create_study(direction="maximize")
    study.add_trial(completed_scalar_trial(recorded))
    return OptimizationResult(
        solutions=(solution,),
        recommended=solution,
        trial_count=1,
        study=study,
    )


def test_the_summary_names_the_best_trial_throughput_and_count() -> None:
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert "best trial: 3" in summary
    assert "best throughput: 42.00 tps" in summary
    assert "trials run: 5" in summary


def test_a_clean_summary_reports_zero_failures() -> None:
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert "failed trials: 0/5 (0%)" in summary


def test_the_hardware_line_follows_the_failure_line() -> None:
    """The footprint sits with the sweep's other record lines, before the config."""
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    records = [
        line.split(":", 1)[0]
        for line in summary.split("\n")
        if line.startswith(
            ("best trial:", "trials run:", "failed trials:", "hardware (all trials):")
        )
    ]

    assert records == ["best trial", "trials run", "failed trials", "hardware (all trials)"]


def test_a_measured_trial_reports_its_lifetime_and_measured_peaks() -> None:
    measured = recorded_result(
        projection(
            lifetime_vram=gib(20.4),
            lifetime_ram=gib(6.2),
            runs=((gib(16.5), gib(5.5)), (gib(17.1), gib(5.9))),
        )
    )

    summary = render_summary(measured, target=TARGET, image=IMAGE)

    assert (
        "hardware (all trials): 20.4 GiB vram peak (measured 17.1 GiB), "
        "6.2 GiB ram peak (measured 5.9 GiB)" in summary
    )


def test_the_measured_peak_is_the_maximum_over_the_runs() -> None:
    """The peak spans the run windows, so the later, larger run decides it."""
    measured = recorded_result(
        projection(
            lifetime_vram=gib(24.0),
            runs=((gib(17.1), None), (gib(21.0), None), (gib(3.0), None)),
        )
    )

    summary = render_summary(measured, target=TARGET, image=IMAGE)

    assert "hardware (all trials): 24.0 GiB vram peak (measured 21.0 GiB)" in summary


def test_a_partially_measured_trial_renders_the_missing_fields_as_na() -> None:
    """A CPU target has RAM but no VRAM, and a run may measure only one half."""
    measured = recorded_result(
        projection(lifetime_ram=gib(6.2), runs=((None, gib(5.9)), (None, None)))
    )

    summary = render_summary(measured, target=TARGET, image=IMAGE)

    assert (
        "hardware (all trials): n/a vram peak (measured n/a), 6.2 GiB ram peak (measured 5.9 GiB)"
        in summary
    )


def test_a_trial_without_any_measurement_renders_every_field_as_na() -> None:
    measured = recorded_result(projection(runs=((None, None),)))

    summary = render_summary(measured, target=TARGET, image=IMAGE)

    assert (
        "hardware (all trials): n/a vram peak (measured n/a), n/a ram peak (measured n/a)"
        in summary
    )


def test_a_study_recorded_before_memory_was_measured_still_summarizes() -> None:
    """A legacy projection lacks the keys entirely, which is not an error."""
    summary = render_summary(
        recorded_result({"tokens_per_second": 42.0}), target=TARGET, image=IMAGE
    )

    assert (
        "hardware (all trials): n/a vram peak (measured n/a), n/a ram peak (measured n/a)"
        in summary
    )
    assert "best throughput: 42.00 tps" in summary


def test_a_trial_without_a_recorded_projection_reports_no_measurement() -> None:
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert (
        "hardware (all trials): n/a vram peak (measured n/a), n/a ram peak (measured n/a)"
        in summary
    )


def test_a_multi_summary_splits_the_study_maximum_from_the_recommended_point() -> None:
    """The line is the sweep's heaviest trial; the subsection is the recommended one."""
    result = multi_result(peaks={0: gib(40.0), 1: gib(17.1), 2: gib(8.0)})

    summary = render_summary(result, target=TARGET, image=IMAGE)

    assert "hardware (all trials): 40.0 GiB vram peak (measured 39.0 GiB)" in summary
    assert (
        "winning configuration:\n"
        "memory:\n"
        "  vram: 17.1 GiB lifetime peak (16.1 GiB while serving)\n"
        "  ram:  n/a lifetime peak (n/a while serving)\n"
        "container:"
    ) in summary
    assert summary.index("hardware (all trials):") < summary.index("winning configuration:")
    assert summary.index("pareto front:") < summary.index("winning configuration:")


def test_the_hardware_line_takes_each_maximum_over_every_trial() -> None:
    """The line's two resources are maximums over the study, not one trial's pair."""
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        completed_scalar_trial(
            projection(
                lifetime_vram=gib(18.9),
                lifetime_ram=gib(9.8),
                runs=((gib(17.0), gib(9.1)),),
            )
        )
    )
    study.add_trial(
        completed_scalar_trial(
            projection(
                lifetime_vram=gib(20.4),
                lifetime_ram=gib(6.2),
                runs=((gib(17.1), gib(5.9)),),
            )
        )
    )
    solution = OptimizationSolution(config=base_config(), value=42.0, trial_number=0)
    summary_result = OptimizationResult(
        solutions=(solution,), recommended=solution, trial_count=2, study=study
    )

    summary = render_summary(summary_result, target=TARGET, image=IMAGE)

    assert (
        "hardware (all trials): 20.4 GiB vram peak (measured 17.1 GiB), "
        "9.8 GiB ram peak (measured 9.1 GiB)" in summary
    )
    assert (
        "winning configuration:\n"
        "memory:\n"
        "  vram: 18.9 GiB lifetime peak (17.0 GiB while serving)\n"
        "  ram:  9.8 GiB lifetime peak (9.1 GiB while serving)"
    ) in summary


def test_a_failed_trial_contributes_nothing_to_the_study_maximum() -> None:
    """A trial that never recorded a projection is skipped, not counted as zero."""
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        completed_scalar_trial(projection(lifetime_vram=gib(20.4), runs=((gib(17.1), None),)))
    )
    study.add_trial(failed_trial())
    solution = OptimizationSolution(config=base_config(), value=42.0, trial_number=0)
    summary_result = OptimizationResult(
        solutions=(solution,), recommended=solution, trial_count=2, study=study
    )

    summary = render_summary(summary_result, target=TARGET, image=IMAGE)

    assert "hardware (all trials): 20.4 GiB vram peak (measured 17.1 GiB)" in summary


def test_an_unmeasured_winning_trial_omits_the_memory_subsection() -> None:
    """A study recorded before memory was measured prints only its configuration."""
    summary = render_summary(result(), target=TARGET, image=IMAGE)

    assert "winning configuration:\ncontainer:" in summary
    assert "\nmemory:" not in summary
    assert "winning trial memory:" not in summary


def test_a_measured_winning_trial_keeps_the_document_after_its_memory(
    tmp_path: Path,
) -> None:
    """The subsection precedes the document without becoming part of it."""
    path = tmp_path / "best.yaml"
    measured = recorded_result(
        projection(
            lifetime_vram=gib(20.4),
            lifetime_ram=gib(6.2),
            runs=((gib(17.1), gib(5.9)),),
        )
    )

    write_config(path, base_config(), target=TARGET, image=IMAGE)
    summary = render_summary(measured, target=TARGET, image=IMAGE)

    section = summary.split("winning configuration:\n", 1)[1]
    document = "container:" + section.split("\ncontainer:", 1)[1]
    assert yaml.safe_load(document) == yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_summary_counts_failed_trials_across_the_whole_study() -> None:
    study = optuna.create_study(direction="maximize")
    for trial in (completed_scalar_trial(), completed_scalar_trial(), failed_trial()):
        study.add_trial(trial)
    summary_result = replace(result(), study=study, trial_count=3)

    summary = render_summary(summary_result, target=TARGET, image=IMAGE)

    assert "failed trials: 1/3 (33%)" in summary


def test_the_summary_shows_the_sub_one_percent_floor() -> None:
    study = optuna.create_study(direction="maximize")
    for _ in range(199):
        study.add_trial(completed_scalar_trial())
    study.add_trial(failed_trial())
    summary_result = replace(result(), study=study, trial_count=200)

    summary = render_summary(summary_result, target=TARGET, image=IMAGE)

    assert "failed trials: 1/200 (<1%)" in summary


def test_a_multi_summary_reports_failed_trials() -> None:
    result = multi_result()
    result.study.add_trial(failed_trial())
    summary_result = replace(result, trial_count=4)

    summary = render_summary(summary_result, target=TARGET, image=IMAGE)

    assert "failed trials: 1/4 (25%)" in summary


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
    assert document["llm"]["speculative_method"] is None
    assert document["llm"]["draft_model"] is None
    assert document["llm"]["speculative_mode"] is None
    assert document["llm"]["num_speculative_tokens"] is None
    assert document["llm"]["resolved_draft_path"] is None
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


def test_write_config_uses_the_registered_yaml_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary selects its renderer through the registry, never a concrete class."""
    selected: list[OutputFormat] = []

    class RecordingWriter:
        requires_plan = False

        def render(self, context: object) -> str:
            return "rendered: true\n"

    def fake_writer_for(output_format: OutputFormat) -> RecordingWriter:
        selected.append(output_format)
        return RecordingWriter()

    monkeypatch.setattr("llm_config_optimization.cli.summary.writer_for", fake_writer_for)
    path = tmp_path / "best.yaml"

    write_config(path, base_config(), target=TARGET, image=IMAGE)

    assert selected == [OutputFormat.YAML]
    assert path.read_text(encoding="utf-8") == "rendered: true\n"


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
    assert "trials run: 3" in summary


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
