"""Unit tests for the engine-neutral objective model and scoring functions.

The objective's observable contract is the value a sweep maximizes: the
weighted-harmonic effective rate of a result's phase rates, the combined
throughput read through unchanged, or — for a multi-objective sweep — the two
phase rates themselves. These tests drive the pure functions directly with
hand-built results, so no optimizer, engine, or daemon is involved. They pin
the formula, the zero-weight degeneracy, the zero-rate limit, scaling
invariance, the constructor's rejection of a weights pair that describes no
token mix, and the clear failure when a rate the objective needs is absent.
"""

import pytest

from llm_config_optimization.inference import (
    ExperimentError,
    ExperimentResult,
    HardwareTarget,
    RunMetrics,
)
from llm_config_optimization.optimization import (
    ObjectiveMode,
    ObjectiveWeights,
    combined_score,
    objective_value,
    phase_rates,
    weighted_harmonic_rate,
    weighted_score,
)
from llm_config_optimization.runtime import ContainerResult

COMBINED = 42.0
PREFILL = 1000.0
DECODE = 50.0
WEIGHTS = ObjectiveWeights(prefill=1024, decode=128)
"""A workload's token mix: 1024 prompt and 128 generated tokens per request."""


def _result(
    *,
    prefill: float | None = None,
    decode: float | None = None,
    tokens_per_second: float = COMBINED,
) -> ExperimentResult:
    """Build a one-run result carrying the given rates."""
    return ExperimentResult(
        runs=(
            RunMetrics(
                tokens_per_second=tokens_per_second,
                requests_per_second=4.0,
                elapsed_time=2.5,
                num_prompts=8,
                raw_json={},
                prefill_tokens_per_second=prefill,
                decode_tokens_per_second=decode,
            ),
        ),
        failed_runs=0,
        target=HardwareTarget.CPU,
        image="fake/engine:latest",
        devices=(),
        raw_result=ContainerResult(exit_code=0, stdout=b"", stderr=b"", container_id="fake"),
    )


def test_objective_mode_values_are_the_command_line_spellings() -> None:
    assert [mode.value for mode in ObjectiveMode] == ["multi", "weighted", "combined"]


def test_combined_score_passes_the_result_through_unchanged() -> None:
    assert combined_score(_result(prefill=PREFILL, decode=DECODE)) == COMBINED


def test_combined_score_tolerates_absent_phase_rates() -> None:
    """The escape hatch keeps working on an engine that reports no phases."""
    assert combined_score(_result()) == COMBINED


def test_phase_rates_returns_the_pair_a_multi_sweep_maximizes() -> None:
    assert phase_rates(_result(prefill=PREFILL, decode=DECODE)) == (PREFILL, DECODE)


def test_phase_rates_rejects_each_missing_rate() -> None:
    with pytest.raises(ExperimentError, match="prefill"):
        phase_rates(_result(decode=DECODE))
    with pytest.raises(ExperimentError, match="decode"):
        phase_rates(_result(prefill=PREFILL))


def test_weighted_score_is_the_weighted_harmonic_rate() -> None:
    # (1024 + 128) / (1024/1000 + 128/50) for a 1024-token prompt and a
    # 128-token response.
    result = _result(prefill=PREFILL, decode=DECODE)

    assert weighted_score(result, WEIGHTS) == pytest.approx(1152.0 / 3.584)


def test_weighted_harmonic_rate_scores_a_raw_phase_pair() -> None:
    """The recommendation seam scores rates without an ExperimentResult."""
    assert weighted_harmonic_rate(PREFILL, DECODE, WEIGHTS) == pytest.approx(1152.0 / 3.584)
    assert weighted_harmonic_rate(PREFILL, DECODE, ObjectiveWeights(prefill=0, decode=1)) == DECODE
    assert weighted_harmonic_rate(0.0, DECODE, ObjectiveWeights(prefill=1, decode=1)) == 0.0


def test_a_both_positive_score_lies_between_the_phase_rates() -> None:
    result = _result(prefill=PREFILL, decode=DECODE)

    score = weighted_score(result, ObjectiveWeights(prefill=1, decode=1))

    assert DECODE < score < PREFILL


def test_scaling_both_weights_leaves_the_score_unchanged() -> None:
    """Only the ratio is a token mix; equal mixes score equally."""
    result = _result(prefill=PREFILL, decode=DECODE)

    assert weighted_score(result, ObjectiveWeights(prefill=8, decode=1)) == pytest.approx(
        weighted_score(result, ObjectiveWeights(prefill=8000, decode=1000))
    )


def test_a_zero_weight_degenerates_exactly_to_the_other_phase() -> None:
    result = _result(prefill=PREFILL, decode=DECODE)

    assert weighted_score(result, ObjectiveWeights(prefill=0, decode=1)) == DECODE
    assert weighted_score(result, ObjectiveWeights(prefill=1, decode=0)) == PREFILL


def test_a_zero_weight_does_not_require_the_unused_rate() -> None:
    assert weighted_score(_result(decode=DECODE), ObjectiveWeights(prefill=0, decode=1)) == DECODE
    assert (
        weighted_score(_result(prefill=PREFILL), ObjectiveWeights(prefill=1, decode=0)) == PREFILL
    )


def test_a_zero_phase_rate_yields_a_zero_score() -> None:
    """A phase that moves no tokens is the formula's zero-rate limit."""
    weights = ObjectiveWeights(prefill=1, decode=1)

    assert weighted_score(_result(prefill=0.0, decode=DECODE), weights) == 0.0
    assert weighted_score(_result(prefill=PREFILL, decode=0.0), weights) == 0.0


def test_weighted_score_rejects_each_missing_rate_it_needs() -> None:
    weights = ObjectiveWeights(prefill=1, decode=1)

    with pytest.raises(ExperimentError, match="prefill"):
        weighted_score(_result(decode=DECODE), weights)
    with pytest.raises(ExperimentError, match="decode"):
        weighted_score(_result(prefill=PREFILL), weights)


def test_a_missing_rate_names_the_sources_a_backend_could_report_from() -> None:
    with pytest.raises(ExperimentError) as excinfo:
        weighted_score(_result(decode=DECODE), ObjectiveWeights(prefill=1, decode=1))

    assert "/metrics" in str(excinfo.value)
    assert "timings" in str(excinfo.value)


def test_objective_value_dispatches_by_mode() -> None:
    result = _result(prefill=PREFILL, decode=DECODE)

    assert objective_value(ObjectiveMode.MULTI, result) == (PREFILL, DECODE)
    assert objective_value(ObjectiveMode.WEIGHTED, result, WEIGHTS) == weighted_score(
        result, WEIGHTS
    )
    assert objective_value(ObjectiveMode.COMBINED, result) == combined_score(result)


def test_objective_value_requires_weights_for_the_weighted_mode() -> None:
    with pytest.raises(ValueError, match="weights"):
        objective_value(ObjectiveMode.WEIGHTED, _result(prefill=PREFILL, decode=DECODE))


def test_objective_weights_allow_a_zero_weight() -> None:
    weights = ObjectiveWeights(prefill=0, decode=1)

    assert (weights.prefill, weights.decode) == (0, 1)


def test_objective_weights_reject_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveWeights(prefill=-1, decode=1)


def test_objective_weights_reject_two_zero_weights() -> None:
    with pytest.raises(ValueError, match="zero"):
        ObjectiveWeights(prefill=0, decode=0)


def test_objective_weights_reject_non_finite_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        ObjectiveWeights(prefill=float("nan"), decode=1)
