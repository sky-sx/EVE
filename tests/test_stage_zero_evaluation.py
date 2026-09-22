from copy import deepcopy

import pytest

from experiments.stage_zero.evaluation import classify_evidence, learning_curve, summarize


def row(episode=0, phase="training", target="A", correct=0):
    return {
        "episode": episode, "phase": phase, "target_class": target,
        "target_action": target, "goodness": 0.375, "fractional_goodness": 0.375,
        "correct_exact_match": correct, "target_probability": 0.6,
        "non_target_probability": 0.4, "target_bit_correct": 1,
        "non_target_false_rate": 0.25, "active_action_count": 7,
        "exact_probability": 0.125, "parameter_delta_norm": 0.01,
        "nan_count": 0, "inf_count": 0,
    }


def test_summary_retains_failed_classes_and_descriptive_metrics():
    rows = [row(0, target="A", correct=1), row(1, target="B"), row(2, target="B")]
    result = summarize(rows)
    assert result["episode_count"] == 3
    assert result["mean_goodness"] == 0.375
    assert result["exact_success_rate"] == pytest.approx(1 / 3)
    assert result["expected_success_count"] == 0.375
    assert result["target_bit_correct_rate"] == 1
    assert result["max_parameter_delta_norm"] == 0.01
    assert result["nan_count"] == result["inf_count"] == 0
    assert [item["target_class"] for item in result["per_class"]] == ["A", "B"]
    assert result["per_class"][1]["episodes"] == 2
    assert result["per_class"][1]["exact_success_rate"] == 0


def test_empty_summary_is_explicit_and_finite():
    result = summarize([])
    assert result["episode_count"] == 0
    assert result["per_class"] == []
    assert result["mean_goodness"] == 0


def test_per_class_uses_numeric_environment_order():
    rows = [row(index, target=index) for index in reversed(range(27))]
    result = summarize(rows)
    assert [item["target_class"] for item in result["per_class"]] == list(range(27))


def test_learning_curve_keeps_partial_windows_and_phase_boundaries():
    rows = [row(0, "initial"), row(1, "initial")]
    rows += [row(episode, correct=episode % 2) for episode in range(2, 7)]
    curve = learning_curve(rows, 3)
    assert [(point["phase"], point["count"]) for point in curve] == [
        ("initial", 2), ("training", 3), ("training", 2),
    ]
    assert [(point["start_episode"], point["end_episode"]) for point in curve] == [(0, 1), (2, 4), (5, 6)]
    assert all("per_class" not in point for point in curve)
    assert sum(point["count"] for point in curve) == len(rows)
    assert learning_curve([], 27) == []


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_learning_curve_rejects_invalid_windows(window):
    with pytest.raises(ValueError):
        learning_curve([], window)


def seed_result(initial=0.0, training=0.2, frozen=0.4, count=270):
    def phase(rate, target, non_target):
        return {
            "episode_count": count, "exact_success_rate": rate,
            "target_probability": target, "non_target_probability": non_target,
            "nan_count": 0, "inf_count": 0,
        }
    return {"initial": phase(initial, 0.5, 0.5),
            "training": phase(training, 0.6, 0.4),
            "frozen": phase(frozen, 0.7, 0.3)}


def test_evidence_requires_repeated_confident_frozen_improvement():
    seed = seed_result()
    assert classify_evidence([deepcopy(seed) for _ in range(3)]) == "SUPPORTED"
    assert classify_evidence([seed]) == "PARTIAL"
    failed = seed_result(training=0.0, frozen=0.0)
    assert classify_evidence([seed, seed, failed]) == "PARTIAL"


def test_no_reward_or_only_parameter_and_tendency_movement_is_not_learning():
    seed = seed_result(training=0.0, frozen=0.0)
    assert classify_evidence([seed] * 5) == "NOT SUPPORTED"
    assert classify_evidence([]) == "NOT SUPPORTED"


def test_frozen_retention_and_correct_probability_directions_are_required():
    no_retention = seed_result(frozen=0.0)
    wrong_direction = seed_result()
    wrong_direction["frozen"]["non_target_probability"] = 0.6
    assert classify_evidence([no_retention] * 3) == "NOT SUPPORTED"
    assert classify_evidence([wrong_direction] * 3) == "NOT SUPPORTED"


def test_small_sample_directional_change_cannot_be_supported():
    seed = seed_result(training=1 / 27, frozen=1 / 27, count=27)
    assert classify_evidence([seed] * 3) == "PARTIAL"


def test_unstable_or_empty_phase_cannot_support_evidence():
    seed = seed_result()
    seed["training"]["nan_count"] = 1
    assert classify_evidence([seed] * 3) == "NOT SUPPORTED"
    seed = seed_result(count=0)
    assert classify_evidence([seed] * 3) == "NOT SUPPORTED"


def test_fractional_goodness_improvement_alone_is_not_learning():
    seed = seed_result(training=0.0, frozen=0.0)
    for phase, value in (("initial", 0.1), ("training", 0.6), ("frozen", 0.9)):
        seed[phase]["mean_goodness"] = value
    assert classify_evidence([seed] * 5) == "NOT SUPPORTED"
