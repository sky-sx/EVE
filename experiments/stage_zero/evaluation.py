"""Descriptive Stage Zero metrics and a declared conservative evidence rubric.

No statistic in this module enters ACNT or changes an action or parameter.
"""

import math


EVIDENCE_CRITERIA = (
    "SUPPORTED requires at least three seeds, each with training exact success "
    "above initialization, non-overlapping Wilson 95% intervals (frozen lower "
    "above initial upper), target probability increasing and non-target "
    "probability decreasing in both training and frozen evaluation. PARTIAL "
    "requires at least one seed with higher training and frozen exact success "
    "and the two correct frozen probability directions. Otherwise NOT SUPPORTED. "
    "These labels concern observed learning under this protocol, not the "
    "possibility of learning by the architecture."
)


def _mean(rows, key):
    return math.fsum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _metrics(rows):
    return {
        "episode_count": len(rows),
        "mean_goodness": _mean(rows, "goodness"),  # Training scalar, not behavioral success.
        "exact_success_rate": _mean(rows, "correct_exact_match"),
        "target_probability": _mean(rows, "target_probability"),
        "non_target_probability": _mean(rows, "non_target_probability"),
        "target_bit_correct_rate": _mean(rows, "target_bit_correct"),
        "non_target_false_rate": _mean(rows, "non_target_false_rate"),
        "mean_active_actions": _mean(rows, "active_action_count"),
        "expected_success_count": math.fsum(float(row["exact_probability"]) for row in rows),
        "max_parameter_delta_norm": max((float(row["parameter_delta_norm"]) for row in rows), default=0.0),
        "nan_count": sum(int(row["nan_count"]) for row in rows),
        "inf_count": sum(int(row["inf_count"]) for row in rows),
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize episodes and every observed class, including zero successes."""
    result = _metrics(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["target_class"], row["target_action"]), []).append(row)
    result["per_class"] = [
        {"target_class": class_name, "target_action": action,
         "episodes": len(group), **_metrics(group)}
        for (class_name, action), group in sorted(
            grouped.items(),
            key=lambda pair: (0, pair[0][0]) if isinstance(pair[0][0], int)
            else (1, pair[0][1] == "MOUSE_LEFT", str(pair[0][1])),
        )
    ]
    return result


def learning_curve(rows: list[dict], window: int) -> list[dict]:
    """Aggregate consecutive windows separately by phase, without reordering.

    The runner uses a multiple-of-27 window and balanced 27-class cycles.
    This function preserves any final partial window rather than dropping data.
    """
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    result = []
    chunk = []

    def append_chunk():
        result.append({
            "phase": chunk[0]["phase"],
            "start_episode": chunk[0]["episode"],
            "end_episode": chunk[-1]["episode"],
            "count": len(chunk),
            **_metrics(chunk),
        })

    for row in rows:
        if chunk and (len(chunk) == window or row["phase"] != chunk[0]["phase"]):
            append_chunk()
            chunk = []
        chunk.append(row)
    if chunk:
        append_chunk()
    return result


def _wilson(rate, count):
    if count <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def classify_evidence(seed_summaries: list[dict]) -> str:
    """Apply EVIDENCE_CRITERIA; changed parameters alone cannot qualify."""
    clear = []
    directional = []
    for seed in seed_summaries:
        initial, training, frozen = (seed[key] for key in ("initial", "training", "frozen"))
        complete = all(phase["episode_count"] > 0 for phase in (initial, training, frozen))
        stable = all(phase["nan_count"] == phase["inf_count"] == 0 for phase in (initial, training, frozen))
        reward_improved = (
            training["exact_success_rate"] > initial["exact_success_rate"]
            and frozen["exact_success_rate"] > initial["exact_success_rate"]
        )
        frozen_direction = (
            frozen["target_probability"] > initial["target_probability"]
            and frozen["non_target_probability"] < initial["non_target_probability"]
        )
        training_direction = (
            training["target_probability"] > initial["target_probability"]
            and training["non_target_probability"] < initial["non_target_probability"]
        )
        initial_upper = _wilson(initial["exact_success_rate"], initial["episode_count"])[1]
        frozen_lower = _wilson(frozen["exact_success_rate"], frozen["episode_count"])[0]
        directional.append(complete and stable and reward_improved and frozen_direction)
        clear.append(directional[-1] and training_direction and frozen_lower > initial_upper)
    if len(clear) >= 3 and all(clear):
        return "SUPPORTED"
    if any(directional):
        return "PARTIAL"
    return "NOT SUPPORTED"
