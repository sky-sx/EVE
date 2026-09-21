"""Recompute formal results from raw CSV; never run learning or alter weights.

Usage: python -m experiments.stage_zero.audit runs/stage_zero/formal-20260920
Writes additional audit/pooled summaries, preserving every original run file.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

from .environment import ACTIONS, TeacherTable
from .evaluation import summarize, learning_curve, classify_evidence


NUMERIC = ("goodness", "teacher_goodness", "target_q", "target_probability", "non_target_probability",
           "non_target_false_rate", "exact_probability", "eligibility_norm_at_action",
           "g_bar_before", "g_bar", "parameter_norm", "parameter_delta_norm", "eligibility_norm")
INTEGER = ("correct_pressed", "wrong_count", "episode", "phase_episode", "visual_time_ms", "logical_time_ms", "target_class",
           "goodness_delivery_time", "correct_exact_match", "target_bit_correct", "active_action_count")


def load_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in NUMERIC:
            row[name] = float(row[name])
        for name in INTEGER:
            row[name] = int(row[name])
        for name in ("nan_count", "inf_count"):
            row[name] = int(float(row[name]))
        row["delta"] = None if row["delta"] == "" else float(row["delta"])
        row["learning_enabled"] = row["learning_enabled"] == "True"
        for name in ("q", "p", "action_bits", "sampled_actions"):
            row[name] = json.loads(row[name])
    return rows


def validate_row(row, protocol, teacher_table):
    assert all(math.isfinite(row[key]) for key in NUMERIC), "nonfinite log"
    assert row["nan_count"] == row["inf_count"] == 0, "nonfinite state"
    q, p, bits = (row[key] for key in ("q", "p", "action_bits"))
    assert len(q) == len(p) == len(bits) == 27, "wrong Hand dimensions"
    assert all(type(bit) is bool for bit in bits), "nonbinary action"
    assert all(math.isfinite(x) for x in q + p), "nonfinite Hand output"
    assert all(0 <= value <= 1 for value in p), "invalid probability"
    target = row["target_class"]
    assert row["target_action"] == ACTIONS[target], "target mapping"
    assert 0 <= target < 27, "invalid target"
    correct = int(bits[target])
    wrong = sum(bit for i, bit in enumerate(bits) if i != target)
    exact = int(correct == 1 and wrong == 0)
    reward = teacher_table.lookup(correct, wrong)
    assert row["correct_pressed"] == correct and row["wrong_count"] == wrong, "action bucket mismatch"
    assert row["goodness"] == row["teacher_goodness"] == reward, "Teacher table value mismatch"
    assert row["correct_exact_match"] == exact, "exact match mismatch"
    assert row["sampled_actions"] == [name for name, bit in zip(ACTIONS, bits) if bit], "action log"
    assert row["active_action_count"] == sum(bits), "active count"
    assert row["target_bit_correct"] == int(bits[target]), "target hit"
    assert row["target_probability"] == p[target] and row["target_q"] == q[target], "target tendency"
    assert math.isclose(row["non_target_probability"], sum(x for i, x in enumerate(p) if i != target) / 26), "wrong tendency"
    assert math.isclose(row["non_target_false_rate"], (sum(bits) - bits[target]) / 26), "false rate"
    expected = p[target] * math.prod(1 - value for i, value in enumerate(p) if i != target)
    assert math.isclose(row["exact_probability"], expected, rel_tol=1e-12), "event probability"
    assert row["logical_time_ms"] - row["visual_time_ms"] == 500, "presentation timing"
    assert row["goodness_delivery_time"] - row["logical_time_ms"] == 250, "delivery timing"
    if row["phase"] == "training":
        assert row["learning_enabled"], "training disabled"
        assert row["delta"] == reward - row["g_bar_before"], "scalar delta"
        assert math.isclose(row["g_bar"], row["g_bar_before"] + protocol["ema_alpha"] * row["delta"], rel_tol=1e-12), "baseline"
        expected_trace_norm = row["eligibility_norm_at_action"] * math.exp(-0.250 / protocol["ticktime"]) * protocol["rho"]
        assert math.isclose(row["eligibility_norm"], expected_trace_norm, rel_tol=2e-5, abs_tol=1e-8), "eligibility decay"
    else:
        assert not row["learning_enabled"] and row["delta"] is None, "evaluation learning"
        assert row["parameter_delta_norm"] == 0.0, "evaluation changed parameters"
        assert row["g_bar"] == row["g_bar_before"], "evaluation changed baseline"


def validate_table_artifact(directory, metadata, config):
    table = TeacherTable.load(directory / "teacher_goodness.json")
    assert table.sha256 == metadata["teacher_table_sha256"] == config["teacher_table_sha256"], "Teacher table hash mismatch"
    assert (directory / "teacher_goodness.sha256").read_text().strip() == table.sha256, "Teacher table hash sidecar"
    assert table.metadata == metadata["teacher_calibration"], "Teacher metadata mismatch"
    assert json.loads((directory / "teacher_calibration_metadata.json").read_text(encoding="utf-8")) == table.metadata, "Teacher metadata copy"
    return table


def audit(directory: Path):
    directory = Path(directory)
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert metadata["completed"], "run incomplete"
    teacher_table = validate_table_artifact(directory, metadata, config)
    root = Path(__file__).resolve().parents[2]
    mismatches = [name for name, digest in metadata["source_sha256"].items()
                  if hashlib.sha256((root / name.replace("\\", "/")).read_bytes()).hexdigest() != digest]
    assert not mismatches, f"run sources changed: {mismatches}"
    pooled, seed_metrics = [], []
    aggregate_expected = []
    for seed in config["seeds"]:
        seed_dir = directory / f"seed_{seed:03d}"
        rows = load_rows(seed_dir / "episode_log.csv")
        expected_count = config["training_episodes"] + 2 * config["evaluation_episodes"]
        assert len(rows) == expected_count
        summary = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["teacher_table_sha256"] == teacher_table.sha256, "seed table hash"
        assert (seed_dir / "teacher_goodness.json").read_bytes() == teacher_table.raw, "seed table copy"
        for index, row in enumerate(rows):
            assert row["episode"] == index
            validate_row(row, config, teacher_table)
        for phase, count in (("initial", config["evaluation_episodes"]),
                             ("training", config["training_episodes"]),
                             ("frozen", config["evaluation_episodes"])):
            phase_rows = [r for r in rows if r["phase"] == phase]
            assert len(phase_rows) == count
            assert [r["phase_episode"] for r in phase_rows] == list(range(count))
            assert Counter(r["target_class"] for r in phase_rows) == {i: count // 27 for i in range(27)}
            computed = summarize(phase_rows)
            for key, value in computed.items():
                if isinstance(value, (float, int)):
                    assert math.isclose(value, summary[phase][key], rel_tol=1e-12, abs_tol=1e-15), key
            if phase != "training":
                assert summary[phase]["parameters_unchanged"]
        assert summary["feedback_unchanged"] and summary["all_tensors_on_device"]
        training = [r for r in rows if r["phase"] == "training"]
        changed = [r["phase_episode"] for r in training if r["parameter_delta_norm"] > 0]
        params = json.loads((seed_dir / "parameter_summary.json").read_text(encoding="utf-8"))
        visual = json.loads((seed_dir / "visual_diagnostics.json").read_text(encoding="utf-8"))
        seed_metrics.append({
            "seed": seed, "rows_checked": len(rows),
            "exact_successes": sum(r["correct_exact_match"] for r in rows),
            "positive_teacher_scores": sum(r["teacher_goodness"] > 0 for r in rows),
            "mean_teacher_goodness": math.fsum(r["teacher_goodness"] for r in rows) / len(rows),
            "training_episodes_with_parameter_change": len(changed),
            "last_nonzero_update_training_episode_1based": max(changed) + 1 if changed else None,
            "final_g_bar": training[-1]["g_bar"],
            "expected_training_successes": sum(r["exact_probability"] for r in training),
            "eligibility_action_norm_min": min(r["eligibility_norm_at_action"] for r in training),
            "eligibility_action_norm_max": max(r["eligibility_norm_at_action"] for r in training),
            "parameter_delta_norm": params["delta_norm"],
            "changed_parameter_tensors": params["changed_parameter_tensors"],
            "parameter_tensors": params["parameter_tensors"],
            "maximum_absolute_parameter_change": max(v["max_abs_delta"] for v in params["per_parameter"].values()),
            "parameter_min_delta": min(v["min_delta"] for v in params["per_parameter"].values()),
            "parameter_max_delta": max(v["max_delta"] for v in params["per_parameter"].values()),
            "visual_min_letter_distance": visual["minimum_pairwise_letter_encoding_l2"],
        })
        for row in rows:
            row["seed"] = seed
        pooled.extend(rows)
        aggregate_expected.append(summary)
    phases = {phase: summarize([r for r in pooled if r["phase"] == phase])
              for phase in ("initial", "training", "frozen")}
    curves = []
    for start in range(0, config["training_episodes"], config["window"]):
        window = [r for r in pooled if r["phase"] == "training" and start <= r["phase_episode"] < start + config["window"]]
        curves.append({"training_episode_start_1based": start + 1,
                       "training_episode_end_1based": min(start + config["window"], config["training_episodes"]),
                       **{k: v for k, v in summarize(window).items() if k != "per_class"}})
    return {"audit_date_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": metadata["git_commit"], "all_logged_source_hashes_match": True,
            "teacher_table_sha256": teacher_table.sha256, "teacher_table_verified": True,
            "validated_rows": len(pooled), "seeds": seed_metrics, "phases": phases,
            "pooled_training_curve": curves, "conclusion": classify_evidence(aggregate_expected),
            "raw_files": [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(directory.glob("seed_*/*")) if p.is_file()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("validated_rows", "all_logged_source_hashes_match", "conclusion", "seeds")}, indent=2))


if __name__ == "__main__":
    main()
