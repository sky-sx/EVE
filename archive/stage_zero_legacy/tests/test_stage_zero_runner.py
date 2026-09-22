"""Persisted raw audit artifacts must agree with actual phase execution."""

import csv
import json

import torch

from test_stage_zero_environment import teacher_table

from experiments.stage_zero.harness import Protocol
from experiments.stage_zero.runner import run_seed, source_metadata


def test_seed_artifacts_preserve_raw_data_frozen_weights_and_metadata(tmp_path, teacher_table):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = run_seed(3, output=tmp_path, device="cpu", train_episodes=1,
                           evaluation_episodes=1, window=1, protocol=Protocol(), teacher_table=teacher_table)
    finally:
        torch.set_num_threads(previous_threads)
    directory = tmp_path / "seed_003"
    with (directory / "episode_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["phase"] for row in rows] == ["initial", "training", "frozen"]
    assert [int(row["episode"]) for row in rows] == [0, 1, 2]
    for row in rows:
        bits = json.loads(row["action_bits"])
        target = int(row["target_class"])
        assert len(bits) == len(json.loads(row["q"])) == len(json.loads(row["p"])) == 27
        assert int(row["correct_exact_match"]) == int(bits[target] and sum(bits) == 1)
        assert float(row["goodness"]) == float(row["teacher_goodness"]) == teacher_table.lookup(int(bits[target]), sum(bits) - bits[target])
        assert int(row["goodness_delivery_time"]) - int(row["logical_time_ms"]) == 250
        assert float(row["nan_count"]) == float(row["inf_count"]) == 0
    assert summary["initial"]["parameters_unchanged"]
    assert summary["frozen"]["parameters_unchanged"]
    assert summary["feedback_unchanged"] and summary["all_tensors_on_device"]
    assert summary["initial_parameter_hash"] != summary["final_parameter_hash"]
    assert json.loads((directory / "summary.json").read_text()) == summary
    changes = json.loads((directory / "parameter_summary.json").read_text())
    assert changes["changed_parameter_tensors"] > 0
    assert (directory / "learning_curve.csv").exists()
    assert (directory / "per_class.csv").exists()
    checkpoint = torch.load(directory / "final_state.pt", weights_only=True)
    assert checkpoint["seed"] == 3
    assert checkpoint["protocol"]["blocks"] == 10
    diagnostics = json.loads((directory / "visual_diagnostics.json").read_text())
    assert diagnostics["finite"] and diagnostics["minimum_pairwise_letter_encoding_l2"] > 0
    metadata = source_metadata()
    assert len(metadata["git_commit"]) == 40
    assert "acnt/plasticity.py" in {key.replace("\\", "/") for key in metadata["source_sha256"]}


def test_production_acnt_sources_match_original_stage_zero_commit():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    baseline = "43677b0bce2d10dfdbe217fc8c7670583054644e"
    for path in (root / "acnt").glob("*.py"):
        original = subprocess.check_output(["git", "show", f"{baseline}:acnt/{path.name}"], cwd=root)
        assert path.read_bytes().replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"), path.name
