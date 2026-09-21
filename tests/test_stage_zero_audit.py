"""A raw-log audit must reject corrupt reward, timing and learning records."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

from test_stage_zero_environment import teacher_table

from experiments.stage_zero.audit import validate_row
from experiments.stage_zero.environment import VisualEnvironment
from experiments.stage_zero.harness import StageZero, Protocol
from experiments.stage_zero.runner import run_episode


@pytest.fixture(scope="module")
def actual_row():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = StageZero(17)
        row = run_episode(model, VisualEnvironment(), 2, episode=0, phase_episode=0,
                          phase="training", start_ms=0,
                          generator=torch.Generator().manual_seed(431), learn=True)
        for name in ("q", "p", "action_bits", "sampled_actions"):
            row[name] = json.loads(row[name])
        return row
    finally:
        torch.set_num_threads(previous)


def test_audit_accepts_actual_fractional_delayed_episode(actual_row):
    validate_row(actual_row, asdict(Protocol()))


@pytest.mark.parametrize("field,value", [
    ("goodness", 1.0),
    ("fractional_goodness", 1.0),
    ("correct_pressed", 9),
    ("wrong_count", 99),
    ("correct_exact_match", 9),
    ("delta", 99),
    ("goodness_delivery_time", 500),
    ("eligibility_norm", 0.0),
    ("p", [0.5]),
    ("learning_enabled", False),
    ("parameter_norm", float("nan")),
])
def test_audit_rejects_corrupted_actual_record(actual_row, field, value):
    assert actual_row["goodness"] != 1.0
    corrupted = deepcopy(actual_row)
    corrupted[field] = value
    with pytest.raises(AssertionError):
        validate_row(corrupted, asdict(Protocol()))


def test_audit_rejects_changed_table_bytes(tmp_path, teacher_table):
    from experiments.stage_zero.audit import validate_table_artifact
    teacher_table.save(tmp_path / "teacher_goodness.json")
    (tmp_path / "teacher_goodness.sha256").write_text(teacher_table.sha256)
    (tmp_path / "teacher_calibration_metadata.json").write_text(json.dumps(teacher_table.metadata))
    metadata = {"teacher_table_sha256": teacher_table.sha256, "teacher_calibration": teacher_table.metadata}
    config = {"teacher_table_sha256": teacher_table.sha256}
    assert validate_table_artifact(tmp_path, metadata, config).sha256 == teacher_table.sha256
    (tmp_path / "teacher_goodness.json").write_bytes(teacher_table.raw + b"\n")
    with pytest.raises(AssertionError, match="hash mismatch"):
        validate_table_artifact(tmp_path, metadata, config)
