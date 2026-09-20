"""A raw-log audit must reject corrupt reward, timing and learning records."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

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


def test_audit_accepts_actual_strict_reward_delayed_episode(actual_row):
    validate_row(actual_row, asdict(Protocol()))


@pytest.mark.parametrize("field,value", [
    ("goodness", 1.0),
    ("goodness_delivery_time", 500),
    ("eligibility_norm", 0.0),
    ("p", [0.5]),
    ("learning_enabled", False),
    ("parameter_norm", float("nan")),
])
def test_audit_rejects_corrupted_actual_record(actual_row, field, value):
    assert actual_row["goodness"] == 0.0
    corrupted = deepcopy(actual_row)
    corrupted[field] = value
    with pytest.raises(AssertionError):
        validate_row(corrupted, asdict(Protocol()))
