import json
import subprocess
import sys

import pytest
import torch

from acnt.__main__ import build_mock_runtime
from acnt.hand import read_training_controls


def test_whole_runtime_continues_with_sparse_teacher_and_real_parameter_updates():
    runtime = build_mock_runtime()
    before_hand = runtime.adapters["hand"].network[-1].bias.detach().clone()
    before_route = runtime.adapters["route"].linear.bias.detach().clone()
    generator = torch.Generator().manual_seed(131)
    active_sets = set()
    for tick in range(20):
        teacher = (0.8 if tick % 8 == 0 else 0.2) if tick % 4 == 0 else None
        ag_before = {name: p.clone() for name, p in runtime.adapters["goodness"].named_parameters()}
        baseline = runtime.plasticity.g_bar
        row = runtime.step(now_ms=tick * 250, readins={
            "eye": torch.full((3, 1080, 1920), (tick % 5) / 4.),
            "ear": torch.full((1, 1600), (tick % 3) / 2.),
        }, teacher=teacher, generator=generator)
        json.dumps(row, allow_nan=False)
        assert {0, 1, 5}.issubset(row["updated"])
        assert 5 in row["next_active"]
        assert len(row["hand_discrete"]) == 85 and len(row["hand_continuous"]) == 2
        assert len(row["speak"]) == 30
        assert row["g_eff"] == (row["g"] if teacher is None else teacher)
        assert row["delta"] == pytest.approx(row["g_eff"] - baseline)
        if teacher is None:
            assert row["calibration_loss"] is None
            for name, p in runtime.adapters["goodness"].named_parameters():
                torch.testing.assert_close(p, ag_before[name])
        else:
            assert row["calibration_loss"] == pytest.approx(0.5 * (row["g"] - teacher) ** 2)
        assert runtime._local_z == {}
        assert all(torch.isfinite(p).all() for p in runtime.parameters())
        for channel in runtime.plasticity.banks():
            for bank in channel.values():
                assert all(torch.isfinite(e).all() and not e.requires_grad for e in bank.values.values())
        active_sets.add(tuple(row["next_active"]))
    assert len(active_sets) > 1
    assert len(runtime.mechanical_log.records) == 80
    hands = [r for r in runtime.mechanical_log.records if r["organ"] == "hand"]
    assert all(len(r["signal"]["discrete"]) == 85 for r in hands)
    assert all(len(read_training_controls(r)) == 27 for r in hands)
    assert not torch.equal(before_hand, runtime.adapters["hand"].network[-1].bias)
    assert not torch.equal(before_route, runtime.adapters["route"].linear.bias)


def test_two_rounds_then_delayed_feedback_use_saved_eligibility(make_runtime):
    runtime = make_runtime()
    runtime.enable_plasticity(learning_rate=0.0001)
    generator = torch.Generator().manual_seed(137)
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    runtime.generate_hand(now_ms=0, generator=generator)
    runtime.generate_speak(now_ms=0)
    runtime.generate_route(now_ms=0, generator=generator)
    signal = runtime.generate_goodness(now_ms=0, teacher=0.9)
    before = runtime.adapters["hand"].network[-1].bias.detach().clone()
    delta = runtime.learn_goodness(signal, delivered_ms=200)
    assert delta == pytest.approx(0.4)
    assert not torch.equal(before, runtime.adapters["hand"].network[-1].bias)
    row = runtime.step(now_ms=250, readins={"ear": torch.zeros(1, 32)}, generator=generator)
    assert row["teacher"] is None and row["calibration_loss"] is None


def test_whole_runtime_cli_writes_complete_logs_and_summary(tmp_path):
    log = tmp_path / "mechanical.jsonl"
    summary = tmp_path / "summary.json"
    result = subprocess.run([sys.executable, "-m", "acnt", "--steps", "8", "--log-file", str(log), "--summary-file", str(summary)], capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 8 and rows[-1]["tick"] == 7
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 32
    assert {r["organ"] for r in records} == {"route", "hand", "speak", "goodness"}
    report = json.loads(summary.read_text())
    assert report["finite_parameters"] and report["changed_parameter_tensors"]
    assert report["teacher_steps"] == [0, 4]
    assert report["stage_0_run"] is False
