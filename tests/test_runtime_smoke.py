import json
import subprocess
import sys
import pytest

import torch

from acnt.__main__ import build_mock_runtime


def test_whole_runtime_with_sparse_teacher_and_finite_states():
    runtime=build_mock_runtime()
    generator=torch.Generator().manual_seed(131)
    before={id(p):p.clone() for p in runtime.parameters()}
    for tick in range(5):
        teacher=.8 if tick==0 else None
        row=runtime.step(now_ms=tick*250,readins={
            "eye":torch.zeros(3,1080,1920),"ear":torch.zeros(1,1600)},
            teacher=teacher,generator=generator)
        json.dumps(row,allow_nan=False)
        assert len(row["hand_discrete"])==85
        assert len(row["hand_continuous"])==2
        assert row["goodness_delta"] is None or -1<=row["goodness_delta"]<=1
        assert all(torch.isfinite(p).all() for p in runtime.parameters())
        assert all(torch.isfinite(e).all() for e in runtime.plasticity.delta_w.values())
    assert any(not torch.equal(p,before[id(p)]) for p in runtime.parameters())


def test_delayed_feedback_resolves_exactly_one_movement(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    runtime.learn_goodness(runtime.generate_goodness(now_ms=0, teacher=.9))
    learner.begin_trial()
    moved = {k:p.clone() for k,p in learner.parameters.items()}
    with pytest.raises(RuntimeError, match="unresolved"):
        runtime.step(now_ms=250)
    signal = runtime.generate_goodness(now_ms=250, teacher=.4)
    assert runtime.learn_goodness(signal, delivered_ms=1000) == pytest.approx(-.5)
    assert learner.pending_indices is None
    assert all(torch.equal(p,moved[k]) for k,p in learner.parameters.items())


def test_learn_false_does_not_change_parameters(make_runtime):
    runtime=make_runtime()
    before={name:p.clone() for name,p in runtime.named_parameters()}
    runtime.step(now_ms=0,readins={"ear":torch.zeros(1,32)},learn=False,
                 generator=torch.Generator().manual_seed(5))
    for name,p in runtime.named_parameters():
        torch.testing.assert_close(p,before[name])


def test_whole_runtime_cli_writes_complete_logs_and_summary(tmp_path):
    log=tmp_path/"mechanical.jsonl"
    summary=tmp_path/"summary.json"
    result=subprocess.run([sys.executable,"-m","acnt","--steps","8","--log-file",str(log),
                           "--summary-file",str(summary)],capture_output=True,text=True,check=True)
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==8 and rows[-1]["tick"]==7
    records=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(records)==32
    assert {r["organ"] for r in records}=={"route","hand","speak","goodness"}
    report=json.loads(summary.read_text())
    assert report["finite_parameters"] and report["changed_parameter_tensors"]
    assert report["teacher_steps"]==[0,4]
    assert report["stage_0_run"] is False


def test_runtime_moves_before_readin_and_freezes_without_rng_changes(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    before = {k:p.detach().clone() for k,p in learner.parameters.items()}
    seen = []

    def observe_forward(module, args):
        seen.append(any(not torch.equal(p,before[k]) for k,p in learner.parameters.items()))
        assert not torch.is_grad_enabled()

    hook = runtime.adapters['ear'].register_forward_pre_hook(observe_forward)
    runtime.step(now_ms=0, readins={'ear':torch.ones(1,32)}, teacher=.6)
    runtime.step(now_ms=250, readins={'ear':torch.ones(1,32)}, teacher=.4)
    hook.remove()
    assert seen == [False, True]
    w = {k:p.detach().clone() for k,p in learner.parameters.items()}
    d = {k:v.clone() for k,v in learner.delta_w.items()}
    rng = learner.rng.getstate()
    for time in (500,750):
        runtime.step(now_ms=time, learn=False, teacher=.9)
    assert learner.rng.getstate() == rng
    assert learner.pending_indices is None and learner.previous_goodness is None
    assert all(torch.equal(p,w[k]) for k,p in learner.parameters.items())
    assert all(torch.equal(v,d[k]) for k,v in learner.delta_w.items())
    runtime.step(now_ms=1000, teacher=.1)
    assert all(torch.equal(p,w[k]) for k,p in learner.parameters.items())
    assert all(torch.equal(v,d[k]) for k,v in learner.delta_w.items())
    assert learner.rng.getstate() == rng
    assert all(p.grad is None for p in runtime.parameters())
