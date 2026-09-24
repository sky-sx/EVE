import json
import subprocess
import sys
import pytest

import torch

from acnt.__main__ import build_mock_runtime


def test_whole_runtime_with_sparse_teacher_and_finite_local_states():
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
        assert -1<=row["goodness_modulation"]<=1
        assert all(torch.isfinite(p).all() for p in runtime.parameters())
        assert all(torch.isfinite(e).all() for e in runtime.plasticity.traces.values())
    assert any(not torch.equal(p,before[id(p)]) for p in runtime.parameters())


def test_delayed_feedback_reads_existing_local_state(make_runtime):
    runtime=make_runtime()
    learner=runtime.enable_plasticity(learning_rate=.0001)
    runtime.update_blocks(now_ms=0,readins={"ear":torch.ones(1,32)})
    runtime.generate_hand(now_ms=0,generator=torch.Generator().manual_seed(137))
    signal=runtime.generate_goodness(now_ms=0,teacher=.9)
    adapter=runtime.adapters["hand"]
    terminal=adapter.network[-1]
    e=learner.traces[id(terminal.bias)].clone()
    before=terminal.bias.clone()
    assert runtime.learn_goodness(signal,delivered_ms=1000)==pytest.approx(.4)
    decayed=e*torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(learner.traces[id(terminal.bias)],decayed)
    torch.testing.assert_close(terminal.bias,before+.0001*.4*decayed)


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
