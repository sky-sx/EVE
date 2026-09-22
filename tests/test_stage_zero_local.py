import torch

from experiments.stage_zero.environment import (
    ACTIONS, balanced_colors, balanced_delays, balanced_targets, exact_goodness,
    phase_schedule, render, stimulus_hash,
)
from experiments.stage_zero.harness import StageZero


def test_balanced_independent_environment_schedule():
    for index,count in enumerate((270,1080,270)):
        schedule,seeds=phase_schedule(count,11,index)
        targets=[x[0] for x in schedule]
        delays=[x[2] for x in schedule]
        assert all(targets.count(i)==count//27 for i in range(27))
        assert all(delays.count(i)==count//3 for i in (250,500,1000))
        assert len(set(seeds.values()))==4
        for target in range(26):
            colors=[color for t,color,_ in schedule if t==target]
            assert abs(colors.count("red")-colors.count("blue"))<=1


def test_environment_exact_one_hot_and_color_variants():
    red=render(0,"red")
    blue=render(0,"blue")
    green=render(26,"green")
    assert red.shape==blue.shape==green.shape==(3,1080,1920)
    assert red.dtype==torch.float32
    assert not torch.equal(red,blue)
    assert stimulus_hash(red)!=stimulus_hash(blue)
    assert torch.all(green[1]==1) and torch.all(green[[0,2]]==0)
    action=torch.zeros(27,dtype=torch.bool)
    action[0]=True
    assert exact_goodness(0,action)==1.
    action[1]=True
    assert exact_goodness(0,action)==0.


def test_stage_zero_group_ownership_and_freeze():
    model=StageZero(11)
    assert len(model.core.blocks)==10
    assert all(b.active and b.neuron_size==100 and b.hold_tick==4 and b.ticktime==.25
               for b in model.core.blocks)
    assert set(model.groups)==set(range(10))
    assert {id(p) for p in model.eye.parameters()}<=set(map(id,model.groups[0].values()))
    assert {id(p) for p in model.hand.parameters()}<=set(map(id,model.groups[1].values()))
    assert len(model.plasticity.states)==sum(len(g) for g in model.groups.values())
    before={id(p):p.clone() for p in model.plasticity.parameters.values()}
    model.reset_phase(False)
    row=model.episode(phase="initial",episode=0,target=0,color="red",delay_ms=1000,
                      start_ms=0,generator=torch.Generator().manual_seed(12))
    assert row["frame_times_ms"]==[0,250,500]
    assert row["goodness_time_ms"]==1500
    assert row["plastic_state"]["total"]["l2"]==0
    assert all(torch.equal(p,before[id(p)]) for p in model.plasticity.parameters.values())
    model.reset_phase(True)
    row=model.episode(phase="training",episode=0,target=0,color="blue",delay_ms=500,
                      start_ms=1750,generator=torch.Generator().manual_seed(13))
    assert row["plastic_state"]["total"]["l2"]>0
    model.reset_phase(False)
    assert model.state_stats()["total"]["l2"]==0
