import pytest
import torch

from experiments.stage_zero.environment import (
    ACTIONS, DELAYS_MS, balanced_colors, balanced_delays, balanced_targets, exact_success, potential_goodness,
    phase_schedule, render, stimulus_hash,
)
from experiments.stage_zero.audit import audit_goodness
from experiments.stage_zero.harness import StageZero
from experiments.stage_zero.runner import aggregate


def test_balanced_independent_environment_schedule():
    for index,count in enumerate((270,1080,270)):
        schedule,seeds=phase_schedule(count,11,index)
        targets=[x[0] for x in schedule]
        delays=[x[2] for x in schedule]
        assert all(targets.count(i)==count//27 for i in range(27))
        assert all(delays.count(d)==count//len(DELAYS_MS) for d in DELAYS_MS)
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
    assert potential_goodness(0,action)==1.
    assert exact_success(0,action)
    action[1]=True
    assert potential_goodness(0,action)==.5
    assert not exact_success(0,action)


def test_stage_zero_group_ownership_and_freeze():
    model=StageZero(11)
    assert len(model.core.blocks)==10
    assert all(b.active and b.neuron_size==100 and b.hold_tick==4 and b.ticktime==250
               for b in model.core.blocks)
    assert set(model.groups)==set(range(10))
    assert {id(p) for p in model.eye.parameters()}<=set(map(id,model.groups[0].values()))
    assert {id(p) for p in model.hand.parameters()}<=set(map(id,model.groups[1].values()))
    assert len(model.plasticity.traces)==sum(len(g) for g in model.groups.values())
    before={id(p):p.clone() for p in model.plasticity.parameters.values()}
    model.reset_phase(False)
    row=model.episode(phase="initial",episode=0,target=0,color="red",delay_ms=1000,
                      start_ms=0,generator=torch.Generator().manual_seed(12))
    assert row["frame_times_ms"]==[0,250,500]
    assert row["goodness_time_ms"]==1500
    assert row["eligibility_trace"]["total"]["l2"]==0
    assert all(torch.equal(p,before[id(p)]) for p in model.plasticity.parameters.values())
    model.reset_phase(True)
    row=model.episode(phase="training",episode=0,target=0,color="blue",delay_ms=500,
                      start_ms=1750,generator=torch.Generator().manual_seed(13))
    assert row["eligibility_trace"]["total"]["l2"]>0
    model.reset_phase(False)
    assert model.state_stats()["total"]["l2"]==0

    model.reset_phase(False)

    before={
        id(p): p.clone()
        for p in model.plasticity.parameters.values()
    }

    model.episode(
        phase="frozen",
        episode=0,
        target=0,
        color="red",
        delay_ms=250,
        start_ms=0,
        generator=torch.Generator().manual_seed(1),
    )

    assert all(
        torch.equal(
            block.z,
            block.z_bar,
        )
        for block in model.core.blocks
    )

    assert all(
        trace.count_nonzero() == 0
        for trace in model.plasticity.traces.values()
    )

    assert all(
        torch.equal(
            p,
            before[id(p)],
        )
        for p in model.plasticity.parameters.values()
    )


@pytest.mark.parametrize(
    "target_pressed,wrong_count,expected",
    [(False,0,0.),(False,9,0.),(True,0,1.),(True,1,.5),
     (True,2,1/3),(True,9,.1)],
)
def test_potential_goodness_and_independent_audit(target_pressed,wrong_count,expected):
    action=torch.zeros(27,dtype=torch.bool)
    action[0]=target_pressed
    action[1:1+wrong_count]=True
    assert potential_goodness(0,action)==pytest.approx(expected)
    audited_g,audited_exact=audit_goodness(0,action.tolist())
    assert audited_g==pytest.approx(expected)
    assert audited_exact==exact_success(0,action)==(target_pressed and wrong_count==0)


def test_aggregate_separates_fractional_goodness_from_exact_success():
    rows=[]
    for target_hit,active,g,exact in ((False,9,0.,False),(True,10,.1,False),
                                      (True,1,1.,True)):
        rows.append({
            "target_bit_actual":target_hit,"active_bit_count":active,
            "g_star":g,"exact_success":exact,
            "target_p":.5,"non_target_mean_p":.5,
            "non_target_false_rate":(active-int(target_hit))/26,
            "exact_event_probability":0.,"parameter_norm":1.,
            "parameter_delta_norm":0.,"eligibility_trace":{"total":{"l2":0.}},
            "nan_count":0,"inf_count":0,
        })
    result=aggregate(rows)
    assert result["target_bit_hit_rate"]==pytest.approx(2/3)
    assert result["mean_active_bits"]==pytest.approx(20/3)
    assert result["mean_active_bits_given_target_hit"]==pytest.approx(5.5)
    assert result["mean_g_star"]==pytest.approx(1.1/3)
    assert result["exact_success_rate"]==pytest.approx(1/3)
    assert result["positive_g_star"]==2
