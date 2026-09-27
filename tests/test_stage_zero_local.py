import pytest
import torch

from archive.stage_zero_superseded.environment import (
    ACTIONS, DELAYS_MS, balanced_colors, balanced_delays, balanced_targets, exact_success, potential_goodness,
    phase_schedule, render, stimulus_hash,
)
from archive.stage_zero_superseded.audit import audit_goodness


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
