import math
import pytest
import torch
from acnt.bounded_goodness import combined_goodness, BoundedGoodnessLearner
from acnt.goodness_prediction import GoodnessPredictionKernel


def test_bounded_unique_root_and_primary_task_priority_over_grid():
    for i in range(101):
        r=i/100
        for j in range(101):
            p=j/100
            result=combined_goodness(r,p)
            g,b=result['goodness'],result['main_base']
            assert 0<=g<=1 and abs(g-b+.2*(p-g)**2)<1e-13
            if r==0:
                assert g<=.2+1e-14
            if r==1:
                assert g>=.8-1e-14
            # Independent monotone root bracketing, not the quadratic formula.
            lo,hi=0.,1.
            for _ in range(45):
                mid=(lo+hi)/2
                if mid-b+.2*(p-mid)**2>0:
                    hi=mid
                else:
                    lo=mid
            assert abs(g-(lo+hi)/2)<1e-13


def test_equal_prediction_has_exactly_zero_correction_and_credit():
    for r in (0.,.25,.5,1.):
        base=.2+.8*r
        result=combined_goodness(r,base)
        assert result['goodness']==base
        assert result['evaluation_correction']==0.
    k=GoodnessPredictionKernel(width=2,hold=2,group_size=8)
    model=BoundedGoodnessLearner(k)
    model.pending_prediction=model.teacher_for_commit=.2
    model.evaluation_traces.fill_(1.)
    before=k.effective(model.state).detach().clone()
    model.commit(torch.zeros(k.parameter_size),torch.full((k.group_count,),.5))
    assert model.evaluation_logit_coefficient==0.
    assert model.auxiliary_direction_norm==0.
    torch.testing.assert_close(before,k.effective(model.state),atol=0,rtol=0)


@pytest.mark.parametrize('r',[0.,.3,1.])
@pytest.mark.parametrize('q',[-2.,0.,2.])
def test_total_objective_logit_derivative_matches_finite_difference(r,q):
    p=1/(1+math.exp(-q))
    g=combined_goodness(r,p)['goodness']
    exact=.4*(g-p)/(1+.4*(g-p))*p*(1-p)
    epsilon=1e-5
    gp=combined_goodness(r,1/(1+math.exp(-(q+epsilon))))['goodness']
    gm=combined_goodness(r,1/(1+math.exp(-(q-epsilon))))['goodness']
    assert abs(exact-(gp-gm)/(2*epsilon))<1e-9


def test_wrong_action_boundary_does_not_make_negative_reward():
    result=combined_goodness(0.,1.)
    assert result['goodness']==0.
    assert abs(result['evaluation_correction']+.2)<1e-14
    with pytest.raises(ValueError):
        combined_goodness(0.,.5,floor=.1,penalty=.2)
