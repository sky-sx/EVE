import pytest
import torch
from torch import nn
from acnt.plasticity import CorrelationRule, LocalEvent, Plasticity


def test_dense_and_retention_are_event_count_based():
    rule=CorrelationRule(retention=.95)
    state=torch.zeros((2,2))
    first=rule.F_e(state,LocalEvent(torch.tensor([2.,3.]),torch.tensor([4.,5.])))
    torch.testing.assert_close(first,torch.tensor([[8.,12.],[10.,15.]]))
    second=rule.F_e(first,LocalEvent(torch.tensor([1.,2.]),torch.tensor([3.,4.])))
    torch.testing.assert_close(second,.95*first+torch.tensor([[3.,6.],[4.,8.]]))


def test_conv_and_bias_local_values():
    conv=nn.Conv2d(1,1,2,bias=False)
    pre=torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]])
    post=torch.ones((1,2,2))
    state=torch.zeros_like(conv.weight)
    value=CorrelationRule().F_e(state,LocalEvent(pre,post,"conv",conv))
    torch.testing.assert_close(value,torch.tensor([[[[3.,4.],[6.,7.]]]]))
    bias=CorrelationRule().F_e(torch.zeros(2),LocalEvent(torch.ones(2),torch.tensor([2.,-3.]),"bias"))
    torch.testing.assert_close(bias,torch.tensor([2.,-3.]))


def test_goodness_wait_does_not_decay_or_observe():
    p=nn.Parameter(torch.tensor([1.,2.]))
    learner=Plasticity({0:{"bias":p}},goodness_id=None)
    learner.observe(p,LocalEvent(torch.ones(2),torch.tensor([2.,4.]),"bias"))
    e0=learner.states[id(p)].clone()
    for delay in (250,500,1000):
        before=p.clone()
        assert learner.apply_goodness(1.,now_ms=delay)==1.
        torch.testing.assert_close(learner.states[id(p)],e0)
        torch.testing.assert_close(p,before+.0005*e0)
    with pytest.raises(RuntimeError,match="no goodness"):
        learner.calibrate_goodness(.5)


def test_all_parameters_unique_and_production_goodness_exception(make_runtime):
    runtime=make_runtime()
    learner=runtime.enable_plasticity()
    assert learner.goodness_id==runtime.organ_blocks["goodness"]
    assert len(learner.states)==sum(len(x) for x in learner.groups.values())
    assert len(learner.states)==len({id(p) for g in learner.groups.values() for p in g.values()})
    for i,group in learner.groups.items():
        assert {"block.b","block.W_ij.0","block.W_c.0","block.b_c.0"}<=set(group)
        for p in group.values():
            assert learner.states[id(p)].shape==p.shape
            assert not learner.states[id(p)].requires_grad
    bg=runtime.core.blocks[learner.goodness_id].b
    learner.observe(bg,LocalEvent(torch.ones_like(bg),torch.ones_like(bg),"bias"))
    previous=bg.clone()
    learner.apply_goodness(1.)
    torch.testing.assert_close(bg,previous)
    learner.calibrate_goodness(.5)
    assert not torch.equal(bg,previous)


def test_route_optional_without_adapter():
    weight=nn.Parameter(torch.ones((2,2)))
    learner=Plasticity({0:{"w":weight}},None)
    learner.attach_adapters({})
    assert learner._excluded_controls=={}


def test_no_autograd_learning_graph_and_finite_update():
    p=nn.Parameter(torch.ones((2,2)))
    learner=Plasticity({0:{"w":p}},None)
    pre=torch.tensor([1.,2.],requires_grad=True)
    post=torch.tensor([3.,4.],requires_grad=True)
    learner.observe(p,LocalEvent(pre,post))
    learner.apply_goodness(0.)
    assert p.grad is None and pre.grad is None and post.grad is None
    assert not learner.states[id(p)].requires_grad
    assert torch.isfinite(p).all() and torch.isfinite(learner.states[id(p)]).all()
    learner.clear()
    assert learner.states[id(p)].count_nonzero()==0


def test_duplicate_group_rejected():
    p=nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match="exactly one"):
        Plasticity({0:{"a":p},1:{"b":p}},None)
