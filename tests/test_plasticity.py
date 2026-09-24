import pytest
import torch
from torch import nn
from acnt.plasticity import CorrelationRule, LocalEvent, Plasticity


def test_dense_trace_uses_real_time_exponential_decay():
    p=nn.Parameter(torch.zeros((2,2)))
    learner=Plasticity({0:{"w":p}},None,tau_e_s=1.)
    first=torch.tensor([[8.,12.],[10.,15.]])
    learner.observe(p,LocalEvent(torch.tensor([2.,3.]),torch.tensor([4.,5.]),0))
    torch.testing.assert_close(learner.states[id(p)],first)
    second=torch.tensor([[3.,6.],[4.,8.]])
    learner.observe(p,LocalEvent(torch.tensor([1.,2.]),torch.tensor([3.,4.]),1000))
    expected=torch.exp(torch.tensor(-1.))*first+second
    torch.testing.assert_close(learner.states[id(p)],expected)


def test_conv_and_bias_local_values():
    conv=nn.Conv2d(1,1,2,bias=False)
    pre=torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]])
    post=torch.ones((1,2,2))
    value=CorrelationRule().F_e(LocalEvent(pre,post,0,"conv",conv))
    torch.testing.assert_close(value,torch.tensor([[[[3.,4.],[6.,7.]]]]))
    bias=CorrelationRule().F_e(LocalEvent(torch.ones(2),torch.tensor([2.,-3.]),0,"bias"))
    torch.testing.assert_close(bias,torch.tensor([2.,-3.]))



def test_grouped_nlm_correlation_is_neuron_local():
    rule=CorrelationRule()
    pre=torch.tensor([[1.,2.],[10.,20.]])
    post=torch.tensor([[3.,4.],[30.,40.]])
    value=rule.F_e(LocalEvent(pre,post,0,"grouped_dense"))
    expected=torch.tensor([
        [[3.,6.],[4.,8.]],
        [[300.,600.],[400.,800.]],
    ])
    torch.testing.assert_close(value,expected)
    # No d=0 post is ever multiplied by d=1 pre (or vice versa).
    assert value[0,0,0]==3 and value[1,0,0]==300
    bias=rule.F_e(LocalEvent(torch.ones_like(post),post,0,"grouped_bias"))
    torch.testing.assert_close(bias,post)


@pytest.mark.parametrize("delay",[250,500,1000])
def test_goodness_wait_decays_trace_before_weight_update(delay):
    p=nn.Parameter(torch.tensor([1.,2.]))
    learner=Plasticity({0:{"bias":p}},goodness_id=None,tau_e_s=1.)
    learner.observe(p,LocalEvent(torch.ones(2),torch.tensor([2.,4.]),0,"bias"))
    e0=learner.states[id(p)].clone()
    before=p.clone()
    assert learner.apply_goodness(1.,now_ms=delay)==.5
    decayed=e0*torch.exp(torch.tensor(-delay/1000.))
    torch.testing.assert_close(learner.states[id(p)],decayed)
    torch.testing.assert_close(p,before+.0005*decayed)
    with pytest.raises(RuntimeError,match="no goodness"):
        learner.calibrate_goodness(.5,now_ms=delay)


def test_modulation_uses_old_g_bar_then_updates_baseline():
    p=nn.Parameter(torch.tensor([1.]))
    learner=Plasticity({0:{"w":p}},None,learning_rate=.1,tau_e_s=1.,tau_g_s=2.)
    learner.observe(p,LocalEvent(torch.ones(1),torch.tensor([2.]),0,"bias"))
    assert learner.apply_goodness(1.,now_ms=0)==pytest.approx(.5)
    torch.testing.assert_close(p,torch.tensor([1.1]))
    assert learner.g_bar==pytest.approx(.5)
    e1=2.*torch.exp(torch.tensor(-1.))
    before=p.clone()
    assert learner.apply_goodness(1.,now_ms=1000)==pytest.approx(.5)
    torch.testing.assert_close(p,before+.1*.5*e1)
    retention=torch.exp(torch.tensor(-.5)).item()
    expected=.5*retention+(1-retention)
    assert learner.g_bar==pytest.approx(expected)
    e2=e1*torch.exp(torch.tensor(-1.))
    before=p.clone()
    assert learner.apply_goodness(0.,now_ms=2000)==pytest.approx(-expected)
    torch.testing.assert_close(p,before-.1*expected*e2)
    assert learner.g_bar==pytest.approx(retention*expected)

def test_all_parameters_unique_and_production_goodness_exception(make_runtime):
    runtime=make_runtime()
    learner=runtime.enable_plasticity()
    assert learner.goodness_id==runtime.organ_blocks["goodness"]
    assert len(learner.states)==sum(len(x) for x in learner.groups.values())
    assert len(learner.states)==len({id(p) for g in learner.groups.values() for p in g.values()})
    for i,group in learner.groups.items():
        assert {"block.b","block.W_ij.0","block.nlm.weight1","block.nlm.bias1",
                "block.nlm.weight2","block.nlm.bias2"}<=set(group)
        for p in group.values():
            assert learner.states[id(p)].shape==p.shape
            assert not learner.states[id(p)].requires_grad
    bg=runtime.core.blocks[learner.goodness_id].b
    learner.observe(bg,LocalEvent(torch.ones_like(bg),torch.ones_like(bg),0,"bias"))
    previous=bg.clone()
    learner.apply_goodness(1.,now_ms=1000)
    torch.testing.assert_close(bg,previous)
    learner.calibrate_goodness(.5,now_ms=1000)
    assert not torch.equal(bg,previous)

def test_nlm_events_register_shapes_and_goodness_updates_without_autograd(make_runtime):
    runtime=make_runtime()
    learner=runtime.enable_plasticity()
    block=runtime.core.blocks[0]
    learner.set_event_time(0)
    block.update(now_ms=0,active_z={})
    nlm_parameters={
        "block.nlm.weight1":block.nlm.weight1,
        "block.nlm.bias1":block.nlm.bias1,
        "block.nlm.weight2":block.nlm.weight2,
        "block.nlm.bias2":block.nlm.bias2,
    }
    for name,parameter in nlm_parameters.items():
        assert learner.groups[0][name] is parameter
        assert learner.states[id(parameter)].shape==parameter.shape
        assert not learner.states[id(parameter)].requires_grad
        assert parameter.grad is None
    before={id(p):p.clone() for p in nlm_parameters.values()}
    learner.apply_goodness(1.,now_ms=250)
    assert any(not torch.equal(p,before[id(p)]) for p in nlm_parameters.values())
    assert all(torch.isfinite(p).all() and p.grad is None for p in nlm_parameters.values())


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
    learner.observe(p,LocalEvent(pre,post,0))
    learner.apply_goodness(0.,now_ms=1000)
    assert p.grad is None and pre.grad is None and post.grad is None
    assert not learner.states[id(p)].requires_grad
    assert torch.isfinite(p).all() and torch.isfinite(learner.states[id(p)]).all()
    learner.clear()
    assert learner.states[id(p)].count_nonzero()==0
    assert learner.g_bar==.5
    assert all(t is None for t in learner.state_times_ms.values())


def test_duplicate_group_rejected():
    p=nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match="exactly one"):
        Plasticity({0:{"a":p},1:{"b":p}},None)
