import pytest
import torch
from torch import nn
from acnt.plasticity import Plasticity


def test_trace_uses_real_time_exponential_decay():
    p = nn.Parameter(
        torch.zeros((2, 2))
    )

    learner = Plasticity(
        {0: {"w": p}},
        None,
        tau_q_s=1.0,
    )

    first = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    learner.accumulate(
        p,
        first,
        now_ms=0,
    )

    second = torch.tensor([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    learner.accumulate(
        p,
        second,
        now_ms=1000,
    )

    expected = (
        torch.exp(torch.tensor(-1.0))
        * first
        + second
    )

    torch.testing.assert_close(
        learner.traces[id(p)],
        expected,
    )


def test_goodness_updates_from_trace():
    p = nn.Parameter(
        torch.tensor([1.0])
    )

    learner = Plasticity(
        {0: {"w": p}},
        None,
        learning_rate=0.1,
        tau_q_s=10.0,
    )

    learner.accumulate(
        p,
        torch.tensor([2.0]),
        now_ms=0,
    )

    learner.apply_goodness(
        1.0,
        now_ms=0,
    )

    torch.testing.assert_close(
        p,
        torch.tensor([1.1]),
    )


def test_trace_has_one_scalar_per_parameter_element():
    p = nn.Parameter(
        torch.zeros((3, 4))
    )

    learner = Plasticity(
        {0: {"w": p}},
        None,
    )

    assert (
        learner.traces[id(p)].shape
        == p.shape
    )


def test_clear_resets_traces_and_baseline():
    p = nn.Parameter(
        torch.zeros(2)
    )

    learner = Plasticity(
        {0: {"w": p}},
        None,
    )

    learner.accumulate(
        p,
        torch.ones(2),
        now_ms=0,
    )

    learner.clear()

    assert (
        learner.traces[id(p)]
        .count_nonzero()
        == 0
    )

    assert learner.g_bar == 0.5

    assert all(
        value is None
        for value
        in learner.trace_times_ms.values()
    )


@pytest.mark.parametrize("delay",[250,500,1000])
def test_goodness_wait_decays_trace_before_weight_update(delay):
    p=nn.Parameter(torch.tensor([1.,2.]))
    learner=Plasticity({0:{"bias":p}},goodness_id=None,tau_q_s=1.)
    learner.accumulate(p,torch.tensor([2.,4.]),now_ms=0)
    e0=learner.traces[id(p)].clone()
    before=p.clone()
    assert learner.apply_goodness(1.,now_ms=delay)==.5
    decayed=e0*torch.exp(torch.tensor(-delay/1000.))
    torch.testing.assert_close(learner.traces[id(p)],decayed)
    torch.testing.assert_close(p,before+.0005*decayed)
    with pytest.raises(RuntimeError,match="no goodness"):
        learner.calibrate_goodness(.5,now_ms=delay)


def test_modulation_uses_old_g_bar_then_updates_baseline():
    p=nn.Parameter(torch.tensor([1.]))
    learner=Plasticity({0:{"w":p}},None,learning_rate=.1,tau_q_s=1.,tau_g_s=2.)
    learner.accumulate(p,torch.tensor([2.]),now_ms=0)
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
    assert len(learner.traces)==sum(len(x) for x in learner.groups.values())
    assert len(learner.traces)==len({id(p) for g in learner.groups.values() for p in g.values()})
    for i,group in learner.groups.items():
        assert {"block.b","block.W_ij.0","block.nlm.weight1","block.nlm.bias1",
                "block.nlm.weight2","block.nlm.bias2"}<=set(group)
        for p in group.values():
            assert learner.traces[id(p)].shape==p.shape
            assert not learner.traces[id(p)].requires_grad
    bg=runtime.core.blocks[learner.goodness_id].b
    learner.accumulate(bg,torch.ones_like(bg),now_ms=0)
    previous=bg.clone()
    learner.apply_goodness(1.,now_ms=1000)
    torch.testing.assert_close(bg,previous)
    assert learner.calibrate_goodness(.5,now_ms=1000)==.5
    assert torch.equal(bg,previous)

def test_nlm_events_register_shapes_and_goodness_updates_without_autograd(make_runtime):
    runtime=make_runtime()
    learner=runtime.enable_plasticity()
    block=runtime.core.blocks[0]
    block.set_learning(True,perturbation_scale=learner.perturbation_scale,
                       generator=torch.Generator().manual_seed(0))
    block.update(now_ms=0,active_z={})
    nlm_parameters={
        "block.nlm.weight1":block.nlm.weight1,
        "block.nlm.bias1":block.nlm.bias1,
        "block.nlm.weight2":block.nlm.weight2,
        "block.nlm.bias2":block.nlm.bias2,
    }
    for name,parameter in nlm_parameters.items():
        assert learner.groups[0][name] is parameter
        assert learner.traces[id(parameter)].shape==parameter.shape
        assert not learner.traces[id(parameter)].requires_grad
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
    learner.accumulate(p,post.unsqueeze(1)*pre.unsqueeze(0),now_ms=0)
    learner.apply_goodness(0.,now_ms=1000)
    assert p.grad is None and pre.grad is None and post.grad is None
    assert not learner.traces[id(p)].requires_grad
    assert torch.isfinite(p).all() and torch.isfinite(learner.traces[id(p)]).all()
    learner.clear()
    assert learner.traces[id(p)].count_nonzero()==0
    assert learner.g_bar==.5
    assert all(t is None for t in learner.trace_times_ms.values())


def test_duplicate_group_rejected():
    p=nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match="exactly one"):
        Plasticity({0:{"a":p},1:{"b":p}},None)