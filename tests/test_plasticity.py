import math

import pytest
import torch
from torch import nn

from acnt import Block
from acnt.plasticity import EligibilityBank, Plasticity


def test_mean_derivative_and_timed_eligibility_recurrence():
    weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
    bank = EligibilityBank({"weight": weight}, tau=2.)
    x0 = torch.tensor([2., 4.])
    output = weight @ x0
    individual = [torch.autograd.grad(value, weight, retain_graph=True)[0] for value in output]
    expected_first = torch.stack(individual).mean(0)
    bank.observe_mean(output, now_ms=0)
    torch.testing.assert_close(bank.values["weight"], expected_first)
    torch.testing.assert_close(expected_first, torch.tensor([[1., 2.], [1., 2.]]))
    x1 = torch.tensor([6., -2.])
    bank.observe_mean(weight @ x1, now_ms=1000)
    expected = math.exp(-0.5) * expected_first + torch.tensor([[3., -1.], [3., -1.]])
    torch.testing.assert_close(bank.values["weight"], expected)
    assert not bank.values["weight"].requires_grad


def test_delayed_goodness_updates_parameter_then_rho_and_baseline():
    weight = nn.Parameter(torch.tensor([1., 2.]))
    learner = Plasticity({0: {"weight": weight}}, {0: 2.}, learning_rate=0.2, rho=0.5, ema_alpha=0.1, initial_g_bar=0.25)
    learner.internal[0].observe_mean(weight * torch.tensor([2., 4.]), now_ms=0)
    initial_trace = torch.tensor([1., 2.])
    delta = learner.apply_goodness(0.75, now_ms=1000)
    decayed = math.exp(-0.5) * initial_trace
    assert delta == 0.5
    torch.testing.assert_close(weight, torch.tensor([1., 2.]) + 0.2 * 0.5 * decayed)
    torch.testing.assert_close(learner.internal[0].values["weight"], decayed * 0.5)
    assert learner.g_bar == pytest.approx(0.3)
    learner.clear()
    assert torch.count_nonzero(learner.internal[0].values["weight"]) == 0


def test_all_required_parameter_tensors_have_distinct_traces(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    runtime.update_blocks(now_ms=0, readins={"eye": torch.ones(3, 1080, 1920), "ear": torch.ones(1, 32)})
    ag_ids = {id(p) for p in runtime.adapters["goodness"].parameters()}
    for block_id, parameters in learner.groups.items():
        assert "block.b" in parameters
        assert "block.W_ij.0" in parameters
        assert "block.W_c.0" in parameters and "block.b_c.0" in parameters
        assert not ag_ids.intersection(id(p) for p in parameters.values())
        for bank in (channel[block_id] for channel in learner.banks()):
            assert set(bank.values) == set(parameters)
            for name, parameter in parameters.items():
                trace = bank.values[name]
                assert trace.shape == parameter.shape and trace.dtype == torch.float32
                assert trace.data_ptr() != parameter.data_ptr()
                assert not trace.requires_grad and torch.isfinite(trace).all()
    assert any(name.startswith("adapter.") for name in learner.groups[0])
    assert any(name.startswith("adapter.") for name in learner.groups[1])


def test_local_graph_does_not_backpropagate_into_sources_or_old_history():
    block = Block(0, 3, [3], readin=True)
    source = torch.tensor([1., 2., 3.], requires_grad=True)
    old_a = torch.tensor([1., -1., 2.], requires_grad=True)
    block.A.append(old_a)
    block.At.append(0)
    block.o = torch.tensor([1., 2., -1., 0., 0., 0.], requires_grad=True)
    z = block.update(now_ms=1000, active_z={0: source}, track_grad=True)
    scalar = (z * torch.tensor([1., 2., 4.])).sum()
    assert torch.autograd.grad(scalar, (source, old_a), allow_unused=True, retain_graph=True) == (None, None)
    assert z.requires_grad
    assert all(not getattr(block, name).requires_grad for name in ("z", "a", "r", "h", "o"))
    assert all(not value.requires_grad for value in block.A)


def test_continuous_readout_trace_reaches_its_block_and_adapter(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity(learning_rate=0.0001)
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    runtime.generate_speak(now_ms=0)
    runtime.generate_hand(now_ms=0)
    speak_id = runtime.organ_blocks["speak"]
    trace = learner.continuous[speak_id].values
    assert trace["adapter.linear.bias"].abs().sum() > 0
    assert sum(value.abs().sum() for name, value in trace.items() if name.startswith("block.")) > 0
    hand_trace = learner.continuous[runtime.organ_blocks["hand"]].values
    torch.testing.assert_close(hand_trace["adapter.network.2.bias"][:85], torch.zeros(85))
    torch.testing.assert_close(hand_trace["adapter.network.2.bias"][85:], torch.full((2,), 0.5))
    with torch.no_grad():
        runtime.adapters["goodness"].linear.weight.zero_()
        runtime.adapters["goodness"].linear.bias.fill_(0.9)
    ag_before = {name: value.clone() for name, value in runtime.adapters["goodness"].named_parameters()}
    signal = runtime.generate_goodness(now_ms=0)  # No teacher: A_g cannot learn.
    delta = runtime.learn_goodness(signal, delivered_ms=250)
    assert delta == pytest.approx(0.4)
    assert runtime._local_z == {}
    for name, value in runtime.adapters["goodness"].named_parameters():
        torch.testing.assert_close(value, ag_before[name])


def test_readin_held_value_does_not_replay_adapter_gradient(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    first = {name: value.clone() for name, value in learner.internal[1].values.items() if name.startswith("adapter.")}
    runtime.update_blocks(now_ms=250)
    for name, value in first.items():
        torch.testing.assert_close(learner.internal[1].values[name], math.exp(-1) * value, atol=1e-8, rtol=1e-5)


def test_invalid_feedback_time_rejected_without_parameter_change():
    weight = nn.Parameter(torch.ones(2))
    learner = Plasticity({0: {"w": weight}}, {0: 1.})
    learner.internal[0].observe_mean(weight, now_ms=1000)
    with pytest.raises(ValueError, match="precede"):
        learner.apply_goodness(1., now_ms=999)
    torch.testing.assert_close(weight, torch.ones(2))
