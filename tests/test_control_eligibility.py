import math

import pytest
import torch
from torch import nn

from acnt.control import sample_discrete
from acnt.plasticity import EligibilityBank, Plasticity


def test_score_function_matches_sum_of_per_action_derivatives():
    weight = nn.Parameter(torch.tensor([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.1]]))
    x = torch.tensor([2., -1.])
    signal = sample_discrete(weight @ x, tau=0.5, threshold=0.1, generator=torch.Generator().manual_seed(97))
    bank = EligibilityBank({"w": weight}, tau=0.5)
    score = bank.observe_control(signal, now_ms=0)
    expected_score = (signal.a.float() - signal.p.detach()) / 0.5
    expected_trace = expected_score[:, None] * x[None, :]
    torch.testing.assert_close(score, expected_score)
    torch.testing.assert_close(bank.values["w"], expected_trace)
    assert not signal.a.requires_grad and not score.requires_grad


def test_independent_actions_sum_for_shared_parameter_instead_of_mean():
    shared = nn.Parameter(torch.tensor(0.25))
    coefficients = torch.tensor([1., 2., -1.])
    signal = sample_discrete(shared * coefficients, tau=0.25, generator=torch.Generator().manual_seed(4))
    bank = EligibilityBank({"shared": shared}, tau=0.25)
    score = bank.observe_control(signal, now_ms=0)
    torch.testing.assert_close(bank.values["shared"], (score * coefficients).sum())
    assert bank.values["shared"].abs() > 0.01


def test_discrete_trace_survives_delay_then_updates_parameter():
    parameter = nn.Parameter(torch.tensor([0.2, -0.3]))
    learner = Plasticity({0: {"q": parameter}}, {0: 0.5}, learning_rate=0.1, rho=0.75)
    signal = sample_discrete(parameter * 2, tau=0.5, generator=torch.Generator().manual_seed(3))
    score = learner.control[0].observe_control(signal, now_ms=0)
    before = parameter.detach().clone()
    trace = 2 * score * math.exp(-1.)
    delta = learner.apply_goodness(1., now_ms=500)
    assert delta == 0.5
    torch.testing.assert_close(parameter, before + 0.1 * 0.5 * trace)
    torch.testing.assert_close(learner.control[0].values["q"], 0.75 * trace)
    assert not torch.equal(parameter, before)


def test_control_recurrence_adds_new_sample_to_decayed_previous_sample():
    parameter = nn.Parameter(torch.tensor([0.2, -0.3]))
    bank = EligibilityBank({"q": parameter}, tau=0.5)
    first = sample_discrete(parameter + 0., tau=0.5, generator=torch.Generator().manual_seed(1))
    e0 = bank.observe_control(first, now_ms=0).clone()
    second = sample_discrete(parameter + 0., tau=0.5, generator=torch.Generator().manual_seed(2))
    e1 = bank.observe_control(second, now_ms=500)
    torch.testing.assert_close(bank.values["q"], math.exp(-1) * e0 + e1)


def test_hand_control_keeps_continuous_axes_out_of_score_trace(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    hand = runtime.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(101))
    block_id = runtime.organ_blocks["hand"]
    trace = learner.control[block_id].values["adapter.network.2.bias"]
    expected = (hand.discrete.a.float() - hand.discrete.p.detach()) / hand.discrete.tau
    torch.testing.assert_close(trace[:85], expected)
    torch.testing.assert_close(trace[85:], torch.zeros(2))
    assert sum(value.abs().sum() for name, value in learner.control[block_id].values.items() if name.startswith("block.")) > 0


def test_execution_switch_does_not_change_local_sample_or_eligibility(make_runtime):
    off, on = make_runtime(), make_runtime()
    off.enable_plasticity()
    on.enable_plasticity()
    off.set_execution_enabled("hand", False)
    on.set_execution_enabled("hand", True)
    left = off.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(109))
    right = on.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(109))
    assert torch.equal(left.discrete.a, right.discrete.a)
    for name, trace in off.plasticity.control[2].values.items():
        torch.testing.assert_close(trace, on.plasticity.control[2].values[name])


def test_route_role_override_does_not_replace_sampled_event_in_score(make_runtime):
    runtime = make_runtime()
    runtime.enable_plasticity()
    with torch.no_grad():
        runtime.adapters["route"].linear.weight.zero_()
        runtime.adapters["route"].linear.bias.fill_(-100.)
    signal = runtime.generate_route(now_ms=0, generator=torch.Generator().manual_seed(5))
    assert not signal.a.any()
    assert runtime.core.blocks[5].active
    expected = (signal.a.float() - signal.p.detach()) / signal.tau
    torch.testing.assert_close(runtime.plasticity.control[5].values["adapter.linear.bias"], expected)
