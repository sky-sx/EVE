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
    learner.internal[0].observe_vector(weight * torch.tensor([2., 4.]), torch.tensor([0.5, 0.5]), now_ms=0)
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
    assert delta == pytest.approx(float(torch.sigmoid(torch.tensor(0.9))) - 0.5)
    assert runtime._local_z == {}
    for name, value in runtime.adapters["goodness"].named_parameters():
        torch.testing.assert_close(value, ag_before[name])


def test_readin_pulse_does_not_replay_adapter_gradient(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    runtime.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(17))
    first = {name: value.clone() for name, value in learner.internal[1].values.items() if name.startswith("adapter.")}
    assert sum(value.abs().sum() for value in first.values()) > 1e-4
    runtime.update_blocks(now_ms=250)
    runtime.generate_hand(now_ms=250, generator=torch.Generator().manual_seed(18))
    for name, value in first.items():
        torch.testing.assert_close(learner.internal[1].values[name], math.exp(-1) * value, atol=1e-8, rtol=1e-5)


def test_invalid_feedback_time_rejected_without_parameter_change():
    weight = nn.Parameter(torch.ones(2))
    learner = Plasticity({0: {"w": weight}}, {0: 1.})
    learner.internal[0].observe_vector(weight, torch.ones_like(weight), now_ms=1000)
    with pytest.raises(ValueError, match="precede"):
        learner.apply_goodness(1., now_ms=999)
    torch.testing.assert_close(weight, torch.ones(2))


def test_vector_vjp_matches_explicit_neuron_factorization_and_decay():
    weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.], [2., -1.]]))
    bank = EligibilityBank({"w": weight}, tau=2.)
    signal = torch.tensor([1., -2., 3.], requires_grad=True)
    output = torch.nn.functional.layer_norm(weight @ torch.tensor([2., -1.]), (3,))
    jacobian = torch.stack([torch.autograd.grad(v, weight, retain_graph=True)[0] for v in output])
    expected = (signal.detach()[:, None, None] * jacobian).sum(0)
    bank.observe_vector(output, signal, now_ms=0)
    torch.testing.assert_close(bank.values["w"], expected)
    assert expected.abs().max() > 1e-3
    bank.observe_vector(output, signal, now_ms=1000)
    torch.testing.assert_close(bank.values["w"], (1 + math.exp(-0.5)) * expected)
    assert signal.grad is None and not bank.values["w"].requires_grad


def test_fixed_feedback_produces_nonzero_local_tags_without_mean_residual(make_runtime):
    runtime = make_runtime(neuron_size=5)
    learner = runtime.enable_plasticity(feedback_seed=29, learning_rate=1e-5)
    with torch.no_grad():
        for block in runtime.core.blocks:
            block.b.copy_(torch.linspace(-0.6, 0.8, 10))
    runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
    # Deterministic actual discrete action; no continuous path is involved.
    from acnt.control import sample_discrete
    hand = sample_discrete(runtime.decode_readout("hand")[:85], tau=0.25, generator=torch.Generator().manual_seed(41))
    route = sample_discrete(runtime.decode_readout("route"), tau=0.25, generator=torch.Generator().manual_seed(43))
    scores = {"hand": (hand.a.float() - hand.p).detach() / hand.tau,
              "route": (route.a.float() - route.p).detach() / route.tau}
    expected = {}
    for i in (1, 6):  # ReadIn plus an ordinary Block.
        z = runtime._local_z[i]
        signal = sum(getattr(runtime, f"feedback_{i}_{name}") @ score for name, score in scores.items())
        params = learner.groups[i]
        expected[i] = torch.autograd.grad(z, tuple(params.values()), grad_outputs=signal, allow_unused=True, retain_graph=True)
    runtime._observe_discrete("hand", hand, now_ms=0)
    # Hand alone must drive Stage 0's internal path, without route or dx/dy.
    for i in (1, 6):
        assert learner.internal[i].values["block.b"].abs().max() > 1e-3
    assert learner.internal[1].values["adapter.linear.weight"].abs().max() > 1e-3
    runtime._observe_discrete("route", route, now_ms=0)
    for i in (1, 6):
        for (name, parameter), derivative in zip(learner.groups[i].items(), expected[i]):
            target = torch.zeros_like(parameter) if derivative is None else derivative
            torch.testing.assert_close(learner.internal[i].values[name], target, atol=1e-5, rtol=1e-4)
        assert learner.internal[i].values["block.b"].abs().max() > 1e-3
    assert learner.internal[1].values["adapter.linear.weight"].abs().max() > 1e-3
    buffers = {name: value.clone() for name, value in runtime.named_buffers() if name.startswith("feedback_")}
    assert len(buffers) == 2 * len(runtime.core.blocks)
    assert not set(buffers).intersection(dict(runtime.named_parameters()))
    for name, value in buffers.items():
        assert not value.requires_grad
    before = {name: value.clone() for name, value in learner.groups[6].items()}
    tag = learner.internal[6].values["block.b"].clone()
    learner.apply_goodness(0.9, now_ms=250)
    torch.testing.assert_close(learner.groups[6]["block.b"], before["block.b"] + 1e-5 * 0.4 * math.exp(-1) * tag)
    for name, value in buffers.items():
        torch.testing.assert_close(getattr(runtime, name), value, rtol=0, atol=0)
    other = make_runtime(neuron_size=5)
    other.enable_plasticity(feedback_seed=29)
    for name, value in buffers.items():
        torch.testing.assert_close(getattr(other, name), value, rtol=0, atol=0)
    assert all(torch.isfinite(p).all() for p in runtime.parameters())
    assert all(torch.isfinite(t).all() for channel in learner.banks() for bank in channel.values() for t in bank.values.values())


def test_feedback_seed_does_not_change_forward_or_action_sampling(make_runtime):
    left, right = make_runtime(), make_runtime()
    left.enable_plasticity(feedback_seed=7)
    right.enable_plasticity(feedback_seed=11)
    assert not torch.equal(left.feedback_0_hand, right.feedback_0_hand)
    outputs = []
    for runtime in (left, right):
        runtime.update_blocks(now_ms=0, readins={"ear": torch.ones(1, 32)})
        hand = runtime.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(59))
        outputs.append(hand)
    for a, b in zip(left.core.blocks, right.core.blocks):
        torch.testing.assert_close(a.z, b.z, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0].discrete.q, outputs[1].discrete.q, rtol=0, atol=0)
    assert torch.equal(outputs[0].discrete.a, outputs[1].discrete.a)
    torch.testing.assert_close(outputs[0].continuous, outputs[1].continuous, rtol=0, atol=0)
