import copy
import pytest
import torch
from acnt.original_training import OriginalTrainingRuntime
from acnt.original_online import OriginalOnlineRuntime, DelayedOnlineTrainer, OnlineAction, FixedDelayLedgerTrainer
from experiments.original_acnt_online import build


def test_online_forward_equals_original_async_forced(make_runtime):
    runtime = make_runtime(neuron_size=3, ticktime=2)
    reference = copy.deepcopy(runtime)
    online = OriginalOnlineRuntime(runtime)
    for t, mask in zip((0, 1, 7), ([True] * 8, [False, False, True, False, True, True, False, True], [True] * 8)):
        runtime.core.set_active_mask(mask)
        reference.core.set_active_mask(mask)
        audio = torch.linspace(-1, 1, 32).reshape(1, 32)
        expected = reference.update_blocks(now_ms=t, readins={"ear": audio})
        actual = online.step(now_ms=t, readins={"ear": audio})
        assert set(expected) == set(actual)
        for b, s in zip(reference.core.blocks, online.states):
            torch.testing.assert_close(b.z, s.z, atol=0, rtol=0)
            assert tuple(b.At) == s.times
            for x, y in zip(b.A, s.history):
                torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_streaming_sensitivity_matches_full_history_bptt(make_runtime):
    runtime = make_runtime(neuron_size=3, block_count=6, ticktime=1)
    reference = OriginalTrainingRuntime(copy.deepcopy(runtime))
    online = OriginalOnlineRuntime(runtime)
    audio = torch.linspace(-.6, .6, 32).reshape(1, 32)
    for t in range(4):
        inputs = {"ear": audio} if t == 0 else None
        online.step(now_ms=t, readins=inputs)
        reference.step(now_ms=t, readins=inputs)
    vector = online.pack(reference.states)
    objective = vector.square().sum()
    reference_params = dict(reference.runtime.named_parameters())
    gradients = torch.autograd.grad(objective,
        [reference_params[n] for n, _ in online.named_parameters], allow_unused=True)
    exact = online._flat_parameter_grad(gradients)
    streaming = (2 * online.state) @ online.sensitivity
    torch.testing.assert_close(streaming, exact, atol=2e-4, rtol=3e-4)
    ear = next((n, p) for n, p in online.named_parameters if n == "adapters.ear.linear.weight")
    offset = sum(p.numel() for n, p in online.named_parameters[:[n for n, _ in online.named_parameters].index(ear[0])])
    assert streaming[offset:offset + ear[1].numel()].norm() > 0
    assert online.state.grad_fn is None and online.sensitivity.grad_fn is None


def test_action_score_matches_full_history_policy_derivative(make_runtime):
    runtime = make_runtime(neuron_size=3, block_count=6, ticktime=1)
    reference = OriginalTrainingRuntime(copy.deepcopy(runtime))
    online = OriginalOnlineRuntime(runtime)
    audio = torch.linspace(-.5, .5, 32).reshape(1, 32)
    for t in range(4):
        inputs = {"ear": audio} if t == 0 else None
        online.step(now_ms=t, readins=inputs)
        reference.step(now_ms=t, readins=inputs)
    actual = online.sample_discrete(generator=torch.Generator().manual_seed(9))
    sample = reference.bernoulli(reference.decode("hand")[:1],
        tau=runtime.noise_scale, generator=torch.Generator().manual_seed(9))
    assert torch.equal(actual.action, sample.action)
    parameters = dict(reference.runtime.named_parameters())
    gradients = torch.autograd.grad(sample.log_prob,
        [parameters[n] for n, _ in online.named_parameters], allow_unused=True)
    torch.testing.assert_close(actual.score, online._flat_parameter_grad(gradients), atol=2e-5, rtol=3e-4)


def test_tangent_matches_common_offset_under_exogenous_weight_drift():
    model, _ = build(11, width=3, hold=3)
    model.sensitivity_limit = None
    # Same external weight schedule, same observations; perturb one common
    # parameter coordinate throughout. The learning schedule is not differentiated.
    runtime = copy.deepcopy(model.runtime)
    name = "adapters.ear.linear.weight"
    index = (0, 0)
    audio = torch.tensor([[.7, -.7, .35, -.35]])
    for t in range(4):
        with torch.no_grad():
            dict(model.runtime.named_parameters())[name].add_(.0005)
        model.step(now_ms=t, readins={"ear": audio} if t == 0 else None)
    pidx = [n for n, _ in model.named_parameters].index(name)
    offset = sum(p.numel() for _, p in model.named_parameters[:pidx])
    analytic = model.sensitivity[:, offset]
    def replay(eps):
        rt = copy.deepcopy(runtime)
        ref = OriginalTrainingRuntime(rt)
        with torch.no_grad():
            dict(rt.named_parameters())[name][index] += eps
        for t in range(4):
            with torch.no_grad():
                dict(rt.named_parameters())[name].add_(.0005)
            # One-step no-grad trajectory; preserve values, not an old graph.
            ref.step(now_ms=t, readins={"ear": audio} if t == 0 else None)
            ref.truncate()
        return model.pack(ref.states).detach()
    numeric = (replay(.001) - replay(-.001)) / .002
    torch.testing.assert_close(analytic, numeric, atol=.002, rtol=.04)


def test_delayed_credit_survives_updates_without_changing_history():
    model, trainer = build(11, width=3, hold=3)
    generator = torch.Generator().manual_seed(9)
    for t in range(4):
        trainer.advance(t)
        model.step(now_ms=t, readins={"ear": torch.tensor([[1., -1., .5, -.5]])} if t == 0 else None)
    action = model.sample_discrete(generator=generator)
    trainer.record_action(action, now_ms=3)
    old_state = model.state.clone()
    old_history = [tuple(h.clone() for h in s.history) for s in model.states]
    before = [p.detach().clone() for p in model.parameters]
    trainer.observe_goodness(1., now_ms=8)
    trainer.observe_goodness(0., now_ms=12)
    assert trainer.updates == 2 and trainer.traces.norm() > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters))
    torch.testing.assert_close(old_state, model.state, atol=0, rtol=0)
    for hs, s in zip(old_history, model.states):
        for x, y in zip(hs, s.history):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    model.step(now_ms=13)
    assert model.events == 5
    assert all(s.z.grad_fn is None for s in model.states)


def test_credit_cut_and_fixed_memory_and_validation():
    model, trainer = build(11, width=3, hold=3, mode="cut_delay")
    count = trainer.learning_state_bytes
    model.step(now_ms=0)
    action = model.sample_discrete(generator=torch.Generator().manual_seed(4))
    trainer.record_action(action, now_ms=0)
    assert trainer.traces.norm() > 0
    trainer.advance(1)
    assert trainer.traces.norm() == 0
    with pytest.raises(ValueError): trainer.observe_goodness(float("nan"), now_ms=1)
    with pytest.raises(ValueError): model.step(now_ms=-1)
    for t in range(1, 5):
        model.step(now_ms=t)
    assert count == trainer.learning_state_bytes


def test_baseline_is_captured_before_action_not_at_late_feedback():
    model, trainer = build(11, width=3, hold=3)
    score = torch.zeros(model.param_dim)
    score[0] = 1
    trainer.record_action(OnlineAction(torch.tensor([1.]), torch.tensor([.5]), score), now_ms=0)
    trainer.baseline = .99
    result = trainer.observe_goodness(1., now_ms=10)
    expected = .5 * sum(__import__('math').exp(-10 / tau) for tau in trainer.taus) / len(trainer.taus)
    assert result['direction_norm'] == pytest.approx(expected)
    assert trainer.m[0] == pytest.approx(.1 * expected)


def test_fixed_delay_ledger_uses_old_score_despite_newer_actions():
    model, _ = build(11, width=3, hold=3)
    trainer = FixedDelayLedgerTrainer(model, feedback_delay_ms=10, capacity=2)
    old = torch.zeros(model.param_dim); old[0] = 1
    new = torch.zeros_like(old); new[1] = 1
    trainer.record_action(OnlineAction(torch.tensor([1.]), torch.tensor([.5]), old), now_ms=0)
    trainer.record_action(OnlineAction(torch.tensor([1.]), torch.tensor([.5]), new), now_ms=5)
    trainer.observe_goodness(1., now_ms=10)
    assert trainer.m[0] > 0 and trainer.m[1] == 0
    assert trainer.ledger_times.count(None) == 1
    with pytest.raises(ValueError): trainer.observe_goodness(1., now_ms=11)
