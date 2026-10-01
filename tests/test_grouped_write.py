from copy import deepcopy

import torch
from torch.func import functional_call

from acnt.address_write import AddressWriteKernel
from acnt.grouped_write import GroupedWriteKernel, GroupedWriteLearner


def test_compact_hand_preserves_original_used_action_path():
    old = AddressWriteKernel(seed=11)
    new = GroupedWriteKernel(seed=11)
    assert old.parameter_size-new.parameter_size == 774
    for _ in range(3):
        h = torch.randn(new.neural_size)
        a = torch.cat((h, old.initial_state()[old.neural_size:]))
        b = torch.cat((h, new.initial_state()[new.neural_size:]))
        torch.testing.assert_close(old.logits(a), new.logits(b), rtol=0, atol=0)


def test_shared_group_modulation_covers_all_coordinates_and_writer():
    kernel = GroupedWriteKernel(width=2, hold=2, group_size=8)
    assert kernel.body.group_index.numel() == kernel.parameter_size
    assert kernel.body.group_sizes.sum() == kernel.parameter_size
    assert (kernel.body.group_sizes <= 8).all()
    state = kernel.initial_state()
    kernel.eligibility.copy_((torch.arange(kernel.parameter_size)%2).float()*2-1)
    before = kernel.effective(state).detach()
    obs = torch.tensor([.7, 1., 1., 1.])
    new = kernel(state, tuple(() for _ in range(7)), 0, None, obs)
    change = kernel.effective(new).detach()-before
    assert torch.count_nonzero(change[kernel.writer_indices]) == kernel.writer_indices.numel()
    assert torch.count_nonzero(change) == kernel.parameter_size
    ratio = change/kernel.eligibility
    for group in range(kernel.group_count):
        members = kernel.body.group_index == group
        torch.testing.assert_close(ratio[members], ratio[members][0].expand(int(members.sum())),
                                   atol=3e-5, rtol=.01)


def test_zero_eligibility_or_absent_goodness_prevents_parameter_write():
    kernel = GroupedWriteKernel(width=2, hold=2)
    state = kernel.initial_state()
    for eligible, valid in ((0., 1.), (1., 0.)):
        kernel.eligibility.fill_(eligible)
        new = kernel(state, tuple(() for _ in range(7)), 0, None, torch.tensor([.7, valid, 1., 1.]))
        torch.testing.assert_close(kernel.effective(new), kernel.effective(state), rtol=0, atol=0)


def test_exponential_trace_matches_existing_decay_and_injection():
    model = GroupedWriteLearner(GroupedWriteKernel(width=2, hold=2), lr=0)
    score = torch.linspace(-.01, .01, model.kernel.parameter_size)
    model.record_action(score, now_ms=0)
    model.advance_eligibility(16)
    decay = torch.tensor([torch.exp(torch.tensor(-16./tau)) for tau in model.taus])
    torch.testing.assert_close(model.traces, decay[:, None]*score)
    torch.testing.assert_close(model.kernel.eligibility, model.traces.mean(dim=0))


def test_actor_eligibility_matches_unroll_when_weights_are_fixed():
    kernel = GroupedWriteKernel(width=2, hold=2)
    model = GroupedWriteLearner(kernel, lr=0, write_enabled=False, meta_limit=None)
    weights = kernel.base().detach().requires_grad_(True)
    neural, times = kernel.initial_state()[:kernel.neural_size], tuple(() for _ in range(7))
    cue = torch.tensor([[.7, -.7, .35, -.35]])
    for t in range(3):
        current = cue if t == 0 else None
        obs = torch.zeros(4)
        neural, _ = functional_call(kernel.body,
            (kernel.mapping(weights), dict(kernel.body.named_buffers())),
            (neural, times, t*2, current, obs, weights, kernel.addresses, False))
        times = tuple(tuple((*old, t*2)[-kernel.hold:]) for old in times)
        model.step(now_ms=t*2, cue=current)
    params = {name.removeprefix('hand.network.'): value for name, value in kernel.mapping(weights).items()
              if name.startswith('hand.network.')}
    stride = kernel.width*(1+kernel.hold)
    logit = functional_call(kernel.body.hand.network, params, (neural[2*stride:2*stride+kernel.width],))[0]
    expected = torch.autograd.grad(torch.nn.functional.logsigmoid(logit), weights)[0]
    torch.testing.assert_close(model.policy_score(1), expected, atol=2e-5, rtol=2e-4)


def test_meta_credit_matches_conditional_unroll_with_given_eligibility():
    kernel = GroupedWriteKernel(width=2, hold=2)
    reference = deepcopy(kernel)
    model = GroupedWriteLearner(kernel, lr=0, meta_limit=None)
    state, times = reference.initial_state(), tuple(() for _ in range(7))
    for t in range(3):
        model.traces.fill_(.2)
        model.advance_eligibility(t*2)
        reference.eligibility = kernel.eligibility.detach().clone()
        state = reference(state, times, t*2, None, torch.tensor([.7, 1., 1., 1.]))
        times = tuple(tuple((*old, t*2)[-kernel.hold:]) for old in times)
        model.step(now_ms=t*2, goodness=.7, action=1)
    parameters = [p for p in reference.parameters() if p.requires_grad]
    grads = torch.autograd.grad(state, parameters, grad_outputs=torch.eye(state.numel()),
                                is_grads_batched=True)
    expected = torch.cat([g.reshape(state.numel(), -1) for g in grads], dim=1)
    torch.testing.assert_close(model.state, state.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.meta, expected, atol=2e-5, rtol=2e-4)


def test_real_self_write_credit_and_constant_resource_budget():
    model = GroupedWriteLearner(GroupedWriteKernel(width=2, hold=2), lr=0)
    budget = model.persistent_tensor_bytes
    for t in range(12):
        if t%3 == 2:
            score = model.policy_score(1)
            model.record_action(score, now_ms=t*2)
            model.step(now_ms=t*2, goodness=.7, action=1)
        else:
            model.step(now_ms=t*2)
        assert model.persistent_tensor_bytes == budget
        assert model.meta.grad_fn is None and model.actor_tangent.grad_fn is None
    assert model.self_write_path > 0
