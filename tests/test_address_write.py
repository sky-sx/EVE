from copy import deepcopy

import pytest
import torch
from torch.func import functional_call

from acnt.address_write import AddressWriteKernel, AddressWriteLearner
from acnt.core import Core


def inputs(t):
    return dict(now_ms=t*2, cue=torch.tensor([[.7, -.7, .35, -.35]]) if t%3 == 0 else None,
                goodness=.7 if t%3 == 2 else None, action=1 if t%3 == 2 else None)


def test_every_effective_weight_including_writer_receives_address_write():
    kernel = AddressWriteKernel(width=2, hold=2)
    with torch.no_grad():
        kernel.body.write.network[-1].bias.fill_(.2)
    state = kernel.initial_state()
    before = kernel.effective(state).detach().clone()
    new = kernel(state, tuple(() for _ in range(7)), 0, None, state.new_zeros(4))
    after = kernel.effective(new)
    expected = before + .002*torch.tanh(torch.tensor(.2))
    torch.testing.assert_close(after, expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(after-before) == kernel.parameter_size
    assert torch.count_nonzero((after-before)[kernel.writer_indices]) == kernel.writer_indices.numel()


def test_no_write_matches_original_core_with_all_effective_weights_replaced():
    kernel = AddressWriteKernel(width=2, hold=2)
    canonical = Core([deepcopy(t.block) for t in kernel.body.transitions])
    state, times = kernel.initial_state(), tuple(() for _ in range(7))
    state[kernel.neural_size:] = .001*torch.randn(kernel.parameter_size)
    actual = kernel.mapping(kernel.effective(state))
    for i, block in enumerate(canonical.blocks):
        with torch.no_grad():
            for name, p in block.named_parameters():
                p.copy_(actual[f"transitions.{i}.block.{name}"])
    for t in range(3):
        args = inputs(t)
        obs = state.new_tensor([args['goodness'] or 0., float(args['goodness'] is not None),
                                args['action'] or 0., float(args['action'] is not None)])
        ear = {name.removeprefix('ear.'): value for name, value in actual.items() if name.startswith('ear.')}
        feedback = {name.removeprefix('learning_readin.'): value for name, value in actual.items()
                    if name.startswith('learning_readin.')}
        with torch.no_grad():
            if args['cue'] is not None:
                # EarAdapter.forward_train is a parameterized Linear on the waveform.
                canonical.blocks[1].o.copy_(functional_call(kernel.body.ear, ear, (args['cue'],)))
            canonical.blocks[6].o.copy_(functional_call(kernel.body.learning_readin, feedback, (obs,)))
            canonical.step(now_ms=args['now_ms'])
        state = kernel(state, times, args['now_ms'], args['cue'], obs, write_enabled=False)
        times = tuple(tuple((*old, args['now_ms'])[-kernel.hold:]) for old in times)
        stride = kernel.width*(1+kernel.hold)
        for i, block in enumerate(canonical.blocks):
            torch.testing.assert_close(state[i*stride:i*stride+kernel.width], block.z)


def test_streaming_credit_through_self_writes_matches_full_unroll():
    kernel = AddressWriteKernel(width=2, hold=2)
    with torch.no_grad():
        kernel.body.write.network[-1].weight.fill_(.03)
        kernel.body.write.network[-1].bias.fill_(.1)
    model = AddressWriteLearner(kernel, lr=0, meta_limit=None)
    state, times = kernel.initial_state(), tuple(() for _ in range(7))
    for t in range(4):
        args = inputs(t)
        obs = state.new_tensor([args['goodness'] or 0., float(args['goodness'] is not None),
                                args['action'] or 0., float(args['action'] is not None)])
        state = kernel(state, times, args['now_ms'], args['cue'], obs)
        times = tuple(tuple((*old, args['now_ms'])[-kernel.hold:]) for old in times)
        model.step(**args)
    expected = torch.autograd.grad(state, model.parameters, grad_outputs=torch.eye(state.numel()),
                                   is_grads_batched=True)
    expected = torch.cat([g.reshape(state.numel(), -1) for g in expected], dim=1)
    torch.testing.assert_close(model.state, state.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.meta, expected, atol=2e-5, rtol=2e-4)
    assert model.self_write_path > 0
    assert model.state.grad_fn is None and model.meta.grad_fn is None


@pytest.mark.parametrize('settings', [{'write_enabled': False}, {'cut_write_credit': True}])
def test_cut_writing_removes_writer_behavior_credit(settings):
    model = AddressWriteLearner(AddressWriteKernel(width=2, hold=2), **settings)
    for t in range(4):
        model.step(**inputs(t))
    torch.testing.assert_close(model.policy_score(1), torch.zeros(model.param_size))


def test_bootstrap_preserves_history_and_resource_budget():
    model = AddressWriteLearner(AddressWriteKernel(width=2, hold=2))
    budget = model.persistent_tensor_bytes
    for t in range(6):
        model.step(**inputs(t))
        before, times = model.state.clone(), model.times
        model.learn(model.policy_score(1), 1.)
        torch.testing.assert_close(model.state, before)
        assert times == model.times and budget == model.persistent_tensor_bytes
        assert float(model.kernel.effective(model.state).detach().abs().max()) <= 4.
    with pytest.raises(ValueError):
        model.step(now_ms=model.last_time)
