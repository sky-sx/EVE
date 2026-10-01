import torch
import pytest
from copy import deepcopy
from torch.func import functional_call

from acnt.core import Core
from acnt.self_write import SelfWriteKernel, SelfWriteLearner


def inputs(t):
    return dict(now_ms=t * 2, cue=torch.tensor([[.7, -.7, .35, -.35]]) if t % 3 == 0 else None,
                goodness=.7 if t % 3 == 2 else None, action=1 if t % 3 == 2 else None)


def test_joint_state_forward_matches_canonical_core_with_actual_weight_replacement():
    kernel = SelfWriteKernel(seed=11, width=2, hold=2)
    canonical = Core([deepcopy(transition.block) for transition in kernel.transitions])
    state, times = kernel.initial_state(), tuple(() for _ in range(7))
    # A non-initial weight verifies that packed actor weights actually participate.
    state[kernel.state_size:-1] *= .6
    for t in range(3):
        args = inputs(t)
        obs = state.new_tensor([args['goodness'] or 0., float(args['goodness'] is not None),
                                args['action'] or 0., float(args['action'] is not None)])
        with torch.no_grad():
            canonical.blocks[2].W_ij[1].copy_(state[kernel.state_size:-1].reshape(4, 2))
            if args['cue'] is not None:
                canonical.blocks[1].o.copy_(kernel.ear.forward_train(args['cue']))
            canonical.blocks[6].o.copy_(kernel.learning_readin(obs))
            canonical.step(now_ms=args['now_ms'])
        state = kernel(state, times, args['now_ms'], args['cue'], obs, write_enabled=False)
        times = tuple(tuple((*old, args['now_ms'])[-kernel.hold:]) for old in times)
        unpacked, _ = kernel.unpack(state, times)
        for block, (z, history) in zip(canonical.blocks, unpacked):
            torch.testing.assert_close(z, block.z)
            torch.testing.assert_close(torch.stack(history), torch.stack(tuple(block.A)))


@pytest.mark.parametrize("write_mode", ["target", "delta"])
def test_meta_tangent_includes_two_actual_writes_and_matches_full_history(write_mode):
    kernel = SelfWriteKernel(seed=11, width=2, hold=2, write_mode=write_mode)
    model = SelfWriteLearner(kernel, lr=0, meta_limit=None)
    state = kernel.initial_state()
    times = tuple(() for _ in range(7))
    for t in range(4):
        args = inputs(t)
        obs = state.new_tensor([args['goodness'] or 0., float(args['goodness'] is not None),
                                args['action'] or 0., float(args['action'] is not None)])
        state = kernel(state, times, args['now_ms'], args['cue'], obs)
        times = tuple(tuple((*old, args['now_ms'])[-kernel.hold:]) for old in times)
        model.step(**args)
    grads = torch.autograd.grad(state, model.parameters,
        grad_outputs=torch.eye(state.numel()), is_grads_batched=True, allow_unused=True)
    expected = model._flatten(grads, state.numel())
    torch.testing.assert_close(model.state, state.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.meta, expected, atol=2e-5, rtol=2e-4)
    assert float(model.meta[kernel.state_size:].norm()) > 0
    assert model.state.grad_fn is None and model.meta.grad_fn is None


@pytest.mark.parametrize("write_mode", ["target", "delta"])
def test_controller_score_finite_difference_through_weight_writes(write_mode):
    kernel = SelfWriteKernel(seed=22, width=2, hold=2, write_mode=write_mode)
    model = SelfWriteLearner(kernel, lr=0, meta_limit=None)
    direction = torch.randn(model.param_size, generator=torch.Generator().manual_seed(4))
    direction /= direction.norm()
    for t in range(4):
        model.step(**inputs(t))
    score = model.policy_score(1)
    def evaluate(epsilon):
        replacements = {}
        start = 0
        for name, p in model.named_parameters:
            replacements[name] = p.detach() + epsilon * direction[start:start+p.numel()].reshape_as(p)
            start += p.numel()
        state, times = kernel.initial_state(), tuple(() for _ in range(7))
        for t in range(4):
            args = inputs(t)
            obs = state.new_tensor([args['goodness'] or 0., float(args['goodness'] is not None),
                                    args['action'] or 0., float(args['action'] is not None)])
            state = functional_call(kernel, replacements,
                (state, times, args['now_ms'], args['cue'], obs))
            times = tuple(tuple((*old, args['now_ms'])[-kernel.hold:]) for old in times)
        return torch.nn.functional.logsigmoid(kernel.logits(state)).detach()
    finite = (evaluate(.005) - evaluate(-.005)) / .01
    torch.testing.assert_close(score @ direction, finite, atol=3e-5, rtol=.08)


def test_no_write_and_cut_write_cannot_receive_controller_behavior_credit():
    for settings in ({'write_enabled': False}, {'cut_write_credit': True}):
        model = SelfWriteLearner(SelfWriteKernel(seed=11, width=2, hold=2), **settings)
        for t in range(5):
            model.step(**inputs(t))
        torch.testing.assert_close(model.policy_score(1), torch.zeros(model.param_size))


def test_outer_updates_preserve_state_and_fixed_resources():
    model = SelfWriteLearner(SelfWriteKernel(seed=33, width=2, hold=2))
    byte_count = model.persistent_tensor_bytes
    initial = model.kernel.initial_fast.clone()
    for t in range(8):
        model.step(**inputs(t))
        before, times = model.state.clone(), model.times
        model.learn(model.policy_score(t % 2), float(t % 2))
        torch.testing.assert_close(model.state, before)
        assert model.times == times
        assert model.persistent_tensor_bytes == byte_count
        assert model.state.grad_fn is None and model.meta.grad_fn is None
        assert all(len(x) <= model.kernel.hold for x in model.times)
    assert not torch.equal(initial, model.state[model.kernel.state_size:])
    assert float(model.state[-1].abs()) <= 4


def test_zero_delta_write_preserves_actual_weights():
    kernel = SelfWriteKernel(seed=33, width=2, hold=2, write_mode="delta")
    with torch.no_grad():
        kernel.write.weight.zero_()
        kernel.write.bias.zero_()
    model = SelfWriteLearner(kernel, lr=0)
    initial = kernel.initial_fast.clone()
    for t in range(8):
        model.step(**inputs(t))
        torch.testing.assert_close(model.state[kernel.state_size:], initial, rtol=0, atol=0)
