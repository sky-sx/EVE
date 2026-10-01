import pytest
import torch
from torch.func import functional_call
from acnt.calibrated_write import CalibratedWriteKernel, CalibratedWriteLearner


def test_vector_self_calibration_writes_every_actual_parameter_when_eligible():
    kernel = CalibratedWriteKernel(width=2, hold=2, group_size=8)
    model = CalibratedWriteLearner(kernel)
    before = kernel.effective(model.state).detach().clone()
    model.commit(torch.ones(kernel.parameter_size), torch.full((kernel.group_count,), .5))
    after = kernel.effective(model.state).detach()
    assert torch.count_nonzero(after-before) == kernel.parameter_size
    assert torch.count_nonzero((after-before)[kernel.writer_indices]) == kernel.writer_indices.numel()


def test_reward_calibrates_via_actual_self_write_without_outer_parameter_updates():
    kernel = CalibratedWriteKernel(width=2, hold=2, group_size=8)
    model = CalibratedWriteLearner(kernel)
    origins = kernel.base().detach().clone()
    budget = model.persistent_tensor_bytes
    for event in range(8):
        model.step(now_ms=event, goodness=float(event%2), action=event%2)
        assert model.persistent_tensor_bytes == budget
        assert model.state.grad_fn is None and model.tangent.grad_fn is None
    torch.testing.assert_close(kernel.base(), origins, rtol=0, atol=0)
    assert model.ever_written[kernel.writer_indices].any()
    assert not model.ever_written[~kernel.control_coordinates].any()  # no action eligibility injected


def test_cut_calibration_keeps_controller_actual_weights_fixed():
    kernel = CalibratedWriteKernel(width=2, hold=2, group_size=8)
    model = CalibratedWriteLearner(kernel, mode='cut_calibration')
    model.step(now_ms=0, goodness=1., action=1)
    assert not model.ever_written[kernel.control_coordinates].any()
    with pytest.raises(ValueError):
        model.step(now_ms=0)


def test_streaming_credit_matches_frozen_original_forward_unroll():
    kernel = CalibratedWriteKernel(width=2, hold=2, group_size=8)
    model = CalibratedWriteLearner(kernel, mode='no_write', normalize=False, tangent_limit=None)
    weights = kernel.base().detach().requires_grad_(True)
    neural = model.state[:kernel.neural_size].clone()
    times = tuple(() for _ in range(7))
    cue = torch.tensor([[.7, -.7, .35, -.35]])
    for event in range(3):
        current = cue if event == 0 else None
        neural, _ = functional_call(kernel.body,
            (kernel.mapping(weights), dict(kernel.body.named_buffers())),
            (neural, times, event*2, current, torch.zeros(4), weights, kernel.addresses, False))
        times = tuple(tuple((*old, event*2)[-kernel.hold:]) for old in times)
        model.step(now_ms=event*2, cue=current)
    expected = torch.autograd.grad(neural, weights,
        grad_outputs=torch.eye(kernel.neural_size), is_grads_batched=True)[0]
    torch.testing.assert_close(model.tangent, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('bias', [-4., 4.])
def test_exploration_floor_and_score_conditioned_on_temperature(bias):
    from experiments.calibrated_write_exploration import ExploringKernel, ExploringLearner
    kernel = ExploringKernel(width=2, hold=2, group_size=8)
    with torch.no_grad():
        kernel.body.hand.network[-1].weight.zero_()
        kernel.body.hand.network[-1].bias.fill_(bias)
    model = ExploringLearner(kernel)
    baseline = CalibratedWriteLearner(kernel)
    assert model.persistent_tensor_bytes == baseline.persistent_tensor_bytes+4
    _, probability, _ = model.sample(torch.Generator().manual_seed(1))
    assert .0474 <= probability <= .9526
    assert float(kernel.temperature) > 1
    score = model.policy_score(1)
    index = next(start for name,start,end,_ in kernel.schema if name=='hand.network.2.bias')
    expected = (1-probability)/float(kernel.temperature)
    assert abs(float(score[index])-expected)<1e-6
    assert (model.rates > 0).all()
