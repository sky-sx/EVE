import torch
from acnt.grouped_write_rms import RMSGroupedWriteKernel, RMSGroupedWriteLearner


def test_rms_is_only_actuator_scaling_and_restores_raw_eligibility():
    model = RMSGroupedWriteLearner(RMSGroupedWriteKernel(width=2, hold=2), lr=0)
    score = torch.linspace(-.01, .01, model.kernel.parameter_size)
    model.record_action(score, now_ms=0)
    old = model.kernel.eligibility.clone()
    state = model.state.clone()
    model.step(now_ms=0, goodness=1., action=1)
    torch.testing.assert_close(model.kernel.eligibility, old)
    torch.testing.assert_close(model.traces, score.expand_as(model.traces))
    torch.testing.assert_close(model.kernel.normalizer, old.abs()+1e-8, atol=1e-7, rtol=1e-5)
    assert not torch.equal(model.state[model.kernel.neural_size:], state[model.kernel.neural_size:])


def test_rms_resource_budget_is_fixed():
    model = RMSGroupedWriteLearner(RMSGroupedWriteKernel(width=2, hold=2), lr=0)
    size = model.persistent_tensor_bytes
    for t in range(4):
        model.step(now_ms=t*2, goodness=.7, action=1)
        assert model.persistent_tensor_bytes == size
        assert model.rms.grad_fn is None and model.kernel.normalizer.grad_fn is None
