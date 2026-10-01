import pytest
import torch
from acnt.goodness_prediction import GoodnessPredictionKernel, GoodnessPredictionLearner


def test_new_goodness_and_writer_parameters_all_covered_and_writable():
    k = GoodnessPredictionKernel(width=4, hold=3, group_size=32)
    assert k.parameter_size == 3839 and k.group_count == 148
    assert k.writer_indices.numel() == 740
    assert sum(p.numel() for p in k.body.goodness.parameters()) == 58
    assert k.body.group_index.numel() == k.parameter_size
    model = GoodnessPredictionLearner(k)
    before = k.effective(model.state).detach().clone()
    model.pending_prediction = .5
    model.teacher_for_commit = 1.
    model.commit(torch.ones(k.parameter_size), torch.full((k.group_count,),.5))
    after = k.effective(model.state).detach()
    assert (after != before).all()


def test_prediction_is_before_feedback_with_detached_finite_credit():
    k = GoodnessPredictionKernel(width=2, hold=2, group_size=8)
    model = GoodnessPredictionLearner(k)
    with pytest.raises(RuntimeError):
        model.step(now_ms=0, goodness=1., action=1)
    model.step(now_ms=0, cue=torch.tensor([[.7,-.7,.35,-.35]]))
    prediction = model.judge(1, now_ms=2)
    assert 0 <= prediction['predicted_goodness'] <= 1
    assert model.evaluation_traces.grad_fn is None
    assert torch.isfinite(model.evaluation_traces).all()
    budget = model.persistent_tensor_bytes
    with pytest.raises(RuntimeError):
        model.judge(0, now_ms=2)
    model.step(now_ms=2, goodness=0., action=1)
    assert model.pending_prediction is None and model.teacher_for_commit is None
    assert model.persistent_tensor_bytes == budget


def test_teacher_bce_direction_and_evaluation_trace_decay():
    k = GoodnessPredictionKernel(width=2, hold=2, group_size=8)
    model = GoodnessPredictionLearner(k, taus_ms=(2.,), max_update=1.)
    model.judge(1, now_ms=0)
    trace = model.evaluation_traces.clone()
    model.advance_trace(2)
    torch.testing.assert_close(model.evaluation_traces, trace*torch.exp(torch.tensor(-1.)))
    model.evaluation_traces.copy_(trace)
    index = next(a+1 for name,a,b,_ in k.schema if name=='goodness.network.2.bias')
    before = k.effective(model.state)[index].detach().clone()
    model.teacher_for_commit = 0.
    model.commit(torch.zeros(k.parameter_size), torch.full((k.group_count,),.5))
    assert k.effective(model.state)[index] < before  # selected logit decreases for teacher 0
    assert not model.ever_written[k.writer_indices].any()  # no calibration E in this isolated test


def test_prediction_does_not_substitute_teacher_goodness_in_forward():
    k1 = GoodnessPredictionKernel(seed=7,width=2,hold=2,group_size=8)
    k2 = GoodnessPredictionKernel(seed=7,width=2,hold=2,group_size=8)
    a, b = GoodnessPredictionLearner(k1), GoodnessPredictionLearner(k2)
    a.judge(0,now_ms=0)
    b.judge(0,now_ms=0)
    a.pending_prediction, b.pending_prediction = .01, .99
    # Predictions legitimately affect their supervised loss, but not current
    # sensory G input or actor/Write calibration targets (both use teacher 1).
    da = a.step(now_ms=0,goodness=1.,action=0)
    db = b.step(now_ms=0,goodness=1.,action=0)
    torch.testing.assert_close(a.state[:k1.neural_size],b.state[:k2.neural_size])
    assert da['calibration_loss'] == db['calibration_loss']
