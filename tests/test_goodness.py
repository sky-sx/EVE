import pytest
import torch


def configure(runtime, raw):
    adapter=runtime.adapters["goodness"]
    with torch.no_grad():
        adapter.linear.weight.zero_()
        adapter.linear.bias.fill_(raw)
    return adapter


@pytest.mark.parametrize("raw",[-2.,.25,2.])
def test_goodness_is_clamped_scalar(raw,make_runtime):
    runtime=make_runtime()
    configure(runtime,raw)
    signal=runtime.generate_goodness(now_ms=0)
    assert float(signal.g)==pytest.approx(min(1.,max(0.,raw)))
    assert signal.g_eff==float(signal.g)
    assert signal.teacher is None and signal.calibration_loss is None


@pytest.mark.parametrize("teacher",[0.,.75,1.])
def test_same_time_teacher_is_effective_scalar(teacher,make_runtime):
    runtime=make_runtime()
    adapter=configure(runtime,.25)
    runtime.enable_plasticity()
    before=adapter.linear.bias.clone()
    signal=runtime.generate_goodness(now_ms=100,teacher=teacher,teacher_time_ms=100)
    assert signal.g_eff==teacher
    assert signal.calibration_loss==pytest.approx(.5*(teacher-.25)**2)
    # Teacher calibration reports c_g only; it reuses no eligibility trace.
    assert torch.equal(adapter.linear.bias,before)


def test_local_calibration_excludes_ordinary_groups(make_runtime):
    runtime=make_runtime()
    adapter=configure(runtime,.25)
    learner=runtime.enable_plasticity(learning_rate=.1)
    initial={id(p):p.clone() for p in runtime.parameters()}
    signal=runtime.generate_goodness(now_ms=0,teacher=.75)
    assert signal.g_eff==.75
    assert torch.equal(adapter.linear.bias,initial[id(adapter.linear.bias)])
    for g in learner.groups.values():
        for p in g.values():
            torch.testing.assert_close(p,initial[id(p)])


def test_no_teacher_reuse_and_disabled_goodness(make_runtime):
    runtime=make_runtime()
    configure(runtime,.3)
    runtime.generate_goodness(now_ms=0,teacher=.8)
    later=runtime.generate_goodness(now_ms=250)
    assert later.teacher is None and later.g_eff==float(later.g)
    runtime.set_goodness_active(False)
    off=runtime.generate_goodness(now_ms=500,teacher=.9)
    assert off.g_eff==.9 and off.calibration_loss is None


@pytest.mark.parametrize("timestamp",[0,500,250.])
def test_teacher_timestamp_validation(timestamp,make_runtime):
    runtime=make_runtime()
    with pytest.raises(ValueError,match="timestamp"):
        runtime.generate_goodness(now_ms=250,teacher=.75,teacher_time_ms=timestamp)


@pytest.mark.parametrize("teacher",[-.01,1.01,float("nan"),float("inf")])
def test_invalid_teacher(teacher,make_runtime):
    with pytest.raises(ValueError,match="teacher"):
        make_runtime().generate_goodness(now_ms=250,teacher=teacher)
