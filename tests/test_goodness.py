from dataclasses import fields

import pytest
import torch

from acnt.runtime import GoodnessSignal


def configure_prediction(runtime, bias, z=None):
    adapter = runtime.adapters["goodness"]
    block = runtime.core.blocks[runtime.organ_blocks["goodness"]]
    with torch.no_grad():
        adapter.linear.weight.zero_()
        adapter.linear.bias.fill_(bias)
        block.z.copy_(torch.zeros_like(block.z) if z is None else z)
    return adapter, block


def parameter_snapshot(runtime):
    return {name: value.detach().clone() for name, value in runtime.named_parameters()}


def assert_parameters_unchanged(runtime, before):
    for name, value in runtime.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("raw,expected", [(-2., 0.), (0.25, 0.25), (2., 1.)])
def test_goodness_is_one_clamped_scalar_without_teacher(make_runtime, raw, expected):
    runtime = make_runtime()
    configure_prediction(runtime, raw)
    before = parameter_snapshot(runtime)

    signal = runtime.generate_goodness(now_ms=250)

    assert isinstance(signal, GoodnessSignal)
    assert signal.time_ms == 250
    assert signal.g.shape == ()
    assert signal.g.dtype == torch.float32
    assert not signal.g.requires_grad
    assert float(signal.g) == expected
    assert signal.g_eff == expected
    assert signal.teacher is None
    assert signal.calibration_loss is None
    assert runtime.calibration_eligibility == {}
    assert_parameters_unchanged(runtime, before)
    assert {field.name for field in fields(signal)} == {
        "time_ms", "g", "g_eff", "teacher", "calibration_loss",
    }


@pytest.mark.parametrize("teacher", [0., 0.75, 1.])
def test_same_time_teacher_replaces_effective_signal_but_loss_uses_prediction(make_runtime, teacher):
    runtime = make_runtime()
    adapter, _ = configure_prediction(runtime, 0.25)

    signal = runtime.generate_goodness(now_ms=500, teacher=teacher, calibration_lr=0.1)

    assert float(signal.g) == 0.25
    assert signal.teacher == teacher
    assert signal.g_eff == teacher
    assert signal.calibration_loss == pytest.approx(0.5 * (0.25 - teacher) ** 2)
    assert float(adapter.linear.bias.detach()) == pytest.approx(0.25 - 0.1 * (0.25 - teacher))


def test_calibration_uses_current_local_derivative_and_changes_only_goodness_adapter(make_runtime):
    runtime = make_runtime()
    z = torch.tensor([1., -2., 0.5])
    adapter, _ = configure_prediction(runtime, 0.25, z)
    before = parameter_snapshot(runtime)

    signal = runtime.generate_goodness(
        now_ms=250, teacher=0.75, teacher_time_ms=250, calibration_lr=0.1,
    )

    assert signal.calibration_loss == pytest.approx(0.125)
    assert float(signal.g) == 0.25  # Return the prediction made before calibration.
    torch.testing.assert_close(adapter.linear.weight, 0.05 * z.unsqueeze(0))
    torch.testing.assert_close(adapter.linear.bias, torch.tensor([0.3]))
    assert set(runtime.calibration_eligibility) == {"linear.weight", "linear.bias"}
    torch.testing.assert_close(runtime.calibration_eligibility["linear.weight"], z.unsqueeze(0))
    torch.testing.assert_close(runtime.calibration_eligibility["linear.bias"], torch.ones(1))
    for name, value in runtime.named_parameters():
        if not name.startswith("adapters.goodness."):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert value.grad is None
    for name, eligibility in runtime.calibration_eligibility.items():
        assert eligibility.shape == dict(adapter.named_parameters())[name].shape
        assert not eligibility.requires_grad


def test_sparse_teacher_is_neither_retroactive_nor_reused(make_runtime):
    runtime = make_runtime()
    adapter, block = configure_prediction(runtime, 0.25, torch.tensor([1., 0., 0.]))
    initial = parameter_snapshot(runtime)
    earlier = runtime.generate_goodness(now_ms=0)
    assert_parameters_unchanged(runtime, initial)

    with torch.no_grad():
        block.z.copy_(torch.tensor([0., 2., 0.]))
    matched = runtime.generate_goodness(now_ms=250, teacher=0.75, calibration_lr=0.1)
    # Only the state at t=250 enters the local derivative; t=0 is not replayed.
    torch.testing.assert_close(adapter.linear.weight, torch.tensor([[0., 0.1, 0.]]))
    torch.testing.assert_close(adapter.linear.bias, torch.tensor([0.3]))
    calibrated = parameter_snapshot(runtime)
    with torch.no_grad():
        block.z.copy_(torch.tensor([0., 0., 3.]))
    later = runtime.generate_goodness(now_ms=500)

    assert float(earlier.g) == 0.25
    assert earlier.teacher is None and earlier.calibration_loss is None
    assert float(matched.g) == 0.25 and matched.g_eff == 0.75
    assert float(later.g) == pytest.approx(0.3)
    assert later.g_eff == pytest.approx(0.3)
    assert later.teacher is None and later.calibration_loss is None
    assert runtime.calibration_eligibility == {}
    assert_parameters_unchanged(runtime, calibrated)


@pytest.mark.parametrize("raw,expected", [(-1., 0.), (2., 1.)])
def test_saturated_goodness_has_zero_calibration_derivative(make_runtime, raw, expected):
    runtime = make_runtime()
    configure_prediction(runtime, raw, torch.tensor([1., 2., 3.]))
    before = parameter_snapshot(runtime)

    signal = runtime.generate_goodness(now_ms=0, teacher=0.5, calibration_lr=0.1)

    assert float(signal.g) == expected
    assert signal.g_eff == 0.5
    assert signal.calibration_loss == pytest.approx(0.125)
    assert runtime.calibration_eligibility
    for eligibility in runtime.calibration_eligibility.values():
        assert torch.count_nonzero(eligibility) == 0
    assert_parameters_unchanged(runtime, before)


@pytest.mark.parametrize("teacher_time_ms", [0, 500, 250.0])
def test_mismatched_or_noninteger_teacher_timestamp_rejected_before_mutation(make_runtime, teacher_time_ms):
    runtime = make_runtime()
    configure_prediction(runtime, 0.25)
    before = parameter_snapshot(runtime)

    with pytest.raises(ValueError, match="timestamp"):
        runtime.generate_goodness(now_ms=250, teacher=0.75, teacher_time_ms=teacher_time_ms)

    assert_parameters_unchanged(runtime, before)


@pytest.mark.parametrize("teacher", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_teacher_rejected_before_mutation(make_runtime, teacher):
    runtime = make_runtime()
    configure_prediction(runtime, 0.25)
    before = parameter_snapshot(runtime)

    with pytest.raises(ValueError, match="teacher"):
        runtime.generate_goodness(now_ms=250, teacher=teacher)

    assert_parameters_unchanged(runtime, before)


def test_teacher_timestamp_without_teacher_rejected(make_runtime):
    runtime = make_runtime()
    before = parameter_snapshot(runtime)
    with pytest.raises(ValueError, match="teacher"):
        runtime.generate_goodness(now_ms=250, teacher_time_ms=250)
    assert_parameters_unchanged(runtime, before)
