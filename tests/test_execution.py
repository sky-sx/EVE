import json

import pytest
import torch

from acnt.mechanical import MechanicalLog


@pytest.mark.parametrize("organ", ["hand", "speak", "goodness", "route"])
def test_disabled_readout_still_computes_and_logs_without_executor_feedback(make_runtime, organ):
    runtime = make_runtime()
    runtime.set_execution_enabled(organ, False)
    adapter_calls, executor_calls = [], []
    handle = runtime.adapters[organ].register_forward_hook(lambda *_: adapter_calls.append(1))
    runtime.executors[organ] = lambda signal: executor_calls.append(signal)
    before = {name: value.clone() for name, value in runtime.core.named_buffers()}
    try:
        result = getattr(runtime, f"generate_{organ}")(now_ms=0)
    finally:
        handle.remove()
    assert adapter_calls == [1] and executor_calls == []
    record = runtime.mechanical_log.records[-1]
    assert record["organ"] == organ and record["execution_enabled"] is False
    assert "signal" in record and "status" not in record and "executor_error" not in record
    for name, value in runtime.core.named_buffers():
        torch.testing.assert_close(value, before[name])
    if organ == "goodness":
        assert result.g_eff is None
        assert not runtime.core.blocks[runtime.organ_blocks["goodness"]].active


def test_route_switch_controls_application_but_not_final_signal(make_runtime):
    runtime = make_runtime()
    with torch.no_grad():
        runtime.adapters["route"].linear.weight.zero_()
        runtime.adapters["route"].linear.bias.fill_(-100.)
    runtime.set_execution_enabled("route", False)
    old_active = runtime.core.active_ids
    off = runtime.generate_route(generator=torch.Generator().manual_seed(7))
    assert runtime.core.active_ids == old_active
    final_off = runtime.mechanical_log.records[-1]["signal"]
    runtime.set_execution_enabled("route", True)
    on = runtime.generate_route(generator=torch.Generator().manual_seed(7))
    assert runtime.core.active_ids == (4, 5)
    assert runtime.mechanical_log.records[-1]["signal"] == final_off
    assert torch.equal(off.a, on.a)
    torch.testing.assert_close(off.noise, on.noise)


def test_executor_errors_and_return_codes_stay_only_in_mechanical_boundary(make_runtime):
    runtime = make_runtime()
    runtime.set_execution_enabled("hand", True)
    before = {name: value.clone() for name, value in runtime.core.named_buffers()}
    calls = []
    def fail(signal):
        calls.append(signal)
        raise RuntimeError("mock motor failure")
    runtime.executors["hand"] = fail
    failure_signal = runtime.generate_hand(generator=torch.Generator().manual_seed(5))
    assert len(calls) == 1
    assert runtime.mechanical_log.records[-1]["executor_error"] == "mock motor failure"
    runtime.executors["hand"] = lambda _signal: {"status": "success", "code": 200}
    success_signal = runtime.generate_hand(generator=torch.Generator().manual_seed(5))
    assert "status" not in runtime.mechanical_log.records[-1]
    assert "executor_error" not in runtime.mechanical_log.records[-1]
    assert torch.equal(failure_signal.discrete.a, success_signal.discrete.a)
    for name, value in runtime.core.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_disabled_goodness_does_not_calibrate_but_external_teacher_is_available(make_runtime):
    runtime = make_runtime()
    runtime.set_execution_enabled("goodness", False)
    before = {name: value.clone() for name, value in runtime.adapters["goodness"].named_parameters()}
    signal = runtime.generate_goodness(now_ms=0, teacher=0.8)
    assert signal.g_eff == 0.8 and signal.calibration_loss is None
    assert runtime.calibration_eligibility == {}
    for name, parameter in runtime.adapters["goodness"].named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_complete_log_can_be_persisted_as_jsonl(make_runtime, tmp_path):
    runtime = make_runtime()
    path = tmp_path / "mechanical.jsonl"
    runtime.mechanical_log = MechanicalLog(path)
    runtime.generate_hand(now_ms=10)
    runtime.generate_speak(now_ms=10)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == runtime.mechanical_log.records
    assert len(rows[0]["signal"]["discrete"]) == 85
