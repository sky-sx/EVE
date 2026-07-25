import time

import eve.core.loop as loop_module
from eve.core.loop import run_once
from eve.core.safegate import check, emergency_stop, report_human_activity
from eve.state import (
    ActionCandidate,
    ActionKind,
    OutputMode,
    OutputResult,
    RuntimeState,
)


def action(kind=ActionKind.MOUSE, valid_until_ns=0):
    return ActionCandidate(
        action_id=f"a-{kind.value}",
        kind=kind,
        payload={"action": "click"},
        valid_until_ns=valid_until_ns,
    )


def ready(mode=OutputMode.MOCK):
    return RuntimeState(
        cold_started=True,
        output_mode=mode,
        mouse_allowed=True,
        keyboard_allowed=True,
        speak_allowed=True,
    )


def test_disabled_denied_mock_simulated_and_result_meanings_do_not_mix(tmp_path):
    disabled = run_once(ready(OutputMode.DISABLED), action(), tmp_path / "disabled")
    simulated = run_once(ready(OutputMode.MOCK), action(), tmp_path / "mock")

    assert disabled.blocked and not disabled.simulated and not disabled.executed
    assert simulated.simulated and not simulated.blocked and not simulated.executed


def test_authorized_real_result_has_executed_only(monkeypatch, tmp_path):
    def fake_real_output(candidate, mode):
        return OutputResult(
            action_id=candidate.action_id,
            kind=candidate.kind.value,
            mode=mode.value,
            executed=True,
            reason="fake_real_backend_for_semantics_test",
        )

    monkeypatch.setattr(loop_module, "_dispatch_output", fake_real_output)
    result = run_once(ready(OutputMode.REAL), action(), tmp_path)

    assert result.executed and not result.simulated and not result.blocked


def test_emergency_cold_start_permission_expiry_and_human_takeover():
    state = ready()
    emergency_stop(state)
    assert check(state, action()).reason == "emergency_stopped"

    state = ready()
    state.cold_started = False
    assert check(state, action()).reason == "not_cold_started"

    state = ready()
    state.mouse_allowed = False
    assert check(state, action()).reason == "mouse_not_allowed"

    state = ready()
    expired = action(valid_until_ns=time.monotonic_ns() - 1)
    assert check(state, expired).reason == "action_expired"

    state = ready()
    report_human_activity(state)
    assert "human_takeover_freeze" in check(state, action()).reason
    assert check(state, action(ActionKind.SPEAK)).allowed
