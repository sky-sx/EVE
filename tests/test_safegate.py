import time

import eve.core.loop as loop_module
from eve.core.loop import check_action_permission, create_runtime_state, run_once


def action(action_type="mouse", valid_until_ns=0):
    return {
        "candidate_id": f"a-{action_type}",
        "source": "test",
        "action_type": action_type,
        "payload": {"action": "click"},
        "observed_at_ns": time.monotonic_ns(),
        "generated_at_ns": time.monotonic_ns(),
        "valid_until_ns": valid_until_ns,
    }


def ready(mode="mock"):
    state = create_runtime_state(
        output_mode=mode, allow_mock_actions=True
    )
    state["cold_started"] = True
    return state


def test_disabled_denied_and_mock_simulated(tmp_path):
    disabled = run_once(ready("disabled"), action(), tmp_path / "disabled")
    simulated = run_once(ready("mock"), action(), tmp_path / "mock")

    assert disabled["blocked"] and not disabled["simulated"]
    assert simulated["simulated"] and not simulated["blocked"]


def test_authorized_real_result_has_executed_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        loop_module,
        "_dispatch",
        lambda candidate, mode: {
            "action_id": candidate["candidate_id"],
            "kind": candidate["action_type"],
            "mode": mode,
            "started_at_ns": 0,
            "finished_at_ns": 1,
            "executed": True,
            "simulated": False,
            "blocked": False,
            "reason": "fake_real_backend",
            "payload": {},
        },
    )
    result = run_once(ready("real"), action(), tmp_path)
    assert result["executed"] and not result["simulated"] and not result["blocked"]


def test_emergency_cold_start_permission_expiry_and_human_takeover():
    state = ready()
    state["emergency_stop"] = True
    assert check_action_permission(state, action())["reason"] == "emergency_stopped"

    state = ready()
    state["cold_started"] = False
    assert check_action_permission(state, action())["reason"] == "not_cold_started"

    state = ready()
    state["permissions"]["mouse"] = False
    assert check_action_permission(state, action())["reason"] == "mouse_not_allowed"

    state = ready()
    assert check_action_permission(
        state, action(valid_until_ns=time.monotonic_ns() - 1)
    )["reason"] == "action_expired"

    state = ready()
    state["human_takeover_until_ns"] = time.monotonic_ns() + 5_000_000_000
    assert check_action_permission(state, action())["reason"] == "human_takeover"
    assert check_action_permission(state, action("speak"))["allowed"]
