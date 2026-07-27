"""Mouse output with disabled, mock, and explicit real modes."""
from __future__ import annotations

import time
from typing import Any


def _result(
    action_id: str,
    mode: str,
    started_ns: int,
    *,
    executed: bool = False,
    simulated: bool = False,
    blocked: bool = False,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "kind": "mouse",
        "mode": mode,
        "started_at_ns": started_ns,
        "finished_at_ns": time.monotonic_ns(),
        "executed": executed,
        "simulated": simulated,
        "blocked": blocked,
        "reason": reason,
        "payload": dict(payload or {}),
    }


def execute(
    action_id: str, payload: dict[str, Any], mode: str
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    if mode == "disabled":
        return _result(
            action_id, mode, started_ns, blocked=True, reason="output_disabled"
        )
    if mode == "mock":
        return _result(
            action_id,
            mode,
            started_ns,
            simulated=True,
            reason="mock_ok",
            payload=payload,
        )
    if mode != "real":
        raise ValueError(f"unknown output mode: {mode}")
    return _execute_real(action_id, payload, started_ns)


def _execute_real(
    action_id: str, payload: dict[str, Any], started_ns: int
) -> dict[str, Any]:
    import pyautogui

    pyautogui.FAILSAFE = False
    action = payload.get("action", "moveTo")
    try:
        if action == "moveTo":
            pyautogui.moveTo(
                payload.get("x", 0),
                payload.get("y", 0),
                duration=payload.get("duration", 0.0),
            )
        elif action == "moveRel":
            pyautogui.moveRel(
                payload.get("dx", 0),
                payload.get("dy", 0),
                duration=payload.get("duration", 0.0),
            )
        elif action in {"click", "doubleClick", "rightClick"}:
            function = getattr(pyautogui, action)
            kwargs = {
                key: payload[key]
                for key in ("x", "y", "button", "clicks")
                if key in payload
            }
            function(**kwargs)
        elif action == "drag":
            pyautogui.moveTo(payload.get("x1", 0), payload.get("y1", 0))
            pyautogui.drag(
                payload.get("x2", 0) - payload.get("x1", 0),
                payload.get("y2", 0) - payload.get("y1", 0),
                duration=payload.get("duration", 0.5),
            )
        elif action == "scroll":
            pyautogui.scroll(payload.get("clicks", 1))
        else:
            return _result(
                action_id,
                "real",
                started_ns,
                blocked=True,
                reason=f"unknown_action_{action}",
            )
        return _result(
            action_id,
            "real",
            started_ns,
            executed=True,
            reason="real_ok",
            payload=payload,
        )
    except Exception as exc:
        return _result(
            action_id,
            "real",
            started_ns,
            blocked=True,
            reason=f"real_error_{exc}",
        )
