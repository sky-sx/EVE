"""Mouse output with disabled, mock, and explicit real modes."""
from __future__ import annotations

import time
import threading
from typing import Any

_STOP = threading.Event()


def stop_all() -> None:
    """Prevent new mouse actions after an emergency stop."""
    _STOP.set()
    try:
        import pyautogui

        for button in ("left", "right", "middle"):
            pyautogui.mouseUp(button=button)
    except Exception:
        pass


def reset_stop() -> None:
    """Re-arm mouse output only after the user explicitly resets EVE."""
    _STOP.clear()


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
        "candidate_id": action_id,
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
        if _STOP.is_set():
            return _result(
                action_id,
                "real",
                started_ns,
                blocked=True,
                reason="emergency_stopped",
                payload=payload,
            )
        if action == "moveTo":
            if not _move_interruptibly(
                pyautogui,
                float(payload.get("x", 0)),
                float(payload.get("y", 0)),
                float(payload.get("duration", 0.0)),
            ):
                return _result(
                    action_id, "real", started_ns, blocked=True,
                    reason="emergency_stopped", payload=payload,
                )
        elif action == "moveRel":
            current_x, current_y = pyautogui.position()
            if not _move_interruptibly(
                pyautogui,
                float(current_x) + float(payload.get("dx", 0)),
                float(current_y) + float(payload.get("dy", 0)),
                float(payload.get("duration", 0.0)),
            ):
                return _result(
                    action_id, "real", started_ns, blocked=True,
                    reason="emergency_stopped", payload=payload,
                )
        elif action in {"click", "doubleClick", "rightClick", "middleClick"}:
            function = getattr(pyautogui, action)
            kwargs = {
                key: payload[key]
                for key in ("x", "y", "button", "clicks")
                if key in payload
            }
            function(**kwargs)
        elif action == "drag":
            pyautogui.moveTo(payload.get("x1", 0), payload.get("y1", 0))
            button = str(payload.get("button", "left"))
            pyautogui.mouseDown(button=button)
            try:
                completed = _move_interruptibly(
                    pyautogui,
                    float(payload.get("x2", 0)),
                    float(payload.get("y2", 0)),
                    float(payload.get("duration", 0.5)),
                )
            finally:
                pyautogui.mouseUp(button=button)
            if not completed:
                return _result(
                    action_id, "real", started_ns, blocked=True,
                    reason="emergency_stopped", payload=payload,
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


def _move_interruptibly(
    pyautogui: Any,
    target_x: float,
    target_y: float,
    duration_s: float,
) -> bool:
    if duration_s <= 0:
        if _STOP.is_set():
            return False
        pyautogui.moveTo(target_x, target_y)
        return True
    start_x, start_y = pyautogui.position()
    steps = max(1, int(duration_s / 0.05) + 1)
    step_duration = duration_s / steps
    for index in range(1, steps + 1):
        if _STOP.is_set():
            return False
        progress = index / steps
        pyautogui.moveTo(
            start_x + (target_x - start_x) * progress,
            start_y + (target_y - start_y) * progress,
            duration=step_duration,
        )
    return True
