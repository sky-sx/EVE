"""
EVE 鼠标输出 — 支持 disabled / mock / real 模式。

REAL 模式通过 pyautogui 执行真实系统调用，仅在模式为 REAL 时导入。
"""
from __future__ import annotations

import time
from typing import Any

from eve.state import OutputMode, OutputResult


def execute(
    action_id: str,
    payload: dict[str, Any],
    mode: OutputMode,
) -> OutputResult:
    """执行鼠标动作。"""
    started_ns = time.monotonic_ns()

    if mode == OutputMode.DISABLED:
        return OutputResult(
            action_id=action_id,
            kind="mouse",
            mode=mode.value,
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason="output_disabled",
        )

    if mode == OutputMode.MOCK:
        return OutputResult(
            action_id=action_id,
            kind="mouse",
            mode=mode.value,
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=True,
            blocked=False,
            reason="mock_ok",
            payload=dict(payload),
        )

    # REAL 模式：仅在此时导入 pyautogui
    return _execute_real(action_id, payload, started_ns)


def _execute_real(
    action_id: str,
    payload: dict[str, Any],
    started_ns: int,
) -> OutputResult:
    import pyautogui

    pyautogui.FAILSAFE = False

    action = payload.get("action", "move")
    result_payload: dict[str, Any] = {}

    try:
        if action == "moveTo":
            x, y = payload.get("x", 0), payload.get("y", 0)
            duration = payload.get("duration", 0.0)
            pyautogui.moveTo(x, y, duration=duration)
            result_payload = {"x": x, "y": y, "duration": duration}

        elif action == "moveRel":
            dx, dy = payload.get("dx", 0), payload.get("dy", 0)
            duration = payload.get("duration", 0.0)
            pyautogui.moveRel(dx, dy, duration=duration)
            result_payload = {"dx": dx, "dy": dy, "duration": duration}

        elif action == "click":
            x = payload.get("x")
            y = payload.get("y")
            button = payload.get("button", "left")
            clicks = payload.get("clicks", 1)
            if x is not None and y is not None:
                pyautogui.click(x, y, clicks=clicks, button=button)
                result_payload = {"x": x, "y": y, "button": button, "clicks": clicks}
            else:
                pyautogui.click(clicks=clicks, button=button)
                result_payload = {"button": button, "clicks": clicks}

        elif action == "doubleClick":
            x = payload.get("x")
            y = payload.get("y")
            button = payload.get("button", "left")
            if x is not None and y is not None:
                pyautogui.doubleClick(x, y, button=button)
                result_payload = {"x": x, "y": y, "button": button}
            else:
                pyautogui.doubleClick(button=button)
                result_payload = {"button": button}

        elif action == "drag":
            x1, y1 = payload.get("x1", 0), payload.get("y1", 0)
            x2, y2 = payload.get("x2", 0), payload.get("y2", 0)
            duration = payload.get("duration", 0.5)
            pyautogui.moveTo(x1, y1)
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
            result_payload = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration}

        elif action == "scroll":
            clicks = payload.get("clicks", 1)
            pyautogui.scroll(clicks)
            result_payload = {"clicks": clicks}

        elif action == "rightClick":
            x = payload.get("x")
            y = payload.get("y")
            if x is not None and y is not None:
                pyautogui.rightClick(x, y)
                result_payload = {"x": x, "y": y}
            else:
                pyautogui.rightClick()
                result_payload = {}

        else:
            return OutputResult(
                action_id=action_id,
                kind="mouse",
                mode="real",
                started_at_ns=started_ns,
                finished_at_ns=time.monotonic_ns(),
                executed=False,
                simulated=False,
                blocked=True,
                reason=f"unknown_action_{action}",
            )

        return OutputResult(
            action_id=action_id,
            kind="mouse",
            mode="real",
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=True,
            simulated=False,
            blocked=False,
            reason="real_ok",
            payload=result_payload,
        )

    except Exception as exc:
        return OutputResult(
            action_id=action_id,
            kind="mouse",
            mode="real",
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason=f"real_error_{exc}",
            payload=result_payload,
        )
