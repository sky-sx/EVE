"""
EVE 键盘输出 — 支持 disabled / mock / real 模式。

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
    """执行键盘动作。"""
    started_ns = time.monotonic_ns()

    if mode == OutputMode.DISABLED:
        return OutputResult(
            action_id=action_id,
            kind="keyboard",
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
            kind="keyboard",
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

    action = payload.get("action", "press")
    result_payload: dict[str, Any] = {}

    try:
        if action == "press":
            keys = payload.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                pyautogui.keyDown(key)
            for key in reversed(keys):
                pyautogui.keyUp(key)
            result_payload = {"keys": keys}

        elif action == "write":
            text = payload.get("text", "")
            interval = payload.get("interval", 0.0)
            pyautogui.write(text, interval=interval)
            result_payload = {"text": text, "interval": interval}

        elif action == "hotkey":
            keys = payload.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            pyautogui.hotkey(*keys)
            result_payload = {"keys": list(keys)}

        elif action == "keyDown":
            key = payload.get("key", "")
            pyautogui.keyDown(key)
            result_payload = {"key": key}

        elif action == "keyUp":
            key = payload.get("key", "")
            pyautogui.keyUp(key)
            result_payload = {"key": key}

        else:
            return OutputResult(
                action_id=action_id,
                kind="keyboard",
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
            kind="keyboard",
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
            kind="keyboard",
            mode="real",
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason=f"real_error_{exc}",
            payload=result_payload,
        )
