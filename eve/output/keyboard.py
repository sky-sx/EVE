"""Keyboard output with disabled, mock, and explicit real modes."""
from __future__ import annotations

import time
import threading
from typing import Any

_STOP = threading.Event()
_PRESSED: set[str] = set()
_PRESSED_LOCK = threading.Lock()
_LAST_RELEASE_ERROR: str | None = None


def stop_all() -> None:
    """Release keys owned by EVE and prevent new keyboard actions."""
    global _LAST_RELEASE_ERROR
    _STOP.set()
    try:
        import pyautogui

        with _PRESSED_LOCK:
            keys = list(_PRESSED)
            _PRESSED.clear()
        for key in keys:
            try:
                pyautogui.keyUp(key)
            except Exception as exc:
                _LAST_RELEASE_ERROR = f"{type(exc).__name__}: {exc}"
    except ImportError:
        with _PRESSED_LOCK:
            _PRESSED.clear()


def reset_stop() -> None:
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
        "action_id": action_id,
        "kind": "keyboard",
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
    action = payload.get("action", "press")
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
        if action == "press":
            keys = payload.get("keys", [])
            keys = [keys] if isinstance(keys, str) else list(keys)
            for key in keys:
                pyautogui.keyDown(key)
                with _PRESSED_LOCK:
                    _PRESSED.add(str(key))
            for key in reversed(keys):
                pyautogui.keyUp(key)
                with _PRESSED_LOCK:
                    _PRESSED.discard(str(key))
        elif action == "write":
            text = str(payload.get("text", ""))
            method = payload.get("method", "write")
            if method in {"paste", "unicode"} or not text.isascii():
                import pyperclip

                previous = pyperclip.paste()
                try:
                    pyperclip.copy(text)
                    pyautogui.hotkey("ctrl", "v")
                finally:
                    pyperclip.copy(previous)
            else:
                pyautogui.write(text, interval=payload.get("interval", 0.0))
        elif action == "hotkey":
            keys = payload.get("keys", [])
            pyautogui.hotkey(*(keys if isinstance(keys, list) else [keys]))
        elif action == "keyDown":
            key = str(payload.get("key", ""))
            pyautogui.keyDown(key)
            with _PRESSED_LOCK:
                _PRESSED.add(key)
        elif action == "keyUp":
            key = str(payload.get("key", ""))
            pyautogui.keyUp(key)
            with _PRESSED_LOCK:
                _PRESSED.discard(key)
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
