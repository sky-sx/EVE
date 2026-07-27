"""The minimum non-bypassable safety boundary before Output."""
from __future__ import annotations

import time
from typing import Any


def check(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    now_ns = time.monotonic_ns()
    reason = _blocked_reason(state, action, now_ns)
    return {
        "allowed": reason == "ok",
        "reason": reason,
        "checked_at_ns": now_ns,
    }


def _blocked_reason(
    state: dict[str, Any], action: dict[str, Any], now_ns: int
) -> str:
    if state.get("emergency_stop"):
        return "emergency_stopped"
    if not state.get("cold_started"):
        return "not_cold_started"
    if state.get("output_mode") == "disabled":
        return "output_disabled"

    action_type = action.get("action_type")
    if action_type not in {"mouse", "keyboard", "speak"}:
        return "invalid_action_type"
    if not state.get("permissions", {}).get(action_type, False):
        return f"{action_type}_not_allowed"

    valid_until_ns = int(action.get("valid_until_ns", 0))
    if valid_until_ns and now_ns > valid_until_ns:
        return "action_expired"
    if (
        action_type in {"mouse", "keyboard"}
        and now_ns < int(state.get("human_takeover_until_ns", 0))
    ):
        return "human_takeover"

    payload = action.get("payload")
    if not isinstance(payload, dict):
        return "invalid_payload"
    if action_type == "mouse":
        for key in ("x", "y", "x1", "y1", "x2", "y2"):
            if key in payload:
                value = payload[key]
                if not isinstance(value, (int, float)) or not -100_000 <= value <= 100_000:
                    return "mouse_range_invalid"
    elif action_type == "keyboard":
        keys = payload.get("keys", payload.get("key", []))
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, (list, tuple)) or len(keys) > 16:
            return "keyboard_range_invalid"
    elif len(str(payload.get("text", ""))) > 2_000:
        return "speak_range_invalid"
    return "ok"


def emergency_stop(state: dict[str, Any]) -> None:
    state["emergency_stop"] = True


def reset_emergency(state: dict[str, Any]) -> None:
    state["emergency_stop"] = False
