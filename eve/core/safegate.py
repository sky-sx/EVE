"""The non-bypassable atomic permission boundary before Output."""
from __future__ import annotations

import string
import time
from typing import Any

MOUSE_ATOMS = (
    "move",
    "left_click",
    "left_double_click",
    "right_click",
    "middle_click",
    "scroll_up",
    "scroll_down",
    "left_drag",
    "right_drag",
    "middle_drag",
)

_NAMED_KEYS = {
    "ENTER", "SPACE", "TAB", "BACKSPACE", "DELETE", "HOME", "END",
    "PAGEUP", "PAGEDOWN", "UP", "DOWN", "LEFT", "RIGHT",
    "SHIFT", "CTRL", "ALT", "WIN", "ESC",
    *(f"F{index}" for index in range(1, 13)),
    *(f"NUM{index}" for index in range(10)),
    "NUMADD", "NUMSUBTRACT", "NUMMULTIPLY", "NUMDIVIDE", "NUMDECIMAL",
    "MINUS", "EQUAL", "LBRACKET", "RBRACKET", "BACKSLASH",
    "SEMICOLON", "APOSTROPHE", "COMMA", "PERIOD", "SLASH", "GRAVE",
}
SUPPORTED_KEYS = tuple(
    sorted(set(string.ascii_uppercase + string.digits) | _NAMED_KEYS)
)

_KEY_ALIASES = {
    "CONTROL": "CTRL",
    "COMMAND": "WIN",
    "WINDOWS": "WIN",
    "RETURN": "ENTER",
    "PGUP": "PAGEUP",
    "PGDN": "PAGEDOWN",
    " ": "SPACE",
    "-": "MINUS",
    "=": "EQUAL",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "\\": "BACKSLASH",
    ";": "SEMICOLON",
    "'": "APOSTROPHE",
    ",": "COMMA",
    ".": "PERIOD",
    "/": "SLASH",
    "`": "GRAVE",
}
_SHIFTED_KEYS = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
    "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "MINUS", "+": "EQUAL", "{": "LBRACKET", "}": "RBRACKET",
    "|": "BACKSLASH", ":": "SEMICOLON", '"': "APOSTROPHE",
    "<": "COMMA", ">": "PERIOD", "?": "SLASH", "~": "GRAVE",
}


def default_permissions(enabled: bool = False) -> dict[str, Any]:
    return {
        "mouse": {name: bool(enabled) for name in MOUSE_ATOMS},
        "keyboard": {name: bool(enabled) for name in SUPPORTED_KEYS},
        "send_text": bool(enabled),
        "speak": bool(enabled),
    }


def normalize_key(value: Any) -> str:
    key = str(value).strip()
    if len(key) == 1 and key.isalpha():
        return key.upper()
    if len(key) == 1 and key.isdigit():
        return key
    upper = key.upper()
    return _KEY_ALIASES.get(upper, _KEY_ALIASES.get(key, upper))


def required_atoms(action: dict[str, Any]) -> list[str]:
    action_type = action.get("action_type")
    payload = action.get("payload", {})
    if action_type == "speak":
        return ["speak"]
    if action_type == "mouse":
        operation = str(payload.get("action", "moveTo"))
        if operation in {"move", "moveTo", "moveRel"}:
            return ["mouse.move"]
        if operation == "doubleClick":
            atoms = ["mouse.left_click", "mouse.left_double_click"]
            if payload.get("x") is not None or payload.get("y") is not None:
                atoms.insert(0, "mouse.move")
            return atoms
        if operation in {"rightClick", "middleClick"}:
            button = "right" if operation == "rightClick" else "middle"
            atoms = [f"mouse.{button}_click"]
            if payload.get("x") is not None or payload.get("y") is not None:
                atoms.insert(0, "mouse.move")
            return atoms
        if operation == "click":
            button = str(payload.get("button", "left")).lower()
            clicks = int(payload.get("clicks", 1))
            atoms = [f"mouse.{button}_click"]
            if button == "left" and clicks >= 2:
                atoms.append("mouse.left_double_click")
            if payload.get("x") is not None or payload.get("y") is not None:
                atoms.insert(0, "mouse.move")
            return atoms
        if operation == "drag":
            button = str(payload.get("button", "left")).lower()
            return [
                "mouse.move",
                f"mouse.{button}_click",
                f"mouse.{button}_drag",
            ]
        if operation == "scroll":
            direction = "up" if float(payload.get("clicks", 1)) > 0 else "down"
            return [f"mouse.scroll_{direction}"]
        return [f"mouse.unknown:{operation}"]
    if action_type == "keyboard":
        operation = str(payload.get("action", "press"))
        if operation in {"press", "hotkey"}:
            keys = payload.get("keys", [])
        elif operation in {"keyDown", "keyUp"}:
            keys = [payload.get("key", "")]
        elif operation == "write":
            keys = list(str(payload.get("text", "")))
        else:
            return [f"keyboard.unknown:{operation}"]
        if isinstance(keys, str):
            keys = [keys]
        method = payload.get("method", "write")
        if operation == "write" and (
            method in {"paste", "unicode"}
            or not str(payload.get("text", "")).isascii()
        ):
            return ["send_text", "keyboard.CTRL", "keyboard.V"]
        atoms: list[str] = []
        for key in keys:
            text = str(key)
            if text in _SHIFTED_KEYS:
                atoms.extend(
                    ["keyboard.SHIFT", f"keyboard.{_SHIFTED_KEYS[text]}"]
                )
            else:
                atoms.append(f"keyboard.{normalize_key(key)}")
        return list(dict.fromkeys(atoms))
    return [f"unknown:{action_type}"]


def check(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    now_ns = time.monotonic_ns()
    atoms = required_atoms(action)
    checked = {
        atom: _permission_enabled(state.get("permissions", {}), atom)
        for atom in atoms
    }
    blocked_atoms = [atom for atom, enabled in checked.items() if not enabled]
    reason = _blocked_reason(state, action, now_ns, atoms, blocked_atoms)
    return {
        "allowed": reason == "ok",
        "blocked_atoms": blocked_atoms,
        "reason": reason,
        "checked_permissions": checked,
        "checked_at_ns": now_ns,
    }


def _permission_enabled(permissions: dict[str, Any], atom: str) -> bool:
    group, separator, name = atom.partition(".")
    if not separator:
        return bool(permissions.get(group, False))
    values = permissions.get(group, {})
    return bool(values.get(name, False)) if isinstance(values, dict) else False


def _blocked_reason(
    state: dict[str, Any],
    action: dict[str, Any],
    now_ns: int,
    atoms: list[str],
    blocked_atoms: list[str],
) -> str:
    if state.get("emergency_stop"):
        return "emergency_stopped"
    if not state.get("cold_started"):
        return "not_cold_started"
    if state.get("paused"):
        return "runtime_paused"
    if state.get("output_mode") == "disabled":
        return "output_disabled"
    if any("unknown:" in atom for atom in atoms):
        return "invalid_action"
    if blocked_atoms:
        action_type = action.get("action_type")
        if action_type in {"mouse", "keyboard", "speak"}:
            return f"{action_type}_not_allowed"
        return "permission_denied"

    action_type = action.get("action_type")
    if action_type not in {"mouse", "keyboard", "speak"}:
        return "invalid_action_type"
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
                if (
                    not isinstance(value, (int, float))
                    or not -100_000 <= value <= 100_000
                ):
                    return "mouse_range_invalid"
    elif action_type == "keyboard" and len(atoms) > 32:
        return "keyboard_range_invalid"
    elif action_type == "speak" and len(str(payload.get("text", ""))) > 2_000:
        return "speak_range_invalid"
    return "ok"


def emergency_stop(state: dict[str, Any]) -> None:
    state["emergency_stop"] = True
    state["action_queue"].clear()
    state["lifecycle"].update(
        {
            "state": "emergency_stopped",
            "changed_at_ns": time.monotonic_ns(),
            "reason": "emergency_stop",
        }
    )


def reset_emergency(state: dict[str, Any]) -> None:
    state["emergency_stop"] = False
    state["paused"] = True
    state["lifecycle"].update(
        {
            "state": "paused",
            "changed_at_ns": time.monotonic_ns(),
            "reason": "user_reset_emergency",
        }
    )
