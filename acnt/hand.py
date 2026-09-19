"""Full mock keyboard controls and the trainer's permitted observation subset."""

from collections.abc import Mapping
from typing import Any


# A declared 82-key mock layout, not a claim about every physical 82-key board.
# Modifiers are independent coordinates so key combinations need no extra axis.
KEYBOARD_KEYS = (
    "ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
    "PRINT_SCREEN", "PAUSE", "DELETE",
    "BACKQUOTE", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "MINUS", "EQUAL", "BACKSPACE",
    "TAB", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "BRACKET_LEFT", "BRACKET_RIGHT", "BACKSLASH",
    "CAPS_LOCK", "A", "S", "D", "F", "G", "H", "J", "K", "L", "SEMICOLON", "QUOTE", "ENTER",
    "SHIFT_LEFT", "Z", "X", "C", "V", "B", "N", "M", "COMMA", "PERIOD", "SLASH", "SHIFT_RIGHT",
    "CTRL_LEFT", "META_LEFT", "ALT_LEFT", "SPACE", "ALT_RIGHT", "FN", "CTRL_RIGHT",
    "ARROW_LEFT", "ARROW_DOWN", "ARROW_UP", "ARROW_RIGHT", "HOME", "END",
)
MOUSE_BUTTONS = ("MOUSE_LEFT", "MOUSE_RIGHT", "MOUSE_MIDDLE")
HAND_DISCRETE_NAMES = KEYBOARD_KEYS + MOUSE_BUTTONS
TRAINING_CONTROL_NAMES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("MOUSE_LEFT",)


def read_training_controls(record: Mapping[str, Any]) -> dict[str, bool]:
    """Read letters and left click without changing the full mechanical log.

    This is a trainer-side projection. It does not mask generated controls,
    remove disallowed keys from stored records, or operate the physical OS.
    """
    if record.get("organ") != "hand":
        return {}
    controls = record["signal"]["discrete"]
    return {name: controls[name] for name in TRAINING_CONTROL_NAMES if name in controls}
