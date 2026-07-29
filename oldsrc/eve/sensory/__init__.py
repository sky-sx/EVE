"""
EVE Sensory Layer — synthetic-only perception capture.

Hot-path modules that produce raw sensory streams. All captures are
deterministic, synthetic-only, and safe by default (no real OS control).
"""

from .screen_capture import ScreenCapture
from .cursor_capture import CursorCapture
from .keyboard_capture import KeyboardCapture
from .audio_capture import AudioCapture

__all__ = [
    "ScreenCapture",
    "CursorCapture",
    "KeyboardCapture",
    "AudioCapture",
]
