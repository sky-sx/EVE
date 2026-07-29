"""Captures keyboard events from an external input source."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


_SYNTHETIC_KEYS = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    "space", "enter", "tab", "escape", "backspace",
    "left", "right", "up", "down",
    "shift", "ctrl", "alt",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
]


@dataclass
class KeyboardPacket:
    """Named packet representing keyboard state at a moment in time.

    Attributes:
        keys: List of currently pressed key name strings.
        events: List of recent key event dicts (key, action, timestamp).
        timestamp: Sample timestamp in seconds.
    """

    keys: list[str]
    events: list[dict]
    timestamp: float


@dataclass
class KeyboardCapture:
    """External keyboard event receiver.

    Attributes:
        event_probability: Probability of generating an event per tick (default 0.1).
        max_pressed: Maximum number of concurrently pressed keys (default 4).
        seed: Random seed for deterministic event generation.
    """

    event_probability: float = 0.1
    max_pressed: int = 4
    seed: int = 42

    _running: bool = field(default=False, init=False, repr=False)
    _pressed: set[str] = field(default_factory=set, init=False, repr=False)
    _events: list[dict] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42), init=False, repr=False)
    _tick_count: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Start keyboard event processing."""
        ...

    def stop(self) -> None:
        """Stop keyboard event processing."""
        ...

    def get_keys(self) -> KeyboardPacket:
        """Get current keyboard state as a named packet.

        Returns:
            KeyboardPacket with keys, events, and timestamp.
        """
        ...

    def get_events(self) -> list[dict]:
        """Get recent key events.

        Returns:
            List of {key, action, timestamp} dicts.
        """
        ...

    def clear_events(self) -> None:
        """Clear the event history."""
        ...

    @property
    def pressed(self) -> list[str]:
        """Alias for get_keys()."""
        ...

    @property
    def running(self) -> bool:
        """Whether keyboard capture is active."""
        ...
