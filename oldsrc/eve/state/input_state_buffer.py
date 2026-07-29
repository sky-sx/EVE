"""Input state buffer module — tracks cursor position and key press history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class InputStateBuffer:
    """Buffer for cursor position and key press history.

    Attributes:
        history_depth: Number of past cursor positions to retain (default 32).
    """

    history_depth: int = 32

    _cursor_x: float = field(default=0.5, init=False, repr=False)
    _cursor_y: float = field(default=0.5, init=False, repr=False)
    _keys: list[str] = field(default_factory=list, init=False, repr=False)
    _cursor_history: deque = field(default_factory=deque, init=False, repr=False)
    _last_timestamp: float = field(default=0.0, init=False, repr=False)

    def update_cursor(self, x: float, y: float) -> None:
        """Record the latest cursor position.

        Args:
            x: Normalized x coordinate.
            y: Normalized y coordinate.
        """
        ...

    def update_keys(self, keys: list[str]) -> None:
        """Record the currently pressed keys.

        Args:
            keys: List of key name strings.
        """
        ...

    def snapshot(self) -> dict:
        """Get a snapshot of current input state.

        Returns:
            Dict with keys: cursor, keys, timestamp, recent_positions.
        """
        ...

    @property
    def cursor_position(self) -> tuple[float, float]:
        """Current cursor position (x, y)."""
        ...

    @property
    def pressed_keys(self) -> list[str]:
        """Currently pressed keys."""
        ...

    @property
    def last_timestamp(self) -> float:
        """Timestamp of last update."""
        ...
