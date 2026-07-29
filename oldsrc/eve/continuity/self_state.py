"""
Self state estimate — maintains a persistent model of the organism's own condition.

Hot-path compatible: mode transitions and snapshot queries are O(1).
Tracks internal dynamics like energy, confidence, learning progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

ModeName = Literal["active", "idle", "learning", "paused", "shadow"]

_VALID_MODES: tuple[ModeName, ...] = ("active", "idle", "learning", "paused", "shadow")


@dataclass
class SelfState:
    """Maintains self state estimate from motor feedback and internal signals.

    Attributes:
        initial_mode: Starting mode (default 'idle').
    """

    initial_mode: ModeName = "idle"

    _mode: ModeName = field(default="idle", init=False)
    _current: dict[str, Any] = field(default_factory=dict)
    _last_update: float = 0.0
    _mode_start: float = 0.0

    def __post_init__(self) -> None:
        ...

    def update(self, feedback: dict[str, Any]) -> None:
        """Update self state from motor feedback and internal signals.

        Args:
            feedback: Dict with optional keys matching snapshot fields.
        """
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return current self state.

        Returns:
            {energy_level, busy_level, confidence, learning_progress,
             action_success_rate, idle_time, mode}
        """
        ...

    def get_mode(self) -> str:
        """Return the current operating mode.

        Returns:
            One of: active, idle, learning, paused, shadow.
        """
        ...

    def transition(self, new_mode: str) -> None:
        """Transition to a new operating mode.

        Args:
            new_mode: Target mode (active, idle, learning, paused, shadow).

        Raises:
            ValueError: If mode is not valid.
        """
        ...

    def reset(self) -> None:
        """Reset to initial state."""
        ...

    @property
    def mode_duration(self) -> float:
        """Seconds spent in the current mode."""
        ...

    @property
    def last_update(self) -> float:
        """Timestamp of last update."""
        ...
