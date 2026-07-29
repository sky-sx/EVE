"""
World state estimate — maintains a persistent model of the external environment.

Hot-path compatible: state updates are deterministic O(1) snapshot operations.
Maintains a bounded history of recent world snapshots for change detection.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldState:
    """Maintains current world state estimate from perception snapshots.

    Attributes:
        max_history: Maximum number of recent snapshots to retain for change detection.
    """

    max_history: int = 60

    _current: dict[str, Any] = field(default_factory=dict)
    _history: deque[dict[str, Any]] = field(default_factory=deque)
    _last_update: float = 0.0

    def update(self, perception: dict[str, Any]) -> None:
        """Update world state from a perception snapshot.

        Args:
            perception: Dict of perception fields to merge.
        """
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return the current world state as a dict.

        Returns:
            {time_of_day, activity_level, visual_complexity, audio_level,
             cursor_active, scene_type, dominant_colors, motion_level}
        """
        ...

    def get_change(self, from_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Compute delta between a previous snapshot and current state.

        Args:
            from_snapshot: A previously captured snapshot dict.

        Returns:
            Dict of {field_name: (old_value, new_value)} for changed fields.
        """
        ...

    def reset(self) -> None:
        """Clear all state and history."""
        ...

    @property
    def history(self) -> list[dict[str, Any]]:
        """Recent snapshot history (read-only copy)."""
        ...

    @property
    def last_update(self) -> float:
        """Timestamp of last update (monotonic seconds)."""
        ...

    @property
    def age(self) -> float:
        """Seconds since last update, or inf if never updated."""
        ...
