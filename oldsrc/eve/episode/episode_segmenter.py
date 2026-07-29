"""
Episode Segmenter — detects boundaries in continuous sensory streams.

Scans state snapshots for transition events that mark episode
boundaries: task changes, user interrupts, idle timeouts, and error
conditions. Emits an EpisodeEvent when a boundary is detected.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EpisodeEventType(Enum):
    """Types of episode boundary events."""
    TASK_CHANGE = "task_change"
    USER_INTERRUPT = "user_interrupt"
    IDLE_TIMEOUT = "idle_timeout"
    ERROR = "error"


@dataclass
class EpisodeEvent:
    """Emitted when a segment boundary is detected.

    Attributes:
        event_type: Kind of boundary event.
        segment_id: Identifier of the just-finished segment.
        timestamp: Time of the boundary event.
        metadata: Optional extra fields (e.g., reason, severity).
    """
    event_type: EpisodeEventType
    segment_id: str
    timestamp: float
    metadata: dict = field(default_factory=dict)


class EpisodeSegmenter:
    """Stateful segmenter that detects episode boundaries from snapshots.

    Monitors a stream of state snapshots and emits an EpisodeEvent
    when a transition warrants splitting the stream into a new episode.

    Attributes:
        idle_timeout_s: Seconds of inactivity before idle_timeout event.
        task_change_threshold: Cosine-distance threshold for task change.
    """

    def __init__(
        self,
        idle_timeout_s: float = 30.0,
        task_change_threshold: float = 0.3,
    ) -> None:
        ...

    def update(self, state_snapshot: dict) -> Optional[EpisodeEvent]:
        """Process a state snapshot and return a boundary event if detected.

        The snapshot may contain:
            timestamp      — float
            state_vector   — list[float] or ndarray
            task_id        — str (optional)
            error_flag     — bool (optional)
            user_interrupt — bool (optional)

        Args:
            state_snapshot: Dictionary containing state snapshot data.

        Returns:
            EpisodeEvent if a boundary was crossed, else None.
        """
        ...

    def get_active_segment(self) -> dict:
        """Return metadata about the currently accumulating segment.

        Returns:
            dict with segment metadata (segment_id, start_s, task_id, etc.).
        """
        ...
