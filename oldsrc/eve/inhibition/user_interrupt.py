"""
User Interrupt Detector — signal handling for pause / resume / stop / override,
plus real-time interrupt detection from input and audio streams.

Monitors input events and audio stream for user interruption signals.
Allows a human operator (or simulated operator) to inject control
signals into the hot path without blocking the main loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class InterruptType(str, Enum):
    """Types of user interrupt signals."""

    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    OVERRIDE = "override"


@dataclass
class InterruptSignal:
    """Signal produced by the interrupt detector.

    Attributes:
        score: Detection confidence in [0, 1].
        type: Interrupt type string (e.g. voice, keypress, gesture).
        timestamp: Time the interrupt was detected.
    """
    score: float
    type: str
    timestamp: float


class UserInterruptDetector:
    """Collects and exposes user interrupt signals.

    Monitors input events and audio stream for user interruption
    signals. Signals are set by *signal(type)* and read by the hot
    path. *pause* is temporary, *stop* is permanent (requires
    clear/reset). *override* forces the safety gate to allow the next
    batch.
    """

    def __init__(self) -> None:
        """Initialize with no active interrupts."""
        ...

    def detect(
        self,
        input_data: dict,
        audio_data: np.ndarray | None = None,
    ) -> InterruptSignal | None:
        """Detect interruption signals from input and/or audio streams.

        Args:
            input_data: Input event data dict (e.g. keyboard, mouse).
            audio_data: Optional raw audio buffer to scan for voice
                interruption.

        Returns:
            InterruptSignal if an interruption was detected, else None.
        """
        ...

    def signal(self, type: str) -> None:
        """Inject an interrupt signal.

        Valid types:
            *pause*   — temporarily pause the loop
            *resume*  — lift a pause
            *stop*    — permanent stop (requires clear())
            *override*— force-allow the next safety-gate batch

        Args:
            type: The interrupt type string (pause/resume/stop/override).
        """
        ...

    def is_paused(self) -> bool:
        """Return True while paused (temporary or permanent).

        Returns:
            True if a pause or stop signal is active.
        """
        ...

    def is_stopped(self) -> bool:
        """Return True when a permanent stop signal was received.

        Returns:
            True if a stop signal is active.
        """
        ...

    def consume_override(self) -> bool:
        """Consume and return the override flag (resets after read).

        Returns:
            True if override was set, and resets the flag.
        """
        ...

    def get_last_interrupt(self) -> dict | None:
        """Return the most recent interrupt entry or None.

        Returns:
            A dict with the last interrupt info, or None.
        """
        ...

    def clear(self) -> None:
        """Clear all interrupt state back to defaults."""
        ...
