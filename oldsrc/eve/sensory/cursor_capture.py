"""Captures mouse events from an external input source."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CursorPacket:
    """Named packet representing a cursor position sample.

    Attributes:
        x: Normalized x coordinate in [0, 1].
        y: Normalized y coordinate in [0, 1].
        timestamp: Sample timestamp in seconds.
    """

    x: float
    y: float
    timestamp: float


@dataclass
class CursorCapture:
    """External cursor position tracker that receives mouse events.

    Attributes:
        speed: Maximum step size per update in normalized units (default 0.01).
        seed: Random seed for deterministic trajectory generation.
    """

    speed: float = 0.01
    seed: int = 42

    _running: bool = field(default=False, init=False, repr=False)
    _x: float = field(default=0.5, init=False, repr=False)
    _y: float = field(default=0.5, init=False, repr=False)
    _vx: float = field(default=0.0, init=False, repr=False)
    _vy: float = field(default=0.0, init=False, repr=False)
    _trajectory: list[tuple[float, float, float]] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42), init=False, repr=False)

    def start(self) -> None:
        """Start generating synthetic cursor movement."""
        ...

    def stop(self) -> None:
        """Stop cursor movement generation."""
        ...

    def get_position(self) -> CursorPacket:
        """Get current cursor position as a named packet.

        Returns:
            CursorPacket with x, y, and timestamp.
        """
        ...

    def get_trajectory(self) -> list[tuple[float, float, float]]:
        """Get recorded trajectory.

        Returns:
            List of (timestamp, x, y) tuples.
        """
        ...

    @property
    def trajectory_length(self) -> int:
        """Number of recorded trajectory points."""
        ...

    @property
    def running(self) -> bool:
        """Whether cursor capture is active."""
        ...
