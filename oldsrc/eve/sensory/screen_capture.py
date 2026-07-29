"""Captures frames from an external screen source."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RetinalFramePacket:
    """Named packet representing a captured screen frame.

    Attributes:
        data: Pixel data as (H, W, 3) uint8 ndarray.
        width: Frame width in pixels.
        height: Frame height in pixels.
        timestamp: Capture timestamp in seconds.
    """

    data: np.ndarray
    width: int
    height: int
    timestamp: float


@dataclass
class ScreenCapture:
    """External screen capture producing frames from an outside video source.

    Attributes:
        width: Frame width in pixels (default 640).
        height: Frame height in pixels (default 480).
        fps: Target frames per second for timing simulation.
    """

    width: int = 640
    height: int = 480
    fps: float = 30.0

    _running: bool = field(default=False, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _latest_frame: np.ndarray | None = field(default=None, init=False, repr=False)
    _seed: int = field(default=42, init=False, repr=False)

    def start(self) -> None:
        """Start the capture session."""
        ...

    def stop(self) -> None:
        """Stop the capture session."""
        ...

    def get_frame(self) -> RetinalFramePacket:
        """Get the next captured frame.

        Returns:
            RetinalFramePacket with pixel data, dimensions, and timestamp.
        """
        ...

    @property
    def frame_count(self) -> int:
        """Total frames generated since start()."""
        ...

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since start()."""
        ...

    @property
    def running(self) -> bool:
        """Whether capture is active."""
        ...

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recently generated frame, or None."""
        ...
