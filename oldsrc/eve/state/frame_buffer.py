"""Frame buffer module — ring buffer for video frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameBuffer:
    """Bounded ring buffer for frames (numpy arrays).

    Attributes:
        capacity: Maximum number of frames to store (default 64).
    """

    capacity: int = 64
    _buffer: deque = field(default_factory=deque, init=False, repr=False)

    def push(self, frame: np.ndarray) -> None:
        """Append a frame. Evicts oldest if at capacity.

        Args:
            frame: numpy array frame to store.
        """
        ...

    def get_latest(self) -> np.ndarray | None:
        """Get the most recent frame.

        Returns:
            np.ndarray or None if buffer is empty.
        """
        ...

    def get_window(self, n: int) -> list[np.ndarray]:
        """Get the last `n` frames.

        Args:
            n: Number of recent frames to retrieve.

        Returns:
            List of copied numpy arrays in chronological order.
        """
        ...

    def __len__(self) -> int:
        ...

    def __bool__(self) -> bool:
        ...
