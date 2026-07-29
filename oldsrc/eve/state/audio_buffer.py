"""Audio buffer module — ring buffer for audio chunks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AudioBuffer:
    """Bounded ring buffer for audio chunks (numpy arrays).

    Attributes:
        capacity: Maximum number of chunks to store (default 256).
    """

    capacity: int = 256
    _buffer: deque = field(default_factory=deque, init=False, repr=False)

    def push(self, chunk: np.ndarray) -> None:
        """Append an audio chunk. Evicts oldest if at capacity.

        Args:
            chunk: numpy array audio chunk to store.
        """
        ...

    def get_latest(self) -> np.ndarray | None:
        """Get the most recent chunk.

        Returns:
            np.ndarray or None if buffer is empty.
        """
        ...

    def get_recent_seconds(self, sr: int, seconds: float) -> np.ndarray:
        """Get audio covering roughly the last `seconds`.

        Args:
            sr: Sample rate in Hz.
            seconds: Desired duration in seconds.

        Returns:
            Concatenated float32 numpy array.
        """
        ...

    def __len__(self) -> int:
        ...

    def __bool__(self) -> bool:
        ...
