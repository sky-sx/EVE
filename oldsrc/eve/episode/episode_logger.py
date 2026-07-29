"""
Episode Logger — hot-path terminal recorder.

Records each step of the hot-path loop into a JSONL file. Uses an
in-memory buffer with periodic flush to stay non-blocking. The logger
is deterministic and synthetic-safe; it writes to the runs/ directory.
"""

import numpy as np
from pathlib import Path
from typing import Optional


_BUFFER_FLUSH_SIZE = 64
_RUNS_ROOT_DEFAULT = Path("runs")


class EpisodeLogger:
    """Non-blocking episode logger for the hot-path loop.

    Writes one JSONL line per logged step. Buffers data in memory
    and flushes periodically to avoid blocking the hot path.

    Attributes:
        runs_root: Base directory for run output (default: runs/).
        buffer_size: Number of steps to buffer before auto-flush.
    """

    def __init__(
        self,
        runs_root: Optional[Path] = None,
        buffer_size: int = _BUFFER_FLUSH_SIZE,
    ) -> None:
        ...

    def start_episode(self, episode_id: str) -> None:
        """Begin a new episode, creating the output JSONL file.

        Args:
            episode_id: Unique identifier for the episode. The output
                file is written to runs/{episode_id}/episode.jsonl.
        """
        ...

    def log_step(self, step_data: dict) -> None:
        """Record one hot-path step.

        Expected keys in step_data:
            timestamp     — float (perf_counter seconds)
            state_vector  — list[float] or ndarray
            action        — dict with action metadata
            reward        — float
            frame_idx     — int

        Args:
            step_data: Dictionary containing step-level data.
        """
        ...

    def end_episode(self) -> None:
        """Finalize the current episode, flush remaining data, close file."""
        ...

    def get_episode_path(self) -> Path:
        """Return the Path to the episode JSONL file.

        Returns:
            Path to the episode JSONL file.

        Raises:
            RuntimeError: If no episode has been started.
        """
        ...

    def flush(self) -> None:
        """Force-write buffered step records to disk."""
        ...

    @property
    def step_count(self) -> int:
        """Current step count within the active episode."""
        ...

    @property
    def active(self) -> bool:
        """Whether an episode is currently being logged."""
        ...
