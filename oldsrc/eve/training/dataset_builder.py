"""
Dataset Builder — builds training datasets from episode logs.

Collects episodes, extracts (state, action, reward) triples,
and exports them as numpy arrays for offline training.
Cold-path only — never blocks the hot path.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetBuilder:
    """Builds training datasets by scanning episode JSONL logs.

    Accumulates state vectors, actions, rewards, and sample weights
    from episode logs. Supports export to numpy format.

    Attributes:
        runs_root: Base directory for run output (default: runs/).
    """

    runs_root: Path = Path("runs")

    _states: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _actions: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _rewards: list[float] = field(default_factory=list, init=False, repr=False)
    _weights: list[float] = field(default_factory=list, init=False, repr=False)
    _episode_ids: list[str] = field(default_factory=list, init=False, repr=False)

    def add_episode(self, episode_id: str) -> None:
        """Load an episode JSONL and add its samples to the dataset.

        Args:
            episode_id: Episode identifier. The file is expected at
                runs/{episode_id}/episode.jsonl.

        Raises:
            FileNotFoundError: If the episode file does not exist.
        """
        ...

    def build(self) -> dict:
        """Compile all accumulated samples into a dataset dict.

        Returns:
            dict with keys:
                states  — np.ndarray of shape (N, state_dim)
                actions — np.ndarray of shape (N, action_dim)
                rewards — np.ndarray of shape (N,)
                weights — np.ndarray of shape (N,)
                num_samples — int
                episode_ids — list[str]
        """
        ...

    def export(self, path: str) -> None:
        """Export the built dataset as .npz file.

        Args:
            path: File path for the .npz output.
        """
        ...

    def get_statistics(self) -> dict:
        """Return summary statistics of the accumulated dataset.

        Returns:
            dict with mean/std/min/max for states, actions, rewards.
        """
        ...
