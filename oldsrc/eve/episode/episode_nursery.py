"""
Episode Nursery — manages episode lifecycle.

Creates, tracks, and closes episodes. Organizes output into
runs/{timestamp}_{id}/ directories and writes summary.json per episode.
"""

from pathlib import Path
from typing import Optional


_RUNS_ROOT_DEFAULT = Path("runs")


class EpisodeNursery:
    """Episode lifecycle manager.

    Creates episode directories under runs/, tracks active episodes,
    and writes per-episode summary metadata.

    Attributes:
        runs_root: Base directory for run output (default: runs/).
    """

    def __init__(self, runs_root: Optional[Path] = None) -> None:
        ...

    def create_episode(self, episode_id: Optional[str] = None) -> str:
        """Create a new episode directory and return its episode_id.

        Args:
            episode_id: Optional explicit ID. If omitted, an ID is
                generated from the current timestamp and a random suffix.

        Returns:
            The episode_id string.
        """
        ...

    def close_episode(self, episode_id: str) -> None:
        """Cleans and finalizes the episode, writing cleaned data to summary.json.

        Args:
            episode_id: Episode identifier returned by create_episode().

        Raises:
            KeyError: If episode_id is unknown.
        """
        ...

    def clean_episode(self, episode_id: str) -> dict:
        """Clean an episode and return the cleaned episode data.

        Args:
            episode_id: Episode identifier.

        Returns:
            Cleaned episode data dict.

        Raises:
            KeyError: If episode_id is unknown.
        """
        ...

    def list_episodes(self) -> list[str]:
        """Return a list of known episode IDs (sorted by creation time).

        Returns:
            list[str] of episode IDs.
        """
        ...

    def get_episode_info(self, episode_id: str) -> dict:
        """Return metadata dict for an episode.

        Args:
            episode_id: Episode identifier.

        Returns:
            dict with episode metadata.

        Raises:
            KeyError: If episode_id is unknown.
        """
        ...

    def update_step_count(self, episode_id: str, count: int) -> None:
        """Update the recorded step count for an episode.

        Args:
            episode_id: Episode identifier.
            count: New step count value.
        """
        ...
