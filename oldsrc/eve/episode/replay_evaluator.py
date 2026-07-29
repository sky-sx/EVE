"""
Replay Evaluator — offline policy evaluation from episode logs.

Re-reads episode JSONL logs and recomputes metrics including average
reward, success rate, step count, distance metrics, and latency
statistics.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union


class ReplayEvaluator:
    """Offline evaluator that replays episode logs to compute metrics.

    Reads episode.jsonl files from run directories and produces
    structured evaluation results. Stateless — all state is carried
    in the return dict.
    """

    @staticmethod
    def evaluate(
        model: callable,
        replay: Union[str, Path, dict],
        episode_id_or_dir: Union[str, Path, None] = None,
    ) -> dict:
        """Evaluate a policy model against replayed episode data.

        Args:
            model: callable — policy model to evaluate. Accepts state
                dict and returns action dict.
            replay: Episode data to replay, either a Path/str to the
                run directory, or a pre-loaded episode dict.
            episode_id_or_dir: (Deprecated) Legacy path or episode ID.

        Returns:
            dict with keys: avg_reward, success_rate, steps, initial_distance,
            final_distance, distance_reduction_ratio, metrics.

        Raises:
            FileNotFoundError: If no episode.jsonl is found.
        """
        ...

    @staticmethod
    def compare(
        episode_a: Union[str, Path],
        episode_b: Union[str, Path],
    ) -> dict:
        """Compare two episodes side-by-side.

        Args:
            episode_a: Path or episode_id for first episode.
            episode_b: Path or episode_id for second episode.

        Returns:
            dict with keys: a, b, delta (metric-wise difference).
        """
        ...

    @staticmethod
    def aggregate(episode_ids: list[Union[str, Path]]) -> dict:
        """Aggregate metrics across multiple episodes.

        Args:
            episode_ids: List of paths or episode_id strings.

        Returns:
            dict with averaged metrics across all episodes.
        """
        ...
