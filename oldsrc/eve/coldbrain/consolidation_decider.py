"""
Consolidation decider — decides what to consolidate into memory (cold path).

Evaluates episode summaries and determines which items are worth keeping,
forgetting, and where they should be stored. Uses heuristic scoring based
on novelty, reward, repetition, and error signals.

Runs in cold path only; all processing is batched via process_pending().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_TARGETS: tuple[str, ...] = (
    "short_term", "mid_term", "long_term", "habit", "skill", "discard"
)


@dataclass
class ConsolidationDecider:
    """Heuristic consolidation decider for episode memories.

    Evaluates episode items and assigns importance scores and target
    memory tiers. Uses simple heuristics; no ML model.

    Attributes:
        novelty_weight: Weight for novelty signal (default 1.0).
        reward_weight: Weight for reward magnitude (default 1.2).
        repetition_weight: Weight for repetition count (default 0.8).
        error_weight: Weight for error/surprise signal (default 0.9).
        importance_threshold: Minimum importance to keep (default 0.2).
    """

    novelty_weight: float = 1.0
    reward_weight: float = 1.2
    repetition_weight: float = 0.8
    error_weight: float = 0.9
    importance_threshold: float = 0.2

    _pending: list[dict[str, Any]] = field(default_factory=list)
    _results: list[list[dict[str, Any]]] = field(default_factory=list)

    def evaluate(self, episode_summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate an episode summary and determine consolidation actions.

        Args:
            episode_summary: Dict with episode data.

        Returns:
            List of consolidation decisions:
            [{type, content, importance, target_memory}].
        """
        ...

    def should_forget(self, item: dict[str, Any]) -> bool:
        """Determine if an item should be forgotten (not consolidated).

        Args:
            item: Item dict with optional 'importance' field.

        Returns:
            True if the item should be discarded.
        """
        ...

    def prioritize(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort items by importance (descending).

        Args:
            items: List of consolidation decision dicts.

        Returns:
            Sorted list, highest importance first.
        """
        ...

    def enqueue(self, episode_summary: dict[str, Any]) -> None:
        """Queue an episode summary for later batch processing."""
        ...

    def process_pending(self) -> list[list[dict[str, Any]]]:
        """Process all queued episode summaries.

        Returns:
            List of consolidation decision lists, one per queued episode.
        """
        ...

    @property
    def pending_count(self) -> int:
        """Number of queued episode summaries awaiting processing."""
        ...
