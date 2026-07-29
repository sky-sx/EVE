"""
Reflection worker — async episode analysis (cold path).

Uses an external LLM/VLM for episode reflection when available. When no
LLM/VLM is provided, simulates offline reflection with synthetic pattern
detection and a configurable processing delay.

Runs in cold path only; all processing is batched via process_pending().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ReflectionWorker:
    """Async reflection worker for episode analysis.

    Maintains a queue of episode IDs to analyze. When an LLM/VLM callable
    is provided, it is invoked for deep episode analysis; otherwise the
    worker simulates pattern detection with a configurable delay.

    Attributes:
        processing_delay: Simulated processing time in seconds when llm_fn is None (default 0.5).
        insight_capacity: Maximum insights stored per episode (default 20).
        llm_fn: Optional external LLM/VLM callable for episode reflection.
    """

    processing_delay: float = 0.5
    insight_capacity: int = 20
    llm_fn: Callable | None = None

    _queue: list[str] = field(default_factory=list)
    _insights: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _processing: bool = False
    _processed_count: int = 0

    def submit(self, episode_id: str) -> None:
        """Queue an episode for reflection.

        Args:
            episode_id: Identifier for the episode to analyze.
        """
        ...

    def process_pending(self) -> None:
        """Process one pending reflection from the queue.

        Invokes LLM/VLM callable to analyze the episode and generate insights
        when available. Falls back to simulated analysis delay and synthetic
        insights when llm_fn is None.
        Non-blocking: processes at most one episode per call.
        """
        ...

    def get_insights(self, episode_id: str) -> list[dict[str, Any]]:
        """Retrieve insights for a previously processed episode.

        Args:
            episode_id: Episode identifier.

        Returns:
            List of insight dicts with {pattern, confidence, suggestion}.
        """
        ...

    def is_busy(self) -> bool:
        """Check if the worker is currently processing.

        Returns:
            True if processing is in progress.
        """
        ...

    @property
    def queue_size(self) -> int:
        """Number of episodes pending reflection."""
        ...

    @property
    def processed_count(self) -> int:
        """Total episodes processed so far."""
        ...

    @property
    def known_episodes(self) -> list[str]:
        """Episode IDs with stored insights."""
        ...
