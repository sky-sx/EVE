"""
Motor feedback collector — collects and summarizes motor feedback.

Collects feedback from all synthetic motors to provide aggregate
metrics for monitoring and cold-path analysis.
"""

from __future__ import annotations

from typing import Any

from .body_schema import MotorFeedback


class MotorFeedbackCollector:
    """Collects feedback from multiple motor instances.

    Provides summary statistics over recent feedback entries for
    monitoring, debugging, and cold-path analysis.

    Attributes:
        max_history: Maximum number of feedback entries to retain.
    """

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize the collector with a maximum history size.

        Args:
            max_history: Maximum number of feedback entries to retain.
        """
        ...

    def record(self, feedback: MotorFeedback) -> None:
        """Record a single motor feedback entry.

        Args:
            feedback: MotorFeedback from a motor execution.
        """
        ...

    def get_recent(self, n: int = 10) -> list[MotorFeedback]:
        """Get the n most recent feedback entries.

        Args:
            n: Number of entries to return (clamped to available).

        Returns:
            List of most recent MotorFeedback entries.
        """
        ...

    def success_rate(self) -> float:
        """Compute the success rate over all recorded feedback.

        Returns:
            Ratio of successful feedback entries, or 0.0 if no entries.
        """
        ...

    def avg_latency_ms(self) -> float:
        """Compute the average latency over all recorded feedback.

        Returns:
            Mean latency in milliseconds, or 0.0 if no entries.
        """
        ...

    def summary(self) -> dict[str, Any]:
        """Generate a summary dict of aggregated feedback.

        Returns:
            Dict with keys: total_feedback, success_rate, avg_latency_ms,
            motor_counts, recent_errors.
        """
        ...

    @property
    def count(self) -> int:
        """Total number of feedback entries recorded.

        Returns:
            Integer count of feedback entries.
        """
        ...
