"""
Teacher Orchestra — coordinates fast/slow teachers on the cold path.

Queues labeling requests asynchronously and dispatches them to the
fast teacher first, falling back to the slow teacher when needed.
All processing is deferred and never blocks the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fast_teacher import FastTeacher
from .slow_teacher import SlowTeacher
from .reward_oracle import RewardOracle


@dataclass
class TeacherOrchestra:
    """Coordinates the fast and slow teachers.

    Labeling requests are queued and processed asynchronously from a
    background thread (cold path). The fast teacher attempts to handle
    simple cases first; if it cannot, the slow teacher takes over.

    Attributes:
        fast_teacher: Rule-based instant teacher.
        slow_teacher: Deferred deep-analysis teacher.
        reward_oracle: Ground-truth reward computer.
    """

    fast_teacher: FastTeacher = field(default_factory=FastTeacher)
    slow_teacher: SlowTeacher = field(default_factory=SlowTeacher)
    reward_oracle: RewardOracle = field(default_factory=RewardOracle)

    def request_label(self, state_snapshot: dict, action: dict) -> bool:
        """Queue a labeling request. Never blocks.

        Args:
            state_snapshot: State snapshot dict.
            action: The action dict to label.

        Returns:
            True if queued, False if the orchestra is shut down.
        """
        ...

    def process_queue(self) -> list[dict]:
        """Process all pending labeling requests (called from cold path).

        Returns:
            List of label results in FIFO order.
        """
        ...

    def get_pending_count(self) -> int:
        """Number of labeling requests waiting in the queue.

        Returns:
            Integer count of pending requests.
        """
        ...

    def shutdown(self) -> None:
        """Stop processing and clear the queue."""
        ...
