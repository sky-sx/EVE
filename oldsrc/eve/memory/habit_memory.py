"""
Habit / procedural memory.

Records state-action-reward triplets and forms habits when enough
positive reinforcement is accumulated. A habit fires when a query
state closely matches a recorded pattern above threshold confidence.
"""

import numpy as np
from dataclasses import dataclass, field

_DEFAULT_HABIT_THRESHOLD = 3
_DEFAULT_SIMILARITY_THRESHOLD = 0.85
_DEFAULT_MAX_HABITS = 200


@dataclass
class HabitMemory:
    """Habit/procedural memory for learned stimulus-response patterns.

    Records state-action-reward triplets. When a state-action pair
    accumulates enough positive reward, it becomes a habit. Subsequent
    queries for similar states return the habitual action.

    Attributes:
        habit_threshold: Number of positive reinforcements needed to
            form a habit (default 3).
        similarity_threshold: Cosine similarity minimum for a habit
            match (default 0.85).
    """

    habit_threshold: int = _DEFAULT_HABIT_THRESHOLD
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD

    _records: list[dict] = field(default_factory=list, init=False, repr=False)
    _habits: list[dict] = field(default_factory=list, init=False, repr=False)

    def record(self, state: np.ndarray, action: dict, reward: float) -> None:
        """Record a state-action-reward triplet.

        Args:
            state: 1D numpy array representing the state.
            action: Dict representing the action taken.
            reward: Scalar reward value.
        """
        ...

    def get_habit(self, state: np.ndarray) -> dict | None:
        """Retrieve a habitual action for a given state.

        Args:
            state: 1D numpy array representing the query state.

        Returns:
            Dict with keys 'action' and 'confidence', or None.
        """
        ...

    def reinforce(self, state: np.ndarray, action: dict, reward: float) -> None:
        """Reinforce an existing habit with additional reward.

        Args:
            state: 1D numpy array.
            action: Action dict.
            reward: Reward value.
        """
        ...

    def get_habit_count(self) -> int:
        """Return the number of formed habits."""
        ...

    @property
    def record_count(self) -> int:
        """Number of recorded state-action pairs (not yet habits)."""
        ...
