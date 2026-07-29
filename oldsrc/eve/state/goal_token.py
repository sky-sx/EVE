"""Goal token module — maps instruction / intention to a goal embedding vector."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


_PREDEFINED_GOAL_TYPES = {
    "explore": 0,
    "attend": 1,
    "rest": 2,
    "follow": 3,
}


@dataclass
class GoalToken:
    """Goal token that converts an instruction or intention into an embedding vector.

    Attributes:
        dim: Embedding dimension (default 16).
        seed: Random seed for generating goal embeddings.
    """

    dim: int = 16
    seed: int = 42

    _goal_type: str = field(default="", init=False, repr=False)
    _params: dict = field(default_factory=dict, init=False, repr=False)
    _embedding: np.ndarray | None = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _goal_embeddings: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        ...

    def set(self, goal_type: str, params: dict | None = None) -> None:
        """Set the active goal from an instruction or intention.

        Args:
            goal_type: One of 'explore', 'attend', 'rest', 'follow'.
            params: Optional dict of goal-specific parameters.
        """
        ...

    def get_embedding(self) -> np.ndarray:
        """Get the goal embedding vector.

        Returns:
            Goal embedding, or zero vector if inactive.
        """
        ...

    def is_active(self) -> bool:
        """Check if a goal is currently set.

        Returns:
            True if a goal is active.
        """
        ...

    def clear(self) -> None:
        """Deactivate the current goal."""
        ...

    @property
    def goal_type(self) -> str:
        """The string name of the active goal type, or '' if inactive."""
        ...

    @property
    def params(self) -> dict:
        """Goal-specific parameters dict."""
        ...
