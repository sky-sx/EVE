"""
Mid-term memory for task-level context, consolidated from STM.

Consolidated from short-term memory. Items promoted from STM are
retained here with configurable decay. Supports cosine-similarity
query and forgetting of low-importance items.
"""

import numpy as np
from dataclasses import dataclass, field

_DEFAULT_CAPACITY = 500
_DECAY_RATE = 0.001


@dataclass
class MidTermMemory:
    """Medium-term memory consolidated from short-term memory.

    Important items from STM are promoted here. Items slowly decay
    in importance over time; those falling below a threshold can be
    forgotten.

    Attributes:
        capacity: Maximum number of items (default 500).
        default_embedding_dim: Expected embedding dimensionality.
    """

    capacity: int = _DEFAULT_CAPACITY
    default_embedding_dim: int = 64

    _items: list[dict] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)
    _consolidation_count: int = field(default=0, init=False, repr=False)

    def consolidate(self, stm_items: list[dict]) -> int:
        """Consolidate items from task context into MTM.

        Args:
            stm_items: List of memory item dicts from ShortTermMemory.

        Returns:
            Number of items actually consolidated.
        """
        ...

    def query(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Retrieve top-k items by cosine similarity.

        Args:
            embedding: Query embedding vector (1D).
            k: Number of results to return.

        Returns:
            List of dicts sorted by similarity (highest first).
        """
        ...

    def forget(self, threshold: float = 0.1) -> int:
        """Remove items with importance below the threshold.

        Args:
            threshold: Items with importance below this are removed.

        Returns:
            Number of items forgotten.
        """
        ...

    def get_statistics(self) -> dict:
        """Return summary statistics of the MTM store.

        Returns:
            Dict with total_items, mean_importance,
            consolidation_rounds, capacity.
        """
        ...

    @property
    def size(self) -> int:
        """Current number of stored items."""
        ...
