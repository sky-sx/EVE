"""
Short-term working memory for recent context (~100 items).

Capacity-limited ring buffer with cosine-similarity retrieval.
Holds up to ~100 items; old or low-importance items are evicted
on prune() or when capacity is exceeded.
"""

import numpy as np
from dataclasses import dataclass, field

_DEFAULT_CAPACITY = 100
_DEFAULT_EMBEDDING_DIM = 64


@dataclass
class ShortTermMemory:
    """Short-term working memory with bounded capacity.

    Stores memory items as dicts with at minimum keys:
    {content, embedding, timestamp}. Uses numpy for cosine-distance
    similarity search. When capacity is exceeded, the oldest item
    is evicted.

    Attributes:
        capacity: Maximum number of items to retain (default 100).
        embedding_dim: Expected dimensionality of embedding vectors.
    """

    capacity: int = _DEFAULT_CAPACITY
    embedding_dim: int = _DEFAULT_EMBEDDING_DIM

    _items: list[dict] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    def add(self, item: dict) -> None:
        """Store a new item from recent context.

        Args:
            item: Dict with keys content, embedding, timestamp, importance.
        """
        ...

    def query(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Retrieve top-k items by cosine similarity to query embedding.

        Args:
            embedding: Query embedding vector (1D).
            k: Number of results to return.

        Returns:
            List of dicts sorted by similarity (highest first).
        """
        ...

    def get_recent(self, n: int = 5) -> list[dict]:
        """Return the n most recently added items.

        Args:
            n: Number of recent items to return.

        Returns:
            List of dicts, newest first.
        """
        ...

    def prune(self, importance_threshold: float = 0.2, max_age: float | None = None) -> int:
        """Remove old or low-importance items.

        Args:
            importance_threshold: Minimum importance to retain (0.0-1.0).
            max_age: Optional age cutoff in seconds.

        Returns:
            Number of items pruned.
        """
        ...

    @property
    def size(self) -> int:
        """Current number of stored items."""
        ...

    @property
    def is_full(self) -> bool:
        """Whether the buffer is at capacity."""
        ...
