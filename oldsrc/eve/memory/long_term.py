"""
Long-term memory for important records, persisted to disk.

Persistent, file-backed memory with larger capacity and slower access.
Consolidated from mid-term memory. Supports JSON save/load and
cosine-similarity query.
"""

import numpy as np
from dataclasses import dataclass, field

_DEFAULT_CAPACITY = 5000


@dataclass
class LongTermMemory:
    """Persistent long-term memory with file backing.

    Items consolidated from MidTermMemory are stored here with
    larger capacity. Supports JSON persistence for durability.

    Attributes:
        capacity: Maximum number of items (default 5000).
        default_embedding_dim: Expected embedding dimensionality.
    """

    capacity: int = _DEFAULT_CAPACITY
    default_embedding_dim: int = 64

    _items: list[dict] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    def store(self, item: dict) -> None:
        """Store an important record in LTM.

        Args:
            item: Memory item dict with keys content, embedding,
                  timestamp, importance.
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

    def save(self, path: str) -> None:
        """Persist all items to a JSON file.

        Args:
            path: File path for the JSON output.
        """
        ...

    def load(self, path: str) -> None:
        """Load items from a JSON file.

        Args:
            path: File path to the JSON input.
        """
        ...

    def consolidate(self, mtm_items: list[dict]) -> int:
        """Promote items from mid-term memory into LTM.

        Args:
            mtm_items: List of memory item dicts from MidTermMemory.

        Returns:
            Number of items promoted.
        """
        ...

    @property
    def size(self) -> int:
        """Current number of stored items."""
        ...
