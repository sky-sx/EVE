"""
Trace Store — detailed per-step execution traces.

Stores structured trace events for offline analysis. Each trace includes
sensory inputs, state snapshots, policy outputs, and inhibition decisions.
Uses a ring buffer per episode to bound memory usage.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_RING_SIZE = 10_000
_TRACES_ROOT_DEFAULT = Path("runs")


@dataclass
class TraceStore:
    """Stores detailed execution traces with per-episode ring buffers.

    Each episode gets its own ring buffer. When the buffer is full,
    oldest events are overwritten. Traces are written to disk on prune
    or explicit flush.

    Attributes:
        traces_root: Base directory for trace output.
        ring_size: Maximum events per episode ring buffer.
    """

    traces_root: Path = _TRACES_ROOT_DEFAULT
    ring_size: int = _DEFAULT_RING_SIZE

    _buffers: dict[str, list[dict]] = field(default_factory=dict, init=False, repr=False)
    _heads: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _offsets: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def record(self, event: dict) -> None:
        """Store a single trace event.

        The event dict should contain an 'episode_id' key. It may also
        include: sensory_input, state_snapshot, policy_output,
        inhibition_decision, timestamp, step.

        Args:
            event: Trace event dictionary.
        """
        ...

    def get_trace(self, episode_id: str) -> list[dict]:
        """Return all trace events for an episode in chronological order.

        Args:
            episode_id: Episode identifier.

        Returns:
            list of trace event dicts (empty list if no traces recorded).
        """
        ...

    def prune(self, max_episodes: int) -> None:
        """Remove oldest episode buffers until at most max_episodes remain.

        Traces for removed episodes are written to disk before deletion.

        Args:
            max_episodes: Maximum number of episode buffers to retain.
        """
        ...
