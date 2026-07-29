"""
Action Audit Logger — audit trail for all actions flowing through the safety gate.

Logs commands and their feedback for audit trail. Maintains an in-memory
ring buffer of audit entries, supports JSONL export, and provides summary
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionAuditLogger:
    """In-memory ring-buffer audit trail for motor commands.

    Logs commands and their feedback for audit trail. Every action
    that passes through the InhibitionGate is recorded with its command
    payload, whether it was allowed/blocked, the reason, and a
    timestamp.

    Attributes:
        capacity: Maximum number of audit entries to retain.
    """

    capacity: int = 10_000

    def log(self, cmd: dict, allowed: bool, reason: str) -> None:
        """Record an audit entry for *cmd*.

        Args:
            cmd: The motor command dict.
            allowed: Whether the command was allowed.
            reason: Reason for allow/block decision.
        """
        ...

    def get_history(self, n: int = 100) -> list[dict]:
        """Return up to the last *n* audit entries (most recent first).

        Args:
            n: Maximum number of entries to return.

        Returns:
            List of audit entry dicts, most recent first.
        """
        ...

    def get_blocked_actions(self) -> list[dict]:
        """Return all blocked audit entries (most recent first).

        Returns:
            List of blocked audit entry dicts.
        """
        ...

    def export(self, path: str) -> None:
        """Write the full audit trail to *path* as JSONL.

        Args:
            path: File path for JSONL export.
        """
        ...

    def summary(self) -> dict:
        """Return aggregate statistics.

        Returns:
            Dict with keys: total, allowed, blocked, by_type.
        """
        ...
