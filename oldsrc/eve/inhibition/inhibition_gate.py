"""
Inhibition Gate — the SafetyGate on the hot path.

Filters motor commands through the PermissionManager, blocks unsafe
actions, and supports an emergency-stop mechanism.
"""

from __future__ import annotations

from .permission_manager import PermissionManager


class InhibitionGate:
    """Safety gate that sits between Policy and MotorStub.

    Filters a batch of motor commands, allowing only those that pass
    the permission check. Supports emergency-stop semantics.
    """

    def __init__(self) -> None:
        """Initialize the inhibition gate."""
        ...

    def allow(
        self,
        motor_commands: list[dict],
        permission_mgr: PermissionManager,
    ) -> list[dict]:
        """Filter *motor_commands*, returning only those permitted.

        Blocked commands are logged and counted. When the emergency
        stop is active every command is blocked regardless of
        permissions.

        Args:
            motor_commands: List of motor command dicts to filter.
            permission_mgr: The PermissionManager to check against.

        Returns:
            List of allowed motor commands.
        """
        ...

    def block_all(self) -> None:
        """Emergency stop — blocks all future commands until released."""
        ...

    def release(self) -> None:
        """Release the emergency stop."""
        ...

    def is_blocked(self) -> bool:
        """Return True when emergency stop is active.

        Returns:
            True if emergency stop is engaged.
        """
        ...

    def get_blocked_count(self) -> int:
        """Total number of commands blocked since creation.

        Returns:
            Integer count of blocked commands.
        """
        ...
