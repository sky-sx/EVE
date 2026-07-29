"""
Permission Manager — controls what actions are allowed.

Part of the safety inhibition layer. Maintains a set of granted permissions
and checks whether a motor command is permitted. High-risk actions require
explicit grant before they can pass.
"""

from __future__ import annotations


class PermissionManager:
    """Manages what motor actions are allowed in the current session.

    Default permissions cover synthetic-safe actions. High-risk actions
    (delete, execute, install) must be explicitly granted before use.
    """

    def __init__(self) -> None:
        """Initialize the permission manager with default safe permissions."""
        ...

    def check(self, cmd: dict) -> tuple[bool, str]:
        """Check whether *cmd* is allowed.

        Args:
            cmd: The motor command dict to check.

        Returns:
            (allowed, reason) — allowed is True when the command may
            proceed; reason explains why it was blocked.
        """
        ...

    def is_high_risk(self, cmd: dict) -> bool:
        """Return True if *cmd* targets a high-risk action.

        Args:
            cmd: The motor command dict to check.

        Returns:
            True if the command targets a high-risk action.
        """
        ...

    def grant(self, permission: str) -> None:
        """Explicitly grant a permission (including high-risk ones).

        Args:
            permission: The permission name to grant.
        """
        ...

    def revoke(self, permission: str) -> None:
        """Revoke a previously granted permission.

        Args:
            permission: The permission name to revoke.
        """
        ...

    def get_permissions(self) -> set[str]:
        """Return a copy of the currently granted permissions.

        Returns:
            A set of currently granted permission strings.
        """
        ...
