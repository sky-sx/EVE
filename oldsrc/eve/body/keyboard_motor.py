"""
Synthetic keyboard motor — log-only, no real OS control.

Simulates key presses, releases, and text typing. All operations are
deterministic, synthetic, and safe by default.
"""

from __future__ import annotations

from .body_schema import BodyLimits, MotorCommand, MotorFeedback


class KeyboardMotor:
    """Synthetic keyboard motor.

    All key events are logged; no real keyboard or OS input is affected.

    Attributes:
        limits: Body safety limits used for rate checking.
    """

    def __init__(self, limits: BodyLimits | None = None) -> None:
        """Initialize the keyboard motor with optional safety limits.

        Args:
            limits: Body safety limits used for rate checking.
        """
        ...

    def press(self, key: str) -> MotorFeedback:
        """Log a synthetic key press.

        Args:
            key: Key identifier (e.g. "a", "enter", "shift").

        Returns:
            MotorFeedback indicating success.
        """
        ...

    def release(self, key: str) -> MotorFeedback:
        """Log a synthetic key release.

        Args:
            key: Key identifier to release.

        Returns:
            MotorFeedback indicating success.
        """
        ...

    def type_text(self, text: str) -> MotorFeedback:
        """Log synthetic text typing.

        Simulates pressing and releasing each character in sequence.

        Args:
            text: The string of characters to type.

        Returns:
            MotorFeedback indicating success.
        """
        ...

    def execute(self, cmd: MotorCommand) -> MotorFeedback:
        """Execute a generic MotorCommand on this motor.

        Args:
            cmd: MotorCommand with motor_type="keyboard".

        Returns:
            MotorFeedback from the operation, or error feedback if invalid.
        """
        ...

    @property
    def pressed_keys(self) -> set[str]:
        """Currently held keys.

        Returns:
            Frozen set of currently pressed key identifiers.
        """
        ...

    @property
    def key_count(self) -> int:
        """Total key events produced.

        Returns:
            Integer count of key events.
        """
        ...
