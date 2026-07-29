"""
Synthetic mouse motor — log-only, no real OS control.

Simulates cursor movement, clicks, and scrolling. All operations are
deterministic, synthetic, and safe by default.
"""

from __future__ import annotations

from .body_schema import BodyLimits, MotorCommand, MotorFeedback


class MouseMotor:
    """Synthetic mouse motor with virtual cursor tracking.

    All commands are logged; no real mouse or display is affected.
    The virtual cursor position is maintained internally.

    Attributes:
        limits: Body safety limits used for command validation.
    """

    def __init__(self, limits: BodyLimits | None = None) -> None:
        """Initialize the mouse motor with optional safety limits.

        Args:
            limits: Body safety limits used for command validation.
        """
        ...

    @property
    def position(self) -> tuple[float, float]:
        """Current virtual cursor position as (x, y).

        Returns:
            A tuple of (x, y) coordinates.
        """
        ...

    def move_to(self, x: float, y: float) -> MotorFeedback:
        """Move virtual cursor to absolute position (x, y).

        Args:
            x: Target x-coordinate (clamped to limits).
            y: Target y-coordinate (clamped to limits).

        Returns:
            MotorFeedback with the new virtual position.
        """
        ...

    def click(self, button: str = "left") -> MotorFeedback:
        """Log a synthetic mouse click.

        Args:
            button: Button name ("left", "right", "middle").

        Returns:
            MotorFeedback indicating success.
        """
        ...

    def scroll(self, dy: float) -> MotorFeedback:
        """Log a synthetic scroll event.

        Args:
            dy: Scroll delta (positive = up, negative = down).

        Returns:
            MotorFeedback indicating success.
        """
        ...

    def execute(self, cmd: MotorCommand) -> MotorFeedback:
        """Execute a generic MotorCommand on this motor.

        Args:
            cmd: MotorCommand with motor_type="mouse".

        Returns:
            MotorFeedback from the operation, or error feedback if invalid.
        """
        ...

    @property
    def click_count(self) -> int:
        """Total synthetic clicks performed.

        Returns:
            Integer count of clicks.
        """
        ...

    @property
    def scroll_total(self) -> float:
        """Cumulative scroll delta.

        Returns:
            Total accumulated scroll amount.
        """
        ...
