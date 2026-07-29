"""
Body schema definitions — dataclasses, validators, and the BodySchema processor.

Defines the structural contracts for body state processing and all synthetic
motor outputs. Deterministic, synthetic-only, safe by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MotorCommand:
    """A command issued to a synthetic motor.

    Attributes:
        motor_type: Target motor identifier (e.g. "mouse", "keyboard", "voice", "avatar").
        params: Arbitrary key-value parameters for the command.
        energy: Estimated energy cost of the command.
        timestamp: Unix timestamp when the command was created.
    """

    motor_type: str
    params: dict[str, Any] = field(default_factory=dict)
    energy: float = 0.0
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(slots=True)
class BodyLimits:
    """Safety and performance limits for the synthetic body.

    Attributes:
        max_speed: Maximum cursor speed in pixels per step (default 20.0).
        max_range_x: Maximum horizontal range in pixels (default 1920).
        max_range_y: Maximum vertical range in pixels (default 1080).
        key_rate_limit: Maximum key events per second (default 10).
        voice_rate_limit: Maximum speak events per second (default 2).
    """

    max_speed: float = 20.0
    max_range_x: float = 1920.0
    max_range_y: float = 1080.0
    key_rate_limit: float = 10.0
    voice_rate_limit: float = 2.0


@dataclass(slots=True)
class MotorFeedback:
    """Feedback returned after executing a motor command.

    Attributes:
        success: Whether the command was accepted and executed.
        actual_position: The virtual cursor position after execution as (x, y), or None.
        error: Error message if the command failed, empty string otherwise.
        latency_ms: Simulated execution latency in milliseconds.
        motor_type: The motor that produced this feedback.
        timestamp: Unix timestamp when feedback was generated.
    """

    success: bool
    actual_position: tuple[float, float] | None = None
    error: str = ""
    latency_ms: float = 0.0
    motor_type: str = ""
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(slots=True)
class BodyState:
    """Current state of the synthetic body.

    Attributes:
        posture: Current posture label (e.g. "idle", "active", "alert").
        position: Virtual position as (x, y).
        expression: Current facial/avatar expression string.
        is_speaking: Whether the body is currently speaking.
        is_moving: Whether the body is currently in motion.
    """

    posture: str
    position: tuple[float, float]
    expression: str
    is_speaking: bool
    is_moving: bool


class BodySchema:
    """Body schema processor — converts input + feedback into BodyState.

    Processes raw input data and motor feedback to produce a coherent
    body state representation.
    """

    def process(
        self, input_data: dict[str, Any], feedback_data: dict[str, Any]
    ) -> BodyState:
        """Process input and feedback into a body state.

        Args:
            input_data: Dict of raw input values (sensors, commands, etc.).
            feedback_data: Dict of motor feedback entries keyed by motor type.

        Returns:
            BodyState with posture, position, expression, and activity flags.
        """
        ...


def validate_command(cmd: MotorCommand, limits: BodyLimits) -> bool:
    """Validate a motor command against body safety limits.

    Args:
        cmd: The motor command to validate.
        limits: The body limits to check against.

    Returns:
        True if the command passes all limit checks.
    """
    ...
