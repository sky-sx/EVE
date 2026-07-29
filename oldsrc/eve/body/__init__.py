"""
EVE Body Layer — synthetic motor output.

Hot-path modules that convert policy decisions into motor commands.
All motors are deterministic, synthetic-only, and safe by default
(no real OS control, no hardware access).
"""

from .body_schema import BodySchema, BodyState, BodyLimits, MotorCommand, MotorFeedback, validate_command
from .mouse_motor import MouseMotor
from .keyboard_motor import KeyboardMotor
from .voice_motor import VoiceMotor
from .avatar_motor import AvatarMotor
from .motor_feedback import MotorFeedbackCollector

__all__ = [
    "BodySchema",
    "BodyState",
    "BodyLimits",
    "MotorCommand",
    "MotorFeedback",
    "validate_command",
    "MouseMotor",
    "KeyboardMotor",
    "VoiceMotor",
    "AvatarMotor",
    "MotorFeedbackCollector",
]
