"""
Synthetic voice output motor — log-only, no real audio hardware.

Generates mock audio buffers of silence and logs speech events.
Deterministic, synthetic, and safe by default.
"""

from __future__ import annotations

import numpy as np

from .body_schema import BodyLimits, MotorCommand, MotorFeedback


class VoiceMotor:
    """Synthetic voice/speech output motor.

    All speak operations generate mock silent audio buffers and log
    the text. No real audio hardware or TTS engine is used.

    Attributes:
        limits: Body safety limits used for rate checking.
    """

    def __init__(self, limits: BodyLimits | None = None) -> None:
        """Initialize the voice motor with optional safety limits.

        Args:
            limits: Body safety limits used for rate checking.
        """
        ...

    def speak(self, text: str, duration_s: float | None = None) -> MotorFeedback:
        """Log synthetic speech and generate a mock silent audio buffer.

        Args:
            text: The text that would be spoken.
            duration_s: Simulated speech duration in seconds. If None,
                estimated as max(0.5, len(text) * 0.08).

        Returns:
            MotorFeedback with the mock audio metadata.
        """
        ...

    def silence(self) -> MotorFeedback:
        """Log a silence command and clear the audio buffer.

        Returns:
            MotorFeedback indicating the voice is now silent.
        """
        ...

    def is_speaking(self) -> bool:
        """Check whether the voice motor is currently speaking.

        Returns:
            True if a speak command is active and has not been silenced.
        """
        ...

    def execute(self, cmd: MotorCommand) -> MotorFeedback:
        """Execute a generic MotorCommand on this motor.

        Args:
            cmd: MotorCommand with motor_type="voice".

        Returns:
            MotorFeedback from the operation, or error feedback if invalid.
        """
        ...

    @property
    def speak_count(self) -> int:
        """Total speak commands issued.

        Returns:
            Integer count of speak commands.
        """
        ...

    @property
    def last_speech_text(self) -> str:
        """Most recent speech text.

        Returns:
            The text of the most recent speech.
        """
        ...

    @property
    def audio_buffer(self) -> np.ndarray | None:
        """Current mock audio buffer, or None if silent.

        Returns:
            NumPy array of audio samples, or None.
        """
        ...
