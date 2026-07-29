"""Captures microphone PCM audio from an external source."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AudioFramePacket:
    """Named packet representing a chunk of audio data.

    Attributes:
        data: Float32 audio samples as 1D ndarray.
        sample_rate: Sample rate in Hz.
        timestamp: Capture timestamp in seconds.
    """

    data: np.ndarray
    sample_rate: int
    timestamp: float


@dataclass
class AudioCapture:
    """External microphone audio stream capture.

    Attributes:
        sample_rate: Samples per second (default 16000).
        frequency: Base sine wave frequency in Hz (default 440.0).
        amplitude: Peak amplitude in [0, 1] (default 0.3).
        noise_level: White noise amplitude (default 0.05).
        seed: Random seed for deterministic noise.
    """

    sample_rate: int = 16000
    frequency: float = 440.0
    amplitude: float = 0.3
    noise_level: float = 0.05
    seed: int = 42

    _running: bool = field(default=False, init=False, repr=False)
    _phase: float = field(default=0.0, init=False, repr=False)
    _sample_count: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42), init=False, repr=False)

    def start(self) -> None:
        """Begin audio capture."""
        ...

    def stop(self) -> None:
        """Stop audio capture."""
        ...

    def get_chunk(self, samples: int = 1024) -> AudioFramePacket:
        """Get a chunk of captured audio.

        Args:
            samples: Number of samples to capture.

        Returns:
            AudioFramePacket with float32 audio data, sample rate, and timestamp.
        """
        ...

    def get_level(self) -> float:
        """Get current audio level.

        Returns:
            Float in [0, 1].
        """
        ...

    @property
    def running(self) -> bool:
        """Whether audio capture is active."""
        ...

    @property
    def total_samples(self) -> int:
        """Total number of samples captured."""
        ...

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since start()."""
        ...
