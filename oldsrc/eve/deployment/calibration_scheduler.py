"""
Calibration Scheduler — triggers recalibration based on performance signals.

Monitors policy performance metrics and triggers recalibration when
deterioration or drift is detected. Cold-path only.
"""

from dataclasses import dataclass, field


@dataclass
class CalibrationScheduler:
    """Schedules policy recalibration based on performance signals.

    Evaluates metrics from ShadowRunner or other monitoring sources and
    decides whether a model component needs recalibration.

    Attributes:
        performance_drop_threshold: Fractional drop in reward that triggers
            recalibration (default 0.2 = 20%).
        drift_threshold: Divergence value above which drift is detected
            (default 0.5).
    """

    performance_drop_threshold: float = 0.2
    drift_threshold: float = 0.5

    _pending: list[dict] = field(default_factory=list, init=False, repr=False)
    _baseline_reward: float | None = field(default=None, init=False, repr=False)

    def should_recalibrate(self, metrics: dict) -> bool:
        """Check whether recalibration is needed based on current metrics.

        Evaluates performance drop (current avg_reward vs baseline) and
        drift detection (divergence_from_live). Returns True if either
        threshold is exceeded.

        Args:
            metrics: Dict with at minimum:
                avg_reward (float): Current average reward.
                divergence_from_live (float): Policy divergence estimate.
                Optionally:
                steps (int): Total evaluation steps.

        Returns:
            True if recalibration is recommended.
        """
        ...

    def schedule(self, component: str, reason: str) -> None:
        """Schedule a recalibration for a given component.

        Args:
            component: Name of the model component to recalibrate
                (e.g. 'visual_servo_policy').
            reason: Human-readable reason for recalibration
                (e.g. 'performance_drop', 'drift_detected', 'user_request').
        """
        ...

    def get_pending(self) -> list[dict]:
        """Return list of pending recalibration entries.

        Returns:
            List of dicts with keys 'component' and 'reason'.
        """
        ...

    def clear(self) -> None:
        """Clear all pending recalibration entries and reset baseline."""
        ...

    @property
    def pending_count(self) -> int:
        """Number of pending recalibration entries."""
        ...
