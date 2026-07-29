"""Action energy module — scales motor impulse energy via confidence."""

from __future__ import annotations

from typing import Any

import numpy as np


class ActionEnergyScaler:
    """Scale motor impulse energy based on confidence.

    Energy cost is deterministic: proportional to movement magnitude
    with penalties for multiple simultaneous actions.

    Attributes:
        threshold: Minimum energy required; actions below this are suppressed.
        movement_cost: Factor for movement energy.
        action_penalty: Penalty for each additional action.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        movement_cost: float = 0.5,
        action_penalty: float = 0.1,
    ) -> None:
        """Initialize the action energy scaler.

        Args:
            threshold: Minimum energy threshold for action execution.
            movement_cost: Factor for movement energy.
            action_penalty: Penalty for each additional action.
        """
        ...

    def compute(
        self, impulse: dict[str, Any], confidence: float
    ) -> dict[str, Any]:
        """Scale motor impulse energy by confidence.

        Args:
            impulse: MotorImpulse as a dict with keys dx, dy, action_type, energy.
            confidence: Confidence score ([0, 1]) used to scale energy.

        Returns:
            Scaled motor impulse with adjusted energy.
        """
        ...

    @property
    def threshold(self) -> float:
        """Minimum energy threshold for action execution."""
        ...

    @threshold.setter
    def threshold(self, value: float) -> None:
        ...
