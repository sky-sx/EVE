"""
Homeostatic need/drive state — reward- and novelty-driven affective state model.

Each need decays slowly over time (representing satisfaction decreasing)
and increases when unsatisfied. The dominant need can bias policy selection.

Hot-path compatible: all operations are deterministic O(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_NEED_NAMES: tuple[str, ...] = (
    "exploration_drive",
    "social_drive",
    "rest_drive",
    "competence_drive",
    "novelty_drive",
    "safety_drive",
)

_DEFAULT_DECAY_RATES: dict[str, float] = {
    "exploration_drive": 0.01,
    "social_drive": 0.008,
    "rest_drive": 0.005,
    "competence_drive": 0.012,
    "novelty_drive": 0.015,
    "safety_drive": 0.003,
}

_DEFAULT_GROWTH_RATES: dict[str, float] = {
    "exploration_drive": 0.02,
    "social_drive": 0.015,
    "rest_drive": 0.03,
    "competence_drive": 0.018,
    "novelty_drive": 0.025,
    "safety_drive": 0.01,
}


@dataclass
class NeedState:
    """Homeostatic need/drive state — produces an affective state summary.

    Each need is a float in [0.0, 1.0]. Higher values mean stronger drive.
    Needs are modulated by reward signals and novelty measures.

    Attributes:
        initial_needs: Optional initial values for each need (default 0.3 each).
    """

    initial_needs: dict[str, float] | None = None

    _needs: dict[str, float] = field(default_factory=dict)
    _decay_rates: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_DECAY_RATES))
    _growth_rates: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_GROWTH_RATES))

    def __post_init__(self) -> None:
        ...

    def update(self, reward: float, novelty: float) -> None:
        """Update affective state from reward and novelty signals.

        Args:
            reward: Current reward signal driving need satisfaction/growth.
            novelty: Novelty measure affecting exploration and novelty drives.
        """
        ...

    def snapshot(self) -> dict[str, float]:
        """Return current affective state.

        Returns:
            {exploration_drive, social_drive, rest_drive,
             competence_drive, novelty_drive, safety_drive}
        """
        ...

    def get_dominant_need(self) -> str:
        """Return the name of the highest drive.

        Returns:
            Name of need with maximum current value.
        """
        ...

    def satisfy(self, need: str, amount: float) -> None:
        """Reduce a specific need by the given amount.

        Args:
            need: Name of the need to satisfy.
            amount: Amount to reduce (clamped to [0, current_value]).

        Raises:
            KeyError: If need name is not valid.
        """
        ...

    def reset(self) -> None:
        """Reset all needs to their initial values."""
        ...

    @property
    def need_names(self) -> tuple[str, ...]:
        """Tuple of all need names."""
        ...
