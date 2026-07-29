"""
Reward Oracle — ground-truth reward computation (synthetic, cold path).

Deterministic reward function based on state transitions. Computes
scalar rewards in [-1, 1] and provides a per-component breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardOracle:
    """Ground-truth reward oracle for synthetic environments.

    Rewards are computed deterministically from the state transition
    captured in the episode trace. Components:
      - proximity:  positive when moving towards target
      - energy_efficiency: penalises large moves
      - collision:  negative when hitting boundaries
      - idle:       small negative for no-op when far from target

    Attributes:
        proximity_scale: Multiplier for proximity reward.
        energy_scale:    Multiplier for energy efficiency penalty.
        collision_penalty: Reward penalty for boundary collision.
        idle_penalty:    Reward penalty for no-op far from target.
        target_key:      Key in state dict for target position.
        cursor_key:      Key in state dict for cursor position.
    """

    proximity_scale: float = 1.0
    energy_scale: float = -0.05
    collision_penalty: float = -0.3
    idle_penalty: float = -0.02
    target_key: str = "target"
    cursor_key: str = "cursor"
    bounds: tuple[float, float] = (0.0, 1.0)

    def compute(self, episode_trace: dict) -> float:
        """Compute scalar reward from a full episode trace.

        Args:
            episode_trace: Dict containing the full episode trace
                (states, actions, metadata) rather than a single step.

        Returns:
            Scalar reward in [-1, 1].
        """
        ...

    def reward_components(self, episode_trace: dict) -> dict[str, float]:
        """Return a breakdown of individual reward components.

        Args:
            episode_trace: Dict containing the full episode trace
                (states, actions, metadata).

        Returns:
            Dict mapping component name → scalar reward.
        """
        ...
