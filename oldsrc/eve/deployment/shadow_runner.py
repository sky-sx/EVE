"""
Shadow Runner — parallel policy evaluation.

Runs a candidate policy in parallel with the live policy, comparing
performance without affecting the hot path. Cold-path only.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ShadowRunner:
    """Evaluates a shadow policy alongside the live policy.

    The shadow runner receives the same state vectors as the live policy
    but uses a separate set of policy weights. It tracks reward divergence
    and provides metrics for comparison.

    Attributes:
        default_reward: Fallback reward when shadow policy is not active.
    """

    default_reward: float = 0.0

    _active: bool = field(default=False, init=False, repr=False)
    _weights: dict = field(default_factory=dict, init=False, repr=False)
    _total_reward: float = field(default=0.0, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)
    _reward_history: list[float] = field(default_factory=list, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)

    def start(self, shadow_policy_weights: dict) -> None:
        """Start shadow evaluation with the given policy weights.

        Args:
            shadow_policy_weights: Dictionary mapping parameter names to
                numpy arrays representing the candidate policy weights.
        """
        ...

    def step(self, state_vector: np.ndarray) -> dict:
        """Run one shadow evaluation step.

        Computes a shadow action from the state vector using the shadow
        policy weights, then estimates a synthetic reward.

        Args:
            state_vector: Current state as a 1D numpy array.

        Returns:
            Dict with keys 'action' (numpy array) and 'reward' (float).
        """
        ...

    def get_metrics(self) -> dict:
        """Return shadow evaluation metrics.

        Returns:
            Dict with keys:
                avg_reward: Mean reward across all shadow steps.
                steps: Total number of shadow steps executed.
                divergence_from_live: Estimate of how different the shadow
                    policy output is from the live policy.
        """
        ...

    def stop(self) -> None:
        """Stop shadow evaluation."""
        ...

    @property
    def active(self) -> bool:
        """Whether shadow evaluation is currently running."""
        ...

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since start()."""
        ...
