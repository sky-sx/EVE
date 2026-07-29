"""
Policy output viewer — current action, action energy, policy weights
summary.
"""

from dataclasses import dataclass


@dataclass
class PolicyViewer:
    """Renders the policy's current decision and internal state."""

    def render(self, state: dict) -> str:
        """Render policy output panel.

        Args:
            state: Dict with keys for current_action, action_params,
                   action_energy, policy_weights, policy_name, etc.

        Returns:
            Formatted ASCII string.
        """
        ...
