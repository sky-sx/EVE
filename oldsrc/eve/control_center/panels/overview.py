"""
Overview panel — compact single-screen system status summary.

Shows: uptime, fps, active modules, episode count, mode.
"""

from dataclasses import dataclass


@dataclass
class OverviewPanel:
    """Renders a compact system overview from a state snapshot."""

    def render(self, state: dict) -> str:
        """Render system overview panel.

        Args:
            state: Dict with keys for uptime, fps, active_modules,
                   episode_count, mode, frame_id, etc.

        Returns:
            Formatted ASCII string.
        """
        ...
