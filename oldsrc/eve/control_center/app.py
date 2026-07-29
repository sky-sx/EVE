"""
ControlCenter — text-based read-only inspection dashboard for the EVE hot path.

Provides a keyboard-driven console interface that samples state from a
provider function and renders the current panel. No GUI, no curses, no
Textual — pure print-based output.
"""

from typing import Callable

from .panels import (
    OverviewPanel,
    SensoryViewer,
    StateViewer,
    AttentionViewer,
    PolicyViewer,
    MotorViewer,
    MemoryViewer,
    TrainingViewer,
    LogExplorer,
)


class ControlCenter:
    """Read-only inspection dashboard.

    Takes a `state_provider` callable that returns a dictionary snapshot
    of the current hot-path state. The dashboard samples this
    periodically and renders panels.

    Panels are navigated via key presses. Press 'q' to quit.
    """

    def __init__(self, state_provider: Callable[[], dict]) -> None:
        """Initialize the control center.

        Args:
            state_provider: Callable returning a dict state snapshot.
        """
        ...

    def run(self) -> None:
        """Start the interactive control loop (blocking)."""
        ...

    def stop(self) -> None:
        """Signal the control loop to stop (thread-safe)."""
        ...

    def get_menu(self) -> list[str]:
        """Return the list of available panel keys.

        Returns:
            List of panel key strings.
        """
        ...

    def open_panel(self, name: str) -> None:
        """Switch the active panel to `name`.

        Args:
            name: Panel key to switch to.

        Raises:
            KeyError: If name is not a known panel.
        """
        ...
