"""
Control Center panels — read-only text renderers for hot-path state.

Each panel exposes a single public method:

    render(state: dict) -> str

which returns a formatted ASCII string suitable for console display.
Panels are stateless renderers — they hold no configuration beyond
what is needed for formatting.
"""

from .overview import OverviewPanel
from .sensory_viewer import SensoryViewer
from .state_viewer import StateViewer
from .attention_viewer import AttentionViewer
from .policy_viewer import PolicyViewer
from .motor_viewer import MotorViewer
from .memory_viewer import MemoryViewer
from .training_viewer import TrainingViewer
from .log_explorer import LogExplorer

__all__ = [
    "OverviewPanel",
    "SensoryViewer",
    "StateViewer",
    "AttentionViewer",
    "PolicyViewer",
    "MotorViewer",
    "MemoryViewer",
    "TrainingViewer",
    "LogExplorer",
]
