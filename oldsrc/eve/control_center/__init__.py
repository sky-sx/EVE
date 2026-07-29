"""
EVE Control Center — Read-Only Inspection Dashboard.

This is a COLD-PATH viewer. It does NOT control the hot path, does NOT
inject commands, and does NOT modify runtime state. It reads state
snapshots and renders human-readable panels.

The ControlCenter app and all panels are re-exported here for
convenient access.
"""

from .app import ControlCenter
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

__all__ = [
    "ControlCenter",
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
