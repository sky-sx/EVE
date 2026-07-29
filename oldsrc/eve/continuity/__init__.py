"""
EVE Continuity Model — persistent sense of self and world.

Hot-path modules that maintain state continuity across frames:
- WorldState: external environment estimate with change detection
- SelfState: internal condition with operating modes
- NeedState: homeostatic drive model
"""

from .world_state import WorldState
from .self_state import SelfState
from .need_state import NeedState

__all__ = [
    "WorldState",
    "SelfState",
    "NeedState",
]
