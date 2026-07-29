"""
EVE State Layer — current state representation (State Slots).

Hot-path modules that maintain the organism's real-time state:
ring buffers, input tracking, retina encoding, object slots,
goal token, and the aggregate HotState.
"""

from .frame_buffer import FrameBuffer
from .audio_buffer import AudioBuffer
from .input_state_buffer import InputStateBuffer
from .retina_encoder import RetinaEncoderNet, RetinaCode
from .object_slots import ObjectSlotBuffer, ObjectSlot
from .goal_token import GoalToken
from .hot_state import HotStateAssembler, HotState

__all__ = [
    "FrameBuffer",
    "AudioBuffer",
    "InputStateBuffer",
    "RetinaEncoderNet",
    "RetinaCode",
    "ObjectSlotBuffer",
    "ObjectSlot",
    "GoalToken",
    "HotStateAssembler",
    "HotState",
]
