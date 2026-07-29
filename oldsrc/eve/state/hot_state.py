"""Hot state assembler — aggregate all current state pieces into a unified snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .frame_buffer import FrameBuffer
from .audio_buffer import AudioBuffer
from .input_state_buffer import InputStateBuffer
from .retina_encoder import RetinaEncoderNet
from .object_slots import ObjectSlotBuffer
from .goal_token import GoalToken


@dataclass
class HotState:
    """Unified snapshot of all current state pieces for the policy hot path.

    Attributes:
        version: Monotonic update counter.
        timestamp: Timestamp of the snapshot in seconds.
        retina_embedding: Latest retina code embedding vector.
        audio_level: Current audio level in [0, 1].
        cursor_position: Current cursor (x, y) position.
        pressed_keys: List of currently pressed key names.
        object_slots: List of active object slot dicts.
        goal_active: Whether a goal is currently set.
        goal_embedding: Current goal embedding vector.
        goal_type: String name of the active goal type.
    """

    version: int
    timestamp: float
    retina_embedding: np.ndarray | None = None
    audio_level: float = 0.0
    cursor_position: tuple[float, float] = (0.5, 0.5)
    pressed_keys: list[str] = field(default_factory=list)
    object_slots: list[dict] = field(default_factory=list)
    goal_active: bool = False
    goal_embedding: np.ndarray | None = None
    goal_type: str = ""


@dataclass
class HotStateAssembler:
    """Assembles all state modules into a unified HotState snapshot.

    Holds references to all state modules and produces a combined
    HotState snapshot and a flattened vector for policy input.

    Attributes:
        frame_buffer: Ring buffer for recent frames.
        audio_buffer: Ring buffer for audio chunks.
        input_state: Buffered cursor and key state.
        retina: Frame-to-embedding encoder neural network.
        object_slots: Object slot buffer.
        goal_token: Active goal representation.
    """

    frame_buffer: FrameBuffer = field(default_factory=FrameBuffer)
    audio_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    input_state: InputStateBuffer = field(default_factory=InputStateBuffer)
    retina: RetinaEncoderNet = field(default_factory=RetinaEncoderNet)
    object_slots: ObjectSlotBuffer = field(default_factory=ObjectSlotBuffer)
    goal_token: GoalToken = field(default_factory=GoalToken)

    _version: int = field(default=0, init=False, repr=False)
    _last_timestamp: float = field(default=0.0, init=False, repr=False)

    def update_from(
        self,
        frame: np.ndarray | None = None,
        audio_chunk: np.ndarray | None = None,
        cursor_pos: tuple[float, float] | None = None,
        keys: list[str] | None = None,
    ) -> None:
        """Feed new sensory data into the state system.

        Args:
            frame: Optional new RGB frame.
            audio_chunk: Optional new audio chunk.
            cursor_pos: Optional (x, y) normalized cursor position.
            keys: Optional list of currently pressed key names.
        """
        ...

    def assemble(self) -> HotState:
        """Assemble all state pieces into a unified HotState snapshot.

        Returns:
            HotState containing all current slot values: version, timestamp,
            retina_embedding, audio_level, cursor, keys, object_slots,
            goal_active, goal_embedding, and goal_type.
        """
        ...

    def as_vector(self) -> np.ndarray:
        """Get a flattened state vector for policy input.

        Returns:
            1D float32 numpy array.
        """
        ...

    @property
    def version(self) -> int:
        """Monotonic update counter."""
        ...
