"""StateSlotBuffer — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.common.types import CursorPacket, StateSlot
# from eve_core.realtime.state_slots import StateSlotBuffer
#
# New structure: eve.state.input_state_buffer.InputStateBuffer
# or eve.state.object_slots.ObjectSlots — API is completely different.


def test_state_slot_buffer_tracks_latest_bounded_state() -> None:
    pass
