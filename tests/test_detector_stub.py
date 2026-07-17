"""RedBallDetectorStub — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.realtime.detector_stub import RedBallDetectorStub
# from eve_core.womb.synthetic_world import RedBallWorld
#
# New structure: eve.state.object_slots.ObjectSlots has synthetic detection
# but completely different API. No RedBallWorld or RedBallDetectorStub.


def test_detector_returns_red_ball_center_slot() -> None:
    pass
