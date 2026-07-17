"""Confidence cascade — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.common.types import CursorPacket, GoalToken, HotState, StateSlot
# from eve_core.realtime.confidence import clamp_confidence, synthetic_detection_confidence
# from eve_core.realtime.detector_stub import RedBallDetectorStub
# from eve_core.realtime.policy_stub import VisualServoPolicyStub
#
# New structure: eve.state.hot_state.HotState, eve.state.goal_token.GoalToken
# — all stub-only, no confidence helpers, no detector/policy stubs.


def test_confidence_helpers_clamp_values() -> None:
    pass


def test_low_confidence_causes_hold() -> None:
    pass


def test_high_confidence_allows_move() -> None:
    pass


def test_detector_confidence_cascade_to_policy_hold() -> None:
    pass
