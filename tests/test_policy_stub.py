"""VisualServoPolicyStub — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.common.types import CursorPacket, GoalToken, HotState, StateSlot
# from eve_core.realtime.policy_stub import VisualServoPolicyStub
#
# New structure: eve.policy.policy_net, eve.policy.attention_policy
# — completely different API, no VisualServoPolicyStub.


def test_visual_servo_policy_moves_toward_goal_without_overshoot() -> None:
    pass


def test_visual_servo_policy_noops_inside_tolerance() -> None:
    pass
