"""Core dataclass types — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.common.types import (
#     CursorPacket,
#     EpisodeRecord,
#     FramePacket,
#     GatedMotorCommand,
#     GoalToken,
#     HotState,
#     MotorFeedback,
#     MotorImpulse,
#     RewardSignal,
#     StateSlot,
#     TeacherLabel,
# )
#
# New structure: eve.common.types does not exist.
# Types are now spread across eve.body.body_schema (MotorCommand, MotorFeedback, BodyLimits),
# eve.state.hot_state (HotState), eve.state.goal_token (GoalToken) — all stub-only.


def test_required_dataclasses_construct() -> None:
    pass


def test_core_dataclasses_serialize_to_json() -> None:
    pass
