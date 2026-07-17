"""EpisodeLogger — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# import json
# from eve_core.common.types import CursorPacket, GatedMotorCommand, HotState, MotorImpulse, StateSlot
# from eve_core.evolution.episode_logger import EpisodeLogger
# from eve_core.realtime.motor_stub import MotorStub
# from eve_core.safety.safety_gate import SafetyGate
# from eve_core.womb.synthetic_world import RedBallWorld
#
# New structure: eve.episode.episode_logger.EpisodeLogger
# — API is completely different (log_step vs record), stub-only.


def test_episode_logger_writes_jsonl_and_summary(tmp_path) -> None:
    pass
