"""RedBallWorld synthetic environment — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.womb.synthetic_world import RedBallWorld
#
# New structure: No RedBallWorld in eve. The synthetic world module
# (eve_core.womb) does not exist in the new package structure.


def test_red_ball_world_is_synthetic_and_cursor_moves_toward_delta() -> None:
    pass


def test_reset_is_deterministic() -> None:
    pass


def test_reset_with_different_seed_produces_different_state() -> None:
    pass
