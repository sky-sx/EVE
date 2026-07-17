"""MotorStub — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.common.types import GatedMotorCommand, MotorImpulse
# from eve_core.realtime.motor_stub import MotorStub
# from eve_core.womb.synthetic_world import RedBallWorld
#
# New structure: eve.body.mouse_motor.MouseMotor, eve.body.body_schema.MotorCommand
# — completely different API, no MotorStub, no GatedMotorCommand or MotorImpulse types.


def test_motor_stub_updates_only_synthetic_cursor() -> None:
    pass
