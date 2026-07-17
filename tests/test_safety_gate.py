"""Safety gate — adapted to InhibitionGate API, stub-only."""

import pytest

pytest.skip("InhibitionGate.allow() is stub-only — remove this skip once implemented", allow_module_level=True)

# New API adapted test — ready when stubs are filled in:
# from eve.inhibition.inhibition_gate import InhibitionGate
# from eve.inhibition.permission_manager import PermissionManager


def test_inhibition_gate_allows_only_safe_synthetic_actions() -> None:
    gate = InhibitionGate()
    perm_mgr = PermissionManager()

    # move and noop are safe commands — they should pass the gate
    move_result = gate.allow([{"action": "move", "dx": 1.0}], perm_mgr)
    noop_result = gate.allow([{"action": "noop"}], perm_mgr)
    assert len(move_result) == 1  # move allowed
    assert len(noop_result) == 1  # noop allowed

    # click and type are unsafe — they should be blocked
    click_result = gate.allow([{"action": "click"}], perm_mgr)
    type_result = gate.allow([{"action": "type", "text": "x"}], perm_mgr)
    assert len(click_result) == 0  # click blocked
    assert len(type_result) == 0  # type blocked

    assert gate.get_blocked_count() == 2
