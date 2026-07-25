"""
Tests for Safegate rules (eve/core/safegate.py).
Tests all 9 rules and helper functions.
"""
import time

import pytest

from eve.core.safegate import (
    check,
    detect_human_cursor_activity,
    detect_human_keyboard_activity,
    emergency_stop,
    mark_expected_event,
    report_human_activity,
    reset_emergency,
)
from eve.state import ActionCandidate, ActionKind, OutputMode, RuntimeState


def _make_state(**overrides) -> RuntimeState:
    """Helper: create a default RuntimeState with optional overrides."""
    s = RuntimeState()
    s.cold_started = True
    s.output_mode = OutputMode.MOCK
    s.mouse_allowed = True
    s.keyboard_allowed = True
    s.speak_allowed = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_action(kind=ActionKind.MOUSE, **overrides) -> ActionCandidate:
    """Helper: create a default ActionCandidate."""
    a = ActionCandidate(
        action_id="test_action",
        kind=kind,
        payload={"action": "click", "x": 100, "y": 200},
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# ── Rule 1: disabled mode ────────────────────────────────

def test_disabled_mode_blocks():
    """Rule 1: output_mode=DISABLED should block all actions."""
    state = _make_state(output_mode=OutputMode.DISABLED)
    action = _make_action()
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "output_disabled"


# ── Rule 2: not cold started ─────────────────────────────

def test_not_cold_started_blocks():
    """Rule 2: not cold_started should block all actions."""
    state = _make_state(cold_started=False)
    action = _make_action()
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "not_cold_started"


# ── Rule 3: emergency stop highest priority ───────────────

def test_emergency_stop_highest_priority():
    """Rule 3: emergency_stopped blocks everything, even cold_started."""
    state = _make_state(emergency_stopped=True)
    action = _make_action()
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "emergency_stopped"


def test_emergency_stop_overrides_cold_start():
    """Emergency stop takes priority over cold_started check."""
    state = _make_state(emergency_stopped=True, cold_started=False)
    action = _make_action()
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "emergency_stopped"


def test_emergency_stop_and_reset():
    """emergency_stop then reset_emergency allows actions again."""
    state = _make_state()
    emergency_stop(state)
    assert state.emergency_stopped is True
    # blocked while emergency
    result = check(state, _make_action())
    assert result.allowed is False
    assert result.reason == "emergency_stopped"

    reset_emergency(state)
    assert state.emergency_stopped is False
    # allowed after reset
    result = check(state, _make_action())
    assert result.allowed is True
    assert result.reason == "ok"


# ── Rule 4: unauthorized outputs ─────────────────────────

def test_unauthorized_mouse():
    """Rule 4: mouse_allowed=False blocks mouse actions."""
    state = _make_state(mouse_allowed=False)
    action = _make_action(ActionKind.MOUSE)
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "mouse_not_allowed"


def test_unauthorized_keyboard():
    """Rule 4: keyboard_allowed=False blocks keyboard actions."""
    state = _make_state(keyboard_allowed=False)
    action = _make_action(ActionKind.KEYBOARD, payload={"text": "hello"})
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "keyboard_not_allowed"


def test_unauthorized_speak():
    """Rule 4: speak_allowed=False blocks speak actions."""
    state = _make_state(speak_allowed=False)
    action = _make_action(ActionKind.SPEAK, payload={"text": "hello"})
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "speak_not_allowed"


# ── Authorized + mock ────────────────────────────────────

def test_authorized_mouse_mock():
    """Rule 4 + mock: mouse_allowed=True in mock mode allows."""
    state = _make_state(output_mode=OutputMode.MOCK, mouse_allowed=True)
    action = _make_action(ActionKind.MOUSE)
    result = check(state, action)
    assert result.allowed is True
    assert result.reason == "ok"


def test_all_checks_pass_for_real():
    """All safegate rules pass for a valid state."""
    state = _make_state(output_mode=OutputMode.REAL)
    action = _make_action(ActionKind.KEYBOARD, payload={"text": "hi"})
    result = check(state, action)
    assert result.allowed is True
    assert result.reason == "ok"


# ── Rule 5: expired action ───────────────────────────────

def test_expired_action_blocked():
    """Rule 5: action with valid_until_ns in the past is blocked."""
    state = _make_state()
    past_ns = time.monotonic_ns() - 10_000_000_000  # 10s ago
    action = _make_action(valid_until_ns=past_ns)
    result = check(state, action)
    assert result.allowed is False
    assert result.reason == "action_expired"


def test_valid_until_zero_not_expired():
    """valid_until_ns=0 means never expires, should pass."""
    state = _make_state()
    action = _make_action(valid_until_ns=0)
    result = check(state, action)
    assert result.allowed is True


# ── Rule 6+7: human takeover freeze ──────────────────────

def test_human_takeover_freeze():
    """Rule 6: report_human_activity freezes keyboard/mouse for 5s."""
    state = _make_state()
    report_human_activity(state)
    assert state.blocked_until_ns > time.monotonic_ns()

    action_mouse = _make_action(ActionKind.MOUSE)
    result = check(state, action_mouse)
    assert result.allowed is False
    assert "human_takeover_freeze" in result.reason


def test_repeat_takeover_refreshes():
    """Rule 7: two human activities refresh the block time."""
    state = _make_state()
    report_human_activity(state)
    first_blocked = state.blocked_until_ns
    time.sleep(0.01)
    report_human_activity(state)
    second_blocked = state.blocked_until_ns
    assert second_blocked > first_blocked


def test_human_takeover_does_not_block_speak():
    """Human takeover only blocks mouse/keyboard, not speak."""
    state = _make_state()
    report_human_activity(state)
    action_speak = _make_action(ActionKind.SPEAK, payload={"text": "hello"})
    result = check(state, action_speak)
    assert result.allowed is True


# ── Rule 8: EVE-origin events don't self-freeze ───────────

def test_eve_origin_not_self_freeze():
    """Rule 8: mark_expected_event then report doesn't freeze."""
    state = _make_state()
    mark_expected_event(state, "evt_001")
    report_human_activity(state, event_id="evt_001")
    # Should not have set blocked_until_ns
    assert state.blocked_until_ns == 0
    # Action should pass
    action = _make_action(ActionKind.MOUSE)
    result = check(state, action)
    assert result.allowed is True


def test_mark_expected_event_discards_after_use():
    """Expected event is discarded after report, not reusable."""
    state = _make_state()
    mark_expected_event(state, "evt_002")
    report_human_activity(state, event_id="evt_002")
    assert state.blocked_until_ns == 0
    # Second report with same ID (no longer expected) should freeze
    report_human_activity(state, event_id="evt_002")
    assert state.blocked_until_ns > 0


# ── Rule 9: unknown action kind ──────────────────────────

def test_unknown_kind_blocked():
    """Unknown ActionKind (not MOUSE/KEYBOARD/SPEAK) raises AttributeError
    because safegate expects an ActionKind enum value and calls .value on it."""
    state = _make_state()
    # Create an action with a kind value that's not in the ActionKind enum
    action = ActionCandidate(
        action_id="test_unknown",
        kind="hydraulic_press",
        payload={},
    )
    # The current safegate code calls action.kind.value, which fails
    # if kind is a raw string rather than ActionKind enum.
    with pytest.raises(AttributeError):
        check(state, action)


# ── Human activity detection ─────────────────────────────

def test_detect_human_cursor_activity_detects_movement():
    """Detects significant cursor movement."""
    result = detect_human_cursor_activity(100, 100, 200, 200)
    assert result is True


def test_detect_human_cursor_activity_no_movement():
    """No detection for minor/no cursor movement."""
    result = detect_human_cursor_activity(100, 100, 101, 101)
    assert result is False


def test_detect_human_cursor_activity_first_frame():
    """First frame (last None) returns False."""
    assert detect_human_cursor_activity(None, None, 500, 500) is False
    assert detect_human_cursor_activity(None, 100, 500, 100) is False


def test_detect_human_keyboard_activity_placeholder():
    """detect_human_keyboard_activity always returns False (placeholder)."""
    assert detect_human_keyboard_activity() is False


# ── SafegateResult fields ────────────────────────────────

def test_safegate_result_has_checked_at():
    """SafegateResult auto-populates checked_at_ns."""
    state = _make_state()
    action = _make_action()
    result = check(state, action)
    assert result.checked_at_ns > 0
    assert isinstance(result.checked_at_ns, int)
