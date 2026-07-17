"""Phase 1 Safegate 测试。

覆盖：
- 默认 disabled
- 未 cold start
- 无权限
- mock 允许
- emergency stop
- 动作过期
- 五秒冻结
- 重复冻结刷新
- EVE-origin 不自我冻结
"""
from __future__ import annotations

import time

import pytest

from eve.core import safegate
from eve.state import ActionCandidate, ActionKind, OutputMode, RuntimeState


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def state() -> RuntimeState:
    """默认 disabled 的 RuntimeState。"""
    return RuntimeState()


@pytest.fixture
def ready_state() -> RuntimeState:
    """mock 模式、全部授权、已 cold start 的 RuntimeState。"""
    s = RuntimeState()
    s.cold_started = True
    s.output_mode = OutputMode.MOCK
    s.mouse_allowed = True
    s.keyboard_allowed = True
    s.speak_allowed = True
    return s


@pytest.fixture
def mouse_action() -> ActionCandidate:
    return ActionCandidate(
        action_id="test-mouse-001",
        kind=ActionKind.MOUSE,
        payload={"x": 100, "y": 200},
        origin="test",
    )


@pytest.fixture
def keyboard_action() -> ActionCandidate:
    return ActionCandidate(
        action_id="test-key-001",
        kind=ActionKind.KEYBOARD,
        payload={"key": "a"},
        origin="test",
    )


@pytest.fixture
def speak_action() -> ActionCandidate:
    return ActionCandidate(
        action_id="test-speak-001",
        kind=ActionKind.SPEAK,
        payload={"text": "hello"},
        origin="test",
    )


# ── 规则 1: 默认 disabled ─────────────────────────────────


def test_default_disabled(state: RuntimeState, mouse_action: ActionCandidate):
    """规则 1: 默认 disabled 模式，所有动作应被阻断。"""
    result = safegate.check(state, mouse_action)
    assert result.allowed is False
    # 未 cold start 也会阻断，但 disabled 检查在前（规则 1 在规则 2 之前）
    # 实际上我们的代码中 emergency_stop → not_cold_started → disabled
    # 因为 state.cold_started=False 所以先命中规则 2
    assert "not_cold_started" in result.reason


def test_disabled_even_after_cold_start(state: RuntimeState, mouse_action: ActionCandidate):
    """默认 disabled，即使 cold start 后也应阻断。"""
    state.cold_started = True
    result = safegate.check(state, mouse_action)
    assert result.allowed is False
    assert result.reason == "output_disabled"


# ── 规则 2: 未 cold start ─────────────────────────────────


def test_not_cold_started(state: RuntimeState, mouse_action: ActionCandidate):
    """规则 2: 未 cold start 时不允许动作。"""
    state.output_mode = OutputMode.MOCK
    state.mouse_allowed = True
    result = safegate.check(state, mouse_action)
    assert result.allowed is False
    assert result.reason == "not_cold_started"


# ── 规则 3: emergency stop ─────────────────────────────────


def test_emergency_stop(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """规则 3: emergency stop 优先级最高。"""
    safegate.emergency_stop(ready_state)
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is False
    assert result.reason == "emergency_stopped"


def test_emergency_stop_beats_all(ready_state: RuntimeState):
    """emergency stop 阻断所有类型动作。"""
    safegate.emergency_stop(ready_state)
    for kind in ActionKind:
        action = ActionCandidate(action_id=f"test-{kind.value}", kind=kind, origin="test")
        result = safegate.check(ready_state, action)
        assert result.allowed is False
        assert result.reason == "emergency_stopped"


# ── 规则 4: 对应输出未授权 ─────────────────────────────────


def test_mouse_not_allowed(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """规则 4: 鼠标未授权时阻断。"""
    ready_state.mouse_allowed = False
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is False
    assert "mouse_not_allowed" in result.reason


def test_keyboard_not_allowed(ready_state: RuntimeState, keyboard_action: ActionCandidate):
    """规则 4: 键盘未授权时阻断。"""
    ready_state.keyboard_allowed = False
    result = safegate.check(ready_state, keyboard_action)
    assert result.allowed is False
    assert "keyboard_not_allowed" in result.reason


def test_speak_not_allowed(ready_state: RuntimeState, speak_action: ActionCandidate):
    """规则 4: 语音未授权时阻断。"""
    ready_state.speak_allowed = False
    result = safegate.check(ready_state, speak_action)
    assert result.allowed is False
    assert "speak_not_allowed" in result.reason


# ── 规则 5: 动作过期 ─────────────────────────────────────


def test_action_expired(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """规则 5: 超过 valid_until_ns 的动作应被阻断。"""
    mouse_action.valid_until_ns = time.monotonic_ns() - 1  # 已过期
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is False
    assert result.reason == "action_expired"


def test_action_not_expired(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """未过期的动作应通过。"""
    mouse_action.valid_until_ns = time.monotonic_ns() + 60_000_000_000  # 未来 60 秒
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is True


# ── 规则 6+7: 人类接管冻结 ──────────────────────────────────


def test_human_takeover_freeze_mouse(
    ready_state: RuntimeState, mouse_action: ActionCandidate
):
    """规则 6: 人类活动后冻结鼠标输出。"""
    safegate.report_human_activity(ready_state)
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is False
    assert "human_takeover_freeze" in result.reason


def test_human_takeover_freeze_keyboard(
    ready_state: RuntimeState, keyboard_action: ActionCandidate
):
    """规则 6: 人类活动后冻结键盘输出。"""
    safegate.report_human_activity(ready_state)
    result = safegate.check(ready_state, keyboard_action)
    assert result.allowed is False
    assert "human_takeover_freeze" in result.reason


def test_human_takeover_not_affect_speak(
    ready_state: RuntimeState, speak_action: ActionCandidate
):
    """规则 6: 人类接管只冻结键鼠，不冻结语音。"""
    safegate.report_human_activity(ready_state)
    result = safegate.check(ready_state, speak_action)
    assert result.allowed is True


def test_freeze_refresh_on_repeat(
    ready_state: RuntimeState, mouse_action: ActionCandidate
):
    """规则 7: 重复人类活动刷新 blocked_until_ns。"""
    safegate.report_human_activity(ready_state)
    first_block = ready_state.blocked_until_ns
    time.sleep(0.01)  # 确保时间差
    safegate.report_human_activity(ready_state)
    second_block = ready_state.blocked_until_ns
    assert second_block > first_block, "重复活动应刷新冻结时间"


# ── 规则 8: EVE-origin 不自我冻结 ──────────────────────────────


def test_eve_origin_no_self_freeze(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """规则 8: EVE 预期的事件不触发冻结。"""
    evt_id = "eve-expected-move-001"
    safegate.mark_expected_event(ready_state, evt_id)
    safegate.report_human_activity(ready_state, event_id=evt_id)
    # 因为事件被标记为 EVE-origin，不应设置冻结
    assert ready_state.blocked_until_ns == 0
    # 且事件应从预期集合中移除
    assert evt_id not in ready_state.eve_expected_events


def test_unknown_event_triggers_freeze(ready_state: RuntimeState):
    """未标记的事件应触发冻结。"""
    safegate.report_human_activity(ready_state, event_id="unknown-event-001")
    assert ready_state.blocked_until_ns > 0


# ── mock 允许 ─────────────────────────────────────────────


def test_mock_allowed(ready_state: RuntimeState, mouse_action: ActionCandidate):
    """mock 模式 + 授权 + 未冻结 → 应允许。"""
    result = safegate.check(ready_state, mouse_action)
    assert result.allowed is True
    assert result.reason == "ok"


# ── 回归: 新测试不导入 src.eve ──────────────────────────────


def test_no_src_eve_import() -> None:
    """确保 safegate 测试模块不导入 src.eve。"""
    import sys
    assert "src.eve" not in sys.modules, "test_safegate 不应导入 src.eve"
