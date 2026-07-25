"""
EVE Safegate — 最小安全门。

规则：
1. 默认 disabled。
2. 未 cold start 时不允许动作。
3. Esc/emergency stop 优先级最高。
4. 对应输出未授权时阻断。
5. 动作超过 valid_until_ns 时阻断。
6. 人类鼠标或键盘活动后，键鼠输出冻结 5 秒。
7. 重复人类活动刷新 blocked_until_ns。
8. EVE-origin 的预期事件不触发自我冻结。
9. Safegate 只做最低安全判断，不做任务理解。
"""
from __future__ import annotations

import time

from eve.state import (
    ActionCandidate,
    ActionKind,
    OutputMode,
    RuntimeState,
    SafegateResult,
)

_FREEZE_NS = 5 * 1_000_000_000  # 5 秒冻结（纳秒）


def check(state: RuntimeState, action: ActionCandidate) -> SafegateResult:
    """对单个 ActionCandidate 执行 Safegate 判定。"""

    # ── 规则 3: emergency stop 优先级最高 ──
    if state.emergency_stopped:
        return SafegateResult(allowed=False, reason="emergency_stopped")

    # ── 规则 2: 未 cold start ──
    if not state.cold_started:
        return SafegateResult(allowed=False, reason="not_cold_started")

    # ── 规则 1: disabled 模式 ──
    if state.output_mode == OutputMode.DISABLED:
        return SafegateResult(allowed=False, reason="output_disabled")

    # ── 规则 4: 对应输出未授权 ──
    if not _output_authorized(state, action.kind):
        return SafegateResult(allowed=False, reason=f"{action.kind.value}_not_allowed")

    # ── 规则 6+7: 人类接管冻结（仅键鼠） ──
    if action.kind in (ActionKind.MOUSE, ActionKind.KEYBOARD):
        now_ns = time.monotonic_ns()
        if now_ns < state.blocked_until_ns:
            remaining = (state.blocked_until_ns - now_ns) / 1e9
            return SafegateResult(
                allowed=False,
                reason=f"human_takeover_freeze_remaining_{remaining:.1f}s",
            )

    # ── 规则 5: 动作过期 ──
    if action.valid_until_ns > 0:
        now_ns = time.monotonic_ns()
        if now_ns > action.valid_until_ns:
            return SafegateResult(allowed=False, reason="action_expired")

    # ── 所有检查通过 ──
    return SafegateResult(allowed=True, reason="ok")


def mark_expected_event(state: RuntimeState, event_id: str) -> None:
    """登记 EVE 预期产生的鼠标/键盘事件，用于规则 8。"""
    state.eve_expected_events.add(event_id)


def report_human_activity(
    state: RuntimeState, event_id: str | None = None
) -> None:
    """
    报告人类鼠标/键盘活动。

    规则 8: 如果 event_id 在 EVE 预期事件集合中，不触发冻结。
    规则 6+7: 否则冻结键鼠输出 5 秒，重复活动刷新冻结时间。
    """
    # 规则 8: EVE-origin 不自我冻结
    if event_id and event_id in state.eve_expected_events:
        state.eve_expected_events.discard(event_id)
        return

    # 规则 6+7: 冻结 5 秒，刷新
    state.blocked_until_ns = time.monotonic_ns() + _FREEZE_NS


def emergency_stop(state: RuntimeState) -> None:
    """规则 3: 触发急停。"""
    state.emergency_stopped = True


def reset_emergency(state: RuntimeState) -> None:
    """重置急停状态（仅用于测试）。"""
    state.emergency_stopped = False


# ── 人类活动检测 ──────────────────────────────────────────


def detect_human_cursor_activity(
    last_x: int | None,
    last_y: int | None,
    current_x: int,
    current_y: int,
    threshold: int = 50,
) -> bool:
    """
    检测两次光标采样之间是否有显著的人类移动。

    若 last_x/last_y 为 None 则返回 False（首帧不判定）。
    """
    if last_x is None or last_y is None:
        return False
    dx = current_x - last_x
    dy = current_y - last_y
    return (dx * dx + dy * dy) > (threshold * threshold)


# ── 内部 ──────────────────────────────────────────────────


def _output_authorized(state: RuntimeState, kind: ActionKind) -> bool:
    if kind == ActionKind.MOUSE:
        return state.mouse_allowed
    if kind == ActionKind.KEYBOARD:
        return state.keyboard_allowed
    if kind == ActionKind.SPEAK:
        return state.speak_allowed
    return False
