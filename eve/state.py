"""
EVE Phase 1 最小运行时状态与数据结构。

只包含本阶段需要的字段，不引入 world/myself/blackboard/Memory/TNN/激素。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── 枚举 ──────────────────────────────────────────────────


class OutputMode(str, Enum):
    DISABLED = "disabled"
    MOCK = "mock"


class ActionKind(str, Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    SPEAK = "speak"


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class ActionCandidate:
    """最小动作候选。"""
    action_id: str
    kind: ActionKind
    payload: dict[str, Any] = field(default_factory=dict)
    created_at_ns: int = 0
    valid_until_ns: int = 0
    origin: str = ""

    def __post_init__(self):
        if self.created_at_ns == 0:
            self.created_at_ns = time.monotonic_ns()


@dataclass
class SafegateResult:
    """Safegate 判定结果。"""
    allowed: bool
    reason: str
    checked_at_ns: int = 0

    def __post_init__(self):
        if self.checked_at_ns == 0:
            self.checked_at_ns = time.monotonic_ns()


@dataclass
class OutputResult:
    """输出执行结果。

    executed:  电脑上真实发生了动作（仅未来 real 模式为 True）
    simulated: Mock 后端处理了动作但未真实执行
    三种状态：
      disabled/阻断 → executed=False, simulated=False
      mock          → executed=False, simulated=True
      未来real      → executed=True,  simulated=False
    """
    action_id: str
    kind: str
    mode: str
    started_at_ns: int = 0
    finished_at_ns: int = 0
    executed: bool = False
    simulated: bool = False
    blocked: bool = False
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeState:
    """Phase 1 最小运行时状态。"""
    cold_started: bool = False
    emergency_stopped: bool = False
    output_mode: OutputMode = OutputMode.DISABLED
    mouse_allowed: bool = False
    keyboard_allowed: bool = False
    speak_allowed: bool = False
    blocked_until_ns: int = 0
    # EVE 预期产生的鼠标/键盘事件 ID 集合（用于区分 EVE-origin 与人类接管）
    eve_expected_events: set[str] = field(default_factory=set)
    # 当前假 state（Phase 1 阶段用一个简单 dict 占位）
    current_state: dict[str, Any] = field(default_factory=dict)
    # pending 动作
    pending_action: ActionCandidate | None = None
    # 最近一次输出结果
    latest_output: OutputResult | None = None
