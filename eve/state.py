"""
EVE 运行时状态与数据结构。

包含 ActionCandidate、SafegateResult、OutputResult、RuntimeState、
以及 Phase 2-4 新增的 WorldState、MyselfState、Blackboard、TimedEntry。
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
    REAL = "real"


class ActionKind(str, Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    SPEAK = "speak"


# ── 核心数据结构 ──────────────────────────────────────────


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

    executed:  电脑上真实发生了动作（real 模式为 True）
    simulated: Mock 后端处理了动作但未真实执行
    三种状态：
      disabled/阻断 → executed=False, simulated=False
      mock          → executed=False, simulated=True
      real          → executed=True,  simulated=False
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
    """EVE 运行时状态。"""
    cold_started: bool = False
    emergency_stopped: bool = False
    output_mode: OutputMode = OutputMode.DISABLED
    mouse_allowed: bool = False
    keyboard_allowed: bool = False
    speak_allowed: bool = False
    blocked_until_ns: int = 0
    # EVE 预期产生的鼠标/键盘事件 ID 集合（用于区分 EVE-origin 与人类接管）
    eve_expected_events: set[str] = field(default_factory=set)
    # 当前假 state（用一个简单 dict 占位）
    current_state: dict[str, Any] = field(default_factory=dict)
    # pending 动作
    pending_action: ActionCandidate | None = None
    # 最近一次输出结果
    latest_output: OutputResult | None = None
    # Phase 2-4: 人类活动检测
    human_activity_detected_at_ns: int = 0


# ── Phase 2-4 新增 ────────────────────────────────────────


@dataclass
class TimedEntry:
    """Blackboard 中的一项结果。"""
    entry_id: str
    kind: str          # 结果类型标识
    producer: str      # 谁产生的
    produced_at_ns: int
    valid_until_ns: int  # 0 = 永不过期
    payload: Any = None

    def __post_init__(self):
        if self.produced_at_ns == 0:
            self.produced_at_ns = time.monotonic_ns()


@dataclass
class WorldState:
    """EVE 对外部世界的认识。"""
    scene: str = ""
    sub_scene: str = ""
    active_window: str = ""
    visible_objects: list[str] = field(default_factory=list)
    detected_text: str = ""
    visual_results: dict[str, Any] = field(default_factory=dict)
    uncertainty: str = ""
    updated_at_ns: int = 0


@dataclass
class MyselfState:
    """EVE 对自身状态的认识。"""
    what_im_thinking: str = ""
    current_task: str = ""
    task_progress: str = ""
    loaded_tnn: list[str] = field(default_factory=list)
    available_tnn_summary: list[str] = field(default_factory=list)
    resource_status: dict[str, Any] = field(default_factory=dict)
    hormone_levels: dict[str, float] = field(default_factory=dict)
    tendencies: dict[str, float] = field(default_factory=dict)
    control_summary: str = ""
    updated_at_ns: int = 0


@dataclass
class Blackboard:
    """不同频率节点交换临时结果的共享区。"""
    entries: dict[str, list[TimedEntry]] = field(default_factory=dict)

    def write(self, entry: TimedEntry) -> None:
        """写入一项结果。"""
        if entry.kind not in self.entries:
            self.entries[entry.kind] = []
        self.entries[entry.kind].append(entry)

    def read(self, kind: str, producer: str | None = None) -> list[TimedEntry]:
        """读取指定类型的结果，自动清除过期项。"""
        entries = self.entries.get(kind, [])
        now_ns = time.monotonic_ns()
        result: list[TimedEntry] = []
        kept: list[TimedEntry] = []
        for e in entries:
            expired = e.valid_until_ns > 0 and now_ns > e.valid_until_ns
            if expired:
                continue
            kept.append(e)
            if producer is None or e.producer == producer:
                result.append(e)
        self.entries[kind] = kept
        return result

    def latest(self, kind: str) -> TimedEntry | None:
        """获取最新不超期的结果。"""
        entries = self.read(kind)
        return entries[-1] if entries else None

    def clear_expired(self) -> None:
        """清除所有过期项。"""
        now_ns = time.monotonic_ns()
        for kind in list(self.entries.keys()):
            kept = [
                e for e in self.entries[kind]
                if e.valid_until_ns == 0 or now_ns <= e.valid_until_ns
            ]
            if kept:
                self.entries[kind] = kept
            else:
                del self.entries[kind]
