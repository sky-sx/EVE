"""EVE 六大激素系统。"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field


# ── 激素数据结构 ──


@dataclass
class HormoneLevels:
    """六大激素当前浓度 (0.0 ~ 1.0)。"""
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5
    oxytocin: float = 0.5
    cortisol: float = 0.5
    acetylcholine: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "oxytocin": self.oxytocin,
            "cortisol": self.cortisol,
            "acetylcholine": self.acetylcholine,
        }

    def summary(self) -> str:
        """人类可读的激素状态摘要。"""
        parts: list[str] = []
        if self.dopamine > 0.7:
            parts.append(f"DOP high ({self.dopamine:.2f})")
        elif self.dopamine < 0.3:
            parts.append(f"DOP low ({self.dopamine:.2f})")
        if self.serotonin > 0.7:
            parts.append(f"SER high ({self.serotonin:.2f})")
        elif self.serotonin < 0.3:
            parts.append(f"SER low ({self.serotonin:.2f})")
        if self.norepinephrine > 0.7:
            parts.append(f"NOR high ({self.norepinephrine:.2f})")
        elif self.norepinephrine < 0.3:
            parts.append(f"NOR low ({self.norepinephrine:.2f})")
        if self.oxytocin > 0.7:
            parts.append(f"OXY high ({self.oxytocin:.2f})")
        if self.cortisol > 0.7:
            parts.append(f"CORT high ({self.cortisol:.2f})")
        elif self.cortisol < 0.3:
            parts.append(f"CORT low ({self.cortisol:.2f})")
        if self.acetylcholine > 0.7:
            parts.append(f"ACH high ({self.acetylcholine:.2f})")
        if not parts:
            parts.append("all balanced")
        return " | ".join(parts)


@dataclass
class HormoneEvent:
    """激素变化事件。"""
    hormone: str
    delta: float  # positive or negative
    source: str    # "success" | "failure" | "user_praise" | "user_critique" | "risk" | "novelty" | "safegate_block" | "resource_pressure" | "repeated_failure" | "stable" | "sleep_debt"
    timestamp_ns: int = 0
    description: str = ""

    def __post_init__(self):
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.monotonic_ns()


# ── 事件类型 → 激素映射 ──

_EVENT_HORMONE_MAP: dict[str, list[tuple[str, float]]] = {
    "success": [("dopamine", 0.1), ("serotonin", 0.05)],
    "failure": [("cortisol", 0.1), ("dopamine", -0.05), ("serotonin", -0.05)],
    "user_praise": [("dopamine", 0.15), ("oxytocin", 0.1), ("serotonin", 0.05)],
    "user_critique": [("cortisol", 0.1), ("serotonin", -0.05)],
    "risk": [("norepinephrine", 0.1), ("cortisol", 0.1)],
    "novelty": [("acetylcholine", 0.1), ("dopamine", 0.05)],
    "safegate_block": [("cortisol", 0.05), ("norepinephrine", 0.05)],
    "resource_pressure": [("cortisol", 0.1), ("norepinephrine", 0.05)],
    "repeated_failure": [("cortisol", 0.15), ("serotonin", -0.1)],
    "stable": [("serotonin", 0.05), ("cortisol", -0.05)],
    "sleep_debt": [("cortisol", 0.05), ("acetylcholine", -0.05)],
    "task_complete": [("dopamine", 0.2), ("serotonin", 0.1)],
}

_HORMONE_NAMES = [
    "dopamine", "serotonin", "norepinephrine",
    "oxytocin", "cortisol", "acetylcholine",
]


# ── 激素管理器 ──


class HormoneManager:
    """六大激素的持续更新管理器。

    更新公式：new = old + event_delta + recovery + long_drift
    - recovery: 向基线(0.5)的自然恢复
    - long_drift: 长期状态的缓慢偏移
    - 所有激素限制在 [0.0, 1.0]
    """

    def __init__(self, recovery_rate: float = 0.01):
        self.levels = HormoneLevels()
        self.base_levels = HormoneLevels()  # 基线 = 0.5
        self.recovery_rate = recovery_rate  # 每轮恢复比例
        self.history: list[HormoneEvent] = []
        self._last_update_ns: int = 0

    def apply_event(self, event_type: str, intensity: float = 1.0,
                    description: str = "") -> None:
        """应用一个激素变化事件。

        event_type → 激素映射：
        - success → dopamine +0.1, serotonin +0.05
        - failure → cortisol +0.1, dopamine -0.05, serotonin -0.05
        - user_praise → dopamine +0.15, oxytocin +0.1, serotonin +0.05
        - user_critique → cortisol +0.1, serotonin -0.05
        - risk → norepinephrine +0.1, cortisol +0.1
        - novelty → acetylcholine +0.1, dopamine +0.05
        - safegate_block → cortisol +0.05, norepinephrine +0.05
        - resource_pressure → cortisol +0.1, norepinephrine +0.05
        - repeated_failure → cortisol +0.15, serotonin -0.1
        - stable → serotonin +0.05, cortisol -0.05
        - sleep_debt → cortisol +0.05, acetylcholine -0.05
        - task_complete → dopamine +0.2, serotonin +0.1

        intensity 是乘数，默认为 1.0。
        """
        deltas = _EVENT_HORMONE_MAP.get(event_type)
        if deltas is None:
            return

        for hormone_name, base_delta in deltas:
            effective_delta = base_delta * intensity
            current = getattr(self.levels, hormone_name)
            new_val = max(0.0, min(1.0, current + effective_delta))
            setattr(self.levels, hormone_name, new_val)

            event = HormoneEvent(
                hormone=hormone_name,
                delta=effective_delta,
                source=event_type,
                description=description,
            )
            self.history.append(event)

    def update_cycle(self) -> None:
        """执行一轮更新：
        1. 向基线自然恢复
        2. 剪切到 [0, 1]
        3. 记录更新时间
        """
        self._last_update_ns = time.monotonic_ns()
        for name in _HORMONE_NAMES:
            current = getattr(self.levels, name)
            baseline = getattr(self.base_levels, name)
            # recovery: 向基线方向移动 recovery_rate 比例的距离
            diff = baseline - current
            recovery = diff * self.recovery_rate
            new_val = current + recovery
            new_val = max(0.0, min(1.0, new_val))
            setattr(self.levels, name, new_val)

    def compute_llm_interval(self, min_s: float = 10.0, max_s: float = 20.0) -> float:
        """根据激素状态计算 LLM 大循环间隔。

        逻辑：
        - 高 norepinephrine(警觉) → 趋向 min_s
        - 高 cortisol(压力) → 趋向 min_s
        - 高 serotonin(稳定) → 趋向 max_s
        - 高 dopamine(任务驱动) → 趋向 min_s
        - 高 acetylcholine(学习) → 趋向 min_s

        在 [min_s, max_s] 之间线性映射。
        """
        # 计算一个 aggressiveness 分数：越高 → 越接近 min_s（更快循环）
        # dopamine, norepinephrine, cortisol, acetylcholine → 加速
        # serotonin → 减速
        push = (
            self.levels.dopamine * 0.25 +
            self.levels.norepinephrine * 0.25 +
            self.levels.cortisol * 0.20 +
            self.levels.acetylcholine * 0.20
        )
        pull = self.levels.serotonin * 0.35
        # aggressiveness: 0 = 偏向 max_s, 1 = 偏向 min_s
        aggressiveness = max(0.0, min(1.0, push - pull + 0.3))

        interval = max_s - aggressiveness * (max_s - min_s)
        return round(interval, 2)

    def get_tendencies(self) -> dict[str, float]:
        """从激素推导当前倾向。
        Returns: {"explore": ..., "exploit": ..., "pause": ..., "sleep": ...,
                  "active_output": ..., "think_more": ..., "train": ...}
        """
        da = self.levels.dopamine
        se = self.levels.serotonin
        ne = self.levels.norepinephrine
        ox = self.levels.oxytocin
        co = self.levels.cortisol
        ac = self.levels.acetylcholine

        # exploit: 继续当前路径 (dopamine + serotonin)
        exploit = max(0.0, min(1.0, (da + se) / 2.0 + 0.1))

        # explore: 探索新路径 (norepinephrine + acetylcholine)
        explore = max(0.0, min(1.0, (ne + ac) / 2.0))

        # pause: 暂停/休息 (cortisol)
        pause = max(0.0, min(1.0, co * 0.8))

        # sleep: 睡眠需求 (high cortisol + low dopamine)
        sleep = max(0.0, min(1.0, co * 0.6 - da * 0.3))

        # active_output: 主动输出 (dopamine + norepinephrine)
        active_output = max(0.0, min(1.0, (da + ne) / 2.0 + 0.1))

        # think_more: 深度思考 (acetylcholine + serotonin)
        think_more = max(0.0, min(1.0, (ac + se) / 2.0))

        # train: 训练倾向 (acetylcholine + dopamine + oxytocin)
        train = max(0.0, min(1.0, (ac + da + ox) / 3.0 + 0.05))

        return {
            "explore": round(explore, 3),
            "exploit": round(exploit, 3),
            "pause": round(pause, 3),
            "sleep": round(sleep, 3),
            "active_output": round(active_output, 3),
            "think_more": round(think_more, 3),
            "train": round(train, 3),
        }

    def get_recent_events(self, count: int = 10) -> list[HormoneEvent]:
        """返回最近的 N 个激素事件。"""
        return self.history[-count:] if self.history else []

    def save_snapshot(self) -> dict:
        """保存可恢复的状态快照。"""
        return {
            "levels": self.levels.to_dict(),
            "base_levels": self.base_levels.to_dict(),
            "recovery_rate": self.recovery_rate,
            "last_update_ns": self._last_update_ns,
            "history_count": len(self.history),
        }

    def restore_snapshot(self, data: dict) -> None:
        """从快照恢复状态。"""
        if "levels" in data:
            for name in _HORMONE_NAMES:
                setattr(self.levels, name, data["levels"].get(name, 0.5))
        if "base_levels" in data:
            for name in _HORMONE_NAMES:
                setattr(self.base_levels, name, data["base_levels"].get(name, 0.5))
        if "recovery_rate" in data:
            self.recovery_rate = data["recovery_rate"]
        if "last_update_ns" in data:
            self._last_update_ns = data["last_update_ns"]
