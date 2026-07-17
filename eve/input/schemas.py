"""EVE 输入层数据结构。

Phase 2 仅包含屏幕帧和光标坐标的 TimedSample。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimedSample:
    """统一时间轴上的输入样本。

    所有时间戳使用 time.monotonic_ns()，确保单调。
    """
    timestamp_ns: int
    kind: str  # "screen" | "cursor"
    value: Any  # screen: np.ndarray, cursor: (x, y)
    index: int = 0  # 全局递增序号


@dataclass
class CursorState:
    """旧光标状态—保留向后兼容。Phase 2 新代码优先使用 TimedSample。"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    timestamp: float = 0.0
    is_active: bool = False
