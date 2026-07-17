"""
极简光标位置捕获。

在 Windows 上使用 pyautogui.position() 作为基础实现。
返回位置、速度、时间戳和活跃状态。
"""
from __future__ import annotations

import time
from typing import Optional

from .schemas import CursorState


class CursorCapture:
    """极简光标位置与速度追踪器。

    通过两次采样的位置差计算速度。
    """

    def __init__(self, movement_threshold: float = 3.0):
        """
        Args:
            movement_threshold: 判断光标活跃的像素移动阈值。
        """
        self._threshold = movement_threshold
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._last_ts: float = 0.0

    def sample(self) -> CursorState:
        """采样当前光标状态。"""
        try:
            import pyautogui
            x, y = pyautogui.position()
        except ImportError:
            x, y = 0.0, 0.0

        now = time.perf_counter()
        vx, vy = 0.0, 0.0
        is_active = False

        if self._last_x is not None and self._last_ts > 0:
            dt = now - self._last_ts
            if dt > 0:
                dx = x - self._last_x
                dy = y - self._last_y
                vx = dx / dt
                vy = dy / dt
                is_active = abs(dx) > self._threshold or abs(dy) > self._threshold

        self._last_x = x
        self._last_y = y
        self._last_ts = now

        return CursorState(x=x, y=y, vx=vx, vy=vy, timestamp=now, is_active=is_active)
