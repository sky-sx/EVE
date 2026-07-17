"""
EVE Phase 2 统一捕获管理器。

管理屏幕捕获（mss）和光标捕获（pyautogui）线程，
将数据统一写入 InputBuffer 并提供 timing 统计。

不加入 YOLO、键盘、音频或 Memory。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eve.input.buffer import InputBuffer

# ── 统计结构 ──────────────────────────────────────────────


@dataclass
class CaptureTiming:
    """捕获 timing 统计。"""
    screen_fps_actual: float = 0.0
    screen_interval_p50_ms: float = 0.0
    screen_interval_p95_ms: float = 0.0
    cursor_hz_actual: float = 0.0
    buffer_screen_count: int = 0
    buffer_cursor_count: int = 0
    memory_growth_mb: float = 0.0
    shutdown_success: bool = False
    screen_interval_samples: list[float] = field(default_factory=list)
    cursor_interval_samples: list[float] = field(default_factory=list)

    def compute(self, run_duration_s: float) -> CaptureTiming:
        """基于采集的数据计算汇总统计。"""
        if self.screen_interval_samples:
            sorted_intervals = sorted(self.screen_interval_samples)
            n = len(sorted_intervals)
            self.screen_fps_actual = len(self.screen_interval_samples) / max(run_duration_s, 0.001)
            self.screen_interval_p50_ms = sorted_intervals[n // 2] * 1000
            self.screen_interval_p95_ms = sorted_intervals[int(n * 0.95)] * 1000
        if self.cursor_interval_samples:
            self.cursor_hz_actual = len(self.cursor_interval_samples) / max(run_duration_s, 0.001)
        return self


# ── 捕获管理器 ────────────────────────────────────────────


class CaptureManager:
    """统一屏幕+光标捕获，输出到 InputBuffer。"""

    def __init__(
        self,
        buffer: InputBuffer,
        monitor_index: int = 1,
        screen_fps: int = 30,
        cursor_hz: int = 60,
    ):
        self._buffer = buffer
        self._monitor_index = monitor_index
        self._screen_interval = 1.0 / screen_fps
        self._cursor_interval = 1.0 / cursor_hz

        self._running = False
        self._screen_thread: Optional[threading.Thread] = None
        self._cursor_thread: Optional[threading.Thread] = None

        # timing 原始数据
        self._timing = CaptureTiming()
        self._screen_intervals: deque[float] = deque(maxlen=10000)
        self._cursor_intervals: deque[float] = deque(maxlen=10000)
        self._last_screen_ns: int = 0
        self._last_cursor_ns: int = 0
        self._start_memory_mb: float = 0.0

    def start(self) -> None:
        """启动屏幕+光标捕获线程。"""
        if self._running:
            return
        self._running = True
        self._start_memory_mb = self._current_memory_mb()
        self._screen_thread = threading.Thread(
            target=self._screen_loop, daemon=True, name="eve-screen"
        )
        self._cursor_thread = threading.Thread(
            target=self._cursor_loop, daemon=True, name="eve-cursor"
        )
        self._screen_thread.start()
        self._cursor_thread.start()

    def stop(self) -> CaptureTiming:
        """停止所有捕获线程并返回 timing 统计。"""
        self._running = False
        if self._screen_thread is not None:
            self._screen_thread.join(timeout=3.0)
            self._screen_thread = None
        if self._cursor_thread is not None:
            self._cursor_thread.join(timeout=3.0)
            self._cursor_thread = None

        timing = self._timing
        timing.buffer_screen_count = self._buffer.count("screen")
        timing.buffer_cursor_count = self._buffer.count("cursor")
        timing.screen_interval_samples = list(self._screen_intervals)
        timing.cursor_interval_samples = list(self._cursor_intervals)
        timing.memory_growth_mb = max(
            0, self._current_memory_mb() - self._start_memory_mb
        )
        timing.shutdown_success = (
            self._screen_thread is None and self._cursor_thread is None
        )
        return timing

    @property
    def running(self) -> bool:
        return self._running

    # ── 内部 ──────────────────────────────────────────────

    def _screen_loop(self) -> None:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[self._monitor_index]
            while self._running:
                t0 = time.monotonic_ns()
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                now_ns = time.monotonic_ns()
                self._buffer.store("screen", frame)
                # 记录帧间隔
                if self._last_screen_ns > 0:
                    interval = (now_ns - self._last_screen_ns) / 1e9
                    self._screen_intervals.append(interval)
                self._last_screen_ns = now_ns
                # 帧率控制
                elapsed = (time.monotonic_ns() - t0) / 1e9
                remaining = self._screen_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    def _cursor_loop(self) -> None:
        import pyautogui
        while self._running:
            t0 = time.monotonic_ns()
            x, y = pyautogui.position()
            now_ns = time.monotonic_ns()
            self._buffer.store("cursor", (x, y))
            # 记录采样间隔
            if self._last_cursor_ns > 0:
                interval = (now_ns - self._last_cursor_ns) / 1e9
                self._cursor_intervals.append(interval)
            self._last_cursor_ns = now_ns
            # 频率控制
            elapsed = (time.monotonic_ns() - t0) / 1e9
            remaining = self._cursor_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    @staticmethod
    def _current_memory_mb() -> float:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
