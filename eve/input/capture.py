"""
EVE Phase 2 统一捕获管理器。

管理屏幕捕获（mss）和光标捕获（pyautogui）线程，
将数据统一写入 InputBuffer 并提供 timing 统计。
同时检测人类鼠标/键盘活动。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
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
    human_activity_events: int = 0

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
    """统一屏幕+光标捕获，输出到 InputBuffer，并检测人类活动。"""

    _HUMAN_CURSOR_THRESHOLD_PX = 50  # 光标位移阈值（像素）

    def __init__(
        self,
        buffer: InputBuffer,
        monitor_index: int = 1,
        screen_fps: int = 30,
        cursor_hz: int = 60,
        human_activity_callback: Callable[[], None] | None = None,
    ):
        self._buffer = buffer
        self._monitor_index = monitor_index
        self._screen_interval = 1.0 / screen_fps
        self._cursor_interval = 1.0 / cursor_hz
        self._human_activity_callback = human_activity_callback

        self._running = False
        self._screen_thread: Optional[threading.Thread] = None
        self._cursor_thread: Optional[threading.Thread] = None

        # 人类活动检测状态
        self._last_cursor_x: float | None = None
        self._last_cursor_y: float | None = None
        self._human_activity_count: int = 0
        self._human_activity_lock = threading.Lock()

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
        self._last_cursor_x = None
        self._last_cursor_y = None
        self._human_activity_count = 0
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
        timing.human_activity_events = self._human_activity_count
        return timing

    @property
    def running(self) -> bool:
        return self._running

    @property
    def human_activity_count(self) -> int:
        with self._human_activity_lock:
            return self._human_activity_count

    def was_human_cursor_recent(self, window_ns: int = 1_000_000_000) -> bool:
        """检查最近 window_ns 内是否有显著光标位移。"""
        if self._last_cursor_x is None or self._last_cursor_y is None:
            return False
        latest = self._buffer.latest("cursor")
        if latest is None:
            return False
        now_ns = time.monotonic_ns()
        if now_ns - latest.timestamp_ns > window_ns:
            return False
        # 查找 window_ns 内的旧样本
        old_samples = self._buffer.range("cursor", now_ns - window_ns, now_ns)
        if not old_samples:
            return False
        x0, y0 = old_samples[0].value
        x1, y1 = latest.value
        dx = x1 - x0
        dy = y1 - y0
        return (dx * dx + dy * dy) > (self._HUMAN_CURSOR_THRESHOLD_PX ** 2)

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

            # 人类光标活动检测
            if self._last_cursor_x is not None and self._last_cursor_y is not None:
                dx = x - self._last_cursor_x
                dy = y - self._last_cursor_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self._HUMAN_CURSOR_THRESHOLD_PX:
                    self._report_human_activity()
            self._last_cursor_x = x
            self._last_cursor_y = y

            # 频率控制
            elapsed = (time.monotonic_ns() - t0) / 1e9
            remaining = self._cursor_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _report_human_activity(self) -> None:
        """报告人类活动事件。"""
        with self._human_activity_lock:
            self._human_activity_count += 1
        if self._human_activity_callback is not None:
            self._human_activity_callback()

    @staticmethod
    def _current_memory_mb() -> float:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
