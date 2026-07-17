"""
基于 mss 的 30fps 实时屏幕捕获
支持回调模式，方便外部获取每一帧
"""
import time
import threading
import numpy as np
from collections import deque
from typing import Callable, Optional

import mss


class ScreenCapture:
    """30fps 实时屏幕捕获器（线程模式）"""

    def __init__(self, monitor_index: int = 1, fps: int = 30,
                 on_frame: Optional[Callable[[np.ndarray], None]] = None):
        self._monitor_index = monitor_index
        self._interval_s = 1.0 / fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_frame = on_frame

        # 统计
        self._frame_count = 0
        self._start_time = 0.0
        self._fps_history: deque = deque(maxlen=30)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def start(self) -> None:
        """启动捕获线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止捕获"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _capture_loop(self) -> None:
        with mss.mss() as sct:
            monitor = sct.monitors[self._monitor_index]
            self._start_time = time.time()

            print(f"[ScreenCapture] 开始捕获 | 监视器 {self._monitor_index} "
                  f"({monitor['width']}x{monitor['height']}) | 目标 {int(1/self._interval_s)}fps")

            while self._running:
                loop_start = time.perf_counter()

                # 抓取一帧
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                self._frame_count += 1

                # 更新最新帧（线程安全）
                with self._frame_lock:
                    self._latest_frame = frame

                # 回调
                if self._on_frame:
                    try:
                        self._on_frame(frame)
                    except Exception:
                        pass

                # 统计瞬时fps
                elapsed = time.perf_counter() - loop_start
                self._fps_history.append(1.0 / max(elapsed, 0.001))

                # 控制帧率
                remaining = self._interval_s - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            elapsed_total = time.time() - self._start_time
            print(f"\n[ScreenCapture] 已停止 | 总计 {self._frame_count} 帧 | "
                  f"运行 {elapsed_total:.1f}s | 平均 {self._frame_count / max(elapsed_total, 0.001):.1f}fps")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新一帧（线程安全）"""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def avg_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        return self._latest_frame


if __name__ == "__main__":
    cap = ScreenCapture(monitor_index=1, fps=30)
    cap.start()
    try:
        while cap.running:
            time.sleep(1)
            print(f"  fps={cap.avg_fps:.1f}  帧数={cap.frame_count}")
    except KeyboardInterrupt:
        cap.stop()
