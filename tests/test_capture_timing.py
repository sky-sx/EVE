"""Phase 2 捕获 timing 测试。

覆盖：
- CaptureManager 启动/停止
- 屏幕 FPS 统计
- 光标采样频率
- buffer 数据量
- 内存增长
- 关闭后无残留线程
- 时间戳单调性
"""
from __future__ import annotations

import threading
import time

import pytest

from eve.input.buffer import InputBuffer
from eve.input.capture import CaptureManager, CaptureTiming


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def buffer() -> InputBuffer:
    return InputBuffer(retention_ns=5_000_000_000)  # 5 秒


# ── 启动/停止 ────────────────────────────────────────────


def test_start_stop(buffer: InputBuffer):
    """CaptureManager 启动后应能正常停止。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=30)
    mgr.start()
    assert mgr.running is True
    time.sleep(0.5)
    timing = mgr.stop()
    assert mgr.running is False
    assert timing.shutdown_success is True


def test_double_start_is_safe(buffer: InputBuffer):
    """重复 start 应安全。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=30)
    mgr.start()
    mgr.start()  # 不应崩溃
    time.sleep(0.3)
    mgr.stop()


# ── FPS 统计 ────────────────────────────────────────────


def test_screen_fps_recorded(buffer: InputBuffer):
    """运行 ~2 秒后应有 FPS 统计。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=10)
    mgr.start()
    time.sleep(2.0)
    timing = mgr.stop()

    assert isinstance(timing, CaptureTiming)
    assert timing.screen_interval_samples, "应该有帧间隔数据"
    timing.compute(run_duration_s=2.0)
    assert timing.screen_fps_actual > 0
    assert timing.screen_interval_p50_ms > 0
    assert timing.screen_interval_p95_ms > 0


def test_cursor_hz_recorded(buffer: InputBuffer):
    """光标采样应有数据统计。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=30)
    mgr.start()
    time.sleep(1.0)
    timing = mgr.stop()

    timing.compute(run_duration_s=1.0)
    assert timing.cursor_hz_actual > 0


# ── buffer 数据量 ────────────────────────────────────────


def test_buffer_has_data(buffer: InputBuffer):
    """捕获后 buffer 应包含屏幕和光标数据。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=30)
    mgr.start()
    time.sleep(1.0)
    mgr.stop()

    screen_count = buffer.count("screen")
    cursor_count = buffer.count("cursor")
    assert screen_count > 0, f"buffer 应包含屏幕帧: count={screen_count}"
    assert cursor_count > 0, f"buffer 应包含光标数据: count={cursor_count}"


def test_buffer_latest_works(buffer: InputBuffer):
    """捕获后 latest 应返回有效数据。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=10)
    mgr.start()
    time.sleep(1.5)
    mgr.stop()

    screen = buffer.latest("screen")
    cursor = buffer.latest("cursor")
    assert screen is not None
    assert screen.kind == "screen"
    assert screen.value is not None  # np.ndarray
    assert cursor is not None
    assert cursor.kind == "cursor"
    assert isinstance(cursor.value, tuple)
    assert len(cursor.value) == 2


# ── 时间戳单调 ──────────────────────────────────────────


def test_timestamps_monotonic(buffer: InputBuffer):
    """同 kind 内时间戳应单调递增。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=30)
    mgr.start()
    time.sleep(1.5)
    mgr.stop()

    for kind in ("screen", "cursor"):
        samples = buffer.range(
            kind, 0, time.monotonic_ns() + 1_000_000_000
        )
        if len(samples) < 2:
            continue
        for i in range(1, len(samples)):
            assert samples[i].timestamp_ns >= samples[i - 1].timestamp_ns, (
                f"{kind} 时间戳非单调: {samples[i-1].timestamp_ns} > {samples[i].timestamp_ns}"
            )


# ── 内存 ────────────────────────────────────────────────


def test_memory_growth_tracked(buffer: InputBuffer):
    """timing 应报告内存增长。"""
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=10)
    mgr.start()
    time.sleep(1.5)
    timing = mgr.stop()

    assert timing.memory_growth_mb >= 0  # 0 或正数


# ── 线程清理 ───────────────────────────────────────────


def test_threads_cleaned_up(buffer: InputBuffer):
    """stop 后线程应退出。"""
    before = threading.active_count()
    mgr = CaptureManager(buffer, screen_fps=30, cursor_hz=10)
    mgr.start()
    time.sleep(1.0)
    mgr.stop()
    # 给 daemon 线程一点时间完全退出
    time.sleep(0.2)
    after = threading.active_count()
    assert after <= before + 1, (
        f"捕获线程未清理: before={before} after={after}"
    )


# ── 不导入 src.eve ────────────────────────────────────


def test_no_src_eve_import() -> None:
    import sys
    assert "src.eve" not in sys.modules
