"""
Tests for CaptureManager timing (eve/input/capture.py).
Tests CaptureManager lifecycle, timing statistics, and thread management.
Some tests may be skipped if no display is available.
"""
import threading
import time

import pytest

from eve.input.buffer import InputBuffer
from eve.input.capture import CaptureManager, CaptureTiming

# Check if we can import the dependencies needed for actual capture
try:
    import mss  # noqa: F401
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False

try:
    import pyautogui  # noqa: F401
    _HAS_PYAUTOGUI = True
except ImportError:
    _HAS_PYAUTOGUI = False

_HAS_DISPLAY = _HAS_MSS and _HAS_PYAUTOGUI


# ── CaptureTiming unit tests ─────────────────────────────

def test_capture_timing_defaults():
    """CaptureTiming has expected default fields."""
    ct = CaptureTiming()
    assert ct.screen_fps_actual == 0.0
    assert ct.screen_interval_p50_ms == 0.0
    assert ct.screen_interval_p95_ms == 0.0
    assert ct.cursor_hz_actual == 0.0
    assert ct.buffer_screen_count == 0
    assert ct.buffer_cursor_count == 0
    assert ct.memory_growth_mb == 0.0
    assert ct.shutdown_success is False
    assert ct.screen_interval_samples == []
    assert ct.cursor_interval_samples == []
    assert ct.human_activity_events == 0


def test_capture_timing_compute():
    """compute() calculates real statistics from samples."""
    ct = CaptureTiming(
        screen_interval_samples=[0.03, 0.04, 0.05, 0.033, 0.035, 0.037, 0.045, 0.048, 0.050, 0.032],
    )
    ct.compute(run_duration_s=1.0)
    assert ct.screen_fps_actual > 0
    assert ct.screen_interval_p50_ms > 0
    assert ct.screen_interval_p95_ms > 0


def test_capture_timing_compute_empty():
    """compute() with empty samples returns 0 values."""
    ct = CaptureTiming()
    ct.compute(run_duration_s=1.0)
    assert ct.screen_fps_actual == 0.0
    assert ct.screen_interval_p50_ms == 0.0


# ── CaptureManager lifecycle ─────────────────────────────

@pytest.mark.skipif(not _HAS_DISPLAY, reason="No display (mss/pyautogui unavailable)")
def test_capture_start_stop():
    """CaptureManager start and stop without crashes."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    assert cm.running is True
    # Let it run briefly
    time.sleep(0.2)
    timing = cm.stop()
    assert cm.running is False
    assert isinstance(timing, CaptureTiming)


@pytest.mark.skipif(not _HAS_DISPLAY, reason="No display (mss/pyautogui unavailable)")
def test_timing_fields_populated():
    """After stop, timing has populated fields."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    time.sleep(0.3)
    timing = cm.stop()

    # Compute and verify
    timing.compute(run_duration_s=0.3)
    assert timing.buffer_screen_count > 0
    assert timing.buffer_cursor_count > 0
    assert timing.shutdown_success is True
    assert isinstance(timing.screen_interval_samples, list)
    assert isinstance(timing.cursor_interval_samples, list)


@pytest.mark.skipif(not _HAS_DISPLAY, reason="No display (mss/pyautogui unavailable)")
def test_capture_threads_exit():
    """After stop, threads are cleaned up."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    time.sleep(0.1)
    timing = cm.stop()
    assert timing.shutdown_success is True
    assert cm.running is False


@pytest.mark.skipif(not _HAS_DISPLAY, reason="No display (mss/pyautogui unavailable)")
def test_buffer_has_samples():
    """After capture, buffer has screen and cursor samples."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    time.sleep(0.2)
    cm.stop()

    # Buffer should have screen and cursor samples
    screen_count = buf.count("screen")
    cursor_count = buf.count("cursor")
    assert screen_count > 0, f"Expected screen samples, got {screen_count}"
    assert cursor_count > 0, f"Expected cursor samples, got {cursor_count}"


@pytest.mark.skipif(not _HAS_DISPLAY, reason="No display (mss/pyautogui unavailable)")
def test_monotonic_timestamps():
    """Screen samples have non-decreasing timestamps."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    time.sleep(0.2)
    cm.stop()

    screen_samples = buf.range("screen", 0)
    if len(screen_samples) >= 2:
        for i in range(len(screen_samples) - 1):
            assert screen_samples[i].timestamp_ns <= screen_samples[i + 1].timestamp_ns


# ── CaptureManager without display (creation only) ───────

def test_capture_manager_creation():
    """CaptureManager can be created without starting."""
    buf = InputBuffer()
    cm = CaptureManager(buf, monitor_index=1, screen_fps=30, cursor_hz=60)
    assert cm.running is False
    assert cm.human_activity_count == 0


def test_capture_manager_creation_with_monitor_index():
    """CaptureManager accepts custom monitor_index."""
    buf = InputBuffer()
    cm = CaptureManager(buf, monitor_index=2)
    assert cm.running is False


def test_capture_manager_creation_with_callback():
    """CaptureManager accepts a human_activity_callback."""
    call_count = [0]

    def callback():
        call_count[0] += 1

    buf = InputBuffer()
    cm = CaptureManager(buf, human_activity_callback=callback)
    assert cm.running is False
    # callback not called during creation
    assert call_count[0] == 0


def test_capture_human_activity_count_initial():
    """Human activity count starts at 0."""
    buf = InputBuffer()
    cm = CaptureManager(buf)
    assert cm.human_activity_count == 0


def test_capture_was_human_cursor_recent_empty_buffer():
    """was_human_cursor_recent returns False on empty buffer."""
    buf = InputBuffer()
    cm = CaptureManager(buf)
    assert cm.was_human_cursor_recent() is False


@pytest.mark.skipif(not _HAS_MSS, reason="mss not available")
def test_double_start_is_safe():
    """Starting an already-started manager is safe."""
    buf = InputBuffer()
    cm = CaptureManager(buf, screen_fps=30, cursor_hz=60)
    cm.start()
    assert cm.running is True
    cm.start()  # Second start should be a no-op
    assert cm.running is True
    cm.stop()
