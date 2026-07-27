"""Independent screen and cursor capture loops on one monotonic timeline."""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eve.input.buffer import InputBuffer


@dataclass(frozen=True)
class ScreenFrame:
    frame_id: int
    captured_at_ns: int
    image: Any


@dataclass(frozen=True)
class CursorState:
    frame_id: int
    captured_at_ns: int
    x: int
    y: int
    velocity_x: float
    velocity_y: float
    speed: float

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(frozen=True)
class CaptureError:
    timestamp_ns: int
    exception_type: str
    message: str
    traceback: str
    source: str
    recovery_action: str


class Capture:
    """Capture screen and cursor without either source blocking the other."""

    def __init__(
        self,
        buffer: InputBuffer,
        *,
        screen_fps: float = 30.0,
        cursor_hz: float = 60.0,
        screen_reader: Callable[[], Any] | None = None,
        cursor_reader: Callable[[], tuple[int, int]] | None = None,
        cursor_callback: Callable[[CursorState], None] | None = None,
        error_callback: Callable[[CaptureError], None] | None = None,
    ) -> None:
        if screen_fps <= 0 or cursor_hz <= 0:
            raise ValueError("capture frequencies must be positive")
        self.buffer = buffer
        self.screen_period_ns = int(1_000_000_000 / screen_fps)
        self.cursor_period_ns = int(1_000_000_000 / cursor_hz)
        self._screen_reader = screen_reader
        self._cursor_reader = cursor_reader
        self._cursor_callback = cursor_callback
        self._error_callback = error_callback
        self._stop_event = threading.Event()
        self._screen_ready = threading.Event()
        self._cursor_ready = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._frame_id = 0
        self._started_at_ns = 0
        self._screen_count = 0
        self._cursor_count = 0
        self._screen_latency_ns = 0
        self._cursor_latency_ns = 0
        self.last_error: CaptureError | None = None

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def start(self, startup_timeout_s: float = 5.0) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._screen_ready.clear()
        self._cursor_ready.clear()
        self.last_error = None
        self._started_at_ns = time.monotonic_ns()
        self._threads = [
            threading.Thread(
                target=self._screen_loop,
                name="eve-capture-screen",
            ),
            threading.Thread(
                target=self._cursor_loop,
                name="eve-capture-cursor",
            ),
        ]
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + startup_timeout_s
        for ready in (self._screen_ready, self._cursor_ready):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not ready.wait(remaining):
                self._stop_event.set()
                self.stop()
                raise RuntimeError("capture initialization timed out")
        if self.last_error is not None:
            error = self.last_error
            self.stop()
            raise RuntimeError(
                f"capture initialization failed: {error.exception_type}: {error.message}"
            )

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        deadline = time.monotonic() + timeout_s
        for thread in self._threads:
            if thread is threading.current_thread():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"capture threads did not stop: {alive}")
        self._threads = []

    def stats(self) -> dict[str, float | int]:
        duration_s = max(
            (time.monotonic_ns() - self._started_at_ns) / 1_000_000_000,
            1e-9,
        )
        with self._lock:
            return {
                "screen_frames": self._screen_count,
                "cursor_samples": self._cursor_count,
                "screen_fps": self._screen_count / duration_s,
                "cursor_hz": self._cursor_count / duration_s,
                "screen_average_latency_ms": (
                    self._screen_latency_ns / max(self._screen_count, 1) / 1_000_000
                ),
                "cursor_average_latency_ms": (
                    self._cursor_latency_ns / max(self._cursor_count, 1) / 1_000_000
                ),
            }

    def _next_frame_id(self) -> int:
        with self._lock:
            self._frame_id += 1
            return self._frame_id

    def _screen_loop(self) -> None:
        cleanup: Callable[[], None] = lambda: None
        try:
            reader = self._screen_reader
            if reader is None:
                reader, cleanup = self._open_real_screen_reader()
            next_capture_ns = time.monotonic_ns()
            while not self._stop_event.is_set():
                self._wait_until(next_capture_ns)
                if self._stop_event.is_set():
                    break
                started_ns = time.monotonic_ns()
                image = reader()
                captured_at_ns = time.monotonic_ns()
                frame = ScreenFrame(
                    frame_id=self._next_frame_id(),
                    captured_at_ns=captured_at_ns,
                    image=image,
                )
                self.buffer.store("screen", frame, timestamp_ns=captured_at_ns)
                self._screen_ready.set()
                with self._lock:
                    self._screen_count += 1
                    self._screen_latency_ns += captured_at_ns - started_ns
                next_capture_ns = max(
                    next_capture_ns + self.screen_period_ns,
                    captured_at_ns,
                )
        except Exception as exc:
            self._fail("screen_input", exc)
        finally:
            self._screen_ready.set()
            cleanup()

    def _cursor_loop(self) -> None:
        try:
            reader = self._cursor_reader or self._read_real_cursor
            next_capture_ns = time.monotonic_ns()
            previous: CursorState | None = None
            while not self._stop_event.is_set():
                self._wait_until(next_capture_ns)
                if self._stop_event.is_set():
                    break
                started_ns = time.monotonic_ns()
                x, y = reader()
                captured_at_ns = time.monotonic_ns()
                elapsed_s = (
                    (captured_at_ns - previous.captured_at_ns) / 1_000_000_000
                    if previous is not None
                    else 0.0
                )
                velocity_x = (x - previous.x) / elapsed_s if elapsed_s > 0 else 0.0
                velocity_y = (y - previous.y) / elapsed_s if elapsed_s > 0 else 0.0
                state = CursorState(
                    frame_id=self._next_frame_id(),
                    captured_at_ns=captured_at_ns,
                    x=int(x),
                    y=int(y),
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    speed=(velocity_x * velocity_x + velocity_y * velocity_y) ** 0.5,
                )
                self.buffer.store("cursor", state, timestamp_ns=captured_at_ns)
                self._cursor_ready.set()
                if self._cursor_callback is not None:
                    self._cursor_callback(state)
                previous = state
                with self._lock:
                    self._cursor_count += 1
                    self._cursor_latency_ns += captured_at_ns - started_ns
                next_capture_ns = max(
                    next_capture_ns + self.cursor_period_ns,
                    captured_at_ns,
                )
        except Exception as exc:
            self._fail("cursor_input", exc)
        finally:
            self._cursor_ready.set()

    def _wait_until(self, deadline_ns: int) -> None:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns > 0:
            self._stop_event.wait(remaining_ns / 1_000_000_000)

    def _fail(self, source: str, exc: Exception) -> None:
        error = CaptureError(
            timestamp_ns=time.monotonic_ns(),
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
            source=source,
            recovery_action="capture_stopped_no_output",
        )
        with self._lock:
            if self.last_error is not None:
                return
            self.last_error = error
        self._stop_event.set()
        if self._error_callback is not None:
            self._error_callback(error)

    @staticmethod
    def _open_real_screen_reader() -> tuple[Callable[[], Any], Callable[[], None]]:
        import mss
        import numpy as np

        session = mss.mss()
        if len(session.monitors) < 2:
            session.close()
            raise RuntimeError("no desktop monitor is available")
        monitor = session.monitors[1]

        def read_screen() -> Any:
            return np.asarray(session.grab(monitor))

        return read_screen, session.close

    @staticmethod
    def _read_real_cursor() -> tuple[int, int]:
        import ctypes

        class Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        point = Point()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            raise OSError("GetCursorPos failed")
        return int(point.x), int(point.y)
