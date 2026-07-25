"""One capture lifecycle for screen and cursor input."""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from eve.input.buffer import InputBuffer


@dataclass(frozen=True)
class CaptureError:
    timestamp_ns: int
    exception_type: str
    message: str
    traceback: str
    source: str
    recovery_action: str


class Capture:
    def __init__(
        self,
        buffer: InputBuffer,
        *,
        screen_fps: float = 10.0,
        cursor_hz: float = 20.0,
        screen_reader: Callable[[], Any] | None = None,
        cursor_reader: Callable[[], tuple[int, int]] | None = None,
        error_callback: Callable[[CaptureError], None] | None = None,
    ) -> None:
        if screen_fps <= 0 or cursor_hz <= 0:
            raise ValueError("capture frequencies must be positive")
        self.buffer = buffer
        self.screen_period_ns = int(1_000_000_000 / screen_fps)
        self.cursor_period_ns = int(1_000_000_000 / cursor_hz)
        self._screen_reader = screen_reader
        self._cursor_reader = cursor_reader
        self._error_callback = error_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: CaptureError | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="eve-capture")
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout_s)
        if thread.is_alive():
            raise RuntimeError("capture thread did not stop")
        self._thread = None

    def _run(self) -> None:
        cleanup: Callable[[], None] = lambda: None
        try:
            screen_reader = self._screen_reader
            cursor_reader = self._cursor_reader
            if screen_reader is None or cursor_reader is None:
                screen_reader, cursor_reader, cleanup = self._real_readers()
            next_screen = next_cursor = time.monotonic_ns()
            while not self._stop_event.is_set():
                now_ns = time.monotonic_ns()
                if now_ns >= next_screen:
                    self.buffer.store("screen", screen_reader(), timestamp_ns=now_ns)
                    next_screen = now_ns + self.screen_period_ns
                if now_ns >= next_cursor:
                    self.buffer.store("cursor", cursor_reader(), timestamp_ns=now_ns)
                    next_cursor = now_ns + self.cursor_period_ns
                wait_ns = max(1_000_000, min(next_screen, next_cursor) - time.monotonic_ns())
                self._stop_event.wait(wait_ns / 1_000_000_000)
        except Exception as exc:
            error = CaptureError(
                timestamp_ns=time.monotonic_ns(),
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                source="capture",
                recovery_action="capture_stopped_no_output",
            )
            self.last_error = error
            if self._error_callback is not None:
                self._error_callback(error)
            self._stop_event.set()
        finally:
            cleanup()

    @staticmethod
    def _real_readers() -> tuple[
        Callable[[], Any], Callable[[], tuple[int, int]], Callable[[], None]
    ]:
        import mss
        import numpy as np
        import pyautogui

        session = mss.mss()
        monitor = session.monitors[1]

        def read_screen() -> Any:
            return np.asarray(session.grab(monitor))

        def read_cursor() -> tuple[int, int]:
            point = pyautogui.position()
            return int(point.x), int(point.y)

        return read_screen, read_cursor, session.close
