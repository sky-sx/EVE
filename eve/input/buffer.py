"""The sole public Input boundary and owner of the Capture subprocess."""
from __future__ import annotations

import multiprocessing
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from eve.input.capture import capture_process_main


@dataclass(frozen=True)
class TimedSample:
    timestamp_ns: int
    kind: str
    value: Any
    index: int


@dataclass(frozen=True)
class ScreenFrame:
    frame_id: int
    captured_at_ns: int
    slot: int
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


class InputBuffer:
    """Maintain the one-second input window and all Capture IPC resources."""

    def __init__(
        self,
        retention_ns: int = 1_000_000_000,
        max_samples_per_kind: int = 256,
        *,
        profile: str = "smoke",
        screen_fps: float = 30.0,
        cursor_hz: float = 60.0,
        capture_options: dict[str, Any] | None = None,
    ) -> None:
        if retention_ns <= 0:
            raise ValueError("retention_ns must be positive")
        if max_samples_per_kind <= 0:
            raise ValueError("max_samples_per_kind must be positive")
        if profile not in {"smoke", "observe", "control"}:
            raise ValueError(f"unsupported input profile: {profile}")
        self.retention_ns = retention_ns
        self.max_samples_per_kind = max_samples_per_kind
        self.profile = profile
        self.screen_fps = screen_fps
        self.cursor_hz = cursor_hz
        self.capture_options = dict(capture_options or {})
        self._samples: dict[str, deque[TimedSample]] = {}
        self._index = 0
        self._lock = threading.Lock()
        self._closed = False
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._parent_connection: Connection | None = None
        self._stop_event: Any = None
        self._receiver_thread: threading.Thread | None = None
        self._receiver_stop = threading.Event()
        self._startup_event = threading.Event()
        self._shared_screen: SharedMemory | None = None
        self._screen_ring: dict[str, Any] | None = None
        self._capture_error: dict[str, Any] | None = None
        self._capture_health: dict[str, Any] = {"state": "not_started"}
        self._capture_started_at_ns = 0
        self._screen_count = 0
        self._cursor_count = 0
        self._keyboard_count = 0
        self._window_count = 0
        self._dropped_screen_frames = 0
        self._last_screen_frame_id = 0
        self._last_cursor_position: tuple[int, int] | None = None
        self._expected_mouse_until_ns = 0
        self._expected_mouse_target: tuple[int, int] | None = None
        self._expected_keyboard_until_ns = 0
        self.human_activity_detected_at_ns = 0
        self.human_takeover_until_ns = 0

    def start_capture(self, startup_timeout_s: float = 7.0) -> None:
        """Start Capture and wait until both real/synthetic sources are healthy."""
        if self._closed:
            raise RuntimeError("input buffer is closed")
        if self.capture_running:
            return
        parent, child = self._context.Pipe(duplex=True)
        self._parent_connection = parent
        self._stop_event = self._context.Event()
        self._receiver_stop.clear()
        self._startup_event.clear()
        self._capture_error = None
        self._capture_started_at_ns = time.monotonic_ns()
        config = {
            "screen_fps": self.screen_fps,
            "cursor_hz": self.cursor_hz,
            "screen_slots": max(4, int(self.screen_fps * 1.25)),
            "screen_mode": "synthetic" if self.profile == "smoke" else "real",
            "cursor_mode": "synthetic" if self.profile == "smoke" else "real",
            "keyboard_mode": "synthetic" if self.profile == "smoke" else "real",
            "window_mode": "synthetic" if self.profile == "smoke" else "real",
            "startup_timeout_s": startup_timeout_s,
            **self.capture_options,
        }
        self._process = self._context.Process(
            target=capture_process_main,
            args=(child, self._stop_event, config),
            name="eve-capture",
        )
        self._receiver_thread = threading.Thread(
            target=self._receive_capture_messages,
            name="eve-input-ipc",
        )
        self._capture_health = {"state": "starting"}
        self._receiver_thread.start()
        self._process.start()
        child.close()
        if not self._startup_event.wait(startup_timeout_s):
            self.stop_capture()
            raise RuntimeError("capture process initialization timed out")
        if self._capture_error is not None:
            error = dict(self._capture_error)
            self.stop_capture()
            raise RuntimeError(
                "capture process initialization failed: "
                f"{error.get('exception_type')}: {error.get('message')}"
            )
        if not self.capture_running:
            self.stop_capture()
            raise RuntimeError("capture process exited during initialization")

    def stop_capture(self, timeout_s: float = 3.0) -> None:
        """Stop Capture, join its process/IPC thread, and release shared memory."""
        connection = self._parent_connection
        if connection is not None:
            try:
                connection.send({"type": "control", "command": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self._stop_event is not None:
            self._stop_event.set()
        process = self._process
        if process is not None:
            process.join(timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(1.0)
                if process.is_alive():
                    raise RuntimeError("capture process did not stop")
        self._receiver_stop.set()
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        receiver = self._receiver_thread
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout_s)
            if receiver.is_alive():
                raise RuntimeError("capture IPC receiver did not stop")
        self._receiver_thread = None
        self._parent_connection = None
        self._process = None
        self._stop_event = None
        self._release_shared_screen()
        self._capture_health["state"] = (
            "failed" if self._capture_error is not None else "stopped"
        )

    def start(self) -> None:
        self.start_capture()

    def stop(self) -> None:
        self.stop_capture()

    @property
    def capture_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def capture_process_id(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def capture_error(self) -> dict[str, Any] | None:
        return dict(self._capture_error) if self._capture_error is not None else None

    def capture_health(self) -> dict[str, Any]:
        self._check_capture_process()
        return dict(self._capture_health)

    def capture_stats(self) -> dict[str, float | int]:
        duration_s = max(
            (time.monotonic_ns() - self._capture_started_at_ns) / 1_000_000_000,
            1e-9,
        )
        final = self._capture_health
        final_duration_s = final.get("duration_ns", 0) / 1_000_000_000
        if final_duration_s > 0:
            duration_s = final_duration_s
        screen_count = int(final.get("screen_frames", self._screen_count))
        cursor_count = int(final.get("cursor_samples", self._cursor_count))
        return {
            "screen_frames": screen_count,
            "cursor_samples": cursor_count,
            "keyboard_samples": int(
                final.get("keyboard_samples", self._keyboard_count)
            ),
            "window_samples": int(
                final.get("window_samples", self._window_count)
            ),
            "dropped_screen_frames": self._dropped_screen_frames,
            "screen_fps": screen_count / duration_s,
            "cursor_hz": cursor_count / duration_s,
            "screen_average_latency_ms": (
                int(final.get("screen_latency_ns", 0))
                / max(screen_count, 1)
                / 1_000_000
            ),
            "cursor_average_latency_ms": (
                int(final.get("cursor_latency_ns", 0))
                / max(cursor_count, 1)
                / 1_000_000
            ),
        }

    def store(
        self,
        kind: str,
        value: Any,
        *,
        timestamp_ns: int | None = None,
    ) -> TimedSample:
        """Store a small sample; used internally by IPC and focused tests."""
        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        with self._lock:
            if self._closed:
                raise RuntimeError("input buffer is closed")
            self._index += 1
            sample = TimedSample(timestamp_ns, kind, value, self._index)
            items = self._samples.setdefault(kind, deque())
            if items and timestamp_ns < items[-1].timestamp_ns:
                raise ValueError("input timestamps must be monotonic per kind")
            items.append(sample)
            cutoff = timestamp_ns - self.retention_ns
            while items and items[0].timestamp_ns < cutoff:
                items.popleft()
            while len(items) > self.max_samples_per_kind:
                items.popleft()
            return sample

    def latest(self, kind: str) -> TimedSample | None:
        with self._lock:
            items = self._samples.get(kind)
            return items[-1] if items else None

    def range(
        self,
        kind: str,
        start_ns: int,
        end_ns: int | None = None,
    ) -> list[TimedSample]:
        end_ns = time.monotonic_ns() if end_ns is None else end_ns
        with self._lock:
            return [
                sample
                for sample in self._samples.get(kind, ())
                if start_ns <= sample.timestamp_ns < end_ns
            ]

    def snapshot(
        self,
        duration_ns: int = 1_000_000_000,
    ) -> dict[str, list[TimedSample]]:
        now_ns = time.monotonic_ns()
        with self._lock:
            return {
                kind: [
                    sample
                    for sample in items
                    if sample.timestamp_ns >= now_ns - duration_ns
                ]
                for kind, items in self._samples.items()
            }

    def get_state(self) -> dict[str, Any]:
        window = self.snapshot(min(self.retention_ns, 1_000_000_000))
        screen = window.get("screen", [])
        cursor = window.get("cursor", [])
        keyboard = window.get("keyboard_activity", [])
        active_window = window.get("active_window", [])
        return {
            "screen": screen,
            "cursor": cursor,
            "keyboard_activity": keyboard,
            "active_window": active_window,
            "latest": {
                "screen": screen[-1] if screen else None,
                "cursor": cursor[-1] if cursor else None,
                "keyboard_activity": keyboard[-1] if keyboard else None,
                "active_window": active_window[-1] if active_window else None,
            },
            "capture": self.capture_health(),
            "human_activity_detected_at_ns": self.human_activity_detected_at_ns,
            "human_takeover_until_ns": self.human_takeover_until_ns,
            "dropped_screen_frames": self._dropped_screen_frames,
        }

    def mark_eve_mouse_action(
        self,
        action_id: str,
        *,
        target: tuple[int, int] | None,
        duration_s: float = 0.0,
    ) -> None:
        """Mark the short motion window so EVE movement is not user takeover."""
        del action_id
        now_ns = time.monotonic_ns()
        self._expected_mouse_until_ns = now_ns + int(
            (max(0.0, duration_s) + 0.35) * 1_000_000_000
        )
        self._expected_mouse_target = target

    def mark_eve_keyboard_action(
        self, action_id: str, *, duration_s: float = 0.25
    ) -> None:
        del action_id
        self._expected_keyboard_until_ns = time.monotonic_ns() + int(
            max(0.1, duration_s) * 1_000_000_000
        )

    def submit_user_text(self, text: str) -> TimedSample:
        return self.store(
            "user_text",
            {"text": str(text)},
            timestamp_ns=time.monotonic_ns(),
        )

    def get_latest_screen(self) -> TimedSample | None:
        samples = self.snapshot(min(self.retention_ns, 1_000_000_000)).get(
            "screen", []
        )
        return samples[-1] if samples else None

    def get_latest_cursor(self) -> TimedSample | None:
        samples = self.snapshot(min(self.retention_ns, 1_000_000_000)).get(
            "cursor", []
        )
        return samples[-1] if samples else None

    def count(self, kind: str) -> int:
        with self._lock:
            return len(self._samples.get(kind, ()))

    @property
    def kinds(self) -> list[str]:
        with self._lock:
            return sorted(self._samples)

    def close(self) -> None:
        self.stop_capture()
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _receive_capture_messages(self) -> None:
        connection = self._parent_connection
        if connection is None:
            return
        try:
            while not self._receiver_stop.is_set():
                if not connection.poll(0.05):
                    self._check_capture_process()
                    if (
                        self._process is not None
                        and not self._process.is_alive()
                    ):
                        break
                    continue
                message = connection.recv()
                message_type = message.get("type")
                if message_type == "screen_ring":
                    self._attach_screen_ring(message)
                elif message_type == "screen":
                    self._receive_screen(message)
                elif message_type == "cursor":
                    self._receive_cursor(message)
                elif message_type == "keyboard_activity":
                    self._receive_keyboard_activity(message)
                elif message_type == "active_window":
                    self._receive_active_window(message)
                elif message_type == "health":
                    self._capture_health = dict(message)
                    if message.get("state") == "running":
                        self._startup_event.set()
                elif message_type == "error":
                    self._capture_error = dict(message)
                    self._capture_health = {
                        "state": "failed",
                        "timestamp_ns": message.get("timestamp_ns"),
                    }
                    self._startup_event.set()
        except (EOFError, OSError):
            if not self._receiver_stop.is_set() and self._capture_error is None:
                self._capture_error = {
                    "type": "error",
                    "source": "capture_ipc",
                    "timestamp_ns": time.monotonic_ns(),
                    "exception_type": "EOFError",
                    "message": "capture IPC closed unexpectedly",
                    "traceback": "",
                }
                self._startup_event.set()

    def _attach_screen_ring(self, message: dict[str, Any]) -> None:
        self._shared_screen = SharedMemory(name=message["name"])
        self._screen_ring = dict(message)

    def _receive_screen(self, message: dict[str, Any]) -> None:
        if self._shared_screen is None or self._screen_ring is None:
            return
        import numpy as np

        offset = int(message["slot"]) * int(self._screen_ring["frame_bytes"])
        image = np.ndarray(
            tuple(message["shape"]),
            dtype=np.dtype(message["dtype"]),
            buffer=self._shared_screen.buf,
            offset=offset,
        )
        image.flags.writeable = False
        frame = ScreenFrame(
            frame_id=int(message["frame_id"]),
            captured_at_ns=int(message["timestamp_ns"]),
            slot=int(message["slot"]),
            image=image,
        )
        self.store("screen", frame, timestamp_ns=frame.captured_at_ns)
        if self._last_screen_frame_id:
            self._dropped_screen_frames += max(
                0, frame.frame_id - self._last_screen_frame_id - 1
            )
        self._last_screen_frame_id = frame.frame_id
        self._screen_count += 1

    def _receive_cursor(self, message: dict[str, Any]) -> None:
        cursor = CursorState(
            frame_id=int(message["frame_id"]),
            captured_at_ns=int(message["timestamp_ns"]),
            x=int(message["x"]),
            y=int(message["y"]),
            velocity_x=float(message["velocity_x"]),
            velocity_y=float(message["velocity_y"]),
            speed=float(message["speed"]),
        )
        current = (cursor.x, cursor.y)
        if self._last_cursor_position is not None and current != self._last_cursor_position:
            expected = self._is_expected_mouse_motion(current, cursor.captured_at_ns)
            if not expected:
                self.human_activity_detected_at_ns = cursor.captured_at_ns
                self.human_takeover_until_ns = (
                    cursor.captured_at_ns + 5_000_000_000
                )
        self._last_cursor_position = current
        self.store("cursor", cursor, timestamp_ns=cursor.captured_at_ns)
        self._cursor_count += 1

    def _is_expected_mouse_motion(
        self, current: tuple[int, int], timestamp_ns: int
    ) -> bool:
        if timestamp_ns > self._expected_mouse_until_ns:
            return False
        target = self._expected_mouse_target
        previous = self._last_cursor_position
        if target is None or previous is None:
            return False
        previous_distance = (
            (previous[0] - target[0]) ** 2 + (previous[1] - target[1]) ** 2
        )
        current_distance = (
            (current[0] - target[0]) ** 2 + (current[1] - target[1]) ** 2
        )
        return current_distance <= previous_distance + 4

    def _receive_keyboard_activity(self, message: dict[str, Any]) -> None:
        timestamp_ns = int(message["timestamp_ns"])
        value = {
            "active": bool(message["active"]),
            "active_key_count": int(message["active_key_count"]),
            "last_activity_ns": timestamp_ns if message["active"] else 0,
        }
        if value["active"] and timestamp_ns > self._expected_keyboard_until_ns:
            self.human_activity_detected_at_ns = timestamp_ns
            self.human_takeover_until_ns = timestamp_ns + 5_000_000_000
        self.store("keyboard_activity", value, timestamp_ns=timestamp_ns)
        self._keyboard_count += 1

    def _receive_active_window(self, message: dict[str, Any]) -> None:
        timestamp_ns = int(message["timestamp_ns"])
        self.store(
            "active_window",
            {
                "title": str(message.get("title", "")),
                "process": str(message.get("process", "")),
                "updated_at_ns": timestamp_ns,
            },
            timestamp_ns=timestamp_ns,
        )
        self._window_count += 1

    def _check_capture_process(self) -> None:
        process = self._process
        if (
            process is not None
            and process.pid is not None
            and not process.is_alive()
            and self._capture_health.get("state") not in {"stopped", "failed"}
            and not self._receiver_stop.is_set()
        ):
            self._capture_error = {
                "type": "error",
                "source": "capture_process",
                "timestamp_ns": time.monotonic_ns(),
                "exception_type": "ProcessExit",
                "message": f"capture process exited with code {process.exitcode}",
                "traceback": "",
            }
            self._capture_health = {"state": "failed"}
            self._startup_event.set()

    def _release_shared_screen(self) -> None:
        with self._lock:
            self._samples.pop("screen", None)
        shared = self._shared_screen
        self._shared_screen = None
        self._screen_ring = None
        if shared is None:
            return
        try:
            shared.close()
        except BufferError:
            pass
        try:
            shared.unlink()
        except FileNotFoundError:
            pass
