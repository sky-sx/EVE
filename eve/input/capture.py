"""Capture subprocess implementation.

The parent process controls this module exclusively through ``InputBuffer``.
Large screen frames are written into a shared-memory ring; the Pipe carries
only metadata, cursor samples, health, errors, and stop control.
"""
from __future__ import annotations

import ctypes
import math
import threading
import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable


def capture_process_main(
    connection: Connection,
    stop_event: Event,
    config: dict[str, Any],
) -> None:
    """Run screen and cursor capture threads inside the Capture subprocess."""
    send_lock = threading.Lock()
    id_lock = threading.Lock()
    screen_ready = threading.Event()
    cursor_ready = threading.Event()
    shared: SharedMemory | None = None
    frame_id = 0
    stats = {
        "screen_frames": 0,
        "cursor_samples": 0,
        "screen_latency_ns": 0,
        "cursor_latency_ns": 0,
    }
    started_at_ns = time.monotonic_ns()

    def next_frame_id() -> int:
        nonlocal frame_id
        with id_lock:
            frame_id += 1
            return frame_id

    def send(message: dict[str, Any]) -> None:
        with send_lock:
            try:
                connection.send(message)
            except (BrokenPipeError, EOFError, OSError):
                stop_event.set()

    def fail(source: str, exc: Exception) -> None:
        send(
            {
                "type": "error",
                "source": source,
                "timestamp_ns": time.monotonic_ns(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        stop_event.set()

    def wait_until(deadline_ns: int) -> None:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns > 0:
            stop_event.wait(remaining_ns / 1_000_000_000)

    def screen_loop() -> None:
        nonlocal shared
        cleanup: Callable[[], None] = lambda: None
        try:
            import numpy as np

            reader, cleanup = _screen_reader(config)
            period_ns = int(1_000_000_000 / float(config["screen_fps"]))
            slots = int(config.get("screen_slots", 40))
            next_capture_ns = time.monotonic_ns()
            shape: tuple[int, ...] | None = None
            dtype: Any = None
            frame_bytes = 0
            while not stop_event.is_set():
                wait_until(next_capture_ns)
                if stop_event.is_set():
                    break
                capture_started_ns = time.monotonic_ns()
                image = np.ascontiguousarray(reader())
                captured_at_ns = time.monotonic_ns()
                if shared is None:
                    shape = tuple(image.shape)
                    dtype = image.dtype
                    frame_bytes = int(image.nbytes)
                    shared = SharedMemory(create=True, size=frame_bytes * slots)
                    send(
                        {
                            "type": "screen_ring",
                            "name": shared.name,
                            "slots": slots,
                            "shape": shape,
                            "dtype": dtype.str,
                            "frame_bytes": frame_bytes,
                        }
                    )
                if tuple(image.shape) != shape or image.dtype != dtype:
                    raise RuntimeError("screen shape or dtype changed during capture")
                current_id = next_frame_id()
                slot = current_id % slots
                target = np.ndarray(
                    shape,
                    dtype=dtype,
                    buffer=shared.buf,
                    offset=slot * frame_bytes,
                )
                np.copyto(target, image)
                send(
                    {
                        "type": "screen",
                        "frame_id": current_id,
                        "timestamp_ns": captured_at_ns,
                        "slot": slot,
                        "shape": shape,
                        "dtype": dtype.str,
                    }
                )
                stats["screen_frames"] += 1
                stats["screen_latency_ns"] += captured_at_ns - capture_started_ns
                screen_ready.set()
                next_capture_ns = max(
                    next_capture_ns + period_ns,
                    captured_at_ns,
                )
        except Exception as exc:
            fail("screen_input", exc)
        finally:
            screen_ready.set()
            cleanup()

    def cursor_loop() -> None:
        try:
            reader = _cursor_reader(config)
            period_ns = int(1_000_000_000 / float(config["cursor_hz"]))
            next_capture_ns = time.monotonic_ns()
            previous: tuple[int, int, int] | None = None
            while not stop_event.is_set():
                wait_until(next_capture_ns)
                if stop_event.is_set():
                    break
                capture_started_ns = time.monotonic_ns()
                x, y = reader()
                captured_at_ns = time.monotonic_ns()
                if previous is None:
                    velocity_x = velocity_y = 0.0
                else:
                    previous_x, previous_y, previous_ns = previous
                    elapsed_s = (captured_at_ns - previous_ns) / 1_000_000_000
                    velocity_x = (
                        (x - previous_x) / elapsed_s if elapsed_s > 0 else 0.0
                    )
                    velocity_y = (
                        (y - previous_y) / elapsed_s if elapsed_s > 0 else 0.0
                    )
                current_id = next_frame_id()
                send(
                    {
                        "type": "cursor",
                        "frame_id": current_id,
                        "timestamp_ns": captured_at_ns,
                        "x": int(x),
                        "y": int(y),
                        "velocity_x": velocity_x,
                        "velocity_y": velocity_y,
                        "speed": math.hypot(velocity_x, velocity_y),
                    }
                )
                previous = (int(x), int(y), captured_at_ns)
                stats["cursor_samples"] += 1
                stats["cursor_latency_ns"] += captured_at_ns - capture_started_ns
                cursor_ready.set()
                next_capture_ns = max(
                    next_capture_ns + period_ns,
                    captured_at_ns,
                )
        except Exception as exc:
            fail("cursor_input", exc)
        finally:
            cursor_ready.set()

    threads = [
        threading.Thread(target=screen_loop, name="capture-screen"),
        threading.Thread(target=cursor_loop, name="capture-cursor"),
    ]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + float(config.get("startup_timeout_s", 5.0))
        while not stop_event.is_set():
            if screen_ready.is_set() and cursor_ready.is_set():
                send(
                    {
                        "type": "health",
                        "state": "running",
                        "pid_ready": True,
                        "timestamp_ns": time.monotonic_ns(),
                    }
                )
                break
            if time.monotonic() >= deadline:
                fail("capture_process", TimeoutError("capture startup timed out"))
                break
            stop_event.wait(0.01)

        next_health_ns = time.monotonic_ns() + 1_000_000_000
        while not stop_event.is_set():
            if connection.poll(0.05):
                message = connection.recv()
                if (
                    isinstance(message, dict)
                    and message.get("type") == "control"
                    and message.get("command") == "stop"
                ):
                    stop_event.set()
                    break
            now_ns = time.monotonic_ns()
            if now_ns >= next_health_ns:
                duration_s = max((now_ns - started_at_ns) / 1_000_000_000, 1e-9)
                send(
                    {
                        "type": "health",
                        "state": "running",
                        "timestamp_ns": now_ns,
                        "screen_fps": stats["screen_frames"] / duration_s,
                        "cursor_hz": stats["cursor_samples"] / duration_s,
                    }
                )
                next_health_ns = now_ns + 1_000_000_000
    except (EOFError, OSError) as exc:
        fail("capture_control", exc)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(3.0)
        send(
            {
                "type": "health",
                "state": "stopped",
                "timestamp_ns": time.monotonic_ns(),
                **stats,
                "duration_ns": time.monotonic_ns() - started_at_ns,
            }
        )
        if shared is not None:
            shared.close()
        connection.close()


def _screen_reader(
    config: dict[str, Any],
) -> tuple[Callable[[], Any], Callable[[], None]]:
    mode = config.get("screen_mode", "real")
    if mode == "synthetic":
        import numpy as np

        shape = tuple(config.get("synthetic_screen_shape", (64, 64, 4)))
        image = np.zeros(shape, dtype=np.uint8)
        return lambda: image, lambda: None
    if mode == "error":
        raise OSError("capture unavailable")
    if mode != "real":
        raise ValueError(f"unknown screen capture mode: {mode}")
    import mss
    import numpy as np

    session = mss.mss()
    if len(session.monitors) < 2:
        session.close()
        raise RuntimeError("no desktop monitor is available")
    monitor = session.monitors[1]
    return lambda: np.asarray(session.grab(monitor)), session.close


def _cursor_reader(config: dict[str, Any]) -> Callable[[], tuple[int, int]]:
    mode = config.get("cursor_mode", "real")
    if mode == "synthetic":
        point = tuple(config.get("synthetic_cursor", (10, 10)))
        return lambda: (int(point[0]), int(point[1]))
    if mode == "error":
        raise OSError("cursor capture unavailable")
    if mode != "real":
        raise ValueError(f"unknown cursor capture mode: {mode}")

    class Point(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    def read_cursor() -> tuple[int, int]:
        point = Point()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            raise OSError("GetCursorPos failed")
        return int(point.x), int(point.y)

    return read_cursor
