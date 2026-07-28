"""Speech output boundary with one bounded, interruptible TTS worker."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

_QUEUE: queue.Queue[tuple[str, str, dict[str, Any]] | None] = queue.Queue(
    maxsize=16
)
_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_ENGINE: Any | None = None
_LAST_ERROR: str | None = None


def _worker() -> None:
    global _ENGINE, _LAST_ERROR
    try:
        import pyttsx3

        _ENGINE = pyttsx3.init()
        while not _STOP.is_set():
            try:
                item = _QUEUE.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            _, text, options = item
            if _STOP.is_set():
                return
            try:
                if options.get("rate") is not None:
                    _ENGINE.setProperty("rate", int(options["rate"]))
                if options.get("volume") is not None:
                    _ENGINE.setProperty("volume", float(options["volume"]))
                _ENGINE.say(text)
                _ENGINE.runAndWait()
            except Exception as exc:
                _LAST_ERROR = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        _ENGINE = None


def _ensure_worker() -> bool:
    global _THREAD
    with _LOCK:
        if _STOP.is_set():
            return False
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_worker, name="eve-tts")
            _THREAD.start()
    return True


def stop_all(timeout_s: float = 3.0) -> None:
    """Cancel queued speech and interrupt the active utterance where supported."""
    global _THREAD, _LAST_ERROR
    _STOP.set()
    while True:
        try:
            _QUEUE.get_nowait()
        except queue.Empty:
            break
    engine = _ENGINE
    if engine is not None:
        try:
            engine.stop()
        except Exception as exc:
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
    try:
        _QUEUE.put_nowait(None)
    except queue.Full:
        pass
    thread = _THREAD
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout_s)
    _THREAD = None


def reset_stop() -> None:
    _STOP.clear()


def status() -> dict[str, Any]:
    thread = _THREAD
    return {
        "running": bool(thread and thread.is_alive()),
        "queue_depth": _QUEUE.qsize(),
        "last_error": _LAST_ERROR,
    }


def execute(
    action_id: str, payload: dict[str, Any], mode: str
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    common = {
        "action_id": action_id,
        "kind": "speak",
        "mode": mode,
        "started_at_ns": started_ns,
        "finished_at_ns": time.monotonic_ns(),
        "executed": False,
        "payload": dict(payload),
    }
    if mode == "disabled":
        return {
            **common,
            "simulated": False,
            "blocked": True,
            "reason": "output_disabled",
        }
    if mode == "mock":
        return {
            **common,
            "simulated": True,
            "blocked": False,
            "reason": "mock_ok",
        }
    if mode != "real":
        raise ValueError(f"unknown output mode: {mode}")
    text = str(payload.get("text", ""))
    if not text:
        return {
            **common,
            "simulated": False,
            "blocked": True,
            "reason": "empty_text",
        }
    if not _ensure_worker():
        return {
            **common,
            "simulated": False,
            "blocked": True,
            "reason": "emergency_stopped",
        }
    try:
        _QUEUE.put_nowait((action_id, text, dict(payload)))
    except queue.Full:
        return {
            **common,
            "simulated": False,
            "blocked": True,
            "reason": "tts_queue_full",
        }
    return {
        **common,
        "finished_at_ns": time.monotonic_ns(),
        "executed": True,
        "simulated": False,
        "blocked": False,
        "reason": "tts_queued",
    }
