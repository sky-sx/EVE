"""Safe process entry point; Main controls Input only through InputBuffer."""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from eve.core.loop import CoreLoop, create_runtime_state, log_event
from eve.core.safegate import emergency_stop
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer


class EVEApplication:
    """Own one runtime and enforce its shutdown order."""

    def __init__(
        self,
        *,
        profile: str = "smoke",
        mode: str | None = None,
        run_dir: str | Path = "runs",
        memory_dir: str | Path | None = None,
        input_buffer: InputBuffer | None = None,
        tnn_id: str | None = None,
        allow_mock_actions: bool = False,
    ) -> None:
        if profile not in {"smoke", "observe"}:
            raise ValueError(f"unsupported runtime profile: {profile}")
        normalized_mode = getattr(mode, "value", mode) or "mock"
        if normalized_mode == "real":
            raise ValueError("real output is disabled in this integration stage")
        self.profile = profile
        self.run_dir = Path(run_dir)
        self.buffer = input_buffer or InputBuffer(profile=profile)
        self.state = create_runtime_state(
            output_mode=normalized_mode,
            allow_mock_actions=allow_mock_actions,
        )
        self._critical_event = threading.Event()
        self._stop_requested = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self.started_at_ns = 0
        self.finished_at_ns = 0
        self.exit_reason = "not_started"
        self.memory = Memorizer(
            Path(memory_dir) if memory_dir is not None else self.run_dir / "memory",
            writer_error_callback=self._memory_error,
        )
        self.core = CoreLoop(
            self.buffer,
            self.memory,
            state=self.state,
            log_dir=self.run_dir,
            tnn_id=tnn_id,
            smoke_node=tnn_id is None,
        )

    @property
    def running(self) -> bool:
        return self.state["cold_started"] and not self._stop_requested.is_set()

    @property
    def critical_failure(self) -> bool:
        return self._critical_event.is_set() or self.core.failed

    def start(self, *, load_smoke_node: bool = True) -> None:
        if self.running:
            return
        self.core.smoke_node = load_smoke_node and self.core.tnn_id is None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_ns = time.monotonic_ns()
        self.exit_reason = "running"
        self._critical_event.clear()
        self._stop_requested.clear()
        self.memory.start_writer()
        self.state["resource_status"].update(
            {"memory_writer": "running", "capture": "starting", "core": "starting"}
        )
        try:
            self.buffer.start_capture()
            self.state["resource_status"]["capture"] = "running"
            self.core.start()
            self.state["resource_status"]["core"] = "running"
            self._watch_thread = threading.Thread(
                target=self._watch_stop, name="eve-stop-watch"
            )
            self._watch_thread.start()
            log_event(
                self.run_dir,
                "runtime_started",
                profile=self.profile,
                output_mode=self.state["output_mode"],
                permissions=self.state["permissions"],
                tnn_id=self.core.tnn_id or "smoke_rule",
                capture_pid=self.buffer.capture_process_id,
            )
        except Exception:
            self.exit_reason = "startup_error"
            self._critical_event.set()
            self._stop_requested.set()
            try:
                self.stop()
            except Exception:
                pass
            raise

    def wait(self, duration_s: float | None = None) -> bool:
        deadline = time.monotonic() + duration_s if duration_s is not None else None
        while not self._stop_requested.wait(0.02):
            if deadline is not None and time.monotonic() >= deadline:
                self.exit_reason = "duration_elapsed"
                self._stop_requested.set()
                break
        if self.core.failed:
            self._critical_event.set()
            self.exit_reason = "core_error"
        return not self.critical_failure

    def request_stop(self, reason: str = "requested") -> None:
        if self.exit_reason == "running":
            self.exit_reason = reason
        self._stop_requested.set()

    def stop(self) -> None:
        """Stop Loop and Memory first, then Buffer closes Capture and shared memory."""
        self._stop_requested.set()
        failures: list[Exception] = []
        for stop in (self.core.stop, self.memory.stop_writer, self.buffer.close):
            try:
                stop()
            except Exception as exc:
                failures.append(exc)
        self.state["resource_status"].update(
            {
                "core": "stopped",
                "memory_writer": "stopped",
                "capture": "stopped",
                "input_buffer": "closed",
            }
        )
        watch = self._watch_thread
        if watch is not None and watch is not threading.current_thread():
            watch.join(3.0)
            if watch.is_alive():
                failures.append(RuntimeError("stop watch thread did not stop"))
        self._watch_thread = None
        try:
            self.core.save_snapshot(self.run_dir / "state_snapshot.json")
        except Exception as exc:
            failures.append(exc)
        self.finished_at_ns = time.monotonic_ns()
        if self.exit_reason == "running":
            self.exit_reason = "stopped"
        if failures:
            self._critical_event.set()
            self.exit_reason = "shutdown_error"
            raise RuntimeError(
                "shutdown failure(s): " + "; ".join(str(item) for item in failures)
            )
        log_event(self.run_dir, "shutdown_complete", reason=self.exit_reason)

    def summary(self) -> dict[str, Any]:
        finished_ns = self.finished_at_ns or time.monotonic_ns()
        duration_s = (
            max(0, finished_ns - self.started_at_ns) / 1_000_000_000
            if self.started_at_ns else 0.0
        )
        capture = self.buffer.capture_stats()
        core = self.core.stats()
        memory = self.memory.writer_stats()
        stats = self.state["runtime_stats"]
        latest_error = self.state["latest_error"]
        return {
            "profile": self.profile,
            "duration_s": duration_s,
            "screen_fps": capture["screen_fps"],
            "cursor_hz": capture["cursor_hz"],
            "screen_latency_ms": capture["screen_average_latency_ms"],
            "cursor_latency_ms": capture["cursor_average_latency_ms"],
            "core_loop_hz": core["loop_hz"],
            "tnn_invocations": core["tnn_invocations"],
            "safegate_allowed": stats.get("safegate_allowed", 0),
            "safegate_blocked": stats.get("safegate_blocked", 0),
            "mock_outputs": stats.get("mock_outputs", 0),
            "real_output_calls": 0,
            "memory_written": memory["written"],
            "memory_dropped": memory["dropped"],
            "memory_failed": memory["failed"],
            "critical_error": (
                latest_error.get("message") if latest_error else memory["last_error"]
            ),
            "exit_reason": self.exit_reason,
            "threads_stopped": not (
                self.buffer.capture_running
                or self.core.running
                or self.memory.writer_running
                or (
                    self._watch_thread is not None
                    and self._watch_thread.is_alive()
                )
            ),
            "capture_process_stopped": not self.buffer.capture_running,
            "log": str(self.run_dir / "eve.jsonl"),
        }

    def _watch_stop(self) -> None:
        while not self._stop_requested.wait(0.01):
            if _global_escape_pressed():
                emergency_stop(self.state)
                self.exit_reason = "escape_key"
                log_event(self.run_dir, "emergency_stop", source="global_escape")
                self._stop_requested.set()
                return
            capture_error = self.buffer.capture_error
            if capture_error is not None:
                self.state["latest_error"] = capture_error
                self._critical_event.set()
                self.exit_reason = "capture_error"
                self._stop_requested.set()
                return
            if not self.buffer.capture_running:
                self.state["latest_error"] = {
                    "loop_node": "capture_process",
                    "exception_type": "ProcessExit",
                    "message": "Capture exited unexpectedly",
                }
                self._critical_event.set()
                self.exit_reason = "capture_error"
                self._stop_requested.set()
                return
            if self.core.failed:
                self._critical_event.set()
                self.exit_reason = "core_error"
                self._stop_requested.set()
                return

    def _memory_error(self, error: Exception) -> None:
        self.state["latest_error"] = {
            "timestamp_ns": time.time_ns(),
            "loop_node": "memory_writer",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "recovery_action": "runtime_stopped_for_memory_integrity",
        }
        self._critical_event.set()
        self.exit_reason = "memory_error"
        self._stop_requested.set()


def _global_escape_pressed() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except (AttributeError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the safe EVE integration")
    parser.add_argument(
        "--profile", choices=("smoke", "observe", "control"), default="smoke"
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--tnn-id")
    parser.add_argument("--memory-dir")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.profile == "control":
        print(
            "control profile is not enabled in this integration stage",
            file=sys.stderr,
        )
        return 2
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    duration = args.duration
    if args.profile == "smoke" and duration is None:
        duration = 1.0
    application = EVEApplication(
        profile=args.profile,
        run_dir=args.run_dir,
        memory_dir=args.memory_dir,
        tnn_id=args.tnn_id,
        allow_mock_actions=False,
    )
    exit_code = 0
    try:
        application.start()
        if not application.wait(duration):
            exit_code = 1
    except KeyboardInterrupt:
        emergency_stop(application.state)
        application.exit_reason = "keyboard_interrupt"
    except Exception as exc:
        exit_code = 1
        if application.state["latest_error"] is None:
            application.state["latest_error"] = {
                "timestamp_ns": time.time_ns(),
                "loop_node": "main",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "recovery_action": "startup_or_runtime_aborted",
            }
    finally:
        try:
            application.stop()
        except Exception:
            exit_code = 1
    print(json.dumps(application.summary(), ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
