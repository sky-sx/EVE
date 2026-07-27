"""Safe process entry point for smoke and real-input observation profiles."""
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

from eve.core.loop import CoreLoop, log_event
from eve.core.safegate import emergency_stop, report_human_activity
from eve.core.tnn import SmokeActionNode, SourceRef, load_tnn
from eve.input.buffer import InputBuffer
from eve.input.capture import Capture, CaptureError, CursorState
from eve.memory.memorizer import Memorizer
from eve.state import OutputMode, RuntimeErrorRecord, RuntimeState


class EVEApplication:
    """Own one complete runtime lifecycle and its verified shutdown order."""

    def __init__(
        self,
        *,
        profile: str = "smoke",
        mode: OutputMode | None = None,
        run_dir: str | Path = "runs",
        memory_dir: str | Path | None = None,
        capture: Capture | None = None,
        tnn_id: str | None = None,
        allow_mock_actions: bool | None = None,
    ) -> None:
        if profile not in {"smoke", "observe"}:
            raise ValueError(f"unsupported runtime profile: {profile}")
        if mode == OutputMode.REAL:
            raise ValueError("real output is disabled in this integration stage")
        self.profile = profile
        self.run_dir = Path(run_dir)
        self.tnn_id = tnn_id
        self.buffer = capture.buffer if capture is not None else InputBuffer()
        legacy_permissions = mode is not None
        if allow_mock_actions is None:
            allow_mock_actions = legacy_permissions
        self.state = RuntimeState(
            output_mode=mode or OutputMode.MOCK,
            mouse_allowed=allow_mock_actions,
            keyboard_allowed=allow_mock_actions,
            speak_allowed=allow_mock_actions,
        )
        self._critical_event = threading.Event()
        self._stop_requested = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._previous_cursor: tuple[int, int] | None = None
        self.started_at_ns = 0
        self.finished_at_ns = 0
        self.exit_reason = "not_started"
        self.memory = Memorizer(
            Path(memory_dir) if memory_dir is not None else self.run_dir / "memory",
            writer_error_callback=self._memory_error,
        )
        if capture is not None:
            self.capture = capture
            self.capture._error_callback = self._capture_error
        elif profile == "smoke":
            self.capture = Capture(
                self.buffer,
                screen_fps=30.0,
                cursor_hz=60.0,
                screen_reader=lambda: {"source": "synthetic_smoke"},
                cursor_reader=lambda: (10, 10),
                error_callback=self._capture_error,
            )
        else:
            self.capture = Capture(
                self.buffer,
                screen_fps=30.0,
                cursor_hz=60.0,
                cursor_callback=self._cursor_observed,
                error_callback=self._capture_error,
            )
        self.core = CoreLoop(
            self.state,
            self.buffer,
            self.memory,
            log_dir=self.run_dir,
        )

    @property
    def running(self) -> bool:
        return self.state.cold_started and not self._stop_requested.is_set()

    @property
    def critical_failure(self) -> bool:
        return self._critical_event.is_set() or self.core.failed

    def start(self, *, load_smoke_node: bool = True) -> None:
        if self.state.cold_started:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_ns = time.monotonic_ns()
        self.exit_reason = "running"
        self._critical_event.clear()
        self._stop_requested.clear()
        self.memory.start_writer()
        self.state.resource_status.update(
            {
                "memory_writer": "running",
                "capture": "starting",
                "core": "starting",
            }
        )
        self.state.cold_started = True
        try:
            if self.tnn_id is not None:
                self.state.active_tnn.add(self.tnn_id)
                artifact = self.memory.resolve_tnn_artifact(self.tnn_id)
                structure = json.loads(
                    Path(artifact["structure_path"]).read_text(encoding="utf-8")
                )
                refs = {
                    name: SourceRef(
                        f"state:{name}"
                        if name in {"screen", "cursor"}
                        else f"blackboard:{name}"
                    )
                    for name in structure.get("input_schema", {})
                }
                self.core.load_tnn_runtime(self.tnn_id, input_refs=refs)
            elif load_smoke_node:
                self.state.active_tnn.add("smoke_rule")
                load_tnn(self.state, SmokeActionNode(), activate=False)
            self.capture.start()
            self.state.resource_status["capture"] = "running"
            self.core.start()
            self.state.resource_status["core"] = "running"
            self._watch_thread = threading.Thread(
                target=self._watch_stop,
                name="eve-stop-watch",
            )
            self._watch_thread.start()
            log_event(
                self.run_dir,
                "runtime_started",
                profile=self.profile,
                output_mode=self.state.output_mode.value,
                permissions=self.state.permissions,
                tnn_id=self.tnn_id or "smoke_rule",
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
        deadline = (
            time.monotonic() + duration_s if duration_s is not None else None
        )
        while not self._stop_requested.wait(0.02):
            if deadline is not None and time.monotonic() >= deadline:
                self.exit_reason = "duration_elapsed"
                self._stop_requested.set()
                break
        if self.core.failed and self.state.latest_error is not None:
            self._critical_event.set()
            self.exit_reason = "core_error"
        return not self.critical_failure

    def request_stop(self, reason: str = "requested") -> None:
        if self.exit_reason == "running":
            self.exit_reason = reason
        self._stop_requested.set()

    def stop(self) -> None:
        self._stop_requested.set()
        failures: list[Exception] = []
        try:
            self.capture.stop()
        except Exception as exc:
            failures.append(exc)
        try:
            self.core.stop()
        except Exception as exc:
            failures.append(exc)
        try:
            self.memory.stop_writer()
        except Exception as exc:
            failures.append(exc)
        self.buffer.close()
        self.state.resource_status.update(
            {
                "capture": "stopped",
                "core": "stopped",
                "memory_writer": "stopped",
                "input_buffer": "closed",
            }
        )
        self.state.cold_started = False
        watch = self._watch_thread
        if watch is not None and watch is not threading.current_thread():
            watch.join(3.0)
            if watch.is_alive():
                failures.append(RuntimeError("stop watch thread did not stop"))
        self._watch_thread = None
        try:
            self.state.save_snapshot(self.run_dir / "state_snapshot.json")
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
            if self.started_at_ns
            else 0.0
        )
        capture_stats = self.capture.stats()
        core_stats = self.core.stats()
        memory_stats = self.memory.writer_stats()
        return {
            "profile": self.profile,
            "duration_s": duration_s,
            "screen_fps": capture_stats["screen_fps"],
            "cursor_hz": capture_stats["cursor_hz"],
            "screen_latency_ms": capture_stats[
                "screen_average_latency_ms"
            ],
            "cursor_latency_ms": capture_stats[
                "cursor_average_latency_ms"
            ],
            "core_loop_hz": core_stats["loop_hz"],
            "tnn_invocations": core_stats["tnn_invocations"],
            "safegate_allowed": self.state.runtime_stats.get(
                "safegate_allowed", 0
            ),
            "safegate_blocked": self.state.runtime_stats.get(
                "safegate_blocked", 0
            ),
            "mock_outputs": self.state.runtime_stats.get("mock_outputs", 0),
            "real_output_calls": 0,
            "memory_written": memory_stats["written"],
            "memory_dropped": memory_stats["dropped"],
            "memory_failed": memory_stats["failed"],
            "critical_error": (
                self.state.latest_error.message
                if self.state.latest_error is not None
                else memory_stats["last_error"]
            ),
            "exit_reason": self.exit_reason,
            "threads_stopped": not (
                self.capture.running
                or self.core.running
                or self.memory.writer_running
                or (
                    self._watch_thread is not None
                    and self._watch_thread.is_alive()
                )
            ),
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
            if self.core.failed:
                self._critical_event.set()
                self.exit_reason = "core_error"
                self._stop_requested.set()
                return

    def _cursor_observed(self, cursor: CursorState) -> None:
        current = (cursor.x, cursor.y)
        if self._previous_cursor is not None and current != self._previous_cursor:
            self.state.human_activity_detected_at_ns = cursor.captured_at_ns
            report_human_activity(self.state)
        self._previous_cursor = current

    def _capture_error(self, error: CaptureError) -> None:
        self.state.latest_error = RuntimeErrorRecord(
            timestamp_ns=error.timestamp_ns,
            loop_node=error.source,
            exception_type=error.exception_type,
            message=error.message,
            traceback=error.traceback,
            relevant_source=error.source,
            recovery_action=error.recovery_action,
        )
        self._critical_event.set()
        self.exit_reason = "capture_error"
        self._stop_requested.set()
        log_event(self.run_dir, "runtime_error", **error.__dict__)

    def _memory_error(self, error: Exception) -> None:
        self.state.latest_error = RuntimeErrorRecord(
            timestamp_ns=time.time_ns(),
            loop_node="memory_writer",
            exception_type=type(error).__name__,
            message=str(error),
            traceback="".join(traceback.format_exception(error)),
            relevant_source="memory",
            recovery_action="runtime_stopped_for_memory_integrity",
        )
        self._critical_event.set()
        self.exit_reason = "memory_error"
        self._stop_requested.set()


def _global_escape_pressed() -> bool:
    """Use the Windows global key state; console focus is not required."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except (AttributeError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the safe EVE integration")
    parser.add_argument(
        "--profile",
        choices=("smoke", "observe", "control"),
        default="smoke",
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
        if application.state.latest_error is None:
            application.state.latest_error = RuntimeErrorRecord(
                timestamp_ns=time.time_ns(),
                loop_node="main",
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                relevant_source=args.profile,
                recovery_action="startup_or_runtime_aborted",
            )
    finally:
        try:
            application.stop()
        except Exception:
            exit_code = 1
    summary = application.summary()
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
