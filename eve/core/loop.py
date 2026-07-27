"""Readable EVE runtime chain: TNN -> action -> Safegate -> output -> Memory."""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eve.core import safegate
from eve.core.tnn import (
    SourceRef,
    due_tnn_ids,
    load_tnn_runtime as attach_tnn_runtime,
    run_node,
    unload_tnn_runtime as detach_tnn_runtime,
)
from eve.output import keyboard, mouse, speak
from eve.state import (
    ActionCandidate,
    ActionKind,
    OutputMode,
    OutputResult,
    RuntimeErrorRecord,
    RuntimeState,
    TimedValue,
)


def _dispatch_output(action: ActionCandidate, mode: OutputMode) -> OutputResult:
    if action.kind == ActionKind.MOUSE:
        return mouse.execute(action.action_id, action.payload, mode)
    if action.kind == ActionKind.KEYBOARD:
        return keyboard.execute(action.action_id, action.payload, mode)
    if action.kind == ActionKind.SPEAK:
        return speak.execute(action.action_id, action.payload, mode)
    raise ValueError(f"unknown action kind: {action.kind}")


def _blocked_result(
    state: RuntimeState, action: ActionCandidate, reason: str
) -> OutputResult:
    now_ns = time.monotonic_ns()
    return OutputResult(
        action_id=action.action_id,
        kind=action.kind.value,
        mode=state.output_mode.value,
        started_at_ns=now_ns,
        finished_at_ns=now_ns,
        blocked=True,
        reason=f"safegate_{reason}",
    )


def run_once(
    state: RuntimeState,
    action: ActionCandidate,
    log_dir: str | Path = "runs",
) -> OutputResult:
    """Synchronous safe action chain retained for focused tests and callers."""
    decision = safegate.check(state, action)
    state.publish(
        "latest_safegate_result",
        TimedValue(
            value=decision,
            produced_at_ns=decision.checked_at_ns,
            producer="safegate",
        ),
    )
    state.increment_stat(
        "safegate_allowed" if decision.allowed else "safegate_blocked"
    )
    result = (
        _dispatch_output(action, state.output_mode)
        if decision.allowed
        else _blocked_result(state, action, decision.reason)
    )
    state.latest_output = result
    state.publish(
        "latest_output_feedback",
        TimedValue(
            value=result,
            produced_at_ns=result.finished_at_ns,
            producer="output",
        ),
    )
    if result.simulated:
        state.increment_stat("mock_outputs")
    log_event(
        log_dir,
        "action_result",
        action=asdict(action),
        safegate=asdict(decision),
        output=asdict(result),
    )
    return result


def log_event(log_dir: str | Path, event: str, **fields: Any) -> Path:
    path = Path(log_dir) / "eve.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_ns": time.time_ns(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
    return path


class CoreLoop:
    """One independent runtime lifecycle with explicit start and verified stop."""

    def __init__(
        self,
        state: RuntimeState,
        input_buffer: Any,
        memorizer: Any,
        *,
        log_dir: str | Path = "runs",
        interval_s: float = 0.02,
        runtime_device: str = "cpu",
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.state = state
        self.input_buffer = input_buffer
        self.memorizer = memorizer
        self.log_dir = Path(log_dir)
        self.interval_s = interval_s
        self.runtime_device = runtime_device
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_ns = 0
        self._last_input_snapshot_ns = time.monotonic_ns()

    def load_tnn_runtime(
        self,
        tnn_id: str,
        version: str | None = None,
        *,
        input_refs: dict[str, SourceRef | str] | None = None,
        run_frequency_hz: float = 1.0,
        output_ttl_ns: int = 1_000_000_000,
        action_output: str | None = None,
        factory: str = "create_tnn",
    ) -> Any:
        """Load a persisted TNN into this Core lifecycle."""
        return attach_tnn_runtime(
            self.state,
            self.memorizer,
            tnn_id,
            version,
            device=self.runtime_device,
            input_refs=input_refs,
            run_frequency_hz=run_frequency_hz,
            output_ttl_ns=output_ttl_ns,
            action_output=action_output,
            factory=factory,
        )

    def unload_tnn_runtime(self, tnn_id: str) -> None:
        """Unload a persisted TNN and clear its live outputs and schedule state."""
        detach_tnn_runtime(self.state, tnn_id)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failed(self) -> bool:
        return self._failed_event.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._failed_event.clear()
        self._started_at_ns = time.monotonic_ns()
        self._last_input_snapshot_ns = self._started_at_ns
        self.state.loop_status["core"] = "running"
        self._thread = threading.Thread(target=self._run, name="eve-core")
        self._thread.start()
        log_event(self.log_dir, "core_started")

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout_s)
        if thread.is_alive():
            raise RuntimeError("core thread did not stop")
        self._thread = None
        self.state.loop_status["core"] = "failed" if self.failed else "stopped"
        log_event(self.log_dir, "core_stopped")

    def stats(self) -> dict[str, float | int]:
        duration_s = max(
            (time.monotonic_ns() - self._started_at_ns) / 1_000_000_000,
            1e-9,
        )
        iterations = self.state.runtime_stats.get("core_iterations", 0)
        return {
            "iterations": iterations,
            "loop_hz": iterations / duration_s,
            "tnn_invocations": self.state.runtime_stats.get(
                "tnn_invocations", 0
            ),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started_ns = time.monotonic_ns()
            try:
                self.step(started_ns)
            except Exception as exc:
                self._record_error(
                    "core",
                    exc,
                    {"cold_started": self.state.cold_started},
                    "core_stopped_no_real_output",
                    critical=True,
                )
            if self.failed:
                self._stop_event.set()
            elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
            self._stop_event.wait(max(0.001, self.interval_s - elapsed_s))

    def step(self, now_ns: int | None = None) -> list[OutputResult]:
        if (
            not self.state.cold_started
            or self.state.myself.sleep_requested
            or self.state.emergency_stopped
        ):
            return []
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        self.state.increment_stat("core_iterations")
        self.state.publish(
            "latest_input_summary",
            TimedValue(
                value=self._input_summary(),
                produced_at_ns=now_ns,
                producer="input_buffer",
            ),
        )
        self._maybe_remember_input(now_ns)
        for tnn_id in due_tnn_ids(self.state, now_ns):
            try:
                outputs = run_node(self.state, self.input_buffer, tnn_id, now_ns)
                if outputs is not None:
                    log_event(
                        self.log_dir,
                        "tnn_output",
                        tnn_id=tnn_id,
                        output_names=sorted(outputs),
                    )
            except Exception as exc:
                self.state.myself.active_tnn.discard(tnn_id)
                self.state.tnn_status[tnn_id] = "failed"
                self._record_error(
                    f"tnn:{tnn_id}",
                    exc,
                    {"tnn_id": tnn_id},
                    "node_paused_no_output",
                    critical=True,
                )

        results: list[OutputResult] = []
        while True:
            action = self.state.consume_action()
            if action is None:
                break
            try:
                result = run_once(self.state, action, self.log_dir)
                results.append(result)
                self._remember_chain(action, result)
            except Exception as exc:
                self._record_error(
                    "output",
                    exc,
                    {"action_id": action.action_id, "kind": action.kind.value},
                    "action_failed_no_retry",
                    critical=True,
                )
        self.state.myself.settle_hormones()
        return results

    def _remember_chain(
        self, action: ActionCandidate, result: OutputResult
    ) -> None:
        try:
            decision_value = self.state.read_latest("latest_safegate_result")
            records = [
                ("action_candidate", asdict(action)),
                (
                    "safegate_result",
                    asdict(decision_value.value) if decision_value else {},
                ),
                ("output_result", asdict(result)),
            ]
            recorded_ids: list[str] = []
            for payload_type, payload in records:
                memory_id = self.memorizer.enqueue(
                    payload,
                    payload_type,
                    priority="critical",
                )
                if memory_id is not None:
                    self.state.memory_ids.append(memory_id)
                    recorded_ids.append(memory_id)
            log_event(
                self.log_dir,
                "memory_recorded",
                action_id=action.action_id,
                memory_ids=recorded_ids,
            )
        except Exception as exc:
            self._record_error(
                "memory",
                exc,
                {"action_id": action.action_id},
                "result_kept_in_runtime_memory",
                critical=True,
            )

    def _maybe_remember_input(self, now_ns: int) -> None:
        if now_ns - self._last_input_snapshot_ns < 1_000_000_000:
            return
        memory_id = self.memorizer.enqueue(
            self._input_summary(),
            "input_snapshot",
            priority="low",
        )
        self._last_input_snapshot_ns = now_ns
        if memory_id is not None:
            self.state.memory_ids.append(memory_id)

    def _input_summary(self) -> dict[str, Any]:
        screen = self.input_buffer.latest("screen")
        cursor = self.input_buffer.latest("cursor")
        screen_value = screen.value if screen is not None else None
        cursor_value = cursor.value if cursor is not None else None
        if cursor_value is not None and hasattr(cursor_value, "x"):
            cursor_x = cursor_value.x
            cursor_y = cursor_value.y
        elif isinstance(cursor_value, (tuple, list)) and len(cursor_value) >= 2:
            cursor_x, cursor_y = cursor_value[:2]
        else:
            cursor_x = cursor_y = None
        return {
            "screen": (
                {
                    "frame_id": getattr(screen_value, "frame_id", screen.index),
                    "timestamp_ns": screen.timestamp_ns,
                    "shape": list(getattr(
                        getattr(screen_value, "image", None),
                        "shape",
                        (),
                    )),
                }
                if screen is not None
                else None
            ),
            "cursor": (
                {
                    "frame_id": getattr(cursor_value, "frame_id", cursor.index),
                    "timestamp_ns": cursor.timestamp_ns,
                    "x": cursor_x,
                    "y": cursor_y,
                    "speed": getattr(cursor_value, "speed", 0.0),
                }
                if cursor is not None
                else None
            ),
        }

    def _record_error(
        self,
        loop_node: str,
        exc: Exception,
        relevant_source: Any,
        recovery_action: str,
        *,
        critical: bool = False,
    ) -> None:
        error = RuntimeErrorRecord(
            timestamp_ns=time.time_ns(),
            loop_node=loop_node,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
            relevant_source=relevant_source,
            recovery_action=recovery_action,
        )
        self.state.latest_error = error
        if critical:
            self._failed_event.set()
        log_event(self.log_dir, "runtime_error", **asdict(error))
        if loop_node != "memory":
            self.memorizer.enqueue(
                asdict(error),
                "runtime_error",
                priority="critical",
            )
