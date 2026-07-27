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
    result = (
        _dispatch_output(action, state.output_mode)
        if decision.allowed
        else _blocked_result(state, action, decision.reason)
    )
    state.latest_output = result
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
        self._thread: threading.Thread | None = None

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

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
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
        log_event(self.log_dir, "core_stopped")

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
                    "iteration_skipped_no_real_output",
                )
            elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
            self._stop_event.wait(max(0.001, self.interval_s - elapsed_s))

    def step(self, now_ns: int | None = None) -> list[OutputResult]:
        if not self.state.cold_started or self.state.myself.sleep_requested:
            return []
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
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
                self._record_error(
                    f"tnn:{tnn_id}",
                    exc,
                    {"tnn_id": tnn_id},
                    "node_paused_no_output",
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
                )
        self.state.myself.settle_hormones()
        return results

    def _remember_chain(
        self, action: ActionCandidate, result: OutputResult
    ) -> None:
        try:
            snapshot = {
                kind: [
                    {
                        "timestamp_ns": sample.timestamp_ns,
                        "index": sample.index,
                        "value": sample.value,
                    }
                    for sample in samples
                ]
                for kind, samples in self.input_buffer.snapshot().items()
            }
            records = [
                ("input_snapshot", snapshot),
                (
                    "tnn_output",
                    {
                        "origin": action.origin,
                        "outputs": {
                            name: asdict(value)
                            for name, value in self.state.tnn_outputs.get(
                                action.origin, {}
                            ).items()
                        },
                    },
                ),
                ("output_result", asdict(result)),
            ]
            for payload_type, payload in records:
                memory_id = self.memorizer.create(payload, payload_type)
                self.state.memory_ids.append(memory_id)
            log_event(
                self.log_dir,
                "memory_recorded",
                action_id=action.action_id,
                memory_ids=self.state.memory_ids[-3:],
            )
        except Exception as exc:
            self._record_error(
                "memory",
                exc,
                {"action_id": action.action_id},
                "result_kept_in_runtime_memory",
            )

    def _record_error(
        self,
        loop_node: str,
        exc: Exception,
        relevant_source: Any,
        recovery_action: str,
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
        log_event(self.log_dir, "runtime_error", **asdict(error))
