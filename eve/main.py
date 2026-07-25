"""Safe executable entry point for the minimal EVE runtime."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eve.core.loop import CoreLoop, log_event
from eve.core.safegate import emergency_stop
from eve.core.tnn import SmokeActionNode, load_tnn
from eve.input.buffer import InputBuffer
from eve.input.capture import Capture, CaptureError
from eve.memory.memorizer import Memorizer
from eve.state import OutputMode, RuntimeErrorRecord, RuntimeState


class EVEApplication:
    """Owns the process-level resources and their shutdown order."""

    def __init__(
        self,
        *,
        mode: OutputMode = OutputMode.MOCK,
        run_dir: str | Path = "runs",
        capture: Capture | None = None,
    ) -> None:
        if mode == OutputMode.REAL:
            raise ValueError("the formal CLI does not enable real output")
        self.run_dir = Path(run_dir)
        self.state = RuntimeState(
            output_mode=mode,
            mouse_allowed=True,
            keyboard_allowed=True,
            speak_allowed=True,
        )
        self.buffer = InputBuffer()
        self.capture = capture or Capture(
            self.buffer,
            screen_reader=lambda: {"source": "synthetic_smoke"},
            cursor_reader=lambda: (10, 10),
            error_callback=self._capture_error,
        )
        self.memory = Memorizer(self.run_dir / "memory")
        self.core = CoreLoop(
            self.state,
            self.buffer,
            self.memory,
            log_dir=self.run_dir,
        )

    def start(self, *, load_smoke_node: bool = True) -> None:
        if self.state.cold_started:
            return
        self.state.cold_started = True
        if load_smoke_node:
            load_tnn(self.state, SmokeActionNode())
        self.capture.start()
        self.core.start()
        log_event(
            self.run_dir,
            "cold_start",
            output_mode=self.state.output_mode.value,
            smoke_rule=load_smoke_node,
        )

    def stop(self) -> None:
        failures: list[Exception] = []
        try:
            self.core.stop()
        except Exception as exc:
            failures.append(exc)
        try:
            self.capture.stop()
        except Exception as exc:
            failures.append(exc)
        try:
            self.state.save_snapshot(self.run_dir / "state_snapshot.json")
        except Exception as exc:
            failures.append(exc)
        self.state.cold_started = False
        if not failures:
            log_event(self.run_dir, "shutdown_complete")
            return
        raise RuntimeError(
            "shutdown failure(s): " + "; ".join(str(item) for item in failures)
        )

    def _capture_error(self, error: CaptureError) -> None:
        self.state.latest_error = RuntimeErrorRecord(
            timestamp_ns=error.timestamp_ns,
            loop_node="capture",
            exception_type=error.exception_type,
            message=error.message,
            traceback=error.traceback,
            relevant_source=error.source,
            recovery_action=error.recovery_action,
        )
        log_event(self.run_dir, "runtime_error", **error.__dict__)


def _esc_pressed() -> bool:
    try:
        import msvcrt

        return bool(msvcrt.kbhit() and msvcrt.getwch() == "\x1b")
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the safe EVE minimal loop")
    parser.add_argument("--mode", choices=("disabled", "mock"), default="mock")
    parser.add_argument("--smoke-seconds", type=float, default=1.0)
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args(argv)
    if args.smoke_seconds <= 0:
        parser.error("--smoke-seconds must be positive")

    app = EVEApplication(mode=OutputMode(args.mode), run_dir=args.run_dir)
    try:
        app.start()
        deadline = time.monotonic() + args.smoke_seconds
        while time.monotonic() < deadline:
            if _esc_pressed():
                emergency_stop(app.state)
                log_event(app.run_dir, "emergency_stop", source="escape_key")
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        emergency_stop(app.state)
        log_event(app.run_dir, "emergency_stop", source="keyboard_interrupt")
    finally:
        app.stop()

    summary = {
        "output_mode": app.state.output_mode.value,
        "executed": bool(app.state.latest_output and app.state.latest_output.executed),
        "simulated": bool(app.state.latest_output and app.state.latest_output.simulated),
        "memory_units": len(app.state.memory_ids),
        "error": app.state.latest_error.message if app.state.latest_error else None,
        "threads_stopped": not app.core.running and not app.capture.running,
        "log": str(app.run_dir / "eve.jsonl"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
