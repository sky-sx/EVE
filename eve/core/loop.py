"""EVE Core loop, runtime state, and live TNN execution."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable

from eve.core import safegate
from eve.output import keyboard, mouse, speak


def create_runtime_state(
    *, output_mode: str = "mock", allow_mock_actions: bool = False
) -> dict[str, Any]:
    """Create the small authoritative state owned by Core."""
    if output_mode not in {"disabled", "mock", "real"}:
        raise ValueError(f"unknown output mode: {output_mode}")
    return {
        "cold_started": False,
        "world": {},
        "myself": {},
        "blackboard": {},
        "active_tnn": set(),
        "loaded_tnn": {},
        "tnn_status": {},
        "loop_status": {"core": "not_started"},
        "permissions": {
            name: bool(allow_mock_actions)
            for name in ("mouse", "keyboard", "speak")
        },
        "resource_status": {},
        "emergency_stop": False,
        "latest_error": None,
        "output_mode": output_mode,
        "human_activity_detected_at_ns": 0,
        "human_takeover_until_ns": 0,
        "action_queue": deque(),
        "consumed_action_ids": set(),
        "last_run_ns": {},
        "tnn_outputs": {},
        "runtime_stats": {},
        "latest_output": None,
        "memory_ids": [],
    }


def register_runtime_tnn(
    state: dict[str, Any],
    tnn_id: str,
    run: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    inputs: dict[str, str] | None = None,
    outputs: tuple[str, ...] = (),
    run_frequency_hz: float = 1.0,
    output_ttl_ns: int = 1_000_000_000,
    action_output: str | None = None,
    model: Any = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Register one live node without a descriptor/adapter hierarchy."""
    if not tnn_id:
        raise ValueError("tnn_id is required")
    if tnn_id in state["loaded_tnn"]:
        raise ValueError(f"TNN already loaded: {tnn_id}")
    if run_frequency_hz <= 0:
        raise ValueError("run_frequency_hz must be positive")
    node = {
        "tnn_id": tnn_id,
        "run": run,
        "inputs": dict(inputs or {}),
        "outputs": tuple(outputs),
        "run_frequency_hz": float(run_frequency_hz),
        "output_ttl_ns": int(output_ttl_ns),
        "action_output": action_output,
        "model": model,
    }
    state["loaded_tnn"][tnn_id] = node
    state["tnn_status"][tnn_id] = "loaded"
    if activate:
        state["active_tnn"].add(tnn_id)
    return node


def unregister_runtime_tnn(state: dict[str, Any], tnn_id: str) -> Any:
    node = state["loaded_tnn"].pop(tnn_id, None)
    state["active_tnn"].discard(tnn_id)
    state["tnn_status"][tnn_id] = "unloaded"
    state["last_run_ns"].pop(tnn_id, None)
    state["tnn_outputs"].pop(tnn_id, None)
    return node


def _timed(
    value: Any, produced_at_ns: int, ttl_ns: int = 0, producer: str = ""
) -> dict[str, Any]:
    return {
        "value": value,
        "produced_at_ns": produced_at_ns,
        "valid_until_ns": produced_at_ns + ttl_ns if ttl_ns else 0,
        "producer": producer,
    }


def _resolve(
    state: dict[str, Any], input_buffer: Any, reference: str, now_ns: int
) -> tuple[Any, int]:
    if reference.startswith("tnn:"):
        tnn_id, separator, field = reference[4:].rpartition(".")
        if not separator or not tnn_id or not field:
            raise ValueError(f"invalid TNN input reference: {reference}")
        item = state["tnn_outputs"].get(tnn_id, {}).get(field)
    else:
        source, separator, key = reference.partition(":")
        if not separator or source not in {
            "state", "world", "myself", "blackboard"
        }:
            raise ValueError(f"invalid input reference: {reference}")
        if source == "state":
            sample = input_buffer.latest(key)
            return (
                (sample.value, sample.timestamp_ns)
                if sample is not None
                else (None, 0)
            )
        if source in {"world", "myself"}:
            return state[source].get(key), now_ns
        item = state["blackboard"].get(key)
    if item is None:
        return None, 0
    valid_until_ns = int(item.get("valid_until_ns", 0))
    if valid_until_ns and now_ns > valid_until_ns:
        return None, 0
    return item.get("value"), int(item.get("produced_at_ns", 0))


def _enqueue_action(
    state: dict[str, Any],
    node: dict[str, Any],
    value: dict[str, Any],
    now_ns: int,
    observed_at_ns: int,
) -> None:
    candidate_id = str(
        value.get("candidate_id", value.get("action_id", f"{node['tnn_id']}:{now_ns}"))
    )
    if candidate_id in state["consumed_action_ids"] or any(
        item["candidate_id"] == candidate_id for item in state["action_queue"]
    ):
        return
    ttl_ns = int(value.get("horizon_ns", node["output_ttl_ns"]))
    action = {
        "candidate_id": candidate_id,
        "source": node["tnn_id"],
        "action_type": str(value.get("action_type", value.get("kind", ""))),
        "payload": dict(value.get("payload", {})),
        "observed_at_ns": observed_at_ns or now_ns,
        "generated_at_ns": now_ns,
        "valid_until_ns": now_ns + ttl_ns if ttl_ns else 0,
    }
    state["action_queue"].append(action)
    state["blackboard"]["latest_action_candidate"] = _timed(
        action, now_ns, ttl_ns, node["tnn_id"]
    )


def _run_tnn(
    state: dict[str, Any], input_buffer: Any, tnn_id: str, now_ns: int
) -> dict[str, Any] | None:
    node = state["loaded_tnn"][tnn_id]
    inputs: dict[str, Any] = {}
    observed_times: list[int] = []
    for name, reference in node["inputs"].items():
        value, observed_ns = _resolve(state, input_buffer, reference, now_ns)
        if value is None:
            return None
        inputs[name] = value
        observed_times.append(observed_ns)
    outputs = node["run"](inputs)
    if not isinstance(outputs, dict):
        raise TypeError(f"{tnn_id} must return a dict")
    unknown = set(outputs) - set(node["outputs"])
    if unknown:
        raise ValueError(f"{tnn_id} produced undeclared outputs: {sorted(unknown)}")
    stats = state["runtime_stats"]
    stats["tnn_invocations"] = stats.get("tnn_invocations", 0) + 1
    state["last_run_ns"][tnn_id] = now_ns
    state["tnn_outputs"][tnn_id] = {
        name: _timed(value, now_ns, node["output_ttl_ns"], tnn_id)
        for name, value in outputs.items()
    }
    state["blackboard"]["latest_tnn_output"] = _timed(
        {"tnn_id": tnn_id, "outputs": outputs},
        now_ns,
        node["output_ttl_ns"],
        tnn_id,
    )
    action_output = node["action_output"]
    if action_output and action_output in outputs:
        if not isinstance(outputs[action_output], dict):
            raise TypeError("action output must be a dict")
        _enqueue_action(
            state,
            node,
            outputs[action_output],
            now_ns,
            max(observed_times, default=now_ns),
        )
    return outputs


def _dispatch(action: dict[str, Any], mode: str) -> dict[str, Any]:
    executors = {"mouse": mouse, "keyboard": keyboard, "speak": speak}
    executor = executors.get(action["action_type"])
    if executor is None:
        raise ValueError(f"unknown action type: {action['action_type']}")
    return executor.execute(action["candidate_id"], action["payload"], mode)


def run_once(
    state: dict[str, Any],
    action: dict[str, Any],
    log_dir: str | Path = "runs",
) -> dict[str, Any]:
    """Run the non-bypassable Core -> Safegate -> Output chain."""
    decision = safegate.check(state, action)
    state["blackboard"]["latest_safegate_result"] = _timed(
        decision, decision["checked_at_ns"], producer="safegate"
    )
    stat = "safegate_allowed" if decision["allowed"] else "safegate_blocked"
    state["runtime_stats"][stat] = state["runtime_stats"].get(stat, 0) + 1
    if decision["allowed"]:
        result = _dispatch(action, state["output_mode"])
    else:
        now_ns = time.monotonic_ns()
        result = {
            "action_id": action["candidate_id"],
            "kind": action["action_type"],
            "mode": state["output_mode"],
            "started_at_ns": now_ns,
            "finished_at_ns": now_ns,
            "executed": False,
            "simulated": False,
            "blocked": True,
            "reason": f"safegate_{decision['reason']}",
            "payload": {},
        }
    state["latest_output"] = result
    state["blackboard"]["latest_output_feedback"] = _timed(
        result, result["finished_at_ns"], producer="output"
    )
    if result.get("simulated"):
        state["runtime_stats"]["mock_outputs"] = (
            state["runtime_stats"].get("mock_outputs", 0) + 1
        )
    log_event(log_dir, "action_result", action=action, safegate=decision, output=result)
    return result


def log_event(log_dir: str | Path, event: str, **fields: Any) -> Path:
    path = Path(log_dir) / "eve.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"timestamp_ns": time.time_ns(), "event": event, **fields},
                ensure_ascii=False,
                default=repr,
            )
            + "\n"
        )
    return path


class CoreLoop:
    """Own runtime state, TNN lifecycles, and the Core worker."""

    def __init__(
        self,
        input_buffer: Any,
        memorizer: Any,
        *,
        state: dict[str, Any] | None = None,
        log_dir: str | Path = "runs",
        interval_s: float = 0.02,
        runtime_device: str | None = None,
        tnn_id: str | None = None,
        smoke_node: bool = False,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.state = state or create_runtime_state()
        self.input_buffer = input_buffer
        self.memorizer = memorizer
        self.log_dir = Path(log_dir)
        self.interval_s = interval_s
        self.runtime_device = runtime_device
        self.tnn_id = tnn_id
        self.smoke_node = smoke_node
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_ns = 0
        self._last_input_snapshot_ns = time.monotonic_ns()

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
        self.state["cold_started"] = True
        self.state["loop_status"]["core"] = "starting"
        try:
            if self.tnn_id:
                self.state["active_tnn"].add(self.tnn_id)
                self.load_tnn_runtime(self.tnn_id)
            elif self.smoke_node:
                self._load_smoke_rule()
        except Exception as exc:
            self._record_error("tnn_load", exc, critical=True)
            self.state["loop_status"]["core"] = "failed"
            raise
        self._thread = threading.Thread(target=self._run, name="eve-core")
        self._thread.start()
        self.state["loop_status"]["core"] = "running"
        log_event(self.log_dir, "core_started")

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError("core thread did not stop")
        self._thread = None
        for tnn_id in list(self.state["loaded_tnn"]):
            self.unload_tnn_runtime(tnn_id)
        self.state["cold_started"] = False
        self.state["loop_status"]["core"] = "failed" if self.failed else "stopped"
        log_event(self.log_dir, "core_stopped")

    def load_tnn_runtime(
        self,
        tnn_id: str,
        version: str | None = None,
        *,
        input_refs: dict[str, str] | None = None,
        run_frequency_hz: float = 1.0,
        output_ttl_ns: int = 1_000_000_000,
        action_output: str | None = None,
        factory: str = "create_tnn",
    ) -> dict[str, Any]:
        """Load a persisted TinyNN artifact into this Core lifecycle."""
        import torch
        from eve.dock.tinynn import TinyNN

        artifact = self.memorizer.resolve_tnn_artifact(tnn_id, version)
        path = Path(artifact["model_path"]).resolve()
        module_name = f"_eve_runtime_tnn_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import TNN model: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            creator = getattr(module, factory, None)
            if not callable(creator):
                raise AttributeError(f"{path} does not define {factory}()")
            model = creator()
        finally:
            sys.modules.pop(module_name, None)
        if not isinstance(model, TinyNN):
            raise TypeError(
                f"{factory}() must return TinyNN, got {type(model).__name__}"
            )
        device = torch.device(
            self.runtime_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.load_weights(artifact["weights_path"], map_location=device)
        model.to(device)
        model.eval()
        schema = model.get_input_schema()
        refs = input_refs or {
            name: (
                f"state:{name}"
                if name in {"screen", "cursor"}
                else f"blackboard:{name}"
            )
            for name in schema
        }
        if set(refs) != set(schema):
            raise ValueError("runtime input references must match input schema")

        def infer(inputs: dict[str, Any]) -> dict[str, Any]:
            prepared: dict[str, Any] = {}
            for name, value in inputs.items():
                dtype_name = schema.get(name, {}).get("dtype")
                dtype = getattr(torch, dtype_name, None) if dtype_name else None
                if hasattr(value, "image"):
                    value = value.image
                if hasattr(value, "x") and hasattr(value, "y"):
                    value = [value.x, value.y]
                if isinstance(value, torch.Tensor):
                    prepared[name] = value.to(device=device, dtype=dtype)
                elif isinstance(dtype, torch.dtype):
                    prepared[name] = torch.as_tensor(
                        value, device=device, dtype=dtype
                    )
                else:
                    prepared[name] = value
            return model.infer(prepared)

        node = register_runtime_tnn(
            self.state,
            tnn_id,
            infer,
            inputs=refs,
            outputs=tuple(model.get_output_schema()),
            run_frequency_hz=run_frequency_hz,
            output_ttl_ns=output_ttl_ns,
            action_output=action_output,
            model=model,
        )
        node["device"] = str(device)
        self.state["resource_status"]["tnn_device"] = str(device)
        return node

    def unload_tnn_runtime(self, tnn_id: str) -> None:
        node = unregister_runtime_tnn(self.state, tnn_id)
        if not node or node.get("model") is None:
            return
        import torch

        model = node["model"]
        was_cuda = any(parameter.is_cuda for parameter in model.parameters())
        model.to("cpu")
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def step(self, now_ns: int | None = None) -> list[dict[str, Any]]:
        if not self.state["cold_started"] or self.state["emergency_stop"]:
            return []
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        stats = self.state["runtime_stats"]
        stats["core_iterations"] = stats.get("core_iterations", 0) + 1
        input_state = self.input_buffer.get_state()
        for name in (
            "human_activity_detected_at_ns", "human_takeover_until_ns"
        ):
            self.state[name] = input_state[name]
        self.state["resource_status"]["capture"] = input_state["capture"]
        summary = self._input_summary(input_state)
        self.state["blackboard"]["latest_input_summary"] = _timed(
            summary, now_ns, producer="input_buffer"
        )
        self._maybe_remember_input(now_ns, summary)

        for tnn_id in sorted(self.state["active_tnn"]):
            node = self.state["loaded_tnn"].get(tnn_id)
            if node is None:
                self.state["tnn_status"][tnn_id] = "requested_not_loaded"
                continue
            interval_ns = int(1_000_000_000 / node["run_frequency_hz"])
            if now_ns - self.state["last_run_ns"].get(tnn_id, 0) < interval_ns:
                continue
            try:
                outputs = _run_tnn(self.state, self.input_buffer, tnn_id, now_ns)
                if outputs is not None:
                    log_event(
                        self.log_dir,
                        "tnn_output",
                        tnn_id=tnn_id,
                        output_names=sorted(outputs),
                    )
            except Exception as exc:
                self.state["tnn_status"][tnn_id] = "failed"
                self._record_error(f"tnn:{tnn_id}", exc, critical=True)
                break

        results: list[dict[str, Any]] = []
        while self.state["action_queue"]:
            action = self.state["action_queue"].popleft()
            candidate_id = action["candidate_id"]
            if candidate_id in self.state["consumed_action_ids"]:
                continue
            self.state["consumed_action_ids"].add(candidate_id)
            try:
                result = run_once(self.state, action, self.log_dir)
                results.append(result)
                self._remember_chain(action, result)
            except Exception as exc:
                self._record_error("output", exc, critical=True)
                break
        return results

    def stats(self) -> dict[str, float | int]:
        duration_s = max(
            (time.monotonic_ns() - self._started_at_ns) / 1_000_000_000, 1e-9
        )
        iterations = self.state["runtime_stats"].get("core_iterations", 0)
        return {
            "iterations": iterations,
            "loop_hz": iterations / duration_s,
            "tnn_invocations": self.state["runtime_stats"].get(
                "tnn_invocations", 0
            ),
        }

    def save_snapshot(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            name: self.state[name]
            for name in (
                "world", "myself", "tnn_status", "loop_status", "permissions",
                "resource_status", "emergency_stop", "latest_error"
            )
        }
        snapshot["active_tnn"] = sorted(self.state["active_tnn"])
        snapshot["loaded_tnn"] = sorted(self.state["loaded_tnn"])
        destination.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )

    def _load_smoke_rule(self) -> None:
        emitted = False

        def run(inputs: dict[str, Any]) -> dict[str, Any]:
            nonlocal emitted
            if emitted:
                return {}
            emitted = True
            cursor = inputs["cursor"]
            x, y = (
                (cursor.x, cursor.y)
                if hasattr(cursor, "x")
                else tuple(cursor)[:2]
            )
            return {
                "action_candidate": {
                    "candidate_id": "smoke-action-1",
                    "action_type": "mouse",
                    "payload": {"action": "moveTo", "x": x, "y": y},
                    "horizon_ns": 1_000_000_000,
                }
            }

        register_runtime_tnn(
            self.state,
            "smoke_rule",
            run,
            inputs={"cursor": "state:cursor"},
            outputs=("action_candidate",),
            run_frequency_hz=20.0,
            action_output="action_candidate",
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started_ns = time.monotonic_ns()
            try:
                self.step(started_ns)
            except Exception as exc:
                self._record_error("core", exc, critical=True)
            if self.failed:
                self._stop_event.set()
            elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
            self._stop_event.wait(max(0.001, self.interval_s - elapsed_s))

    def _input_summary(self, input_state: dict[str, Any]) -> dict[str, Any]:
        screen = input_state["latest"]["screen"]
        cursor = input_state["latest"]["cursor"]
        screen_value = screen.value if screen is not None else None
        cursor_value = cursor.value if cursor is not None else None
        return {
            "screen": (
                {
                    "frame_id": getattr(screen_value, "frame_id", screen.index),
                    "timestamp_ns": screen.timestamp_ns,
                    "shape": list(
                        getattr(getattr(screen_value, "image", None), "shape", ())
                    ),
                }
                if screen is not None else None
            ),
            "cursor": (
                {
                    "frame_id": getattr(cursor_value, "frame_id", cursor.index),
                    "timestamp_ns": cursor.timestamp_ns,
                    "x": getattr(cursor_value, "x", None),
                    "y": getattr(cursor_value, "y", None),
                    "speed": getattr(cursor_value, "speed", 0.0),
                }
                if cursor is not None else None
            ),
            "human_activity_detected_at_ns": input_state[
                "human_activity_detected_at_ns"
            ],
        }

    def _maybe_remember_input(
        self, now_ns: int, summary: dict[str, Any]
    ) -> None:
        if now_ns - self._last_input_snapshot_ns < 1_000_000_000:
            return
        memory_id = self.memorizer.enqueue(
            summary, "input_snapshot", priority="low"
        )
        self._last_input_snapshot_ns = now_ns
        if memory_id is not None:
            self.state["memory_ids"].append(memory_id)

    def _remember_chain(
        self, action: dict[str, Any], result: dict[str, Any]
    ) -> None:
        try:
            decision = self.state["blackboard"].get(
                "latest_safegate_result", {}
            ).get("value", {})
            for payload_type, payload in (
                ("action_candidate", action),
                ("safegate_result", decision),
                ("output_result", result),
            ):
                memory_id = self.memorizer.enqueue(
                    payload, payload_type, priority="critical"
                )
                if memory_id is not None:
                    self.state["memory_ids"].append(memory_id)
        except Exception as exc:
            self._record_error("memory", exc, critical=True)

    def _record_error(
        self, loop_node: str, exc: Exception, *, critical: bool = False
    ) -> None:
        error = {
            "timestamp_ns": time.time_ns(),
            "loop_node": loop_node,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "recovery_action": "runtime_stopped_no_output",
        }
        self.state["latest_error"] = error
        if critical:
            self._failed_event.set()
        log_event(self.log_dir, "runtime_error", **error)
        if loop_node != "memory":
            self.memorizer.enqueue(error, "runtime_error", priority="critical")
