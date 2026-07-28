"""EVE Core loop, runtime state, and live TNN execution."""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
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

MAX_LOADED_TNN = 5
CORE_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_LLM_PATH = str(CORE_MODEL_DIR / "deepseek-7b")
DEFAULT_VLM_PATH = str(CORE_MODEL_DIR / "qwen")
DEFAULT_YOLO_PATH = str(CORE_MODEL_DIR / "yolo26" / "weights" / "yolo26n.pt")
DEFAULT_HORMONES = {
    "dopamine": 0.5,
    "serotonin": 0.5,
    "norepinephrine": 0.5,
    "oxytocin": 0.5,
    "cortisol": 0.5,
    "acetylcholine": 0.5,
}


def create_runtime_state(
    *, output_mode: str = "mock", allow_mock_actions: bool = False
) -> dict[str, Any]:
    """Create the small authoritative state owned by Core."""
    if output_mode not in {"disabled", "mock", "real"}:
        raise ValueError(f"unknown output mode: {output_mode}")
    return {
        "cold_started": False,
        "world": {},
        "myself": {
            "current_task": "",
            "what_im_thinking": "",
            "hormones": dict(DEFAULT_HORMONES),
            "tendencies": {
                name: {
                    "strength": 0.0,
                    "updated_at_ns": 0,
                    "suppressed": True,
                    "reason": "permission_disabled",
                }
                for name in (
                    "mouse", "keyboard", "send_text", "speak",
                    "thinking", "pause", "sleep",
                )
            },
        },
        "blackboard": {},
        "active_tnn": set(),
        "loaded_tnn": {},
        "tnn_status": {},
        "loop_status": {"core": "not_started"},
        "node_status": {},
        "model_status": {
            "local_llm": {"state": "not_configured", "device": None},
            "vlm": {
                "state": "not_configured",
                "device": None,
                "role": "teacher",
            },
            "yolo": {
                "state": "not_configured",
                "device": None,
                "role": "runtime_visual",
            },
            "cloud_llm": {"state": "disabled", "device": "cloud"},
        },
        "model_config": {
            "local_llm_path": "",
            "vlm_path": "",
            "yolo_model_path": "",
            "cloud_base_url": "",
            "cloud_model": "",
            "cloud_timeout_s": 30.0,
            "cloud_enabled": False,
        },
        "permissions": safegate.default_permissions(allow_mock_actions),
        "resource_status": {},
        "emergency_stop": False,
        "paused": False,
        "lifecycle": {
            "state": "gui_only",
            "changed_at_ns": time.monotonic_ns(),
            "reason": "application_initialized",
            "escape_triggered_at_ns": 0,
        },
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
        "conversation": [],
        "visual_result": None,
        "teacher_visual_result": None,
        "cloud_result": None,
        "cuda_status": {},
        "last_feedback": None,
        "_state_lock": threading.RLock(),
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
    if len(state["loaded_tnn"]) >= MAX_LOADED_TNN:
        raise RuntimeError(f"maximum loaded TNN count is {MAX_LOADED_TNN}")
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
        "status": "active" if activate else "paused",
        "last_run_ns": 0,
        "last_duration_ms": 0.0,
        "average_duration_ms": 0.0,
        "run_count": 0,
        "last_output_summary": {},
        "last_error": None,
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
        "created_at_ns": produced_at_ns,
        "valid_until_ns": produced_at_ns + ttl_ns if ttl_ns else 0,
        "producer": producer,
        "source": producer,
        "status": "active",
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
    generated_at_ns = time.monotonic_ns()
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
        "generated_at_ns": generated_at_ns,
        "valid_until_ns": generated_at_ns + ttl_ns if ttl_ns else 0,
    }
    state["action_queue"].append(action)
    state["blackboard"]["latest_action_candidate"] = _timed(
        action, generated_at_ns, ttl_ns, node["tnn_id"]
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
    started_ns = time.monotonic_ns()
    outputs = node["run"](inputs)
    duration_ms = (time.monotonic_ns() - started_ns) / 1_000_000
    if not isinstance(outputs, dict):
        raise TypeError(f"{tnn_id} must return a dict")
    unknown = set(outputs) - set(node["outputs"])
    if unknown:
        raise ValueError(f"{tnn_id} produced undeclared outputs: {sorted(unknown)}")
    stats = state["runtime_stats"]
    stats["tnn_invocations"] = stats.get("tnn_invocations", 0) + 1
    state["last_run_ns"][tnn_id] = now_ns
    node["last_run_ns"] = now_ns
    node["last_duration_ms"] = duration_ms
    node["run_count"] += 1
    node["average_duration_ms"] += (
        duration_ms - node["average_duration_ms"]
    ) / node["run_count"]
    node["last_output_summary"] = {
        name: _value_summary(value) for name, value in outputs.items()
    }
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
    if (
        "detections" in outputs
        and any(
            reference == "state:screen"
            for reference in node["inputs"].values()
        )
    ):
        visual_result = {
            "source": "tnn",
            "role": "runtime_visual",
            "tnn_id": tnn_id,
            "completed_at_ns": now_ns,
            "detections": outputs["detections"],
            "status": "current",
        }
        state["visual_result"] = visual_result
        state["blackboard"]["current_visual_result"] = _timed(
            visual_result,
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


def _value_summary(value: Any) -> Any:
    if hasattr(value, "shape"):
        summary = {
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "cpu")),
        }
        try:
            summary.update(
                {
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.float().mean()),
                }
            )
        except Exception as exc:
            summary["summary_error"] = f"{type(exc).__name__}: {exc}"
        return summary
    if isinstance(value, dict):
        return {str(key): _value_summary(item) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    text = str(value)
    return text[:240] + ("…" if len(text) > 240 else "")


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
    before_output: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the non-bypassable Core -> Safegate -> Output chain."""
    started_ns = time.monotonic_ns()
    decision = safegate.check(state, action)
    checked_ns = time.monotonic_ns()
    _record_state_node(state, "safegate", started_ns, checked_ns)
    state["blackboard"]["latest_safegate_result"] = _timed(
        decision, decision["checked_at_ns"], producer="safegate"
    )
    stat = "safegate_allowed" if decision["allowed"] else "safegate_blocked"
    state["runtime_stats"][stat] = state["runtime_stats"].get(stat, 0) + 1
    if decision["allowed"]:
        if before_output is not None:
            before_output(action)
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
    _record_state_node(
        state,
        f"{action['action_type']}_output",
        checked_ns,
        int(result["finished_at_ns"]),
        None if not result.get("blocked") else str(result.get("reason")),
    )
    state["blackboard"]["latest_output_feedback"] = _timed(
        result, result["finished_at_ns"], producer="output"
    )
    if result.get("simulated"):
        state["runtime_stats"]["mock_outputs"] = (
            state["runtime_stats"].get("mock_outputs", 0) + 1
        )
    log_event(log_dir, "action_result", action=action, safegate=decision, output=result)
    return result


def _record_state_node(
    state: dict[str, Any],
    name: str,
    started_ns: int,
    finished_ns: int,
    error: str | None = None,
) -> None:
    node = state["node_status"].setdefault(
        name,
        {
            "state": "running",
            "last_run_ns": 0,
            "last_duration_ms": 0.0,
            "average_duration_ms": 0.0,
            "run_count": 0,
            "last_error": None,
        },
    )
    duration_ms = max(0, finished_ns - started_ns) / 1_000_000
    node["last_run_ns"] = finished_ns
    node["last_duration_ms"] = duration_ms
    node["run_count"] = int(node.get("run_count", 0)) + 1
    previous = float(node.get("average_duration_ms", 0.0))
    node["average_duration_ms"] = previous + (
        duration_ms - previous
    ) / node["run_count"]
    node["last_error"] = error
    node["state"] = "error" if error else "running"


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
        local_llm_backend: Callable[[dict[str, Any]], Any] | None = None,
        vlm_backend: Callable[[dict[str, Any]], Any] | None = None,
        runtime_visual_backend: Callable[[Any], Any] | None = None,
        cloud_backend: Callable[[dict[str, Any]], Any] | None = None,
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
        self.local_llm_backend = local_llm_backend
        self.vlm_backend = vlm_backend
        self.runtime_visual_backend = runtime_visual_backend
        self.cloud_backend = cloud_backend
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._model_stop = threading.Event()
        self._cancel_generation = threading.Event()
        self._cancel_vlm = threading.Event()
        self._cancel_cloud = threading.Event()
        self._llm_requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4)
        self._vlm_requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self._cloud_requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self._tnn_requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self._model_threads: list[threading.Thread] = []
        self._local_model: Any = None
        self._local_tokenizer: Any = None
        self._vlm_model: Any = None
        self._vlm_processor: Any = None
        self._yolo_detector: Any = None
        self._yolo_force_event = threading.Event()
        self._model_load_lock = threading.Lock()
        self._request_serial = 0
        self._last_autonomous_ns = time.monotonic_ns()
        self._last_hormone_ns = time.monotonic_ns()
        self._thread: threading.Thread | None = None
        self._started_at_ns = 0
        self._last_input_snapshot_ns = time.monotonic_ns()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failed(self) -> bool:
        return self._failed_event.is_set()

    def configure_models(self, config: dict[str, Any]) -> None:
        allowed = set(self.state["model_config"])
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown model config: {sorted(unknown)}")
        self.state["model_config"].update(config)

    def submit_user_message(self, message: str) -> str:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before sending a message")
        status = self.state["model_status"]["local_llm"]
        if status.get("state") not in {"ready", "running", "queued"}:
            raise RuntimeError(
                status.get("error") or f"local LLM is {status.get('state')}"
            )
        text = str(message).strip()
        if not text:
            raise ValueError("message is empty")
        self._request_serial += 1
        request_id = f"chat_{time.time_ns()}_{self._request_serial}"
        self.input_buffer.submit_user_text(text)
        memory_id = self.memorizer.enqueue(
            {"request_id": request_id, "text": text},
            "user_text",
            priority="critical",
        )
        request = {
            "request_id": request_id,
            "kind": "user",
            "message": text,
            "requested_at_ns": time.monotonic_ns(),
            "memory_id": memory_id,
        }
        try:
            self._llm_requests.put_nowait(request)
        except queue.Full as exc:
            raise RuntimeError("local LLM request queue is full") from exc
        status["queued_request_id"] = request_id
        if status.get("state") == "ready":
            status["state"] = "queued"
        log_event(
            self.log_dir,
            "llm_chat_requested",
            request_id=request_id,
            message_length=len(text),
        )
        return request_id

    def cancel_generation(self) -> None:
        self._cancel_generation.set()
        self._cancel_vlm.set()
        self._cancel_cloud.set()
        status = self.state["model_status"]["local_llm"]
        if status.get("state") in {"queued", "running"}:
            status["state"] = "cancel_requested"

    def submit_teacher_review(
        self, *, prompt: str = "请复核当前屏幕并生成可用于训练的视觉标签。"
    ) -> str:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before requesting VLM teacher")
        status = self.state["model_status"]["vlm"]
        if status.get("state") not in {
            "configured", "teacher_idle", "ready"
        }:
            raise RuntimeError(
                status.get("error")
                or f"VLM teacher is {status.get('state')}"
            )
        sample = self.input_buffer.get_latest_screen()
        if sample is None:
            raise RuntimeError("no screen frame is available")
        import numpy as np

        frame = sample.value
        image = np.array(frame.image, copy=True)
        self._request_serial += 1
        request_id = f"vlm_{time.time_ns()}_{self._request_serial}"
        request = {
            "request_id": request_id,
            "prompt": str(prompt),
            "frame_id": frame.frame_id,
            "frame_timestamp_ns": frame.captured_at_ns,
            "requested_at_ns": time.monotonic_ns(),
            "image": image,
        }
        try:
            self._vlm_requests.put_nowait(request)
        except queue.Full as exc:
            raise RuntimeError("VLM request queue is full") from exc
        self.state["model_status"]["vlm"].update(
            {"state": "queued", "request_id": request_id}
        )
        log_event(
            self.log_dir,
            "vlm_teacher_review_requested",
            request_id=request_id,
            reference_frame_id=frame.frame_id,
        )
        return request_id

    def submit_visual_request(
        self, *, prompt: str = "请复核当前屏幕并生成可用于训练的视觉标签。"
    ) -> str:
        """Backward-compatible alias for an explicit VLM teacher review."""
        return self.submit_teacher_review(prompt=prompt)

    def submit_runtime_visual_analysis(self) -> str:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before requesting visual analysis")
        status = self.state["model_status"]["yolo"]
        if status.get("state") != "ready":
            raise RuntimeError(
                status.get("error")
                or f"runtime visual node is {status.get('state')}"
            )
        sample = self.input_buffer.get_latest_screen()
        if sample is None:
            raise RuntimeError("no screen frame is available")
        request_id = f"visual_{time.time_ns()}"
        self._yolo_force_event.set()
        log_event(
            self.log_dir,
            "runtime_visual_requested",
            request_id=request_id,
            reference_frame_id=sample.value.frame_id,
        )
        return request_id

    def submit_cloud_request(self, message: str) -> str:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before requesting cloud model")
        if not self.state["model_config"].get("cloud_enabled"):
            raise RuntimeError("cloud model is disabled")
        self._request_serial += 1
        request_id = f"cloud_{time.time_ns()}_{self._request_serial}"
        try:
            self._cloud_requests.put_nowait(
                {
                    "request_id": request_id,
                    "message": str(message),
                    "requested_at_ns": time.monotonic_ns(),
                }
            )
        except queue.Full as exc:
            raise RuntimeError("cloud model request queue is full") from exc
        return request_id

    def pause(self, reason: str = "user_pause") -> None:
        self.state["paused"] = True
        self.state["lifecycle"].update(
            {
                "state": "paused",
                "changed_at_ns": time.monotonic_ns(),
                "reason": reason,
            }
        )

    def resume(self) -> None:
        if self.state["emergency_stop"]:
            raise RuntimeError("reset emergency stop before resuming")
        self.state["paused"] = False
        self.state["lifecycle"].update(
            {
                "state": "running",
                "changed_at_ns": time.monotonic_ns(),
                "reason": "user_resume",
            }
        )

    def set_active_tnn(self, requested: set[str] | list[str]) -> bool:
        desired = set(requested)
        if len(desired) > MAX_LOADED_TNN:
            self._publish_tnn_error(
                f"active_tnn exceeds maximum {MAX_LOADED_TNN}"
            )
            return False
        missing = desired - set(self.state["loaded_tnn"])
        if missing:
            self._publish_tnn_error(
                f"active_tnn contains unloaded TNN: {sorted(missing)}"
            )
            return False
        self.state["active_tnn"] = desired
        for tnn_id, node in self.state["loaded_tnn"].items():
            active = tnn_id in desired
            node["status"] = "active" if active else "paused"
            self.state["tnn_status"][tnn_id] = node["status"]
        return True

    def activate_tnn(self, tnn_id: str) -> bool:
        return self.set_active_tnn(set(self.state["active_tnn"]) | {tnn_id})

    def pause_tnn(self, tnn_id: str) -> bool:
        if tnn_id not in self.state["loaded_tnn"]:
            self._publish_tnn_error(f"unknown loaded TNN: {tnn_id}")
            return False
        return self.set_active_tnn(set(self.state["active_tnn"]) - {tnn_id})

    def request_tnn_load(
        self,
        tnn_id: str,
        version: str | None = None,
        **options: Any,
    ) -> None:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before loading a TNN")
        try:
            self._tnn_requests.put_nowait(
                {
                    "operation": "load",
                    "tnn_id": str(tnn_id),
                    "version": version,
                    "options": options,
                }
            )
        except queue.Full as exc:
            raise RuntimeError("TNN lifecycle request queue is full") from exc

    def request_tnn_unload(self, tnn_id: str) -> None:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before unloading a TNN")
        try:
            self._tnn_requests.put_nowait(
                {"operation": "unload", "tnn_id": str(tnn_id)}
            )
        except queue.Full as exc:
            raise RuntimeError("TNN lifecycle request queue is full") from exc

    def feedback(self, kind: str) -> dict[str, Any]:
        if not self.state["cold_started"]:
            raise RuntimeError("cold start EVE before recording feedback")
        if kind not in {"praise", "criticism"}:
            raise ValueError("feedback must be praise or criticism")
        now_ns = time.monotonic_ns()
        deltas = (
            {"dopamine": 0.08, "oxytocin": 0.06, "cortisol": -0.03}
            if kind == "praise"
            else {"dopamine": -0.04, "norepinephrine": 0.05, "cortisol": 0.08}
        )
        self._adjust_hormones(deltas, f"user_{kind}", now_ns)
        record = {
            "kind": kind,
            "timestamp_ns": now_ns,
            "current_task": self.state["myself"].get("current_task", ""),
            "recent_context": self.state["conversation"][-2:],
        }
        memory_id = self.memorizer.enqueue(
            record, f"user_{kind}", priority="critical"
        )
        record["memory_id"] = memory_id
        self.state["last_feedback"] = record
        self.state["blackboard"]["latest_feedback"] = _timed(
            record, now_ns, producer="user"
        )
        return record

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._failed_event.clear()
        self._started_at_ns = time.monotonic_ns()
        self._last_input_snapshot_ns = self._started_at_ns
        self.state["cold_started"] = True
        self.state["paused"] = False
        self.state["lifecycle"].update(
            {
                "state": "starting",
                "changed_at_ns": self._started_at_ns,
                "reason": "cold_start",
            }
        )
        self.state["loop_status"]["core"] = "starting"
        try:
            self.verify_cuda()
            if self.tnn_id:
                self.state["active_tnn"].add(self.tnn_id)
                self.load_tnn_runtime(self.tnn_id)
            elif self.smoke_node:
                self._load_smoke_rule()
        except Exception as exc:
            self._record_error("tnn_load", exc, critical=True)
            self.state["loop_status"]["core"] = "failed"
            raise
        self._start_model_workers()
        self._thread = threading.Thread(target=self._run, name="eve-core")
        self._thread.start()
        self.state["loop_status"]["core"] = "running"
        self.state["lifecycle"].update(
            {
                "state": "running",
                "changed_at_ns": time.monotonic_ns(),
                "reason": "cold_start_complete",
            }
        )
        log_event(self.log_dir, "core_started")

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        self._stop_model_workers(timeout_s)
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError("core thread did not stop")
        self._thread = None
        for tnn_id in list(self.state["loaded_tnn"]):
            self.unload_tnn_runtime(tnn_id)
        self.state["cold_started"] = False
        self.state["paused"] = False
        self.state["loop_status"]["core"] = "failed" if self.failed else "stopped"
        self.state["lifecycle"].update(
            {
                "state": "stopped",
                "changed_at_ns": time.monotonic_ns(),
                "reason": "normal_stop" if not self.failed else "core_failed",
            }
        )
        log_event(self.log_dir, "core_stopped")

    def verify_cuda(self) -> dict[str, Any]:
        import torch

        status: dict[str, Any] = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": None,
            "is_rtx_5080": False,
            "compute_capability": None,
            "arch_list": torch.cuda.get_arch_list(),
            "tensor_test_passed": False,
            "error": None,
            "checked_at_ns": time.monotonic_ns(),
        }
        if status["available"] and status["device_count"]:
            try:
                status["device_name"] = torch.cuda.get_device_name(0)
                status["is_rtx_5080"] = (
                    "RTX 5080" in str(status["device_name"]).upper()
                )
                status["compute_capability"] = list(
                    torch.cuda.get_device_capability(0)
                )
                value = (
                    torch.tensor([2.0, 3.0], device="cuda")
                    * torch.tensor([4.0, 5.0], device="cuda")
                ).sum()
                torch.cuda.synchronize()
                status["tensor_test_passed"] = float(value.cpu()) == 23.0
            except Exception as exc:
                status["error"] = f"{type(exc).__name__}: {exc}"
        else:
            status["error"] = "CUDA is unavailable"
        self.state["cuda_status"] = status
        self.state["resource_status"]["cuda"] = status
        log_event(self.log_dir, "cuda_verified", **status)
        return status

    def _start_model_workers(self) -> None:
        self._model_stop.clear()
        self._cancel_generation.clear()
        self._cancel_vlm.clear()
        self._cancel_cloud.clear()
        workers = (
            ("eve-local-llm", self._local_llm_worker),
            ("eve-vlm", self._vlm_worker),
            ("eve-yolo", self._yolo_worker),
            ("eve-cloud-llm", self._cloud_worker),
            ("eve-tnn-lifecycle", self._tnn_lifecycle_worker),
        )
        self._model_threads = [
            threading.Thread(target=target, name=name)
            for name, target in workers
        ]
        for thread in self._model_threads:
            thread.start()

    def _stop_model_workers(self, timeout_s: float) -> None:
        self._model_stop.set()
        self._cancel_generation.set()
        self._cancel_vlm.set()
        self._cancel_cloud.set()
        self._yolo_force_event.set()
        for requests in (
            self._llm_requests,
            self._vlm_requests,
            self._cloud_requests,
            self._tnn_requests,
        ):
            try:
                requests.put_nowait({"kind": "stop"})
            except queue.Full:
                pass
        for thread in self._model_threads:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError(f"model worker did not stop: {thread.name}")
        self._model_threads.clear()
        self._local_model = None
        self._local_tokenizer = None
        self._vlm_model = None
        self._vlm_processor = None
        self._yolo_detector = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _yolo_worker(self) -> None:
        status = self.state["model_status"]["yolo"]
        detector = None
        try:
            path = self.state["model_config"].get("yolo_model_path", "")
            if self.runtime_visual_backend is None:
                if not path:
                    status.update(
                        {
                            "state": "not_configured",
                            "error": "YOLO model path is empty",
                        }
                    )
                    return
                from eve.core.yolo26.detector import YOLODetector

                detector = YOLODetector(model_path=path)
                status.update({"state": "loading", "path": path})
                with self._model_load_lock:
                    if not detector.load():
                        raise RuntimeError("YOLO detector failed to load")
                self._yolo_detector = detector
                status.update(
                    {
                        "state": "ready",
                        "device": "cuda:0",
                        "model": path,
                        "role": "runtime_visual",
                        "error": None,
                    }
                )
            else:
                status.update(
                    {
                        "state": "ready",
                        "device": "injected",
                        "model": "test_backend",
                        "role": "runtime_visual",
                        "error": None,
                    }
                )
            self._yolo_frame_loop(detector)
        except Exception as exc:
            status.update(
                {
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at_ns": time.monotonic_ns(),
                }
            )
            log_event(
                self.log_dir,
                "runtime_visual_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record_error("yolo", exc, critical=False)
        finally:
            if detector is not None:
                detector.unload()

    def _yolo_frame_loop(self, detector: Any) -> None:
        status = self.state["model_status"]["yolo"]
        log_event(
            self.log_dir,
            "runtime_visual_ready",
            model=status.get("model"),
            device=status.get("device"),
        )
        last_frame_id = -1
        while not self._model_stop.is_set():
            sample = self.input_buffer.get_latest_screen()
            if sample is not None:
                frame = sample.value
                if int(frame.frame_id) != last_frame_id:
                    last_frame_id = int(frame.frame_id)
                    self._run_yolo_frame(frame, detector)
            self._yolo_force_event.wait(0.1)
            self._yolo_force_event.clear()

    def _run_yolo_frame(self, frame: Any, detector: Any) -> None:
        status = self.state["model_status"]["yolo"]
        started_ns = time.monotonic_ns()
        try:
            raw = (
                self.runtime_visual_backend(frame.image)
                if self.runtime_visual_backend is not None
                else detector.detect(frame.image)
            )
            result = self._normalize_yolo_result(frame, raw, started_ns)
            finished_ns = int(result["completed_at_ns"])
            with self.state["_state_lock"]:
                self.state["visual_result"] = result
                self.state["blackboard"]["current_visual_result"] = _timed(
                    result,
                    finished_ns,
                    ttl_ns=1_000_000_000,
                    producer="yolo",
                )
            stats = self.state["runtime_stats"]
            stats["yolo_calls"] = stats.get("yolo_calls", 0) + 1
            status.update(
                {
                    "state": "ready",
                    "finished_at_ns": finished_ns,
                    "last_duration_ms": result["duration_ms"],
                    "last_request_error": None,
                    "error": None,
                }
            )
        except Exception as exc:
            status.update(
                {
                    "state": "ready",
                    "last_request_error": f"{type(exc).__name__}: {exc}",
                    "finished_at_ns": time.monotonic_ns(),
                }
            )
            log_event(
                self.log_dir,
                "runtime_visual_frame_failed",
                reference_frame_id=getattr(frame, "frame_id", None),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _normalize_yolo_result(
        self, frame: Any, raw: Any, started_ns: int
    ) -> dict[str, Any]:
        detections, inference_ms = self._yolo_detection_payload(
            raw, started_ns
        )
        return {
            "source": "yolo",
            "role": "runtime_visual",
            "model": self.state["model_status"]["yolo"].get("model"),
            "reference_frame_id": int(frame.frame_id),
            "reference_frame_timestamp_ns": int(frame.captured_at_ns),
            "completed_at_ns": time.monotonic_ns(),
            "duration_ms": inference_ms,
            "detections": detections,
            "detection_count": len(detections),
            "status": "current",
        }

    @staticmethod
    def _yolo_detection_payload(
        raw: Any, started_ns: int
    ) -> tuple[list[dict[str, Any]], float]:
        if isinstance(raw, dict):
            return (
                list(raw.get("detections", [])),
                float(
                    raw.get(
                        "inference_time_ms",
                        (time.monotonic_ns() - started_ns) / 1_000_000,
                    )
                ),
            )
        detections = []
        for item in raw.detections:
            detections.append(
                {
                    "bbox": [
                        float(item.x1),
                        float(item.y1),
                        float(item.x2),
                        float(item.y2),
                    ],
                    "confidence": float(item.confidence),
                    "class_id": int(item.class_id),
                    "class_name": str(item.class_name),
                }
            )
        return detections, float(raw.inference_time_ms)

    def _tnn_lifecycle_worker(self) -> None:
        while not self._model_stop.is_set():
            try:
                request = self._tnn_requests.get(timeout=0.25)
            except queue.Empty:
                continue
            if request.get("kind") == "stop":
                return
            operation = request.get("operation")
            tnn_id = str(request.get("tnn_id", ""))
            try:
                if operation == "load":
                    self.load_tnn_runtime(
                        tnn_id,
                        request.get("version"),
                        **request.get("options", {}),
                    )
                elif operation == "unload":
                    self.unload_tnn_runtime(tnn_id)
                else:
                    raise ValueError(f"unknown TNN operation: {operation}")
            except Exception as exc:
                self.state["tnn_status"][tnn_id] = "error"
                self._record_error(f"tnn_lifecycle:{tnn_id}", exc, critical=False)

    def _local_llm_worker(self) -> None:
        status = self.state["model_status"]["local_llm"]
        try:
            if self.local_llm_backend is None:
                path = self.state["model_config"].get("local_llm_path", "")
                if path:
                    with self._model_load_lock:
                        self._load_local_llm(path)
                else:
                    status.update(
                        {"state": "not_configured", "error": "model path is empty"}
                    )
            else:
                status.update(
                    {
                        "state": "ready",
                        "device": "injected",
                        "quantization": "test_backend",
                    }
                )
            while not self._model_stop.is_set():
                try:
                    request = self._llm_requests.get(timeout=0.25)
                except queue.Empty:
                    self._maybe_enqueue_autonomous()
                    continue
                if request.get("kind") == "stop":
                    return
                self._cancel_generation.clear()
                try:
                    self._process_llm_request(request)
                except Exception as exc:
                    status.update(
                        {
                            "state": "ready",
                            "error": None,
                            "last_request_state": "error",
                            "last_request_error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "finished_at_ns": time.monotonic_ns(),
                        }
                    )
                    log_event(
                        self.log_dir,
                        "llm_request_failed",
                        request_id=request.get("request_id"),
                        request_kind=request.get("kind"),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self._record_error("local_llm", exc, critical=False)
        except Exception as exc:
            status.update(
                {
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at_ns": time.monotonic_ns(),
                }
            )
            self._record_error("local_llm", exc, critical=False)

    def _load_local_llm(self, path: str) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_path = Path(path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"local LLM path does not exist: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit local LLM requires an available CUDA device")
        try:
            import bitsandbytes  # noqa: F401
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "4-bit local LLM requires bitsandbytes and accelerate"
            ) from exc
        status = self.state["model_status"]["local_llm"]
        status.update({"state": "loading", "path": str(model_path)})
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._local_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization,
            device_map="auto",
        )
        device = str(next(self._local_model.parameters()).device)
        is_4bit = bool(getattr(self._local_model, "is_loaded_in_4bit", False))
        if not is_4bit:
            raise RuntimeError("local LLM did not actually load in 4-bit mode")
        status.update(
            {
                "state": "ready",
                "device": device,
                "quantization": "4bit-nf4",
                "is_loaded_in_4bit": True,
                "model": str(model_path),
                "error": None,
            }
        )

    def _maybe_enqueue_autonomous(self) -> None:
        if self.state["paused"] or self.state["emergency_stop"]:
            return
        status = self.state["model_status"]["local_llm"]
        if status.get("state") != "ready":
            return
        now_ns = time.monotonic_ns()
        interval_s = self._autonomous_interval_s()
        if now_ns - self._last_autonomous_ns < interval_s * 1_000_000_000:
            return
        self._last_autonomous_ns = now_ns
        try:
            self._llm_requests.put_nowait(
                {
                    "request_id": f"thought_{time.time_ns()}",
                    "kind": "autonomous",
                    "message": "进行一次低频自主状态整理。",
                    "requested_at_ns": now_ns,
                    "memory_id": None,
                }
            )
        except queue.Full:
            pass

    def _autonomous_interval_s(self) -> float:
        hormones = self.state["myself"]["hormones"]
        alert = (
            hormones["norepinephrine"]
            + hormones["cortisol"]
            + hormones["acetylcholine"]
        ) / 3
        return max(10.0, min(20.0, 20.0 - 10.0 * alert))

    def _process_llm_request(self, request: dict[str, Any]) -> None:
        status = self.state["model_status"]["local_llm"]
        if status.get("state") not in {"ready", "queued"}:
            raise RuntimeError(status.get("error") or "local LLM is not ready")
        started_ns = time.monotonic_ns()
        status.update(
            {
                "state": "running",
                "request_id": request["request_id"],
                "started_at_ns": started_ns,
            }
        )
        context = self._llm_context(request)
        if self.local_llm_backend is not None:
            raw = self.local_llm_backend(context)
        elif request["kind"] == "user":
            raw = self._generate_local_chat(context)
        else:
            raw = self._generate_local_llm(context)
        if self._cancel_generation.is_set():
            status.update(
                {
                    "state": "ready",
                    "last_request_state": "cancelled",
                    "finished_at_ns": time.monotonic_ns(),
                }
            )
            return
        try:
            if (
                self.local_llm_backend is None
                and request["kind"] == "user"
            ):
                self._apply_chat_reply(request, str(raw))
            else:
                result = (
                    raw
                    if isinstance(raw, dict)
                    else self._parse_model_json(str(raw))
                )
                self._apply_llm_result(request, result)
        except Exception as exc:
            summary = self._safe_model_output_summary(str(raw))
            error = {
                "request_id": request["request_id"],
                "message": f"{type(exc).__name__}: {exc}",
                "raw_output_summary": summary,
                "timestamp_ns": time.monotonic_ns(),
            }
            status.update(
                {
                    "state": "ready",
                    "error": None,
                    "last_request_state": "error",
                    "last_request_error": error["message"],
                    **error,
                }
            )
            self.state["blackboard"]["local_llm_error"] = _timed(
                error, error["timestamp_ns"], producer="local_llm"
            )
            self.memorizer.enqueue(error, "local_llm_error", priority="critical")
            log_event(
                self.log_dir,
                "llm_request_failed",
                request_id=request["request_id"],
                request_kind=request["kind"],
                error=error["message"],
            )
            return
        finished_ns = time.monotonic_ns()
        status.update(
            {
                "state": "ready",
                "finished_at_ns": finished_ns,
                "last_duration_ms": (finished_ns - started_ns) / 1_000_000,
                "error": None,
                "last_request_state": "completed",
                "last_request_error": None,
            }
        )
        log_event(
            self.log_dir,
            "llm_request_completed",
            request_id=request["request_id"],
            request_kind=request["kind"],
            duration_ms=(finished_ns - started_ns) / 1_000_000,
        )
        stats = self.state["runtime_stats"]
        stats["local_llm_calls"] = stats.get("local_llm_calls", 0) + 1
        self._adjust_hormones(
            {"dopamine": 0.002, "acetylcholine": 0.001},
            "local_llm_success",
            finished_ns,
        )

    def _llm_context(self, request: dict[str, Any]) -> dict[str, Any]:
        related_ids = self.memorizer.search(keyword=request["message"][:32])[-5:]
        return {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "user_message": request["message"],
            "current_task": self.state["myself"].get("current_task", ""),
            "world": self.state["world"],
            "myself": self.state["myself"],
            "blackboard": {
                key: _value_summary(value.get("value"))
                for key, value in list(self.state["blackboard"].items())[-20:]
            },
            "related_memory": [
                {"memory_id": memory_id, "payload": self.memorizer.read(memory_id)}
                for memory_id in related_ids
            ],
            "available_tnn": self.memorizer.list_tnn_artifacts(),
            "loaded_tnn": sorted(self.state["loaded_tnn"]),
            "active_tnn": sorted(self.state["active_tnn"]),
            "hormones": self.state["myself"]["hormones"],
            "tendencies": self.state["myself"]["tendencies"],
        }

    def _generate_local_llm(self, context: dict[str, Any]) -> str:
        system = (
            "你是 EVE 的本地运行模型。只输出一个 JSON 对象，不输出隐藏推理。"
            "字段必须为 reply, thinking_summary, world_update, myself_update, "
            "blackboard_updates, active_tnn, memory_candidates。"
        )
        user = json.dumps(context, ensure_ascii=False, default=_value_summary)
        return self._generate_from_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_new_tokens=512,
            suppress_reasoning=True,
        )

    def _generate_local_chat(self, context: dict[str, Any]) -> str:
        history = []
        for exchange in self.state["conversation"][-8:]:
            user = str(exchange.get("user", "")).strip()
            reply = str(exchange.get("reply", "")).strip()
            if user:
                history.append({"role": "user", "content": user})
            if reply:
                history.append({"role": "assistant", "content": reply})
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 EVE 的本地对话模型。直接、清楚地回答用户。"
                    "不要输出 JSON，不要展示隐藏推理。"
                ),
            },
            *history,
            {"role": "user", "content": str(context["user_message"])},
        ]
        raw = self._generate_from_messages(
            messages,
            max_new_tokens=512,
            suppress_reasoning=True,
        )
        reply = self._visible_model_reply(raw)
        if not reply:
            raise ValueError("local LLM returned no visible reply")
        return reply

    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        suppress_reasoning: bool,
    ) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        cancel_event = self._cancel_generation

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del input_ids, scores, kwargs
                return cancel_event.is_set()

        tokenizer = self._local_tokenizer
        model = self._local_model
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            )
            prompt += "\nassistant:"
        if suppress_reasoning and prompt.rstrip().endswith("<think>"):
            prompt += "</think>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    @staticmethod
    def _visible_model_reply(raw: str) -> str:
        text = str(raw)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text.strip())
        return text.strip()

    def _apply_chat_reply(self, request: dict[str, Any], reply: str) -> None:
        visible_reply = self._visible_model_reply(reply)
        if not visible_reply:
            raise ValueError("local LLM returned an empty reply")
        now_ns = time.monotonic_ns()
        exchange = {
            "request_id": request["request_id"],
            "kind": "user",
            "user": request["message"],
            "reply": visible_reply,
            "thinking_summary": "",
            "timestamp_ns": now_ns,
        }
        with self.state["_state_lock"]:
            self.state["conversation"].append(exchange)
            self.state["conversation"] = self.state["conversation"][-100:]
            self.state["blackboard"]["latest_llm_reply"] = _timed(
                exchange, now_ns, producer="local_llm"
            )
        reply_id = self.memorizer.enqueue(
            exchange, "llm_reply", priority="critical"
        )
        related = [
            memory_id
            for memory_id in (request.get("memory_id"), reply_id)
            if memory_id
        ]
        if related:
            self.memorizer.create_event(
                related,
                summary="LLM user conversation",
                tags=["llm", "chat"],
            )

    @staticmethod
    def _parse_model_json(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                value, _ = decoder.raw_decode(raw[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("model output does not contain a valid JSON object")

    @staticmethod
    def _safe_model_output_summary(raw: str) -> str:
        redacted = re.sub(
            r"<think>.*?</think>", "[hidden reasoning removed]", raw, flags=re.S
        )
        return redacted[:500]

    def _apply_llm_result(
        self, request: dict[str, Any], result: dict[str, Any]
    ) -> None:
        required = {
            "reply", "thinking_summary", "world_update", "myself_update",
            "blackboard_updates", "active_tnn", "memory_candidates",
        }
        missing = required - set(result)
        if missing:
            raise ValueError(f"missing LLM result fields: {sorted(missing)}")
        if not isinstance(result["world_update"], dict) or not isinstance(
            result["myself_update"], dict
        ):
            raise TypeError("world_update and myself_update must be objects")
        if not isinstance(result["blackboard_updates"], list):
            raise TypeError("blackboard_updates must be an array")
        if not isinstance(result["active_tnn"], list):
            raise TypeError("active_tnn must be an array")
        if not isinstance(result["memory_candidates"], list):
            raise TypeError("memory_candidates must be an array")
        desired = {str(item) for item in result["active_tnn"]}
        if len(desired) > MAX_LOADED_TNN:
            raise ValueError(f"active_tnn exceeds {MAX_LOADED_TNN}")
        if desired - set(self.state["loaded_tnn"]):
            raise ValueError("active_tnn contains unloaded TNN")
        updates: list[tuple[str, Any, int]] = []
        for update in result["blackboard_updates"]:
            if not isinstance(update, dict) or "key" not in update:
                raise TypeError("blackboard update must contain key")
            ttl_ns = int(update.get("ttl_ns", 0))
            if ttl_ns < 0:
                raise ValueError("blackboard ttl_ns must be non-negative")
            updates.append((str(update["key"]), update.get("value"), ttl_ns))
        now_ns = time.monotonic_ns()
        exchange = {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "user": request["message"] if request["kind"] == "user" else "",
            "reply": str(result["reply"]),
            "thinking_summary": str(result["thinking_summary"]),
            "timestamp_ns": now_ns,
        }
        with self.state["_state_lock"]:
            self.state["world"].update(result["world_update"])
            self.state["myself"].update(result["myself_update"])
            self.state["myself"]["what_im_thinking"] = str(
                result["thinking_summary"]
            )[:1000]
            for key, value, ttl_ns in updates:
                self.state["blackboard"][key] = _timed(
                    value,
                    now_ns,
                    ttl_ns,
                    "local_llm",
                )
            if not self.set_active_tnn(desired):
                raise ValueError("active_tnn update rejected")
            self.state["conversation"].append(exchange)
            self.state["conversation"] = self.state["conversation"][-100:]
            self.state["blackboard"]["latest_llm_result"] = _timed(
                exchange, now_ns, producer="local_llm"
            )
        reply_id = self.memorizer.enqueue(
            exchange, "llm_reply", priority="critical"
        )
        thought_id = self.memorizer.enqueue(
            {
                "request_id": request["request_id"],
                "summary": str(result["thinking_summary"]),
            },
            "thinking_summary",
            priority="normal",
        )
        candidate_ids = []
        for candidate in result["memory_candidates"][:20]:
            memory_id = self.memorizer.enqueue(
                candidate, "llm_memory_candidate", priority="normal"
            )
            if memory_id:
                candidate_ids.append(memory_id)
        related = [
            memory_id
            for memory_id in (request.get("memory_id"), reply_id, thought_id, *candidate_ids)
            if memory_id
        ]
        if related:
            self.memorizer.create_event(
                related,
                summary=f"LLM {request['kind']} exchange",
                tags=["llm", request["kind"]],
            )

    def _vlm_worker(self) -> None:
        status = self.state["model_status"]["vlm"]
        try:
            if self.vlm_backend is None:
                path = self.state["model_config"].get("vlm_path", "")
                if path:
                    status.update(
                        {
                            "state": "teacher_idle",
                            "path": path,
                            "role": "teacher",
                            "error": None,
                        }
                    )
                else:
                    status.update(
                        {"state": "not_configured", "error": "model path is empty"}
                    )
            else:
                status.update(
                    {
                        "state": "ready",
                        "device": "injected",
                        "quantization": "test_backend",
                    }
                )
            while not self._model_stop.is_set():
                try:
                    request = self._vlm_requests.get(timeout=0.25)
                except queue.Empty:
                    continue
                if request.get("kind") == "stop":
                    return
                self._cancel_vlm.clear()
                try:
                    if self.vlm_backend is None and self._vlm_model is None:
                        path = self.state["model_config"].get("vlm_path", "")
                        with self._model_load_lock:
                            self._load_vlm(path)
                    self._process_vlm_request(request)
                except Exception as exc:
                    reusable = (
                        self._vlm_model is not None
                        or self.vlm_backend is not None
                    )
                    status.update(
                        {
                            "state": "ready" if reusable else "error",
                            "error": (
                                None
                                if reusable
                                else f"{type(exc).__name__}: {exc}"
                            ),
                            "last_request_error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "finished_at_ns": time.monotonic_ns(),
                        }
                    )
                    log_event(
                        self.log_dir,
                        "vlm_teacher_review_failed",
                        request_id=request.get("request_id"),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self._record_error("vlm", exc, critical=False)
        except Exception as exc:
            status.update(
                {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
            self._record_error("vlm", exc, critical=False)

    def _load_vlm(self, path: str) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig

        model_path = Path(path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"VLM path does not exist: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit VLM requires an available CUDA device")
        try:
            import bitsandbytes  # noqa: F401
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("4-bit VLM requires bitsandbytes and accelerate") from exc
        model_class = getattr(
            transformers,
            "AutoModelForImageTextToText",
            getattr(transformers, "AutoModelForVision2Seq", None),
        )
        if model_class is None:
            raise RuntimeError("installed transformers has no supported VLM auto class")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.state["model_status"]["vlm"].update(
            {"state": "loading", "path": str(model_path)}
        )
        self._vlm_processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self._vlm_model = model_class.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization,
            device_map="auto",
        )
        if not bool(getattr(self._vlm_model, "is_loaded_in_4bit", False)):
            raise RuntimeError("VLM did not actually load in 4-bit mode")
        self.state["model_status"]["vlm"].update(
            {
                "state": "ready",
                "device": str(next(self._vlm_model.parameters()).device),
                "quantization": "4bit-nf4",
                "is_loaded_in_4bit": True,
                "model": str(model_path),
                "error": None,
            }
        )

    def _process_vlm_request(self, request: dict[str, Any]) -> None:
        status = self.state["model_status"]["vlm"]
        if status.get("state") not in {"ready", "queued"}:
            raise RuntimeError(status.get("error") or "VLM is not ready")
        started_ns = time.monotonic_ns()
        status.update({"state": "running", "started_at_ns": started_ns})
        if self.vlm_backend is not None:
            analysis = self.vlm_backend(request)
        else:
            analysis = self._generate_vlm(request)
        if self._cancel_vlm.is_set():
            status.update(
                {
                    "state": "ready",
                    "last_request_state": "cancelled",
                    "finished_at_ns": time.monotonic_ns(),
                }
            )
            return
        finished_ns = time.monotonic_ns()
        latest = self.input_buffer.get_latest_screen()
        latest_frame_id = (
            latest.value.frame_id if latest is not None else None
        )
        stale = latest_frame_id != request["frame_id"]
        result = {
            "request_id": request["request_id"],
            "model": status.get("model", "injected"),
            "role": "teacher",
            "reference_frame_id": request["frame_id"],
            "reference_frame_timestamp_ns": request["frame_timestamp_ns"],
            "requested_at_ns": request["requested_at_ns"],
            "completed_at_ns": finished_ns,
            "analysis": str(analysis),
            "status": "stale" if stale else "current",
            "error": None,
        }
        self.state["last_teacher_visual_result"] = result
        if not stale:
            self.state["teacher_visual_result"] = result
            self.state["blackboard"]["latest_teacher_review"] = _timed(
                result, finished_ns, producer="vlm_teacher"
            )
        self.state["blackboard"]["latest_vlm_teacher_result"] = _timed(
            result, finished_ns, producer="vlm_teacher"
        )
        image_id = self.memorizer.enqueue(
            request["image"], "screen_image", priority="normal"
        )
        result_id = self.memorizer.enqueue(
            result, "vlm_teacher_result", priority="critical"
        )
        if image_id and result_id:
            self.memorizer.create_event(
                [image_id, result_id],
                summary="VLM teacher screen review",
                tags=["vlm", "teacher", result["status"]],
            )
        status.update(
            {
                "state": "ready",
                "finished_at_ns": finished_ns,
                "last_duration_ms": (finished_ns - started_ns) / 1_000_000,
                "error": None,
            }
        )
        stats = self.state["runtime_stats"]
        stats["vlm_calls"] = stats.get("vlm_calls", 0) + 1
        log_event(
            self.log_dir,
            "vlm_teacher_review_completed",
            request_id=request["request_id"],
            reference_frame_id=request["frame_id"],
            status=result["status"],
            duration_ms=(finished_ns - started_ns) / 1_000_000,
        )
        self._adjust_hormones(
            {"acetylcholine": 0.002},
            "vlm_success",
            finished_ns,
        )

    def _generate_vlm(self, request: dict[str, Any]) -> str:
        import torch
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        cancel_event = self._cancel_vlm

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del input_ids, scores, kwargs
                return cancel_event.is_set()

        image = request["image"]
        if image.shape[-1] == 4:
            image = image[..., :3][..., ::-1]
        processor = self._vlm_processor
        model = self._vlm_model
        pil_image = Image.fromarray(image)
        if hasattr(processor, "apply_chat_template"):
            inputs = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": request["prompt"]},
                        ],
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                images=pil_image,
                text=request["prompt"],
                return_tensors="pt",
            )
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = output[:, prompt_length:]
        return processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _cloud_worker(self) -> None:
        status = self.state["model_status"]["cloud_llm"]
        while not self._model_stop.is_set():
            try:
                request = self._cloud_requests.get(timeout=0.25)
            except queue.Empty:
                continue
            if request.get("kind") == "stop":
                return
            self._cancel_cloud.clear()
            started_ns = time.monotonic_ns()
            try:
                status.update({"state": "running", "started_at_ns": started_ns})
                if self.cloud_backend is not None:
                    content = self.cloud_backend(request)
                else:
                    from openai import OpenAI

                    config = self.state["model_config"]
                    api_key = self.state.get("_cloud_api_key") or os.getenv(
                        "OPENAI_API_KEY"
                    )
                    if not api_key:
                        raise RuntimeError("cloud API key is not configured")
                    client = OpenAI(
                        base_url=config["cloud_base_url"] or None,
                        api_key=api_key,
                        timeout=float(config["cloud_timeout_s"]),
                    )
                    response = client.chat.completions.create(
                        model=config["cloud_model"],
                        messages=[{"role": "user", "content": request["message"]}],
                    )
                    content = response.choices[0].message.content
                if self._cancel_cloud.is_set():
                    status.update(
                        {
                            "state": "cancelled",
                            "finished_at_ns": time.monotonic_ns(),
                        }
                    )
                    continue
                finished_ns = time.monotonic_ns()
                result = {
                    "request_id": request["request_id"],
                    "content": str(content),
                    "requested_at_ns": request["requested_at_ns"],
                    "completed_at_ns": finished_ns,
                    "model": self.state["model_config"]["cloud_model"],
                }
                self.state["cloud_result"] = result
                self.state["blackboard"]["latest_cloud_result"] = _timed(
                    result, finished_ns, producer="cloud_llm"
                )
                self.memorizer.enqueue(
                    result, "cloud_llm_result", priority="critical"
                )
                status.update(
                    {
                        "state": "ready",
                        "last_duration_ms": (finished_ns - started_ns) / 1_000_000,
                        "error": None,
                    }
                )
                stats = self.state["runtime_stats"]
                stats["cloud_llm_calls"] = (
                    stats.get("cloud_llm_calls", 0) + 1
                )
            except Exception as exc:
                status.update(
                    {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
                )

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

        if len(self.state["loaded_tnn"]) >= MAX_LOADED_TNN:
            message = f"maximum loaded TNN count is {MAX_LOADED_TNN}"
            self._publish_tnn_error(message)
            raise RuntimeError(message)
        artifact = self.memorizer.resolve_tnn_artifact(tnn_id, version)
        device = torch.device(
            self.runtime_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            message = "target CUDA device is unavailable"
            self._publish_tnn_error(message)
            raise RuntimeError(message)
        self._check_tnn_resources(Path(artifact["weights_path"]), device)
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
        description = json.loads(
            Path(artifact["description_path"]).read_text(encoding="utf-8")
        )
        node.update(
            {
                "version": artifact["version"],
                "description": description.get("purpose", ""),
                "model_path": artifact["model_path"],
                "weights_path": artifact["weights_path"],
                "precision": str(
                    next(model.parameters(), torch.empty(0)).dtype
                ),
                "input_schema": model.get_input_schema(),
                "output_schema": model.get_output_schema(),
                "memory_id": artifact["memory_id"],
            }
        )
        self.state["resource_status"]["tnn_device"] = str(device)
        self.state["resource_status"]["last_tnn_load"] = {
            "tnn_id": tnn_id,
            "success": True,
            "timestamp_ns": time.monotonic_ns(),
        }
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
        self.state["resource_status"]["last_tnn_unload"] = {
            "tnn_id": tnn_id,
            "success": True,
            "timestamp_ns": time.monotonic_ns(),
        }

    def _check_tnn_resources(self, weights_path: Path, device: Any) -> None:
        import psutil
        import torch

        required = max(weights_path.stat().st_size * 2, 1_000_000)
        available_ram = int(psutil.virtual_memory().available)
        if available_ram < required:
            message = (
                f"insufficient system memory: need {required}, "
                f"available {available_ram}"
            )
            self._publish_tnn_error(message)
            raise MemoryError(message)
        if device.type == "cuda":
            free_vram, total_vram = torch.cuda.mem_get_info(device)
            if free_vram < required:
                message = (
                    f"insufficient GPU memory: need {required}, "
                    f"free {free_vram}, total {total_vram}"
                )
                self._publish_tnn_error(message)
                raise MemoryError(message)

    def _publish_tnn_error(self, message: str) -> None:
        now_ns = time.monotonic_ns()
        record = {"message": message, "timestamp_ns": now_ns}
        self.state["resource_status"]["last_tnn_error"] = record
        self.state["blackboard"]["tnn_lifecycle_error"] = _timed(
            record, now_ns, producer="core"
        )

    def step(self, now_ns: int | None = None) -> list[dict[str, Any]]:
        if (
            not self.state["cold_started"]
            or self.state["emergency_stop"]
            or self.state["paused"]
        ):
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
        active_window = summary.get("active_window")
        if active_window:
            self.state["world"]["active_window"] = active_window
            self.state["world"]["updated_at_ns"] = now_ns
        self.state["blackboard"]["latest_input_summary"] = _timed(
            summary, now_ns, producer="input_buffer"
        )
        self._clean_blackboard(now_ns)
        self._update_hormones(now_ns)
        self._update_resources(now_ns)
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
                self.state["active_tnn"].discard(tnn_id)
                node["status"] = "failed"
                node["last_error"] = f"{type(exc).__name__}: {exc}"
                self._record_error(f"tnn:{tnn_id}", exc, critical=False)
                continue

        results: list[dict[str, Any]] = []
        while self.state["action_queue"]:
            action = self.state["action_queue"].popleft()
            candidate_id = action["candidate_id"]
            if candidate_id in self.state["consumed_action_ids"]:
                continue
            self.state["consumed_action_ids"].add(candidate_id)
            try:
                result = run_once(
                    self.state,
                    action,
                    self.log_dir,
                    before_output=self._mark_expected_output,
                )
                results.append(result)
                self._remember_chain(action, result)
                if result.get("blocked"):
                    self._adjust_hormones(
                        {"cortisol": 0.004},
                        "safegate_block",
                        now_ns,
                    )
                elif result.get("executed") or result.get("simulated"):
                    self._adjust_hormones(
                        {"dopamine": 0.002},
                        "action_success",
                        now_ns,
                    )
            except Exception as exc:
                self._record_error("output", exc, critical=True)
                break
        return results

    def _mark_expected_output(self, action: dict[str, Any]) -> None:
        if self.state["output_mode"] != "real":
            return
        payload = action["payload"]
        if action["action_type"] == "mouse":
            target = None
            if payload.get("x2") is not None and payload.get("y2") is not None:
                target = (int(payload["x2"]), int(payload["y2"]))
            elif payload.get("x") is not None and payload.get("y") is not None:
                target = (int(payload["x"]), int(payload["y"]))
            self.input_buffer.mark_eve_mouse_action(
                action["candidate_id"],
                target=target,
                duration_s=float(payload.get("duration", 0.0)),
            )
        elif action["action_type"] == "keyboard":
            self.input_buffer.mark_eve_keyboard_action(action["candidate_id"])

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
                "world", "myself", "tnn_status", "loop_status",
                "resource_status", "latest_error", "model_config"
            )
        }
        snapshot["active_tnn"] = sorted(self.state["active_tnn"])
        snapshot["loaded_tnn"] = [
            {
                key: node.get(key)
                for key in (
                    "tnn_id", "version", "description", "model_path",
                    "weights_path", "device", "precision", "run_frequency_hz",
                    "output_ttl_ns", "input_schema", "output_schema",
                    "memory_id",
                )
            }
            for node in self.state["loaded_tnn"].values()
        ]
        snapshot["blackboard"] = dict(
            list(self.state["blackboard"].items())[-50:]
        )
        destination.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )

    def load_snapshot(self, path: str | Path) -> bool:
        """Restore durable state without restoring permissions or emergency state."""
        source = Path(path)
        if not source.is_file():
            return False
        snapshot = json.loads(source.read_text(encoding="utf-8"))
        for name in ("world", "myself", "blackboard"):
            value = snapshot.get(name)
            if isinstance(value, dict):
                self.state[name].update(value)
        saved_model_config = snapshot.get("model_config")
        if isinstance(saved_model_config, dict):
            for name, value in saved_model_config.items():
                if name not in self.state["model_config"]:
                    continue
                if name in {
                    "local_llm_path", "vlm_path", "yolo_model_path"
                } and not value:
                    continue
                self.state["model_config"][name] = value
        self.state["restored_tnn_descriptions"] = list(
            snapshot.get("loaded_tnn", [])
        )
        self.state["requested_tnn_on_restore"] = [
            str(item) for item in snapshot.get("active_tnn", [])
        ][:MAX_LOADED_TNN]
        self.state["emergency_stop"] = False
        self.state["permissions"] = safegate.default_permissions(False)
        return True

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
            finished_ns = time.monotonic_ns()
            elapsed_s = (finished_ns - started_ns) / 1_000_000_000
            self._record_node_timing("core", started_ns, finished_ns)
            self._stop_event.wait(max(0.001, self.interval_s - elapsed_s))

    def _record_node_timing(
        self,
        name: str,
        started_ns: int,
        finished_ns: int,
        error: str | None = None,
    ) -> None:
        node = self.state["node_status"].setdefault(
            name,
            {
                "state": "running",
                "last_run_ns": 0,
                "last_duration_ms": 0.0,
                "average_duration_ms": 0.0,
                "run_count": 0,
                "last_error": None,
            },
        )
        duration_ms = (finished_ns - started_ns) / 1_000_000
        node["last_run_ns"] = finished_ns
        node["last_duration_ms"] = duration_ms
        node["run_count"] += 1
        node["average_duration_ms"] += (
            duration_ms - node["average_duration_ms"]
        ) / node["run_count"]
        node["last_error"] = error
        node["state"] = "error" if error else "running"
        node["actual_hz"] = node["run_count"] / max(
            (finished_ns - self._started_at_ns) / 1_000_000_000,
            1e-9,
        )

    def _clean_blackboard(self, now_ns: int) -> None:
        expired = [
            key
            for key, value in self.state["blackboard"].items()
            if int(value.get("valid_until_ns", 0))
            and now_ns > int(value["valid_until_ns"])
        ]
        for key in expired:
            self.state["blackboard"].pop(key, None)

    def _update_hormones(self, now_ns: int) -> None:
        elapsed_s = max(0.0, (now_ns - self._last_hormone_ns) / 1_000_000_000)
        if elapsed_s < 0.25:
            return
        hormones = self.state["myself"]["hormones"]
        recovery = min(1.0, elapsed_s / 300.0)
        for name in DEFAULT_HORMONES:
            value = float(hormones.get(name, 0.5))
            hormones[name] = min(1.0, max(0.0, value + (0.5 - value) * recovery))
        self._last_hormone_ns = now_ns
        self._update_tendencies(now_ns)

    def _adjust_hormones(
        self, deltas: dict[str, float], reason: str, now_ns: int
    ) -> None:
        hormones = self.state["myself"]["hormones"]
        changes = {}
        for name, delta in deltas.items():
            old = float(hormones.get(name, 0.5))
            new = min(1.0, max(0.0, old + float(delta)))
            hormones[name] = new
            changes[name] = {"old": old, "new": new, "delta": new - old}
        record = {
            "timestamp_ns": now_ns,
            "reason": reason,
            "changes": changes,
        }
        self.state["myself"]["last_hormone_change"] = record
        self.state["blackboard"]["hormone_change"] = _timed(
            record, now_ns, producer="core"
        )
        self._update_tendencies(now_ns)

    def _update_tendencies(self, now_ns: int) -> None:
        hormones = self.state["myself"]["hormones"]
        values = {
            "mouse": 0.3 + hormones["dopamine"] * 0.3 - hormones["cortisol"] * 0.2,
            "keyboard": 0.25 + hormones["dopamine"] * 0.25,
            "send_text": 0.3 + hormones["oxytocin"] * 0.3,
            "speak": 0.2 + hormones["oxytocin"] * 0.35,
            "thinking": 0.3 + hormones["acetylcholine"] * 0.4,
            "pause": hormones["cortisol"] * 0.5,
            "sleep": 0.2 + hormones["serotonin"] * 0.3,
        }
        permissions = self.state["permissions"]
        tendencies = self.state["myself"]["tendencies"]
        for name, raw in values.items():
            allowed = True
            if name == "mouse":
                allowed = any(permissions["mouse"].values())
            elif name == "keyboard":
                allowed = any(permissions["keyboard"].values())
            elif name in {"send_text", "speak"}:
                allowed = bool(permissions[name])
            tendencies[name] = {
                "strength": min(1.0, max(0.0, raw)),
                "updated_at_ns": now_ns,
                "suppressed": not allowed or self.state["emergency_stop"],
                "reason": (
                    "emergency_stop"
                    if self.state["emergency_stop"]
                    else ("permission_disabled" if not allowed else "")
                ),
            }

    def _update_resources(self, now_ns: int) -> None:
        last_ns = int(self.state["resource_status"].get("updated_at_ns", 0))
        if now_ns - last_ns < 1_000_000_000:
            return
        import psutil

        process = psutil.Process()
        memory = psutil.virtual_memory()
        resources = self.state["resource_status"]
        resources.update(
            {
                "updated_at_ns": now_ns,
                "system_memory_used": int(memory.used),
                "system_memory_total": int(memory.total),
                "system_memory_percent": float(memory.percent),
                "process_memory": int(process.memory_info().rss),
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
            }
        )
        try:
            import torch

            if torch.cuda.is_available():
                free_vram, total_vram = torch.cuda.mem_get_info()
                resources.update(
                    {
                        "gpu_memory_allocated": int(torch.cuda.memory_allocated()),
                        "gpu_memory_reserved": int(torch.cuda.memory_reserved()),
                        "gpu_memory_peak": int(torch.cuda.max_memory_allocated()),
                        "gpu_memory_free": int(free_vram),
                        "gpu_memory_total": int(total_vram),
                        "gpu_utilization": int(torch.cuda.utilization(0)),
                    }
                )
                if total_vram and free_vram / total_vram < 0.05:
                    self._adjust_hormones(
                        {"cortisol": 0.005},
                        "gpu_memory_pressure",
                        now_ns,
                    )
        except Exception as exc:
            resources["gpu_error"] = str(exc)
        if float(memory.percent) >= 90.0:
            self._adjust_hormones(
                {"cortisol": 0.005},
                "system_memory_pressure",
                now_ns,
            )
        resources["tnn_summary"] = {
            "loaded_count": len(self.state["loaded_tnn"]),
            "active_count": len(self.state["active_tnn"]),
            "total_inference_ms": sum(
                float(node.get("last_duration_ms", 0))
                for node in self.state["loaded_tnn"].values()
            ),
            "gpu_memory": sum(
                sum(
                    parameter.nelement() * parameter.element_size()
                    for parameter in node["model"].parameters()
                )
                for node in self.state["loaded_tnn"].values()
                if node.get("model") is not None
                and any(p.is_cuda for p in node["model"].parameters())
            ),
        }
        self.state["node_status"]["capture"] = {
            "state": self.input_buffer.capture_health().get("state", "unknown"),
            "last_run_ns": now_ns,
            "last_duration_ms": 0.0,
            "average_duration_ms": 0.0,
            "actual_hz": self.input_buffer.capture_stats().get("screen_fps", 0.0),
            "last_error": self.input_buffer.capture_error,
        }
        self.state["node_status"]["buffer"] = {
            "state": "running",
            "last_run_ns": now_ns,
            "last_duration_ms": 0.0,
            "average_duration_ms": 0.0,
            "actual_hz": self.stats().get("loop_hz", 0.0),
            "last_error": None,
        }
        writer = self.memorizer.writer_stats()
        self.state["node_status"]["memory_writer"] = {
            "state": "running" if self.memorizer.writer_running else "stopped",
            "last_run_ns": now_ns,
            "last_duration_ms": 0.0,
            "average_duration_ms": 0.0,
            "actual_hz": int(writer["written"])
            / max((now_ns - self._started_at_ns) / 1_000_000_000, 1e-9),
            "last_error": str(self.memorizer.last_writer_error or ""),
        }
        review = self.memorizer.review_status()
        self.state["node_status"]["memory_review"] = {
            "state": review.get("state", "idle"),
            "last_run_ns": now_ns if self.memorizer.review_running else 0,
            "last_duration_ms": 0.0,
            "average_duration_ms": 0.0,
            "actual_hz": 0.0,
            "last_error": review.get("last_error"),
        }
        for name in ("local_llm", "yolo", "vlm", "cloud_llm"):
            model = self.state["model_status"][name]
            self.state["node_status"][name] = {
                "state": model.get("state", "unknown"),
                "last_run_ns": model.get("finished_at_ns", 0),
                "last_duration_ms": model.get("last_duration_ms", 0.0),
                "average_duration_ms": model.get("average_duration_ms", 0.0),
                "actual_hz": int(
                    self.state["runtime_stats"].get(f"{name}_calls", 0)
                )
                / max((now_ns - self._started_at_ns) / 1_000_000_000, 1e-9),
                "last_error": model.get("error"),
            }
        for name in (
            "safegate", "mouse_output", "keyboard_output",
            "speak_output", "dock",
        ):
            self.state["node_status"].setdefault(
                name,
                {
                    "state": "not_running",
                    "last_run_ns": 0,
                    "last_duration_ms": 0.0,
                    "average_duration_ms": 0.0,
                    "last_error": None,
                },
            )

    def _input_summary(self, input_state: dict[str, Any]) -> dict[str, Any]:
        screen = input_state["latest"]["screen"]
        cursor = input_state["latest"]["cursor"]
        keyboard_activity = input_state["latest"]["keyboard_activity"]
        active_window = input_state["latest"]["active_window"]
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
            "keyboard_activity": (
                dict(keyboard_activity.value)
                if keyboard_activity is not None
                else None
            ),
            "active_window": (
                dict(active_window.value)
                if active_window is not None
                else None
            ),
            "dropped_screen_frames": input_state["dropped_screen_frames"],
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
        if "myself" in self.state and "hormones" in self.state["myself"]:
            self._adjust_hormones(
                {"cortisol": 0.01},
                f"{loop_node}_error",
                time.monotonic_ns(),
            )
        if critical:
            self._failed_event.set()
        log_event(self.log_dir, "runtime_error", **error)
        if loop_node != "memory":
            self.memorizer.enqueue(error, "runtime_error", priority="critical")
