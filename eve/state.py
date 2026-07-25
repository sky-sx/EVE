"""Authoritative in-memory state for the minimal EVE runtime."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OutputMode(str, Enum):
    DISABLED = "disabled"
    MOCK = "mock"
    REAL = "real"


class ActionKind(str, Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    SPEAK = "speak"


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    kind: ActionKind
    payload: dict[str, Any] = field(default_factory=dict)
    created_at_ns: int = field(default_factory=time.monotonic_ns)
    valid_until_ns: int = 0
    origin: str = ""


@dataclass(frozen=True)
class SafegateResult:
    allowed: bool
    reason: str
    checked_at_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class OutputResult:
    action_id: str
    kind: str
    mode: str
    started_at_ns: int = 0
    finished_at_ns: int = 0
    executed: bool = False
    simulated: bool = False
    blocked: bool = False
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimedValue:
    value: Any
    produced_at_ns: int = field(default_factory=time.monotonic_ns)
    valid_until_ns: int = 0
    producer: str = ""

    def valid(self, now_ns: int | None = None) -> bool:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return self.valid_until_ns == 0 or now_ns <= self.valid_until_ns


def _default_hormones() -> dict[str, float]:
    return {
        "dopamine": 0.5,
        "serotonin": 0.5,
        "norepinephrine": 0.5,
        "oxytocin": 0.5,
        "cortisol": 0.5,
        "acetylcholine": 0.5,
    }


@dataclass
class MyselfState:
    hormones: dict[str, float] = field(default_factory=_default_hormones)
    sleep_requested: bool = False
    active_tnn: set[str] = field(default_factory=set)

    def settle_hormones(self, rate: float = 0.01) -> None:
        """Move the six experimental values slowly toward a neutral baseline."""
        for name in _default_hormones():
            current = min(1.0, max(0.0, float(self.hormones.get(name, 0.5))))
            self.hormones[name] = current + (0.5 - current) * rate


@dataclass
class RuntimeErrorRecord:
    timestamp_ns: int
    loop_node: str
    exception_type: str
    message: str
    traceback: str
    relevant_source: Any
    recovery_action: str


@dataclass
class RuntimeState:
    """The only authority for runtime control, TNN outputs and action queues."""

    cold_started: bool = False
    emergency_stopped: bool = False
    output_mode: OutputMode = OutputMode.DISABLED
    mouse_allowed: bool = False
    keyboard_allowed: bool = False
    speak_allowed: bool = False
    blocked_until_ns: int = 0
    eve_expected_events: set[str] = field(default_factory=set)
    human_activity_detected_at_ns: int = 0
    world: dict[str, Any] = field(default_factory=dict)
    myself: MyselfState = field(default_factory=MyselfState)
    blackboard: dict[str, TimedValue] = field(default_factory=dict)
    action_queue: deque[ActionCandidate] = field(default_factory=deque)
    consumed_action_ids: set[str] = field(default_factory=set)
    loaded_tnn: dict[str, Any] = field(default_factory=dict)
    last_run_ns: dict[str, int] = field(default_factory=dict)
    tnn_outputs: dict[str, dict[str, TimedValue]] = field(default_factory=dict)
    latest_output: OutputResult | None = None
    latest_error: RuntimeErrorRecord | None = None
    memory_ids: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def publish(self, key: str, value: TimedValue) -> None:
        with self._lock:
            self.blackboard[key] = value

    def read_latest(self, key: str, now_ns: int | None = None) -> TimedValue | None:
        with self._lock:
            item = self.blackboard.get(key)
            if item is not None and not item.valid(now_ns):
                self.blackboard.pop(key, None)
                return None
            return item

    def enqueue_action(self, action: ActionCandidate) -> bool:
        with self._lock:
            if action.action_id in self.consumed_action_ids:
                return False
            if any(item.action_id == action.action_id for item in self.action_queue):
                return False
            self.action_queue.append(action)
            return True

    def consume_action(self) -> ActionCandidate | None:
        with self._lock:
            while self.action_queue:
                action = self.action_queue.popleft()
                if action.action_id not in self.consumed_action_ids:
                    self.consumed_action_ids.add(action.action_id)
                    return action
            return None

    def record_tnn_outputs(
        self, tnn_id: str, outputs: dict[str, Any], *, now_ns: int, ttl_ns: int
    ) -> None:
        with self._lock:
            self.tnn_outputs[tnn_id] = {
                name: TimedValue(
                    value=value,
                    produced_at_ns=now_ns,
                    valid_until_ns=now_ns + ttl_ns if ttl_ns else 0,
                    producer=tnn_id,
                )
                for name, value in outputs.items()
            }

    def save_snapshot(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "world": self.world,
            "hormones": self.myself.hormones,
            "sleep_requested": self.myself.sleep_requested,
            "active_tnn": sorted(self.myself.active_tnn),
            "output_mode": self.output_mode.value,
            "emergency_stopped": self.emergency_stopped,
            "latest_error": asdict(self.latest_error) if self.latest_error else None,
        }
        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=repr),
            encoding="utf-8",
        )

    def load_snapshot(self, path: str | Path) -> bool:
        source = Path(path)
        if not source.exists():
            return False
        data = json.loads(source.read_text(encoding="utf-8"))
        self.world = dict(data.get("world", {}))
        saved_hormones = data.get("hormones", {})
        for name in _default_hormones():
            if name in saved_hormones:
                self.myself.hormones[name] = float(saved_hormones[name])
        self.myself.sleep_requested = bool(data.get("sleep_requested", False))
        self.myself.active_tnn = set(data.get("active_tnn", []))
        return True
