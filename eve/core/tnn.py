"""Minimal TNN runtime: descriptors naturally define the live data flow."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from eve.state import ActionCandidate, ActionKind, RuntimeState


@dataclass(frozen=True)
class SourceRef:
    """A single authoritative input reference.

    Supported forms: ``state:<kind>``, ``world:<field>``,
    ``myself:<field>``, ``blackboard:<key>`` and ``tnn:<id>.<field>``.
    """

    value: str

    def parts(self) -> tuple[str, str, str | None]:
        if self.value.startswith("tnn:"):
            target = self.value[4:]
            tnn_id, separator, field_name = target.rpartition(".")
            if not separator or not tnn_id or not field_name:
                raise ValueError(f"invalid TNN SourceRef: {self.value}")
            return "tnn", tnn_id, field_name
        source, separator, key = self.value.partition(":")
        if not separator or source not in {"state", "world", "myself", "blackboard"}:
            raise ValueError(f"invalid SourceRef: {self.value}")
        return source, key, None


@dataclass(frozen=True)
class TNNDescriptor:
    tnn_id: str
    inputs: dict[str, SourceRef] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    run_frequency_hz: float = 1.0
    output_ttl_ns: int = 1_000_000_000
    action_output: str | None = None
    implementation: str = "trained_tnn"


class TNN(Protocol):
    descriptor: TNNDescriptor

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...


def load_tnn(state: RuntimeState, node: TNN) -> None:
    descriptor = node.descriptor
    if descriptor.tnn_id in state.loaded_tnn:
        raise ValueError(f"TNN already loaded: {descriptor.tnn_id}")
    if descriptor.run_frequency_hz <= 0:
        raise ValueError("run_frequency_hz must be positive")
    state.loaded_tnn[descriptor.tnn_id] = node
    state.myself.active_tnn.add(descriptor.tnn_id)


def unload_tnn(state: RuntimeState, tnn_id: str) -> None:
    state.loaded_tnn.pop(tnn_id, None)
    state.myself.active_tnn.discard(tnn_id)
    state.last_run_ns.pop(tnn_id, None)
    state.tnn_outputs.pop(tnn_id, None)


def due_tnn_ids(state: RuntimeState, now_ns: int) -> list[str]:
    due: list[str] = []
    for tnn_id in sorted(state.myself.active_tnn):
        node = state.loaded_tnn.get(tnn_id)
        if node is None:
            continue
        interval_ns = int(1_000_000_000 / node.descriptor.run_frequency_hz)
        if now_ns - state.last_run_ns.get(tnn_id, 0) >= interval_ns:
            due.append(tnn_id)
    return due


def resolve_inputs(
    state: RuntimeState, input_buffer: Any, descriptor: TNNDescriptor, now_ns: int
) -> dict[str, Any] | None:
    resolved: dict[str, Any] = {}
    for input_name, reference in descriptor.inputs.items():
        source, key, field_name = reference.parts()
        value: Any = None
        if source == "state":
            sample = input_buffer.latest(key)
            value = sample.value if sample is not None else None
        elif source == "world":
            value = state.world.get(key)
        elif source == "myself":
            value = getattr(state.myself, key, None)
        elif source == "blackboard":
            timed = state.read_latest(key, now_ns)
            value = timed.value if timed is not None else None
        elif source == "tnn" and field_name is not None:
            timed = state.tnn_outputs.get(key, {}).get(field_name)
            value = timed.value if timed is not None and timed.valid(now_ns) else None
        if value is None:
            return None
        resolved[input_name] = value
    return resolved


def run_node(
    state: RuntimeState, input_buffer: Any, tnn_id: str, now_ns: int | None = None
) -> dict[str, Any] | None:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    node = state.loaded_tnn[tnn_id]
    descriptor = node.descriptor
    inputs = resolve_inputs(state, input_buffer, descriptor, now_ns)
    if inputs is None:
        return None
    outputs = node.run(inputs)
    unknown = set(outputs) - set(descriptor.outputs)
    if unknown:
        raise ValueError(f"{tnn_id} produced undeclared outputs: {sorted(unknown)}")
    state.record_tnn_outputs(
        tnn_id, outputs, now_ns=now_ns, ttl_ns=descriptor.output_ttl_ns
    )
    state.last_run_ns[tnn_id] = now_ns
    if descriptor.action_output and descriptor.action_output in outputs:
        action_data = outputs[descriptor.action_output]
        if not isinstance(action_data, dict):
            raise TypeError("action output must be a mapping")
        action_id = str(action_data.get("action_id", f"{tnn_id}:{now_ns}"))
        kind = ActionKind(action_data["kind"])
        horizon_ns = int(action_data.get("horizon_ns", descriptor.output_ttl_ns))
        state.enqueue_action(
            ActionCandidate(
                action_id=action_id,
                kind=kind,
                payload=dict(action_data.get("payload", {})),
                created_at_ns=now_ns,
                valid_until_ns=now_ns + horizon_ns if horizon_ns else 0,
                origin=tnn_id,
            )
        )
    return outputs


class SmokeActionNode:
    """Explicit rule placeholder used only by the safe CLI smoke run."""

    descriptor = TNNDescriptor(
        tnn_id="smoke_rule",
        inputs={"cursor": SourceRef("state:cursor")},
        outputs=("action_candidate",),
        run_frequency_hz=20.0,
        action_output="action_candidate",
        implementation="rule_placeholder_not_trained",
    )

    def __init__(self) -> None:
        self._emitted = False

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._emitted:
            return {}
        self._emitted = True
        x, y = inputs["cursor"]
        return {
            "action_candidate": {
                "action_id": "smoke-action-1",
                "kind": "mouse",
                "payload": {"action": "moveTo", "x": x, "y": y},
                "horizon_ns": 1_000_000_000,
            }
        }
