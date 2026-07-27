"""Minimal TNN runtime: descriptors naturally define the live data flow."""
from __future__ import annotations

import importlib.util
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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


def load_tnn(
    state: RuntimeState,
    node: TNN,
    *,
    activate: bool = True,
) -> None:
    descriptor = node.descriptor
    if descriptor.tnn_id in state.loaded_tnn:
        raise ValueError(f"TNN already loaded: {descriptor.tnn_id}")
    if descriptor.run_frequency_hz <= 0:
        raise ValueError("run_frequency_hz must be positive")
    state.loaded_tnn[descriptor.tnn_id] = node
    state.tnn_status[descriptor.tnn_id] = "loaded"
    if activate:
        state.myself.active_tnn.add(descriptor.tnn_id)


def unload_tnn(state: RuntimeState, tnn_id: str) -> None:
    state.loaded_tnn.pop(tnn_id, None)
    state.myself.active_tnn.discard(tnn_id)
    state.tnn_status[tnn_id] = "unloaded"
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
    state.increment_stat("tnn_invocations")
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
        observed_times: list[int] = []
        for reference in descriptor.inputs.values():
            source, key, field_name = reference.parts()
            if source == "state":
                sample = input_buffer.latest(key)
                if sample is not None:
                    observed_times.append(sample.timestamp_ns)
            elif source == "tnn" and field_name is not None:
                timed = state.tnn_outputs.get(key, {}).get(field_name)
                if timed is not None:
                    observed_times.append(timed.produced_at_ns)
        observed_at_ns = max(observed_times, default=now_ns)
        state.enqueue_action(
            ActionCandidate(
                action_id=action_id,
                kind=kind,
                payload=dict(action_data.get("payload", {})),
                created_at_ns=now_ns,
                valid_until_ns=now_ns + horizon_ns if horizon_ns else 0,
                origin=tnn_id,
                observed_at_ns=observed_at_ns,
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


class TrainedTNNNode:
    """Adapter from the persisted TinyNN contract to the existing live node API."""

    def __init__(
        self,
        model: TinyNN,
        *,
        tnn_id: str,
        input_refs: dict[str, SourceRef],
        run_frequency_hz: float,
        output_ttl_ns: int,
        action_output: str | None,
    ) -> None:
        import torch

        self.model = model
        self.device = next(
            model.parameters(),
            next(model.buffers(), torch.empty(0)),
        ).device
        self.descriptor = TNNDescriptor(
            tnn_id=tnn_id,
            inputs=input_refs,
            outputs=tuple(model.get_output_schema()),
            run_frequency_hz=run_frequency_hz,
            output_ttl_ns=output_ttl_ns,
            action_output=action_output,
        )

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch

        prepared: dict[str, Any] = {}
        for name, value in inputs.items():
            schema = self.model.get_input_schema().get(name, {})
            dtype_name = schema.get("dtype")
            dtype = getattr(torch, dtype_name, None) if dtype_name else None
            if isinstance(value, torch.Tensor):
                prepared[name] = value.to(device=self.device, dtype=dtype)
            elif isinstance(dtype, torch.dtype):
                prepared[name] = torch.as_tensor(
                    value, device=self.device, dtype=dtype
                )
            else:
                prepared[name] = value
        return self.model.infer(prepared)


def _import_runtime_model(model_path: str | Path, factory: str) -> TinyNN:
    from eve.dock.tinynn import TinyNN

    path = Path(model_path).resolve()
    module_name = f"_eve_runtime_tnn_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TNN model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(path.read_bytes(), str(path), "exec")
        exec(code, module.__dict__)
        creator = getattr(module, factory, None)
        if not callable(creator):
            raise AttributeError(f"{path} does not define callable {factory}()")
        model = creator()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(model, TinyNN):
        raise TypeError(f"{factory}() must return TinyNN, got {type(model).__name__}")
    return model


def load_tnn_runtime(
    state: RuntimeState,
    memorizer: Any,
    tnn_id: str,
    version: str | None = None,
    *,
    device: Any = "cpu",
    input_refs: dict[str, SourceRef | str] | None = None,
    run_frequency_hz: float = 1.0,
    output_ttl_ns: int = 1_000_000_000,
    action_output: str | None = None,
    factory: str = "create_tnn",
) -> TrainedTNNNode:
    """Create the sole live instance from a cataloged, persisted artifact."""
    import torch

    artifact = memorizer.resolve_tnn_artifact(tnn_id, version)
    model = _import_runtime_model(artifact["model_path"], factory)
    resolved_device = torch.device(device)
    model.load_weights(artifact["weights_path"], map_location=resolved_device)
    model.to(resolved_device)
    model.eval()
    refs = input_refs or {
        name: SourceRef(f"blackboard:{name}") for name in model.get_input_schema()
    }
    normalized_refs = {
        name: value if isinstance(value, SourceRef) else SourceRef(value)
        for name, value in refs.items()
    }
    if set(normalized_refs) != set(model.get_input_schema()):
        raise ValueError("runtime input references must match the model input schema")
    node = TrainedTNNNode(
        model,
        tnn_id=tnn_id,
        input_refs=normalized_refs,
        run_frequency_hz=run_frequency_hz,
        output_ttl_ns=output_ttl_ns,
        action_output=action_output,
    )
    load_tnn(state, node)
    return node


def unload_tnn_runtime(state: RuntimeState, tnn_id: str) -> None:
    """Detach one trained node, clear its cached state, and release its device."""
    import torch

    node = state.loaded_tnn.get(tnn_id)
    unload_tnn(state, tnn_id)
    if isinstance(node, TrainedTNNNode):
        was_cuda = any(parameter.is_cuda for parameter in node.model.parameters())
        node.model.to("cpu")
        del node
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
