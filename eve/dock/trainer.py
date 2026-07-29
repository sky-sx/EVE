"""Training orchestration for JSON-defined and Python-defined TinyNN models."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
import shutil
import sys
import textwrap
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from eve.dock.tinynn import TinyNN


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SUPPORTED_OPS = {
    "linear",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "maxpool1d", "maxpool2d", "maxpool3d",
    "avgpool1d", "avgpool2d", "avgpool3d",
    "adaptive_avgpool1d", "adaptive_avgpool2d", "adaptive_avgpool3d",
    "batchnorm1d", "batchnorm2d", "batchnorm3d",
    "layernorm", "groupnorm", "dropout", "embedding",
    "rnn", "gru", "lstm", "multihead_attention",
    "relu", "leaky_relu", "gelu", "silu", "mish", "tanh", "sigmoid", "softmax",
    "flatten", "reshape", "permute", "transpose", "squeeze", "unsqueeze",
    "concat", "stack", "add", "mul", "mean", "sum",
}
ARTIFACT_FILES = {
    "model.py",
    "weights.pt",
    "structure.json",
    "description.json",
    "training.json",
}


@dataclass
class TrainingOrder:
    order_id: str
    priority: str = "medium"
    target_tnn_id: str = ""
    training_data: list[str] = field(default_factory=list)
    experience_query: dict[str, Any] = field(default_factory=dict)
    evaluation_data: list[str] = field(default_factory=list)
    regression_data: list[str] = field(default_factory=list)
    fitness_data: list[str] = field(default_factory=list)
    teacher_mode: str = "existing_label"
    teacher_prompt: str = ""
    task_id: str = ""
    purpose: str = ""
    train_ratio: float = 0.8
    minimum_samples: int = 1
    acceptance: dict[str, float] = field(default_factory=dict)
    minimum_qnn_fitness: float | None = None
    minimum_qnn_margin: float = 0.0
    continue_training: bool = False
    epochs: int | None = None
    definition: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    order_id: str
    tnn_id: str = ""
    version: str = ""
    success: bool = False
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    report_memory_id: str = ""
    weights_path: str = ""
    artifact_path: str = ""
    latency_ms: float = 0.0
    sample_count: int = 0
    accepted: bool = True
    rejection_reason: str | None = None


def shape_locator_definition(*, version: str | None = None) -> dict[str, Any]:
    return {
        "mode": "json",
        "version": version or f"v{time.time_ns()}",
        "structure": {
            "input_schema": {
                "image": {
                    "dtype": "float32",
                    "shape": [3, 64, 64],
                    "preprocess": "screen_rgb_64",
                }
            },
            "output_schema": {
                "target_center": {"dtype": "float32", "shape": [2]}
            },
            "target_schema": {
                "target_center": {"dtype": "float32", "shape": [2]}
            },
            "nodes": [
                {
                    "id": "conv1",
                    "op": "conv2d",
                    "from": ["image"],
                    "params": {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 5,
                        "stride": 2,
                    },
                },
                {
                    "id": "relu1",
                    "op": "relu",
                    "from": ["conv1"],
                    "params": {},
                },
                {
                    "id": "pool1",
                    "op": "maxpool2d",
                    "from": ["relu1"],
                    "params": {"kernel_size": 2},
                },
                {
                    "id": "conv2",
                    "op": "conv2d",
                    "from": ["pool1"],
                    "params": {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                },
                {
                    "id": "relu2",
                    "op": "relu",
                    "from": ["conv2"],
                    "params": {},
                },
                {
                    "id": "adaptive",
                    "op": "adaptive_avgpool2d",
                    "from": ["relu2"],
                    "params": {"output_size": [4, 4]},
                },
                {
                    "id": "flat",
                    "op": "flatten",
                    "from": ["adaptive"],
                    "params": {"start_dim": 1},
                },
                {
                    "id": "hidden",
                    "op": "linear",
                    "from": ["flat"],
                    "params": {"in_features": 256, "out_features": 64},
                },
                {
                    "id": "relu3",
                    "op": "relu",
                    "from": ["hidden"],
                    "params": {},
                },
                {
                    "id": "center",
                    "op": "linear",
                    "from": ["relu3"],
                    "params": {"in_features": 64, "out_features": 2},
                },
                {
                    "id": "normalized",
                    "op": "sigmoid",
                    "from": ["center"],
                    "params": {},
                },
            ],
            "outputs": {"target_center": "normalized"},
            "training": {
                "batch_size": 16,
                "optimizer": {"type": "adam", "lr": 0.001},
                "losses": [
                    {
                        "type": "mse",
                        "output": "target_center",
                        "target": "target_center",
                    }
                ],
            },
            "runtime": {
                "input_refs": {"image": "state:screen"},
                "run_frequency_hz": 2.0,
                "output_ttl_ns": 500_000_000,
                "action_output": "target_center",
                "action_template": {
                    "action_type": "mouse",
                    "action": "click",
                    "coordinates": "normalized_xy",
                    "button": "left",
                },
            },
        },
    }


def qnn_critic_definition(*, version: str | None = None) -> dict[str, Any]:
    """Small critic mapping screen state and a mouse action to expected reward."""
    return {
        "mode": "json",
        "version": version or f"v{time.time_ns()}",
        "structure": {
            "input_schema": {
                "image": {
                    "dtype": "float32",
                    "shape": [3, 64, 64],
                    "preprocess": "screen_rgb_64",
                },
                "action": {"dtype": "float32", "shape": [4]},
            },
            "output_schema": {
                "q_value": {"dtype": "float32", "shape": [1]}
            },
            "target_schema": {
                "reward": {"dtype": "float32", "shape": [1]}
            },
            "nodes": [
                {
                    "id": "conv1",
                    "op": "conv2d",
                    "from": ["image"],
                    "params": {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 5,
                        "stride": 2,
                    },
                },
                {
                    "id": "image_relu1",
                    "op": "relu",
                    "from": ["conv1"],
                    "params": {},
                },
                {
                    "id": "pool1",
                    "op": "maxpool2d",
                    "from": ["image_relu1"],
                    "params": {"kernel_size": 2},
                },
                {
                    "id": "conv2",
                    "op": "conv2d",
                    "from": ["pool1"],
                    "params": {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                },
                {
                    "id": "image_relu2",
                    "op": "relu",
                    "from": ["conv2"],
                    "params": {},
                },
                {
                    "id": "adaptive",
                    "op": "adaptive_avgpool2d",
                    "from": ["image_relu2"],
                    "params": {"output_size": [4, 4]},
                },
                {
                    "id": "flat",
                    "op": "flatten",
                    "from": ["adaptive"],
                    "params": {"start_dim": 1},
                },
                {
                    "id": "image_hidden",
                    "op": "linear",
                    "from": ["flat"],
                    "params": {"in_features": 256, "out_features": 64},
                },
                {
                    "id": "image_features",
                    "op": "relu",
                    "from": ["image_hidden"],
                    "params": {},
                },
                {
                    "id": "action_hidden",
                    "op": "linear",
                    "from": ["action"],
                    "params": {"in_features": 4, "out_features": 16},
                },
                {
                    "id": "action_features",
                    "op": "relu",
                    "from": ["action_hidden"],
                    "params": {},
                },
                {
                    "id": "joint",
                    "op": "concat",
                    "from": ["image_features", "action_features"],
                    "params": {"dim": 1},
                },
                {
                    "id": "joint_hidden",
                    "op": "linear",
                    "from": ["joint"],
                    "params": {"in_features": 80, "out_features": 32},
                },
                {
                    "id": "joint_features",
                    "op": "relu",
                    "from": ["joint_hidden"],
                    "params": {},
                },
                {
                    "id": "raw_q",
                    "op": "linear",
                    "from": ["joint_features"],
                    "params": {"in_features": 32, "out_features": 1},
                },
                {
                    "id": "bounded_q",
                    "op": "tanh",
                    "from": ["raw_q"],
                    "params": {},
                },
            ],
            "outputs": {"q_value": "bounded_q"},
            "training": {
                "batch_size": 16,
                "optimizer": {"type": "adam", "lr": 0.001},
                "losses": [
                    {
                        "type": "mse",
                        "output": "q_value",
                        "target": "reward",
                    }
                ],
            },
            "runtime": {
                "role": "qnn",
                "minimum_action_score": 0.0,
            },
        },
    }


def _validate_structure(structure: dict[str, Any]) -> None:
    if not isinstance(structure, dict):
        raise TypeError("structure must be a mapping")
    inputs = structure.get("input_schema")
    outputs = structure.get("output_schema")
    targets = structure.get("target_schema", outputs)
    nodes = structure.get("nodes")
    output_refs = structure.get("outputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("structure.input_schema must be a non-empty mapping")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("structure.output_schema must be a non-empty mapping")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("structure.target_schema must be a non-empty mapping")
    if not isinstance(nodes, list):
        raise ValueError("structure.nodes must be a list")
    if not isinstance(output_refs, dict):
        raise ValueError("structure.outputs must be a mapping")

    available = set(inputs)
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"node {index} must be a mapping")
        node_id = node.get("id")
        op = node.get("op")
        sources = node.get("from")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node {index} has an invalid id")
        if node_id in available or node_id in node_ids:
            raise ValueError(f"duplicate node id: {node_id}")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"unknown op: {op}")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"node {node_id} must have at least one source")
        unknown_sources = [source for source in sources if source not in available]
        if unknown_sources:
            raise ValueError(
                f"node {node_id} references unknown or future source(s): "
                f"{unknown_sources}"
            )
        if not isinstance(node.get("params", {}), dict):
            raise TypeError(f"node {node_id}.params must be a mapping")
        node_ids.add(node_id)
        available.add(node_id)

    if set(output_refs) != set(outputs):
        raise ValueError("structure.outputs keys must match output_schema keys")
    for output_name, node_id in output_refs.items():
        if node_id not in available:
            raise ValueError(f"output {output_name} references unknown value: {node_id}")

    training = structure.get("training")
    if not isinstance(training, dict):
        raise ValueError("structure.training must be a mapping")
    if not isinstance(training.get("optimizer"), dict):
        raise ValueError("structure.training.optimizer must be a mapping")
    losses = training.get("losses")
    if not isinstance(losses, list) or not losses:
        raise ValueError("structure.training.losses must be a non-empty list")
    supported_losses = {
        "mse", "l1", "smooth_l1", "cross_entropy",
        "binary_cross_entropy", "bce_with_logits", "kl_div",
    }
    for loss in losses:
        if loss.get("type") not in supported_losses:
            raise ValueError(f"unknown loss: {loss.get('type')}")
        if loss.get("output") not in outputs:
            raise ValueError(f"loss references unknown output: {loss.get('output')}")
        if loss.get("target") not in targets:
            raise ValueError(f"loss references unknown target: {loss.get('target')}")


def _generate_model_source(structure: dict[str, Any]) -> str:
    """Compile a validated structure into a portable, concrete ``model.py``."""
    _validate_structure(structure)
    literal = repr(copy.deepcopy(structure))
    return textwrap.dedent(
        f'''
        """Generated by EVE Trainer. The model and training semantics are portable."""
        from __future__ import annotations

        from typing import Any
        import torch
        from torch import nn
        from torch.nn import functional as F
        from eve.dock.tinynn import TinyNN

        STRUCTURE = {literal}


        class GeneratedTNN(TinyNN):
            def __init__(self):
                super().__init__(
                    tnn_id=str(STRUCTURE.get("_tnn_id", "generated_tnn")),
                    version=str(STRUCTURE.get("_version", "1")),
                    input_schema=STRUCTURE["input_schema"],
                    output_schema=STRUCTURE["output_schema"],
                )
                self.layers = nn.ModuleDict()
                self._layer_keys = {{}}
                for index, node in enumerate(STRUCTURE["nodes"]):
                    layer = self._make_layer(node["op"], node.get("params", {{}}))
                    if layer is not None:
                        key = f"layer_{{index}}"
                        self.layers[key] = layer
                        self._layer_keys[node["id"]] = key
                self.optimizer = self._make_optimizer(STRUCTURE["training"]["optimizer"])

            @staticmethod
            def _make_layer(op: str, params: dict[str, Any]):
                classes = {{
                    "linear": nn.Linear,
                    "conv1d": nn.Conv1d, "conv2d": nn.Conv2d, "conv3d": nn.Conv3d,
                    "conv_transpose1d": nn.ConvTranspose1d,
                    "conv_transpose2d": nn.ConvTranspose2d,
                    "conv_transpose3d": nn.ConvTranspose3d,
                    "maxpool1d": nn.MaxPool1d, "maxpool2d": nn.MaxPool2d,
                    "maxpool3d": nn.MaxPool3d,
                    "avgpool1d": nn.AvgPool1d, "avgpool2d": nn.AvgPool2d,
                    "avgpool3d": nn.AvgPool3d,
                    "adaptive_avgpool1d": nn.AdaptiveAvgPool1d,
                    "adaptive_avgpool2d": nn.AdaptiveAvgPool2d,
                    "adaptive_avgpool3d": nn.AdaptiveAvgPool3d,
                    "batchnorm1d": nn.BatchNorm1d, "batchnorm2d": nn.BatchNorm2d,
                    "batchnorm3d": nn.BatchNorm3d,
                    "layernorm": nn.LayerNorm, "groupnorm": nn.GroupNorm,
                    "dropout": nn.Dropout, "embedding": nn.Embedding,
                    "rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM,
                    "multihead_attention": nn.MultiheadAttention,
                }}
                cls = classes.get(op)
                return cls(**params) if cls is not None else None

            def _make_optimizer(self, definition: dict[str, Any]):
                params = dict(definition)
                kind = str(params.pop("type")).lower()
                classes = {{
                    "adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                    "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop,
                    "adagrad": torch.optim.Adagrad,
                }}
                if kind not in classes:
                    raise ValueError(f"unknown optimizer: {{kind}}")
                return classes[kind](self.parameters(), **params)

            def _apply_node(self, node, values):
                op = node["op"]
                params = node.get("params", {{}})
                args = [values[name] for name in node["from"]]
                if node["id"] in self._layer_keys:
                    layer = self.layers[self._layer_keys[node["id"]]]
                    if op == "multihead_attention":
                        if len(args) == 1:
                            args = [args[0], args[0], args[0]]
                        if len(args) != 3:
                            raise ValueError("multihead_attention expects one or three inputs")
                        return layer(*args, need_weights=False)[0]
                    if len(args) != 1:
                        raise ValueError(f"{{op}} expects exactly one input")
                    value = args[0].long() if op == "embedding" else args[0]
                    result = layer(value)
                    return result[0] if op in {{"rnn", "gru", "lstm"}} else result
                x = args[0]
                if op == "relu": return F.relu(x, **params)
                if op == "leaky_relu": return F.leaky_relu(x, **params)
                if op == "gelu": return F.gelu(x, **params)
                if op == "silu": return F.silu(x, **params)
                if op == "mish": return F.mish(x, **params)
                if op == "tanh": return torch.tanh(x)
                if op == "sigmoid": return torch.sigmoid(x)
                if op == "softmax": return torch.softmax(x, dim=params.get("dim", -1))
                if op == "flatten": return torch.flatten(x, **params)
                if op == "reshape": return x.reshape(*params["shape"])
                if op == "permute": return x.permute(*params["dims"])
                if op == "transpose":
                    return x.transpose(params.get("dim0", 0), params.get("dim1", 1))
                if op == "squeeze":
                    return x.squeeze() if "dim" not in params else x.squeeze(params["dim"])
                if op == "unsqueeze": return x.unsqueeze(params["dim"])
                if op == "concat": return torch.cat(args, dim=params.get("dim", 0))
                if op == "stack": return torch.stack(args, dim=params.get("dim", 0))
                if op in {{"add", "mul"}}:
                    result = args[0]
                    for value in args[1:]:
                        result = result + value if op == "add" else result * value
                    return result
                if op in {{"mean", "sum"}}:
                    dim = params.get("dim")
                    keepdim = params.get("keepdim", False)
                    fn = torch.mean if op == "mean" else torch.sum
                    return fn(x) if dim is None else fn(x, dim=dim, keepdim=keepdim)
                raise ValueError(f"unknown op: {{op}}")

            def forward(self, inputs):
                missing = set(STRUCTURE["input_schema"]) - set(inputs)
                if missing:
                    raise KeyError(f"missing model input(s): {{sorted(missing)}}")
                values = dict(inputs)
                for node in STRUCTURE["nodes"]:
                    values[node["id"]] = self._apply_node(node, values)
                return {{name: values[source] for name, source in STRUCTURE["outputs"].items()}}

            @staticmethod
            def _loss(kind, prediction, target, params):
                if kind == "mse": return F.mse_loss(prediction, target, **params)
                if kind == "l1": return F.l1_loss(prediction, target, **params)
                if kind == "smooth_l1": return F.smooth_l1_loss(prediction, target, **params)
                if kind == "cross_entropy":
                    return F.cross_entropy(prediction, target.long(), **params)
                if kind == "binary_cross_entropy":
                    return F.binary_cross_entropy(prediction, target, **params)
                if kind == "bce_with_logits":
                    return F.binary_cross_entropy_with_logits(prediction, target, **params)
                if kind == "kl_div": return F.kl_div(prediction, target, **params)
                raise ValueError(f"unknown loss: {{kind}}")

            def _calculate_loss(self, batch):
                outputs = self.forward(batch["inputs"])
                total = None
                metrics = {{}}
                for definition in STRUCTURE["training"]["losses"]:
                    target_name = definition["target"]
                    if target_name not in batch["targets"]:
                        raise KeyError(f"missing training target: {{target_name}}")
                    value = self._loss(
                        definition["type"], outputs[definition["output"]],
                        batch["targets"][target_name], dict(definition.get("params", {{}})),
                    )
                    weighted = float(definition.get("weight", 1.0)) * value
                    total = weighted if total is None else total + weighted
                    metrics[f"loss_{{definition['output']}}"] = float(value.detach().cpu())
                if total is None:
                    raise ValueError("at least one loss is required")
                return total, metrics

            def training_step(self, batch):
                self.train()
                self.optimizer.zero_grad()
                loss, metrics = self._calculate_loss(batch)
                loss.backward()
                self.optimizer.step()
                metrics["loss"] = float(loss.detach().cpu())
                return metrics

            def evaluation_step(self, batch):
                self.eval()
                with torch.no_grad():
                    loss, metrics = self._calculate_loss(batch)
                metrics["loss"] = float(loss.detach().cpu())
                return metrics


        def create_tnn():
            return GeneratedTNN()
        '''
    ).lstrip()


def _write_generated_model(workspace: str | Path, source: str) -> str:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    model_path = workspace_path / "model.py"
    model_path.write_text(source, encoding="utf-8")
    return str(model_path)


def _import_tnn_model(model_path: str | Path, factory: str = "create_tnn") -> TinyNN:
    path = Path(model_path).resolve()
    module_name = f"_eve_tnn_{uuid.uuid4().hex}"
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
        tnn = creator()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(tnn, TinyNN):
        raise TypeError(f"{factory}() must return TinyNN, got {type(tnn).__name__}")
    return tnn


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return value


def _validate_generated_shapes(tnn: TinyNN, structure: dict[str, Any]) -> None:
    dummy: dict[str, torch.Tensor] = {}
    for name, schema in structure["input_schema"].items():
        dtype = _dtype(schema.get("dtype", "float32"))
        shape = (2, *schema.get("shape", []))
        dummy[name] = (
            torch.randn(shape, dtype=dtype)
            if dtype.is_floating_point
            else torch.zeros(shape, dtype=dtype)
        )
    tnn.eval()
    try:
        with torch.no_grad():
            outputs = tnn(dummy)
    except Exception as exc:
        raise ValueError(f"incompatible model shapes: {exc}") from exc
    for name, schema in structure["output_schema"].items():
        if name not in outputs:
            raise ValueError(f"model did not produce declared output: {name}")
        expected = tuple(schema.get("shape", []))
        actual = tuple(outputs[name].shape[1:])
        if actual != expected:
            raise ValueError(
                f"output {name} shape mismatch: expected (*, {expected}), got {actual}"
            )


def _to_tensor(value: Any, schema: dict[str, Any] | None = None) -> torch.Tensor:
    dtype = _dtype((schema or {}).get("dtype", "float32"))
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _collate(
    samples: list[dict[str, Any]],
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    inputs = {
        name: torch.stack([_to_tensor(sample["inputs"][name], schema) for sample in samples])
        for name, schema in input_schema.items()
    }
    targets = {
        name: torch.stack([_to_tensor(sample["targets"][name], schema) for sample in samples])
        for name, schema in output_schema.items()
    }
    return {"inputs": inputs, "targets": targets}


class Trainer:
    """Owns training instances and emits immutable, uniform TNN artifacts."""

    def __init__(
        self,
        memorizer: Any | None = None,
        legacy_memorizer: Any | None = None,
        *,
        workspace_root: str | Path = "dock/workspace",
        training_device: str | torch.device | None = None,
        **legacy_options: Any,
    ) -> None:
        # The former storage keyword is accepted but deliberately not retained.
        legacy_options.pop("tnn_store", None)
        if legacy_options:
            raise TypeError(
                f"unexpected Trainer option(s): {sorted(legacy_options)}"
            )
        self._memorizer = legacy_memorizer if legacy_memorizer is not None else memorizer
        if self._memorizer is None:
            raise TypeError("memorizer is required")
        self.workspace_root = Path(workspace_root)
        self.training_device = torch.device(
            training_device
            if training_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._queue: list[TrainingOrder] = []
        self._lock = threading.Lock()
        self._current_order: TrainingOrder | None = None
        self._running = False
        self._results: list[TrainingResult] = []

    def enqueue(self, order: TrainingOrder) -> None:
        with self._lock:
            self._queue.append(order)
            self._queue.sort(
                key=lambda item: PRIORITY_ORDER.get(item.priority, 99)
            )

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def get_queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            queue_size = len(self._queue)
        return {
            "queue_size": queue_size,
            "is_training": self._running,
            "current_order": self._current_order.order_id if self._current_order else None,
            "total_completed": len(self._results),
            "success_count": sum(result.success for result in self._results),
            "fail_count": sum(not result.success for result in self._results),
        }

    def process_one(
        self,
        model_adapters: dict[str, Any] | None = None,
    ) -> TrainingResult:
        with self._lock:
            if not self._queue:
                return TrainingResult(order_id="", error="No orders in queue")
            order = self._queue.pop(0)
        return self.process_order(order, model_adapters=model_adapters)

    def process_order(
        self,
        order: TrainingOrder,
        *,
        model_adapters: dict[str, Any] | None = None,
    ) -> TrainingResult:
        started = time.perf_counter()
        self._current_order = order
        self._running = True
        try:
            result = self._execute(order, started, model_adapters)
        except Exception as exc:
            result = TrainingResult(
                order_id=order.order_id,
                tnn_id=order.target_tnn_id,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            self._running = False
            self._current_order = None
        self._results.append(result)
        return result

    def _execute(
        self,
        order: TrainingOrder,
        started: float,
        model_adapters: dict[str, Any] | None,
    ) -> TrainingResult:
        if not order.order_id or not order.target_tnn_id:
            raise ValueError("order_id and target_tnn_id are required")
        if (
            order.minimum_qnn_fitness is not None
            and not -1.0 <= float(order.minimum_qnn_fitness) <= 1.0
        ):
            raise ValueError("minimum_qnn_fitness must be between -1 and 1")
        if not 0.0 <= float(order.minimum_qnn_margin) <= 2.0:
            raise ValueError("minimum_qnn_margin must be between 0 and 2")
        definition = order.definition
        mode = definition.get("mode")
        if mode not in {"json", "python"}:
            raise ValueError("definition.mode must be 'json' or 'python'")
        version = str(definition.get("version") or f"v{time.time_ns()}")
        artifact = self.workspace_root / order.order_id / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)

        if mode == "json":
            structure = copy.deepcopy(definition.get("structure"))
            if not isinstance(structure, dict):
                raise ValueError("JSON mode requires definition.structure")
            structure["_tnn_id"] = order.target_tnn_id
            structure["_version"] = version
            model_path = Path(
                _write_generated_model(artifact, _generate_model_source(structure))
            )
            tnn = _import_tnn_model(model_path)
            _validate_generated_shapes(tnn, structure)
        else:
            structure, model_path, tnn = self._prepare_python_model(
                definition, artifact, order, version
            )
        tnn.to(self.training_device)

        samples = self._load_samples(order.training_data, structure)
        if len(samples) < order.minimum_samples:
            raise ValueError(
                f"not enough valid training samples: "
                f"{len(samples)} < {order.minimum_samples}"
            )
        if not 0.0 < order.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        evaluation_samples = self._load_samples(order.evaluation_data, structure)
        if not evaluation_samples and len(samples) > 1:
            shuffled = list(samples)
            random.Random(order.order_id).shuffle(shuffled)
            split_at = max(
                1,
                min(len(shuffled) - 1, int(len(shuffled) * order.train_ratio)),
            )
            samples = shuffled[:split_at]
            evaluation_samples = shuffled[split_at:]
        training = structure.get("training", {})
        batch_size = int(training.get("batch_size", len(samples)))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batches = list(self._batches(samples, batch_size, tnn, structure))
        evaluation_batches = list(
            self._batches(
                evaluation_samples,
                batch_size,
                tnn,
                structure,
            )
        )
        regression_samples = self._load_samples(
            order.regression_data, structure
        )
        regression_batches = list(
            self._batches(
                regression_samples,
                batch_size,
                tnn,
                structure,
            )
        )
        epochs = order.epochs if order.epochs is not None else 1
        if epochs < 0:
            raise ValueError("epochs cannot be negative")

        train_metrics: list[dict[str, Any]] = []
        for _ in range(epochs):
            for batch in batches:
                metrics = tnn.training_step(batch)
                if not isinstance(metrics, dict):
                    raise TypeError("training_step() must return a metrics mapping")
                train_metrics.append(metrics)

        eval_metrics: list[dict[str, Any]] = []
        for batch in evaluation_batches:
            try:
                metrics = tnn.evaluation_step(batch)
            except NotImplementedError:
                break
            if not isinstance(metrics, dict):
                raise TypeError("evaluation_step() must return a metrics mapping")
            eval_metrics.append(metrics)
        regression_metrics: list[dict[str, Any]] = []
        for batch in regression_batches:
            try:
                metrics = tnn.evaluation_step(batch)
            except NotImplementedError:
                break
            if not isinstance(metrics, dict):
                raise TypeError("evaluation_step() must return a metrics mapping")
            regression_metrics.append(metrics)

        tnn.save_weights(str(artifact / "weights.pt"))
        averaged_train = self._average_metrics(train_metrics)
        averaged_eval = self._average_metrics(eval_metrics)
        averaged_regression = self._average_metrics(regression_metrics)
        accepted, rejection_reason = self._accept_candidate(
            order.acceptance,
            evaluation=averaged_eval,
            regression=averaged_regression,
        )
        qnn_fitness: dict[str, Any] = {}
        requires_qnn = (
            order.minimum_qnn_fitness is not None
            or order.minimum_qnn_margin > 0.0
        )
        if accepted and requires_qnn:
            evaluator = (model_adapters or {}).get("fitness_evaluator")
            if not callable(evaluator):
                accepted = False
                rejection_reason = "QNN fitness evaluator is unavailable"
            elif not order.fitness_data:
                accepted = False
                rejection_reason = "QNN fitness requires fitness_data"
            else:
                evaluation = evaluator(order, tnn, structure)
                if not isinstance(evaluation, dict):
                    raise TypeError(
                        "QNN fitness evaluator must return a mapping"
                    )
                qnn_fitness = dict(evaluation)
                candidate_fitness = qnn_fitness.get("candidate")
                baseline_fitness = qnn_fitness.get("baseline")
                if candidate_fitness is None:
                    accepted = False
                    rejection_reason = "QNN candidate fitness is unavailable"
                elif (
                    order.minimum_qnn_fitness is not None
                    and float(candidate_fitness)
                    < float(order.minimum_qnn_fitness)
                ):
                    accepted = False
                    rejection_reason = (
                        "minimum_qnn_fitness not reached: "
                        f"{candidate_fitness} < {order.minimum_qnn_fitness}"
                    )
                elif (
                    baseline_fitness is not None
                    and float(candidate_fitness) - float(baseline_fitness)
                    < float(order.minimum_qnn_margin)
                ):
                    accepted = False
                    rejection_reason = (
                        "minimum_qnn_margin not reached: "
                        f"{float(candidate_fitness) - float(baseline_fitness)} "
                        f"< {order.minimum_qnn_margin}"
                    )
        self._write_metadata(
            artifact,
            order,
            structure,
            version,
            train_metrics,
            eval_metrics,
            regression_metrics,
            len(samples) + len(evaluation_samples) + len(regression_samples),
            str(self.training_device),
            accepted,
            rejection_reason,
            qnn_fitness,
        )
        memory_id = self._memorizer.store_tnn_artifact(
            source_directory=str(artifact),
            tnn_id=order.target_tnn_id,
            version=version,
        )
        return TrainingResult(
            order_id=order.order_id,
            tnn_id=order.target_tnn_id,
            version=version,
            success=True,
            metrics={
                **averaged_train,
                "evaluation": averaged_eval,
                "regression": averaged_regression,
                "qnn_fitness": qnn_fitness,
            },
            report_memory_id=memory_id,
            weights_path=str(artifact / "weights.pt"),
            artifact_path=str(artifact),
            latency_ms=(time.perf_counter() - started) * 1000,
            sample_count=(
                len(samples) + len(evaluation_samples) + len(regression_samples)
            ),
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def _prepare_python_model(
        self,
        definition: dict[str, Any],
        artifact: Path,
        order: TrainingOrder,
        version: str,
    ) -> tuple[dict[str, Any], Path, TinyNN]:
        model_memory_id = definition.get("model_memory_id")
        if not model_memory_id:
            raise ValueError("Python mode requires model_memory_id")
        resolved = self._memorizer.resolve_tnn_artifact(str(model_memory_id))
        model_path = artifact / "model.py"
        shutil.copy2(resolved["model_path"], model_path)
        tnn = _import_tnn_model(model_path, str(definition.get("factory", "create_tnn")))
        if order.continue_training:
            tnn.load_weights(resolved["weights_path"], map_location="cpu")
        structure = json.loads(Path(resolved["structure_path"]).read_text(encoding="utf-8"))
        structure.setdefault("input_schema", tnn.get_input_schema())
        structure.setdefault("output_schema", tnn.get_output_schema())
        structure.setdefault("training", {})
        structure["_tnn_id"] = order.target_tnn_id
        structure["_version"] = version
        return structure, model_path, tnn

    def _load_samples(
        self, memory_ids: list[str], structure: dict[str, Any]
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for memory_id in memory_ids:
            payload = self._memorizer.read(memory_id)
            candidates = payload if isinstance(payload, list) else [payload]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("experience_version") == 1:
                    converted = self._experience_sample(candidate, structure)
                    if converted is not None:
                        samples.append(converted)
                    continue
                if "inputs" in candidate and "targets" in candidate:
                    samples.append(candidate)
                    continue
                inputs = {
                    name: candidate[name]
                    for name in structure["input_schema"]
                    if name in candidate
                }
                target_schema = structure.get(
                    "target_schema", structure["output_schema"]
                )
                targets = {
                    name: candidate[name]
                    for name in target_schema
                    if name in candidate
                }
                if (
                    len(inputs) == len(structure["input_schema"])
                    and len(targets) == len(target_schema)
                ):
                    samples.append({"inputs": inputs, "targets": targets})
        return samples

    def _experience_sample(
        self,
        experience: dict[str, Any],
        structure: dict[str, Any],
    ) -> dict[str, Any] | None:
        if structure.get("runtime", {}).get("role") == "qnn":
            return self._qnn_experience_sample(experience, structure)
        state = experience.get("state", {})
        teacher = experience.get("teacher", {})
        objects = teacher.get("objects", [])
        if not isinstance(state, dict) or not isinstance(objects, list):
            return None
        target_classes = {
            str(item)
            for item in experience.get("task", {}).get("target_classes", [])
        }
        target = next(
            (
                item
                for item in objects
                if isinstance(item, dict)
                and (
                    not target_classes
                    or str(item.get("class")) in target_classes
                )
            ),
            None,
        )
        if target is None:
            return None
        screen_memory_id = state.get("screen_memory_id")
        if not screen_memory_id:
            return None
        screen = self._memorizer.read(str(screen_memory_id))
        input_schema = structure["input_schema"]
        target_schema = structure.get(
            "target_schema", structure["output_schema"]
        )
        inputs: dict[str, Any] = {}
        for name, schema in input_schema.items():
            if name in {"screen", "image"}:
                inputs[name] = self._prepare_experience_input(screen, schema)
            elif name == "cursor":
                cursor = state.get("cursor", {})
                inputs[name] = [cursor.get("x", 0), cursor.get("y", 0)]
            elif name in state:
                inputs[name] = state[name]
        targets: dict[str, Any] = {}
        for name in target_schema:
            if name in {"target_center", "mouse_position"}:
                center = target.get("center")
                shape = getattr(screen, "shape", ())
                if (
                    name == "target_center"
                    and isinstance(center, (list, tuple))
                    and len(shape) >= 2
                    and shape[0]
                    and shape[1]
                ):
                    targets[name] = [
                        float(center[0]) / float(shape[1]),
                        float(center[1]) / float(shape[0]),
                    ]
                else:
                    targets[name] = center
            elif name == "target_present":
                targets[name] = [1.0]
            elif name == "target_class":
                targets[name] = target.get("class_index", 0)
            elif name in target:
                targets[name] = target[name]
        if len(inputs) != len(input_schema) or len(targets) != len(target_schema):
            return None
        return {"inputs": inputs, "targets": targets}

    def _qnn_experience_sample(
        self,
        experience: dict[str, Any],
        structure: dict[str, Any],
    ) -> dict[str, Any] | None:
        from eve.core.qnn import action_vector

        state = experience.get("state", {})
        action = experience.get("action", {})
        environment = experience.get("environment", {})
        screen_memory_id = (
            state.get("screen_memory_id") if isinstance(state, dict) else None
        )
        if (
            not screen_memory_id
            or not isinstance(action, dict)
            or not isinstance(environment, dict)
        ):
            return None
        screen = self._memorizer.read(str(screen_memory_id))
        shape = getattr(screen, "shape", ())
        if len(shape) < 2 or not shape[0] or not shape[1]:
            return None
        try:
            encoded_action = action_vector(
                action,
                width=int(shape[1]),
                height=int(shape[0]),
            )
        except (TypeError, ValueError):
            return None
        reward = environment.get("reward")
        if reward is None:
            reward = 1.0 if environment.get("hit") else -1.0
        reward = min(1.0, max(-1.0, float(reward)))
        return {
            "inputs": {
                "image": self._prepare_experience_input(
                    screen, structure["input_schema"]["image"]
                ),
                "action": encoded_action,
            },
            "targets": {"reward": [reward]},
        }

    @staticmethod
    def _prepare_experience_input(value: Any, schema: dict[str, Any]) -> Any:
        if schema.get("preprocess") != "screen_rgb_64":
            return value
        tensor = torch.as_tensor(value)
        if tensor.ndim != 3 or tensor.shape[-1] < 3:
            raise ValueError("screen_rgb_64 requires an HWC screen image")
        tensor = tensor[..., :3].flip(-1).permute(2, 0, 1).float() / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).tolist()

    @staticmethod
    def _batches(
        samples: list[dict[str, Any]],
        batch_size: int,
        tnn: TinyNN,
        structure: dict[str, Any],
    ) -> Iterable[dict[str, dict[str, torch.Tensor]]]:
        device = next(
            tnn.parameters(),
            next(tnn.buffers(), torch.empty(0)),
        ).device
        for start in range(0, len(samples), batch_size):
            batch = _collate(
                samples[start : start + batch_size],
                tnn.get_input_schema(),
                structure.get("target_schema", tnn.get_output_schema()),
            )
            yield {
                group: {
                    name: value.to(device)
                    for name, value in values.items()
                }
                for group, values in batch.items()
            }

    @staticmethod
    def _write_metadata(
        artifact: Path,
        order: TrainingOrder,
        structure: dict[str, Any],
        version: str,
        train_metrics: list[dict[str, Any]],
        eval_metrics: list[dict[str, Any]],
        regression_metrics: list[dict[str, Any]],
        sample_count: int,
        training_device: str,
        accepted: bool,
        rejection_reason: str | None,
        qnn_fitness: dict[str, Any],
    ) -> None:
        documents = {
            "structure.json": structure,
            "description.json": {
                "tnn_id": order.target_tnn_id,
                "version": version,
                "purpose": order.purpose,
                "task_id": order.task_id,
                "mode": order.definition["mode"],
            },
            "training.json": {
                "order": asdict(order),
                "sample_count": sample_count,
                "train_memory_ids": order.training_data,
                "test_memory_ids": order.evaluation_data,
                "regression_memory_ids": order.regression_data,
                "training_device": training_device,
                "train_metrics": train_metrics,
                "evaluation_metrics": eval_metrics,
                "regression_metrics": regression_metrics,
                "qnn_fitness": qnn_fitness,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            },
        }
        for name, document in documents.items():
            (artifact / name).write_text(
                json.dumps(document, ensure_ascii=False, indent=2, default=repr),
                encoding="utf-8",
            )
        missing = ARTIFACT_FILES - {path.name for path in artifact.iterdir()}
        if missing:
            raise RuntimeError(f"incomplete TNN artifact: {sorted(missing)}")

    @staticmethod
    def _average_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in {key for item in items for key in item}:
            values = [
                float(item[key])
                for item in items
                if key in item
                and isinstance(item[key], (int, float))
                and math.isfinite(float(item[key]))
            ]
            if values:
                result[key] = sum(values) / len(values)
        return result

    @staticmethod
    def _accept_candidate(
        acceptance: dict[str, float],
        *,
        evaluation: dict[str, Any],
        regression: dict[str, Any],
    ) -> tuple[bool, str | None]:
        checks = (
            ("max_evaluation_loss", evaluation.get("loss")),
            ("max_regression_loss", regression.get("loss")),
        )
        for name, actual in checks:
            if name not in acceptance:
                continue
            if actual is None:
                return False, f"{name} requested but metric is unavailable"
            if float(actual) > float(acceptance[name]):
                return (
                    False,
                    f"{name} exceeded: {actual} > {acceptance[name]}",
                )
        return True, None
