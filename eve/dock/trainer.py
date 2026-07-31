"""Dock training for concrete, model-authored TinyNN implementations."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from eve.dock.tinynn import TinyNN


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ARTIFACT_FILES = {
    "model.py", "weights.pt", "structure.json", "description.json",
    "training.json",
}


@dataclass
class TrainingOrder:
    """Executable order created only after a proposal has been reviewed."""

    order_id: str
    priority: str = "medium"
    target_tnn_id: str = ""
    model_path: str = ""
    model_memory_id: str = ""
    factory: str = "create_tnn"
    weights_path: str = ""
    version: str = ""
    training_data: list[str] = field(default_factory=list)
    evaluation_data: list[str] = field(default_factory=list)
    regression_data: list[str] = field(default_factory=list)
    experience_query: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    purpose: str = ""
    minimum_samples: int = 1
    batch_size: int = 8
    epochs: int = 1
    acceptance: dict[str, float] = field(default_factory=dict)
    continue_training: bool = False
    runtime: dict[str, Any] = field(default_factory=dict)


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
    accepted: bool = False
    rejection_reason: str | None = None


def _import_tnn_model(model_path: str | Path, factory: str) -> TinyNN:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.name != "model.py":
        raise ValueError("Dock accepts one concrete file named model.py")
    module_name = f"_eve_tnn_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TNN model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        creator = getattr(module, factory, None)
        if not callable(creator):
            raise AttributeError(f"{path} does not define callable {factory}()")
        model = creator()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(model, TinyNN):
        raise TypeError(f"{factory}() must return TinyNN")
    return model


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return value


class Trainer:
    """Queue and execute bounded training against an explicit ``model.py``."""

    def __init__(
        self,
        memorizer: Any | None = None,
        legacy_memorizer: Any | None = None,
        *,
        workspace_root: str | Path = "dock/workspace",
        training_device: str | torch.device | None = None,
        **legacy_options: Any,
    ) -> None:
        legacy_options.pop("tnn_store", None)
        if legacy_options:
            raise TypeError(f"unexpected Trainer options: {sorted(legacy_options)}")
        self._memorizer = legacy_memorizer or memorizer
        if self._memorizer is None:
            raise TypeError("memorizer is required")
        self.workspace_root = Path(workspace_root)
        self.training_device = torch.device(training_device or "cpu")
        self._queue: list[TrainingOrder] = []
        self._lock = threading.Lock()
        self._current_order: TrainingOrder | None = None
        self._completed = 0
        self._failed = 0

    def enqueue(self, order: TrainingOrder) -> None:
        self._validate_order(order)
        with self._lock:
            self._queue.append(order)
            self._queue.sort(key=lambda item: PRIORITY_ORDER[item.priority])

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def get_queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued": len(self._queue),
                "current_order": (
                    self._current_order.order_id if self._current_order else None
                ),
                "completed": self._completed,
                "failed": self._failed,
            }

    def process_one(self, *, batch_size: int | None = None) -> TrainingResult:
        with self._lock:
            if not self._queue:
                return TrainingResult(order_id="", error="queue is empty")
            order = self._queue.pop(0)
            self._current_order = order
        try:
            result = self.process_order(order, batch_size=batch_size)
            with self._lock:
                self._completed += int(result.success)
                self._failed += int(not result.success)
            return result
        finally:
            with self._lock:
                self._current_order = None

    def process_order(
        self, order: TrainingOrder, *, batch_size: int | None = None
    ) -> TrainingResult:
        started = time.perf_counter()
        try:
            return self._execute(order, batch_size or order.batch_size)
        except Exception as exc:
            return TrainingResult(
                order_id=order.order_id,
                tnn_id=order.target_tnn_id,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    @staticmethod
    def _validate_order(order: TrainingOrder) -> None:
        if order.priority not in PRIORITY_ORDER:
            raise ValueError("priority must be high, medium, or low")
        if not order.order_id or not order.target_tnn_id:
            raise ValueError("order_id and target_tnn_id are required")
        if bool(order.model_path) == bool(order.model_memory_id):
            raise ValueError("provide exactly one of model_path or model_memory_id")
        if not order.acceptance:
            raise ValueError("an executable order requires acceptance criteria")
        if order.minimum_samples < 1 or order.batch_size < 1 or order.epochs < 0:
            raise ValueError("invalid training bounds")

    def _execute(self, order: TrainingOrder, batch_size: int) -> TrainingResult:
        self._validate_order(order)
        started = time.perf_counter()
        resolved = None
        if order.model_memory_id:
            resolved = self._memorizer.resolve_tnn_artifact(order.model_memory_id)
            source_model = Path(resolved["model_path"])
        else:
            source_model = Path(order.model_path).expanduser().resolve()
        version = order.version or f"v{time.time_ns()}"
        artifact = self.workspace_root / order.order_id
        if artifact.exists():
            raise FileExistsError(f"training workspace exists: {artifact}")
        artifact.mkdir(parents=True)
        model_path = artifact / "model.py"
        shutil.copy2(source_model, model_path)
        model = _import_tnn_model(model_path, order.factory).to(self.training_device)
        if order.continue_training:
            weights = resolved["weights_path"] if resolved else order.weights_path
            if not weights:
                raise ValueError("continue_training requires concrete weights")
            model.load_weights(str(weights), map_location=self.training_device)
        structure = {
            "input_schema": model.get_input_schema(),
            "output_schema": model.get_output_schema(),
            "runtime": dict(order.runtime),
        }
        training = self._load_samples(order.training_data)
        evaluation = self._load_samples(order.evaluation_data)
        regression = self._load_samples(order.regression_data)
        if len(training) < order.minimum_samples:
            raise ValueError(
                f"insufficient training data: {len(training)} < {order.minimum_samples}"
            )
        train_metrics = self._run(
            model, training, structure, batch_size, order.epochs, False
        )
        eval_metrics = self._run(model, evaluation, structure, batch_size, 1, True)
        regression_metrics = self._run(
            model, regression, structure, batch_size, 1, True
        )
        averaged_train = self._average(train_metrics)
        averaged_eval = self._average(eval_metrics)
        averaged_regression = self._average(regression_metrics)
        accepted, reason = self._accept(
            order.acceptance, averaged_eval, averaged_regression
        )
        model.save_weights(str(artifact / "weights.pt"))
        documents = {
            "structure.json": structure,
            "description.json": {
                "tnn_id": order.target_tnn_id,
                "version": version,
                "purpose": order.purpose,
                "task_id": order.task_id,
                "model_source": "model.py",
            },
            "training.json": {
                "order": asdict(order),
                "training_metrics": train_metrics,
                "evaluation_metrics": eval_metrics,
                "regression_metrics": regression_metrics,
                "accepted": accepted,
                "rejection_reason": reason,
            },
        }
        for name, document in documents.items():
            (artifact / name).write_text(
                json.dumps(document, ensure_ascii=False, indent=2, default=repr),
                encoding="utf-8",
            )
        missing = ARTIFACT_FILES - {item.name for item in artifact.iterdir()}
        if missing:
            raise RuntimeError(f"incomplete TNN artifact: {sorted(missing)}")
        report_id = (
            self._memorizer.store_tnn_artifact(
                source_directory=artifact,
                tnn_id=order.target_tnn_id,
                version=version,
            )
            if accepted
            else self._memorizer.create(
                {"order": order.order_id, "accepted": False, "reason": reason},
                "training_report",
            )
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
            },
            report_memory_id=report_id,
            weights_path=str(artifact / "weights.pt"),
            artifact_path=str(artifact),
            latency_ms=(time.perf_counter() - started) * 1000,
            sample_count=len(training) + len(evaluation) + len(regression),
            accepted=accepted,
            rejection_reason=reason,
        )

    def _load_samples(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        samples = []
        for memory_id in memory_ids:
            payload = self._memorizer.read(memory_id)
            for item in payload if isinstance(payload, list) else [payload]:
                if not isinstance(item, dict):
                    continue
                if item.get("experience_version") == 1:
                    item = item.get("training_sample", {})
                if isinstance(item.get("inputs"), dict) and isinstance(
                    item.get("targets"), dict
                ):
                    samples.append(item)
        return samples

    def _batch(
        self, rows: list[dict[str, Any]], structure: dict[str, Any]
    ) -> dict[str, dict[str, torch.Tensor]]:
        result = {}
        for group, schema_name in (("inputs", "input_schema"), ("targets", "output_schema")):
            result[group] = {
                name: torch.stack([
                    torch.as_tensor(row[group][name], dtype=_dtype(schema.get("dtype", "float32")))
                    for row in rows
                ]).to(self.training_device)
                for name, schema in structure[schema_name].items()
            }
        return result

    def _run(
        self, model: TinyNN, samples: list[dict[str, Any]],
        structure: dict[str, Any], batch_size: int, epochs: int,
        evaluation: bool,
    ) -> list[dict[str, Any]]:
        metrics = []
        for _ in range(epochs):
            for start in range(0, len(samples), batch_size):
                batch = self._batch(samples[start:start + batch_size], structure)
                try:
                    value = (
                        model.evaluation_step(batch)
                        if evaluation else model.training_step(batch)
                    )
                except NotImplementedError:
                    if evaluation:
                        return metrics
                    raise
                if not isinstance(value, dict):
                    raise TypeError("training/evaluation step must return a mapping")
                metrics.append(value)
        return metrics

    @staticmethod
    def _average(items: list[dict[str, Any]]) -> dict[str, float]:
        result = {}
        for key in {key for item in items for key in item}:
            values = [
                float(item[key]) for item in items
                if isinstance(item.get(key), (int, float))
                and math.isfinite(float(item[key]))
            ]
            if values:
                result[key] = sum(values) / len(values)
        return result

    @staticmethod
    def _accept(
        acceptance: dict[str, float], evaluation: dict[str, float],
        regression: dict[str, float],
    ) -> tuple[bool, str | None]:
        metrics = {
            **evaluation,
            **{f"regression_{key}": value for key, value in regression.items()},
        }
        for criterion, threshold in acceptance.items():
            prefix, _, name = criterion.partition("_")
            if prefix not in {"max", "min"} or not name:
                return False, f"unknown acceptance criterion: {criterion}"
            actual = metrics.get(name)
            if actual is None:
                return False, f"acceptance metric unavailable: {name}"
            if prefix == "max" and actual > float(threshold):
                return False, f"{name} exceeded: {actual} > {threshold}"
            if prefix == "min" and actual < float(threshold):
                return False, f"{name} below: {actual} < {threshold}"
        return True, None
