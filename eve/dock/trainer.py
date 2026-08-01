"""Dock training for concrete, model-authored TinyNN implementations."""
from __future__ import annotations

import ast
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


def _evaluate_goodness_expression(
    expression: str,
    variables: dict[str, Any],
    declared_variables: list[str] | tuple[str, ...],
) -> float:
    """Evaluate the deliberately tiny numeric subset used by ValueDefinition."""
    declared = {str(item) for item in declared_variables}
    if set(variables) - declared:
        raise ValueError("expression variables were not declared")
    missing = declared - set(variables)
    if missing:
        raise KeyError(f"missing value variables: {sorted(missing)}")
    clean: dict[str, float] = {}
    for name, value in variables.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"value variable must be numeric: {name}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"value variable must be finite: {name}")
        clean[name] = number

    binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
    }
    comparisons = {
        ast.Eq: lambda left, right: left == right,
        ast.NotEq: lambda left, right: left != right,
        ast.Lt: lambda left, right: left < right,
        ast.LtE: lambda left, right: left <= right,
        ast.Gt: lambda left, right: left > right,
        ast.GtE: lambda left, right: left >= right,
    }

    def evaluate(node: ast.AST) -> float | bool:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise TypeError("only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in declared:
                raise NameError(f"undeclared value variable: {node.id}")
            return clean[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](float(evaluate(node.left)), float(evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            number = float(evaluate(node.operand))
            return number if isinstance(node.op, ast.UAdd) else -number
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            for operation, comparator in zip(node.ops, node.comparators):
                if type(operation) not in comparisons:
                    raise TypeError("comparison operator is not allowed")
                right = evaluate(comparator)
                if not comparisons[type(operation)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return evaluate(node.body if bool(evaluate(node.test)) else node.orelse)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise TypeError("only direct calls to min/max/abs/clip are allowed")
            name = node.func.id
            values = [float(evaluate(argument)) for argument in node.args]
            if name == "abs" and len(values) == 1:
                return abs(values[0])
            if name == "min" and values:
                return min(values)
            if name == "max" and values:
                return max(values)
            if name == "clip" and len(values) == 3:
                return max(values[1], min(values[2], values[0]))
            raise TypeError("invalid safe numeric function call")
        raise TypeError(f"disallowed expression node: {type(node).__name__}")

    tree = ast.parse(str(expression), mode="eval")
    result = evaluate(tree.body)
    if isinstance(result, bool):
        raise TypeError("goodness expression must return a number")
    number = float(result)
    if not math.isfinite(number):
        raise ValueError("goodness expression returned NaN or infinity")
    return max(-1.0, min(1.0, number))


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
    value_definition_ids: list[str] = field(default_factory=list)
    goodness_data: list[str] = field(default_factory=list)
    qnn_stage: dict[str, Any] = field(default_factory=dict)
    actor_stage: dict[str, Any] = field(default_factory=dict)
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
        aggregation = str(order.actor_stage.get("goodness_aggregation", "mean"))
        if aggregation not in {"mean", "minimum"}:
            raise ValueError("goodness aggregation must be mean or minimum")
        qnn = order.qnn_stage
        if qnn.get("enabled"):
            if bool(qnn.get("model_path")) == bool(qnn.get("model_memory_id")):
                raise ValueError("QNN requires exactly one model_path or model_memory_id")
            if not order.goodness_data:
                raise ValueError("enabled QNN requires goodness_data")
            mapping = qnn.get("input_mapping")
            if not isinstance(mapping, dict) or not isinstance(
                mapping.get("state"), list
            ) or not isinstance(mapping.get("candidate"), list):
                raise ValueError("QNN input_mapping requires state and candidate fields")
            if int(qnn.get("top_k", 1)) < 1:
                raise ValueError("QNN top_k must be positive")
            if qnn.get("delete_after_training", True) is not True:
                raise ValueError("temporary QNN must be deleted after training")

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
        qnn_report: dict[str, Any] = {"enabled": False, "cleanup_complete": True}
        generated_actor_samples: list[dict[str, Any]] = []
        if order.qnn_stage.get("enabled"):
            generated_actor_samples, qnn_report = self._run_qnn_stage(order, artifact)
            training.extend(generated_actor_samples)
        try:
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
            aggregation = str(order.actor_stage.get("goodness_aggregation", "mean"))
            averaged_train = self._average(train_metrics, aggregation)
            averaged_eval = self._average(eval_metrics, aggregation)
            averaged_regression = self._average(regression_metrics, aggregation)
            accepted, reason = self._accept(
                order.acceptance, averaged_eval, averaged_regression
            )
            if order.qnn_stage.get("enabled"):
                qnn_report["actor_stage_completed"] = True
        finally:
            if order.qnn_stage.get("enabled"):
                self._cleanup_qnn_workspace(artifact, qnn_report)
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
                "goodness_aggregation": aggregation,
                "qnn": qnn_report,
                "generated_actor_samples": [
                    item.get("selection", {}) for item in generated_actor_samples
                ],
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
                {
                    "order": order.order_id,
                    "accepted": False,
                    "reason": reason,
                    "evaluation": averaged_eval,
                    "regression": averaged_regression,
                    "qnn": qnn_report,
                },
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
                "qnn": qnn_report,
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
                if item.get("experience_version") in {1, 2}:
                    item = item.get("training_sample", {})
                if isinstance(item.get("inputs"), dict) and isinstance(
                    item.get("targets"), dict
                ):
                    samples.append(item)
        return samples

    @staticmethod
    def _field(value: dict[str, Any], path: str) -> Any:
        current: Any = value
        for part in str(path).split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(path)
            current = current[part]
        return current

    def _load_value_definitions(
        self, memory_ids: list[str]
    ) -> dict[str, tuple[str, dict[str, Any]]]:
        definitions: dict[str, tuple[str, dict[str, Any]]] = {}
        for memory_id in memory_ids:
            value = self._memorizer.read(memory_id)
            if not isinstance(value, dict) or value.get("value_definition_version") != 1:
                raise ValueError(f"invalid ValueDefinition: {memory_id}")
            version = str(value.get("value_version", ""))
            if not version:
                raise ValueError(f"ValueDefinition has no value_version: {memory_id}")
            definitions[version] = (memory_id, value)
        return definitions

    @staticmethod
    def _score_from_definition(
        definition: dict[str, Any], facts: list[dict[str, Any]]
    ) -> tuple[float, dict[str, float]]:
        if definition.get("mode") != "generated_function":
            raise ValueError("teacher goodness is required for non-function definitions")
        function = definition.get("function")
        if not isinstance(function, dict):
            raise ValueError("generated function definition is missing")
        available = {
            str(item.get("name")): item.get("value")
            for item in facts
            if isinstance(item, dict) and str(item.get("name", ""))
        }
        declared_inputs = {
            str(item.get("name")): bool(item.get("required", False))
            for item in definition.get("inputs", [])
            if isinstance(item, dict) and str(item.get("name", ""))
        }
        variables = [str(item) for item in function.get("variables", [])]
        if set(variables) - set(declared_inputs):
            raise ValueError("function uses variables outside ValueDefinition.inputs")
        missing_required = [
            name for name, required in declared_inputs.items()
            if required and name not in available
        ]
        if missing_required:
            raise KeyError(f"missing required goodness facts: {missing_required}")
        used = {name: available[name] for name in variables if name in available}
        score = _evaluate_goodness_expression(
            str(function.get("expression", "")), used, variables
        )
        return score, {name: float(value) for name, value in used.items()}

    def _load_goodness_candidates(
        self, order: TrainingOrder
    ) -> list[dict[str, Any]]:
        definitions = self._load_value_definitions(order.value_definition_ids)
        candidates: list[dict[str, Any]] = []
        for memory_id in order.goodness_data:
            value = self._memorizer.read(memory_id)
            if not isinstance(value, dict):
                raise ValueError(f"invalid goodness data: {memory_id}")
            record = value if value.get("goodness_version") == 1 else None
            if record is not None:
                target = record.get("target", {})
                target_id = str(target.get("id", ""))
                candidate = self._memorizer.read(target_id)
                if not isinstance(candidate, dict):
                    raise ValueError(f"goodness target is not a candidate sample: {target_id}")
                score: Any = record.get("score")
                value_version = str(record.get("value_basis", {}).get("value_version", ""))
                goodness_memory_id = memory_id
                teacher_source = str(record.get("method", {}).get("producer", "teacher"))
                evidence = record.get("evidence_memory_ids", [])
            else:
                candidate = value
                score = candidate.get("teacher_goodness", candidate.get("goodness"))
                value_version = str(candidate.get("value_version", ""))
                goodness_memory_id = str(candidate.get("goodness_memory_id", ""))
                teacher_source = str(candidate.get("teacher_source", "teacher"))
                evidence = candidate.get("evidence_memory_ids", [])
            if not isinstance(evidence, list) or any(
                self._memorizer.get_unit(str(item)) is None for item in evidence
            ):
                raise KeyError(f"untraceable goodness evidence: {memory_id}")
            if not value_version:
                raise ValueError(f"goodness data has no value_version: {memory_id}")
            used_variables: dict[str, float] = {}
            if score is None:
                if value_version not in definitions:
                    raise ValueError("goodness is waiting for a teacher evaluation")
                definition_id, definition = definitions[value_version]
                score, used_variables = self._score_from_definition(
                    definition, list(candidate.get("facts", ()))
                )
                generated_record = {
                    "goodness_version": 1,
                    "record_id": f"goodness_{time.time_ns()}_{uuid.uuid4().hex[:8]}",
                    "target": {"kind": "candidate_output", "id": memory_id},
                    "score": score,
                    "confidence": 1.0,
                    "value_basis": {
                        "value_version": value_version,
                        "scope": dict(definition.get("scope", {})),
                        "anchors": dict(definition.get("anchors", {})),
                    },
                    "method": {
                        "type": "generated_function",
                        "producer": "dock_safe_ast",
                        "definition_memory_id": definition_id,
                        "qnn_job_id": order.order_id,
                    },
                    "facts": [
                        {
                            "name": name,
                            "value": value,
                            "source": "candidate_facts",
                            "memory_id": memory_id,
                        }
                        for name, value in used_variables.items()
                    ],
                    "reason": "safe ValueDefinition expression",
                    "evidence_memory_ids": list(dict.fromkeys(
                        [memory_id, definition_id, *(str(item) for item in evidence)]
                    )),
                    "created_at_ns": time.time_ns(),
                }
                goodness_memory_id = self._memorizer.record_goodness(
                    generated_record,
                    related_memory_ids=generated_record["evidence_memory_ids"],
                )
                teacher_source = "generated_function"
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(
                float(score)
            ):
                raise ValueError(f"candidate goodness must be finite: {memory_id}")
            if not isinstance(candidate.get("state"), dict) or not isinstance(
                candidate.get("candidate_output"), dict
            ):
                raise ValueError("goodness candidate requires state and candidate_output")
            if record is None and score is not None:
                if goodness_memory_id:
                    existing_record = self._memorizer.read(goodness_memory_id)
                    if not isinstance(existing_record, dict) or existing_record.get(
                        "goodness_version"
                    ) != 1:
                        raise ValueError("goodness_memory_id is not a GoodnessRecord")
                else:
                    definition_id = str(
                        candidate.get("value_definition_memory_id", "")
                    )
                    if definition_id and self._memorizer.get_unit(definition_id) is None:
                        raise KeyError(f"unknown ValueDefinition MemoryID: {definition_id}")
                    confidence = float(candidate.get("goodness_confidence", 1.0))
                    if not math.isfinite(confidence):
                        raise ValueError("teacher goodness confidence must be finite")
                    teacher_record = {
                        "goodness_version": 1,
                        "record_id": f"goodness_{time.time_ns()}_{uuid.uuid4().hex[:8]}",
                        "target": {"kind": "candidate_output", "id": memory_id},
                        "score": max(-1.0, min(1.0, float(score))),
                        "confidence": max(0.0, min(1.0, confidence)),
                        "value_basis": {
                            "value_version": value_version,
                            "scope": dict(candidate.get("value_scope", {})),
                            "anchors": dict(candidate.get("value_anchors", {
                                "negative": -1.0,
                                "neutral": 0.0,
                                "positive": 1.0,
                            })),
                        },
                        "method": {
                            "type": "teacher_direct",
                            "producer": teacher_source,
                            "definition_memory_id": definition_id,
                            "qnn_job_id": order.order_id,
                        },
                        "facts": [dict(item) for item in candidate.get("facts", [])],
                        "reason": str(candidate.get("goodness_reason", "teacher supplied")),
                        "evidence_memory_ids": list(dict.fromkeys(
                            [memory_id, *(str(item) for item in evidence), *(
                                [definition_id] if definition_id else []
                            )]
                        )),
                        "created_at_ns": time.time_ns(),
                    }
                    goodness_memory_id = self._memorizer.record_goodness(
                        teacher_record,
                        related_memory_ids=teacher_record["evidence_memory_ids"],
                    )
            candidates.append(
                {
                    **candidate,
                    "source_memory_id": memory_id,
                    "teacher_goodness": max(-1.0, min(1.0, float(score))),
                    "value_version": value_version,
                    "goodness_memory_id": goodness_memory_id,
                    "teacher_source": teacher_source,
                    "function_variables": used_variables,
                }
            )
        return candidates

    @staticmethod
    def _schema_value(values: list[Any], schema: dict[str, Any]) -> Any:
        shape = list(schema.get("shape", []))
        if not shape:
            if len(values) != 1:
                raise ValueError("scalar schema requires exactly one mapped field")
            return values[0]
        return values

    def _qnn_sample(
        self,
        candidate: dict[str, Any],
        mapping: dict[str, list[str]],
        structure: dict[str, Any],
        target_field: str,
    ) -> dict[str, Any]:
        input_schema = structure["input_schema"]
        output_schema = structure["output_schema"]
        if set(input_schema) != {"state", "candidate_output"}:
            raise ValueError("QNN must expose state and candidate_output inputs")
        if target_field not in output_schema:
            raise ValueError(f"QNN output is missing {target_field}")
        state_values = [
            self._field(candidate["state"], name) for name in mapping["state"]
        ]
        candidate_values = [
            self._field(candidate["candidate_output"], name)
            for name in mapping["candidate"]
        ]
        return {
            "inputs": {
                "state": self._schema_value(state_values, input_schema["state"]),
                "candidate_output": self._schema_value(
                    candidate_values, input_schema["candidate_output"]
                ),
            },
            "targets": {
                target_field: self._schema_value(
                    [candidate["teacher_goodness"]], output_schema[target_field]
                )
            },
        }

    @staticmethod
    def _scalar_output(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                raise ValueError("QNN returned an empty goodness tensor")
            number = float(value.detach().cpu().reshape(-1)[0].item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
        else:
            raise TypeError("QNN goodness output must be numeric")
        if not math.isfinite(number):
            raise ValueError("QNN goodness output must be finite")
        return max(-1.0, min(1.0, number))

    def _run_qnn_stage(
        self, order: TrainingOrder, artifact: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        config = order.qnn_stage
        qnn_dir = artifact / "_temporary_qnn"
        qnn_dir.mkdir()
        report: dict[str, Any] = {
            "enabled": True,
            "cleanup_complete": False,
            "registered_as_tnn": False,
            "actor_stage_completed": False,
        }
        actor_samples: list[dict[str, Any]] = []
        try:
            if config.get("model_memory_id"):
                resolved = self._memorizer.resolve_tnn_artifact(
                    str(config["model_memory_id"])
                )
                source_model = Path(resolved["model_path"])
                model_source = {"model_memory_id": str(config["model_memory_id"])}
            else:
                source_model = Path(str(config["model_path"])).expanduser().resolve()
                model_source = {"model_path": str(source_model)}
            qnn_model_path = qnn_dir / "model.py"
            shutil.copy2(source_model, qnn_model_path)
            qnn = _import_tnn_model(
                qnn_model_path, str(config.get("factory", "create_tnn"))
            ).to(self.training_device)
            structure = {
                "input_schema": qnn.get_input_schema(),
                "output_schema": qnn.get_output_schema(),
            }
            candidates = self._load_goodness_candidates(order)
            mapping = config["input_mapping"]
            target_field = str(config.get("target_field", "goodness"))
            qnn_samples = [
                self._qnn_sample(item, mapping, structure, target_field)
                for item in candidates
            ]
            training_metrics = self._run(
                qnn,
                qnn_samples,
                structure,
                max(1, int(config.get("batch_size", order.batch_size))),
                max(1, int(config.get("epochs", 1))),
                False,
            )
            evaluation_metrics = self._run(
                qnn,
                qnn_samples,
                structure,
                max(1, int(config.get("batch_size", order.batch_size))),
                1,
                True,
            )
            predictions: list[float] = []
            for sample in qnn_samples:
                batch = self._batch([sample], structure)
                output = qnn.infer(batch["inputs"])
                if not isinstance(output, dict) or target_field not in output:
                    raise ValueError("QNN inference did not return predicted goodness")
                predictions.append(self._scalar_output(output[target_field]))
            for candidate, predicted in zip(candidates, predictions):
                candidate["predicted_goodness"] = predicted
            comparable = consistent = 0
            for left in range(len(candidates)):
                for right in range(left + 1, len(candidates)):
                    teacher_delta = (
                        candidates[left]["teacher_goodness"]
                        - candidates[right]["teacher_goodness"]
                    )
                    predicted_delta = predictions[left] - predictions[right]
                    if teacher_delta == 0:
                        continue
                    comparable += 1
                    consistent += int(teacher_delta * predicted_delta > 0)
            group_by = str(config.get("group_by", "state_id"))
            groups: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                group = str(self._field(candidate, group_by))
                groups.setdefault(group, []).append(candidate)
            top_k = int(config.get("top_k", 1))
            selected: list[dict[str, Any]] = []
            for rows in groups.values():
                selected.extend(
                    sorted(
                        rows,
                        key=lambda item: float(item["predicted_goodness"]),
                        reverse=True,
                    )[:top_k]
                )
            for candidate in selected:
                actor = candidate.get("actor_sample")
                if actor is None:
                    actor = {
                        "inputs": candidate["state"],
                        "targets": candidate["candidate_output"],
                    }
                if not isinstance(actor, dict) or not isinstance(
                    actor.get("inputs"), dict
                ) or not isinstance(actor.get("targets"), dict):
                    raise ValueError("selected candidate has no valid Actor sample")
                actor_samples.append(
                    {
                        "inputs": dict(actor["inputs"]),
                        "targets": dict(actor["targets"]),
                        "selection": {
                            "state_id": str(candidate.get("state_id", "")),
                            "candidate_id": str(candidate.get("candidate_id", "")),
                            "teacher_goodness": candidate["teacher_goodness"],
                            "predicted_goodness": candidate["predicted_goodness"],
                            "value_version": candidate["value_version"],
                            "goodness_memory_id": candidate["goodness_memory_id"],
                        },
                    }
                )
            qnn.save_weights(str(qnn_dir / "weights.pt"))
            errors = [
                abs(predicted - candidate["teacher_goodness"])
                for candidate, predicted in zip(candidates, predictions)
            ]
            report.update(
                {
                    "model_source": model_source,
                    "structure": structure,
                    "sample_count": len(candidates),
                    "training_metrics": training_metrics,
                    "evaluation_metrics": evaluation_metrics,
                    "mean_absolute_goodness_error": sum(errors) / len(errors),
                    "ranking_consistency": (
                        consistent / comparable if comparable else 1.0
                    ),
                    "value_versions": sorted(
                        {item["value_version"] for item in candidates}
                    ),
                    "teacher_sources": sorted(
                        {item["teacher_source"] for item in candidates}
                    ),
                    "top_k": top_k,
                    "selected_candidate_ids": [
                        str(item.get("candidate_id", "")) for item in selected
                    ],
                }
            )
        except Exception:
            self._cleanup_qnn_workspace(artifact, report)
            raise
        return actor_samples, report

    @staticmethod
    def _cleanup_qnn_workspace(
        artifact: Path, report: dict[str, Any]
    ) -> None:
        qnn_dir = artifact / "_temporary_qnn"
        if qnn_dir.exists():
            shutil.rmtree(qnn_dir)
        report["cleanup_complete"] = not qnn_dir.exists()

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
    def _average(
        items: list[dict[str, Any]], goodness_aggregation: str = "mean"
    ) -> dict[str, float]:
        result = {}
        for key in {key for item in items for key in item}:
            values = [
                float(item[key]) for item in items
                if isinstance(item.get(key), (int, float))
                and math.isfinite(float(item[key]))
            ]
            if values:
                result[key] = (
                    min(values)
                    if key == "goodness" and goodness_aggregation == "minimum"
                    else sum(values) / len(values)
                )
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
