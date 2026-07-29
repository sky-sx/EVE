"""Trainable state-action value runtime used before Safegate."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import torch

from eve.dock.tinynn import TinyNN


def action_vector(
    action: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[float]:
    """Encode one mouse candidate without adding task-specific shortcuts."""
    if action.get("action_type") != "mouse":
        raise ValueError("QNN v1 only evaluates mouse actions")
    payload = action.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("mouse action payload must be a mapping")
    kind = str(payload.get("action", ""))
    if kind not in {"moveTo", "click", "doubleClick"}:
        raise ValueError(f"QNN v1 cannot encode mouse action: {kind}")
    x = float(payload.get("x", 0.0)) / float(max(width - 1, 1))
    y = float(payload.get("y", 0.0)) / float(max(height - 1, 1))
    return [
        min(1.0, max(0.0, x)),
        min(1.0, max(0.0, y)),
        1.0 if kind in {"click", "doubleClick"} else 0.0,
        1.0 if kind == "moveTo" else 0.0,
    ]


def normalized_action_vector(
    value: Any,
    action_template: dict[str, Any],
) -> list[float]:
    """Encode a normalized TNN action output for offline fitness."""
    if action_template.get("coordinates") != "normalized_xy":
        raise ValueError("QNN fitness requires normalized_xy actions")
    if hasattr(value, "detach"):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
        while values and isinstance(values[0], (list, tuple)):
            values = list(values[0])
    else:
        raise TypeError("normalized action must be a tensor or sequence")
    if len(values) < 2:
        raise ValueError("normalized action requires x and y")
    kind = str(action_template.get("action", "moveTo"))
    return [
        min(1.0, max(0.0, float(values[0]))),
        min(1.0, max(0.0, float(values[1]))),
        1.0 if kind in {"click", "doubleClick"} else 0.0,
        1.0 if kind == "moveTo" else 0.0,
    ]


def prepare_screen(value: Any, device: torch.device) -> torch.Tensor:
    """Convert HWC/BGRA or CHW/BCHW input to one RGB 64x64 batch."""
    if hasattr(value, "image"):
        value = value.image
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("QNN screen batch must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("QNN screen input must be HWC or CHW")
    if tensor.shape[0] == 3:
        chw = tensor.float()
        if float(chw.max()) > 1.0:
            chw = chw / 255.0
    elif tensor.shape[-1] >= 3:
        chw = tensor[..., :3].flip(-1).permute(2, 0, 1).float() / 255.0
    else:
        raise ValueError("QNN screen input requires three color channels")
    return torch.nn.functional.interpolate(
        chw.unsqueeze(0).to(device),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )


class QNNRuntime:
    """Loaded critic with a serialized inference lock."""

    def __init__(
        self,
        *,
        tnn_id: str,
        version: str,
        model: TinyNN,
        device: torch.device,
        artifact: dict[str, Any],
        minimum_action_score: float,
    ) -> None:
        self.tnn_id = tnn_id
        self.version = version
        self.model = model
        self.device = device
        self.artifact = artifact
        self.minimum_action_score = float(minimum_action_score)
        self._lock = threading.Lock()

    def score(self, screen: Any, action: dict[str, Any]) -> float:
        image = getattr(screen, "image", screen)
        height, width = image.shape[:2]
        return self.score_prepared(
            screen,
            action_vector(action, width=width, height=height),
        )

    def score_prepared(
        self,
        screen: Any,
        encoded_action: list[float],
    ) -> float:
        with self._lock, torch.no_grad():
            output = self.model.infer(
                {
                    "image": prepare_screen(screen, self.device),
                    "action": torch.as_tensor(
                        [encoded_action],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                }
            )
        value = output.get("q_value")
        if value is None:
            raise KeyError("QNN did not produce q_value")
        score = float(torch.as_tensor(value).reshape(-1)[0].detach().cpu())
        return min(1.0, max(-1.0, score))

    def close(self) -> None:
        was_cuda = any(parameter.is_cuda for parameter in self.model.parameters())
        self.model.to("cpu")
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_qnn_artifact(
    memorizer: Any,
    tnn_id: str,
    version: str | None,
    *,
    device: torch.device,
    minimum_action_score: float | None = None,
) -> QNNRuntime:
    artifact = memorizer.resolve_tnn_artifact(tnn_id, version)
    structure = json.loads(
        Path(artifact["structure_path"]).read_text(encoding="utf-8")
    )
    runtime = structure.get("runtime", {})
    if runtime.get("role") != "qnn":
        raise ValueError(f"{tnn_id} is not a QNN artifact")
    path = Path(artifact["model_path"]).resolve()
    module_name = f"_eve_runtime_qnn_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import QNN model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        creator = getattr(module, "create_tnn", None)
        if not callable(creator):
            raise AttributeError(f"{path} does not define create_tnn()")
        model = creator()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(model, TinyNN):
        raise TypeError(
            f"create_tnn() must return TinyNN, got {type(model).__name__}"
        )
    if set(model.get_input_schema()) != {"image", "action"}:
        raise ValueError("QNN inputs must be exactly image and action")
    if set(model.get_output_schema()) != {"q_value"}:
        raise ValueError("QNN output must be exactly q_value")
    model.load_weights(artifact["weights_path"], map_location=device)
    model.to(device)
    model.eval()
    threshold = (
        runtime.get("minimum_action_score", 0.0)
        if minimum_action_score is None
        else minimum_action_score
    )
    return QNNRuntime(
        tnn_id=tnn_id,
        version=str(artifact["version"]),
        model=model,
        device=device,
        artifact=artifact,
        minimum_action_score=float(threshold),
    )
