from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eve.core.loop import CoreLoop, create_runtime_state
from eve.dock.trainer import (
    TrainingOrder,
    Trainer,
    _generate_model_source,
    _import_tnn_model,
    _write_generated_model,
)
from eve.memory.memorizer import Memorizer
from eve.input.buffer import InputBuffer


def mixed_structure() -> dict:
    return {
        "format_version": 1,
        "input_schema": {
            "image": {"dtype": "float32", "shape": [1, 32, 32]},
            "cursor": {"dtype": "float32", "shape": [2]},
        },
        "output_schema": {
            "mouse_delta": {"dtype": "float32", "shape": [2]},
            "click_tendency": {"dtype": "float32", "shape": [1]},
        },
        "nodes": [
            {
                "id": "conv1",
                "op": "conv2d",
                "from": ["image"],
                "params": {
                    "in_channels": 1, "out_channels": 8,
                    "kernel_size": 3, "padding": 1,
                },
            },
            {"id": "act1", "op": "silu", "from": ["conv1"], "params": {}},
            {
                "id": "conv2",
                "op": "conv2d",
                "from": ["act1"],
                "params": {
                    "in_channels": 8, "out_channels": 8,
                    "kernel_size": 3, "padding": 1,
                },
            },
            {
                "id": "residual", "op": "add",
                "from": ["act1", "conv2"], "params": {},
            },
            {
                "id": "pool", "op": "maxpool2d", "from": ["residual"],
                "params": {"kernel_size": 2, "stride": 2},
            },
            {
                "id": "conv3", "op": "conv2d", "from": ["pool"],
                "params": {
                    "in_channels": 8, "out_channels": 16,
                    "kernel_size": 3, "padding": 1,
                },
            },
            {
                "id": "adaptive", "op": "adaptive_avgpool2d", "from": ["conv3"],
                "params": {"output_size": [4, 4]},
            },
            {
                "id": "flatten", "op": "flatten", "from": ["adaptive"],
                "params": {"start_dim": 1},
            },
            {
                "id": "visual_fc", "op": "linear", "from": ["flatten"],
                "params": {"in_features": 256, "out_features": 64},
            },
            {
                "id": "cursor_fc", "op": "linear", "from": ["cursor"],
                "params": {"in_features": 2, "out_features": 16},
            },
            {
                "id": "fusion", "op": "concat",
                "from": ["visual_fc", "cursor_fc"], "params": {"dim": -1},
            },
            {
                "id": "hidden", "op": "linear", "from": ["fusion"],
                "params": {"in_features": 80, "out_features": 32},
            },
            {
                "id": "mouse_head", "op": "linear", "from": ["hidden"],
                "params": {"in_features": 32, "out_features": 2},
            },
            {
                "id": "click_head", "op": "linear", "from": ["hidden"],
                "params": {"in_features": 32, "out_features": 1},
            },
        ],
        "outputs": {
            "mouse_delta": "mouse_head",
            "click_tendency": "click_head",
        },
        "training": {
            "batch_size": 4,
            "optimizer": {"type": "adam", "lr": 0.001},
            "losses": [
                {
                    "output": "mouse_delta", "target": "mouse_delta",
                    "type": "smooth_l1", "weight": 1.0,
                },
                {
                    "output": "click_tendency", "target": "click_tendency",
                    "type": "bce_with_logits", "weight": 0.2,
                },
            ],
        },
    }


def test_generated_mixed_model_shapes_and_autograd(tmp_path):
    structure = mixed_structure()
    model_path = _write_generated_model(
        tmp_path, _generate_model_source(structure)
    )
    model = _import_tnn_model(model_path)
    batch = {
        "inputs": {
            "image": torch.randn(4, 1, 32, 32),
            "cursor": torch.randn(4, 2),
        },
        "targets": {
            "mouse_delta": torch.randn(4, 2),
            "click_tendency": torch.rand(4, 1),
        },
    }
    outputs = model(batch["inputs"])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = model.training_step(batch)

    assert outputs["mouse_delta"].shape == (4, 2)
    assert outputs["click_tendency"].shape == (4, 1)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )


def test_train_store_destroy_and_core_reload_are_equivalent(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_ids = [
        memory.create(
            {
                "inputs": {
                    "image": torch.randn(1, 32, 32).tolist(),
                    "cursor": torch.randn(2).tolist(),
                },
                "targets": {
                    "mouse_delta": torch.randn(2).tolist(),
                    "click_tendency": torch.rand(1).tolist(),
                },
            },
            "training_sample",
        )
        for _ in range(4)
    ]
    order = TrainingOrder(
        order_id="mixed-order",
        target_tnn_id="mixed-model",
        training_data=sample_ids,
        epochs=1,
        definition={"mode": "json", "version": "v1", "structure": mixed_structure()},
    )
    trainer = Trainer(memory, workspace_root=tmp_path / "workspace")
    assert trainer.training_device.type == expected_device
    result = trainer.process_order(order)
    assert result.success, result.error
    assert {path.name for path in Path(result.artifact_path).iterdir()} >= {
        "model.py", "weights.pt", "structure.json",
        "description.json", "training.json",
    }
    training_record = json.loads(
        (Path(result.artifact_path) / "training.json").read_text(encoding="utf-8")
    )
    assert training_record["training_device"] == expected_device

    inputs = {
        "image": torch.randn(4, 1, 32, 32),
        "cursor": torch.randn(4, 2),
    }
    artifact = memory.resolve_tnn_artifact("mixed-model", "v1")
    trained = _import_tnn_model(artifact["model_path"])
    trained.load_weights(artifact["weights_path"], map_location="cpu")
    expected = trained.infer(inputs)
    del trained

    state = create_runtime_state()
    loop = CoreLoop(InputBuffer(), memory, state=state)
    node = loop.load_tnn_runtime("mixed-model", "v1")
    actual = node["run"](inputs)
    assert node["device"] == expected_device
    assert next(node["model"].parameters()).device.type == expected_device
    assert all(
        torch.max(torch.abs(expected[name] - actual[name].cpu())).item() < 1e-6
        for name in expected
    )
    loop.unload_tnn_runtime("mixed-model")
    assert "mixed-model" not in state["loaded_tnn"]
    assert "mixed-model" not in state["tnn_outputs"]


def test_explicit_cpu_device_override_remains_available(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    trainer = Trainer(memory, training_device="cpu")
    loop = CoreLoop(InputBuffer(), memory, runtime_device="cpu")

    assert trainer.training_device.type == "cpu"
    assert loop.runtime_device == "cpu"


def test_unknown_op_is_rejected_explicitly():
    structure = mixed_structure()
    structure["nodes"][0]["op"] = "mystery"
    with pytest.raises(ValueError, match="unknown op: mystery"):
        _generate_model_source(structure)


def test_cross_entropy_accepts_scalar_target_schema(tmp_path):
    structure = {
        "input_schema": {"x": {"dtype": "float32", "shape": [2]}},
        "output_schema": {
            "class_logits": {"dtype": "float32", "shape": [3]}
        },
        "target_schema": {"class_id": {"dtype": "int64", "shape": []}},
        "nodes": [
            {
                "id": "classifier",
                "op": "linear",
                "from": ["x"],
                "params": {"in_features": 2, "out_features": 3},
            }
        ],
        "outputs": {"class_logits": "classifier"},
        "training": {
            "optimizer": {"type": "adam", "lr": 0.01},
            "losses": [
                {
                    "output": "class_logits",
                    "target": "class_id",
                    "type": "cross_entropy",
                }
            ],
        },
    }
    model = _import_tnn_model(
        _write_generated_model(tmp_path, _generate_model_source(structure))
    )
    metrics = model.training_step(
        {
            "inputs": {"x": torch.randn(5, 2)},
            "targets": {"class_id": torch.tensor([0, 1, 2, 1, 0])},
        }
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_python_model_uses_its_custom_training_step_and_core_protocol(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    source_artifact = tmp_path / "special-source"
    source_artifact.mkdir()
    source = """
import torch
from eve.dock.tinynn import TinyNN

class SpecialTNN(TinyNN):
    def __init__(self):
        super().__init__(
            "special-source", "v1",
            {"x": {"dtype": "float32", "shape": [1]}},
            {"y": {"dtype": "float32", "shape": [1]}},
        )
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return {"y": inputs["x"] * self.weight}

    def training_step(self, batch):
        prediction = self.forward(batch["inputs"])["y"]
        loss = torch.mean((prediction - batch["targets"]["y"]) ** 2)
        loss.backward()
        with torch.no_grad():
            self.weight -= 0.1 * self.weight.grad
            self.weight.grad.zero_()
        return {"loss": float(loss.detach()), "custom_step": 1.0}

def create_tnn():
    return SpecialTNN()
"""
    (source_artifact / "model.py").write_text(source, encoding="utf-8")
    source_model = _import_tnn_model(source_artifact / "model.py")
    source_model.save_weights(str(source_artifact / "weights.pt"))
    special_structure = {
        "input_schema": {"x": {"dtype": "float32", "shape": [1]}},
        "output_schema": {"y": {"dtype": "float32", "shape": [1]}},
        "training": {"batch_size": 1},
    }
    for name, value in {
        "structure.json": special_structure,
        "description.json": {"tnn_id": "special-source", "version": "v1"},
        "training.json": {},
    }.items():
        (source_artifact / name).write_text(json.dumps(value), encoding="utf-8")
    source_memory_id = memory.store_tnn_artifact(
        str(source_artifact), "special-source", "v1"
    )
    sample_id = memory.create(
        {"inputs": {"x": [1.0]}, "targets": {"y": [1.0]}},
        "training_sample",
    )

    result = Trainer(memory, workspace_root=tmp_path / "workspace").process_order(
        TrainingOrder(
            order_id="special-order",
            target_tnn_id="special-copy",
            training_data=[sample_id],
            epochs=1,
            definition={
                "mode": "python",
                "model_memory_id": source_memory_id,
                "factory": "create_tnn",
                "version": "v2",
            },
        )
    )
    assert result.success, result.error
    assert result.metrics["custom_step"] == 1.0

    state = create_runtime_state()
    loop = CoreLoop(InputBuffer(), memory, state=state)
    node = loop.load_tnn_runtime("special-copy", "v2")
    assert node["run"]({"x": torch.ones(1, 1)})["y"].item() > 0
    loop.unload_tnn_runtime("special-copy")
