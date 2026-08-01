from __future__ import annotations

from pathlib import Path

from eve.dock.trainer import Trainer, TrainingOrder
from eve.memory.memorizer import Memorizer


def write_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "model.py"
    path.write_text(
        """import torch
from eve.dock.tinynn import TinyNN

class Model(TinyNN):
    def __init__(self):
        super().__init__('explicit', '1',
            {'x': {'dtype': 'float32', 'shape': [1]}},
            {'y': {'dtype': 'float32', 'shape': [1]}})
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    def forward(self, inputs): return {'y': inputs['x'] * self.weight}
    def training_step(self, batch):
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self.forward(batch['inputs'])['y'], batch['targets']['y'])
        loss.backward(); self.optimizer.step()
        return {'loss': float(loss.detach())}
    def evaluation_step(self, batch):
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(self.forward(batch['inputs'])['y'], batch['targets']['y'])
        return {'loss': float(loss), 'goodness': 1.0}

def create_tnn(): return Model()
""",
        encoding="utf-8",
    )
    return path


def test_dock_uses_exact_model_py_and_persists_uniform_artifact(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    sample = memory.create(
        {"inputs": {"x": [1.0]}, "targets": {"y": [2.0]}},
        "training_sample",
    )
    trainer = Trainer(memory, workspace_root=tmp_path / "workspace")
    result = trainer.process_order(
        TrainingOrder(
            order_id="explicit-order",
            target_tnn_id="explicit",
            model_path=str(write_model(tmp_path / "source")),
            training_data=[sample],
            evaluation_data=[sample],
            regression_data=[sample],
            acceptance={"min_goodness": 0.5, "min_regression_goodness": 0.5},
        )
    )
    assert result.success and result.accepted
    artifact = memory.resolve_tnn_artifact("explicit")
    assert Path(artifact["model_path"]).read_text(encoding="utf-8").startswith(
        "import torch"
    )
    assert Path(artifact["weights_path"]).is_file()


def test_training_proposal_is_not_a_training_order_surface():
    fields = set(TrainingOrder.__dataclass_fields__)
    assert "definition" not in fields
    assert {"model_path", "model_memory_id", "training_data", "acceptance"} <= fields
