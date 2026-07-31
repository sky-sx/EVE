from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from eve.core.loop import (
    MAX_LOADED_TNN,
    CoreLoop,
    create_runtime_state,
    register_runtime_tnn,
)
from eve.dock.trainer import Trainer, TrainingOrder
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer


def wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def write_linear_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "model.py"
    path.write_text(
        """import torch
from eve.dock.tinynn import TinyNN

class Model(TinyNN):
    def __init__(self):
        super().__init__('generic-regressor', 'v1',
            {'features': {'dtype': 'float32', 'shape': [2]}},
            {'prediction': {'dtype': 'float32', 'shape': [1]}})
        self.layer = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    def forward(self, inputs):
        return {'prediction': self.layer(inputs['features'])}
    def _metrics(self, batch, train):
        if train:
            self.optimizer.zero_grad()
        prediction = self.forward(batch['inputs'])['prediction']
        loss = torch.nn.functional.mse_loss(prediction, batch['targets']['prediction'])
        if train:
            loss.backward(); self.optimizer.step()
        return {'loss': float(loss.detach())}
    def training_step(self, batch): return self._metrics(batch, True)
    def evaluation_step(self, batch): return self._metrics(batch, False)

def create_tnn(): return Model()
""",
        encoding="utf-8",
    )
    return path


def sample_ids(memory: Memorizer, count: int = 6) -> list[str]:
    return [
        memory.create(
            {
                "inputs": {"features": [float(index), 1.0]},
                "targets": {"prediction": [float(index * 2 + 1)]},
            },
            "training_sample",
        )
        for index in range(count)
    ]


def test_generic_order_trains_persists_and_core_loads_tnn(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    trainer = Trainer(memory, workspace_root=tmp_path / "dock" / "workspace")
    ids = sample_ids(memory)
    result = trainer.process_order(
        TrainingOrder(
            order_id="generic-order",
            target_tnn_id="generic-regressor",
            model_path=str(write_linear_model(tmp_path / "source")),
            version="v1",
            training_data=ids,
            evaluation_data=ids,
            minimum_samples=4,
            epochs=1,
            acceptance={"max_loss": 1_000_000.0},
            runtime={
                "input_refs": {"features": "blackboard:features"},
                "run_frequency_hz": 5.0,
                "output_ttl_ns": 500_000_000,
            },
        )
    )

    assert result.success and result.accepted
    assert memory.resolve_tnn_artifact("generic-regressor", "v1")["memory_id"]

    state = create_runtime_state()
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path / "run")
    node = core.load_tnn_runtime("generic-regressor", "v1", activate=False)
    assert node["tnn_id"] == "generic-regressor"
    assert "generic-regressor" in state["loaded_tnn"]
    core.stop()


def test_explicit_model_file_can_continue_training(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    trainer = Trainer(memory, workspace_root=tmp_path / "dock" / "workspace")
    ids = sample_ids(memory)
    model_path = write_linear_model(tmp_path / "source")
    first = trainer.process_order(
        TrainingOrder(
            order_id="first",
            target_tnn_id="portable-model",
            model_path=str(model_path),
            version="v1",
            training_data=ids,
            evaluation_data=ids,
            epochs=1,
            acceptance={"max_loss": 1_000_000.0},
        )
    )
    continued = trainer.process_order(
        TrainingOrder(
            order_id="continued",
            target_tnn_id="portable-model",
            model_path=str(Path(first.artifact_path) / "model.py"),
            weights_path=str(Path(first.artifact_path) / "weights.pt"),
            version="v2",
            training_data=ids,
            evaluation_data=ids,
            epochs=1,
            continue_training=True,
            acceptance={"max_loss": 1_000_000.0},
        )
    )
    assert continued.success and continued.accepted
    assert memory.resolve_tnn_artifact("portable-model", "v2")["version"] == "v2"


def test_executable_order_requires_concrete_model_and_acceptance(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    trainer = Trainer(memory, workspace_root=tmp_path / "dock")
    with pytest.raises(ValueError, match="exactly one"):
        trainer.enqueue(TrainingOrder(order_id="bad", target_tnn_id="bad"))


def test_runtime_has_no_qnn_surface_and_enforces_five_tnn_limit():
    assert not (Path(__file__).parents[1] / "eve" / "core" / "qnn.py").exists()
    state = create_runtime_state()
    assert "qnn_status" not in state
    for index in range(MAX_LOADED_TNN):
        register_runtime_tnn(state, f"node-{index}", lambda _: {}, outputs=())
    with pytest.raises(RuntimeError, match="maximum loaded TNN count"):
        register_runtime_tnn(state, "one-too-many", lambda _: {}, outputs=())


def test_due_scheduler_prevents_reentry_and_slow_node_does_not_block_fast_node(
    tmp_path,
):
    state = create_runtime_state()
    state["cold_started"] = True
    state["resource_status"]["updated_at_ns"] = time.monotonic_ns()
    buffer = InputBuffer()
    buffer.store("cursor", (1, 2))
    memory = Memorizer(tmp_path / "memory")
    active_slow = 0
    max_active_slow = 0
    slow_lock = threading.Lock()

    def slow(_inputs):
        nonlocal active_slow, max_active_slow
        with slow_lock:
            active_slow += 1
            max_active_slow = max(max_active_slow, active_slow)
        try:
            time.sleep(0.12)
            return {"value": 1}
        finally:
            with slow_lock:
                active_slow -= 1

    register_runtime_tnn(
        state,
        "slow",
        slow,
        inputs={"cursor": "state:cursor"},
        outputs=("value",),
        run_frequency_hz=20.0,
    )
    register_runtime_tnn(
        state,
        "fast",
        lambda _inputs: {"value": 1},
        inputs={"cursor": "state:cursor"},
        outputs=("value",),
        run_frequency_hz=20.0,
    )
    core = CoreLoop(buffer, memory, state=state, log_dir=tmp_path / "run")
    try:
        deadline = time.monotonic() + 0.45
        while time.monotonic() < deadline:
            core.step()
            time.sleep(0.005)
        state["active_tnn"].discard("slow")
        completion_deadline = time.monotonic() + 1.0
        while (
            state["loaded_tnn"]["slow"]["running"]
            and time.monotonic() < completion_deadline
        ):
            core.step()
            time.sleep(0.005)

        slow_node = state["loaded_tnn"]["slow"]
        fast_node = state["loaded_tnn"]["fast"]
        assert not slow_node["running"]
        assert max_active_slow == 1
        assert fast_node["run_count"] > slow_node["run_count"]
        assert slow_node["overdue"]
        assert slow_node["skipped_count"] > 0
        assert slow_node["actual_frequency_hz"] > 0
        assert fast_node["actual_frequency_hz"] > 0
    finally:
        core.stop()


def test_memory_catalog_and_semantic_views_are_independent(tmp_path):
    memory = Memorizer(tmp_path / "memory", stm_limit=2)
    ids = [memory.create({"index": index}) for index in range(5)]
    assert set(ids) == set(memory.catalog)
    assert memory.counts()["stm"] == 2
    assert memory.counts()["mtm"] == 0
    memory.load_to_mtm(ids[-1])
    memory.persist_to_ltm(ids[-1])
    assert ids[-1] in memory.stm
    assert ids[-1] in memory.mtm
    assert ids[-1] in memory.ltm

    reloaded = Memorizer(tmp_path / "memory", stm_limit=2)
    assert set(reloaded.catalog) == set(ids)
    assert ids[-1] in set(reloaded.stm) & reloaded.mtm & reloaded.ltm


def test_feedback_requires_exact_candidate_action_time_and_environment_event(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state(output_mode="mock", allow_mock_actions=True)
    state["cold_started"] = True
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path / "run")
    executed_at_ns = time.monotonic_ns()
    action = {
        "candidate_id": "candidate-1",
        "source": "actor",
        "action_type": "mouse",
        "payload": {"action": "click", "x": 20, "y": 30},
        "generated_at_ns": executed_at_ns - 1_000,
        "valid_until_ns": executed_at_ns + 1_000_000_000,
    }
    state["pending_experiences"]["candidate-1"] = {
        "state": {},
        "teacher": {},
        "action": action,
        "output": {
            "candidate_id": "candidate-1",
            "action_id": "candidate-1",
            "finished_at_ns": executed_at_ns,
            "executed": False,
            "simulated": True,
            "blocked": False,
        },
        "started_at_ns": action["generated_at_ns"],
        "related_memory_ids": [],
    }
    event_memory_id = memory.create(
        {
            "candidate_id": "candidate-1",
            "action_id": "candidate-1",
            "action_type": "mouse",
            "payload": {"action": "click", "x": 20, "y": 30},
        },
        "environment_feedback",
    )
    event = memory.create_event(
        [event_memory_id],
        started_at_ns=executed_at_ns,
        ended_at_ns=executed_at_ns + 100,
    )
    experience_id = core.submit_environment_feedback(
        {
            "candidate_id": "candidate-1",
            "action_id": "candidate-1",
            "executed_at_ns": executed_at_ns,
            "environment_event_id": event.event_id,
            "hit": True,
        }
    )
    memory.flush()
    experience = memory.read(experience_id)
    assert experience["environment"]["environment_event_id"] == event.event_id
    assert "candidate-1" not in state["pending_experiences"]

    state["pending_experiences"]["candidate-2"] = {
        **state["blackboard"]["latest_experience"]["value"],
        "state": {},
        "teacher": {},
        "action": {**action, "candidate_id": "candidate-2"},
        "output": {
            "action_id": "candidate-2",
            "finished_at_ns": executed_at_ns,
            "simulated": True,
            "blocked": False,
        },
        "started_at_ns": action["generated_at_ns"],
        "related_memory_ids": [],
    }
    with pytest.raises(ValueError, match="not bound"):
        core.submit_environment_feedback(
            {
                "candidate_id": "candidate-2",
                "action_id": "candidate-2",
                "executed_at_ns": executed_at_ns,
                "environment_event_id": event.event_id,
            }
        )
    assert "candidate-2" in state["pending_experiences"]
    core.stop()
    memory.stop_writer()


def test_emergency_stop_clears_candidate_and_output_queues(tmp_path):
    state = create_runtime_state()
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path / "run")
    state["action_queue"].append({"candidate_id": "candidate"})
    core._output_requests.put_nowait({"candidate_id": "queued"})
    core.emergency_stop("test")
    assert list(state["action_queue"]) == []
    assert core._output_requests.empty()
    assert state["emergency_stop"]
    core.stop()
