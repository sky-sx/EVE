from __future__ import annotations

from pathlib import Path

import pytest

from eve.core.loop import CoreLoop, create_runtime_state
from eve.dock.trainer import (
    Trainer,
    TrainingOrder,
    _evaluate_goodness_expression,
)
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer


def protocol_reply() -> dict:
    return {
        "reply": "",
        "thinking_summary": "visible summary only",
        "world_interpretation_update": {},
        "myself_cognition_update": {},
        "goodness_update": {},
        "goodness_records": [],
        "blackboard_updates": [],
        "active_tnn": [],
        "memory_actions": [],
        "action_candidates": [],
        "training_proposal": None,
        "prompt_request": None,
    }


def write_qnn_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "model.py"
    path.write_text(
        """import torch
from eve.dock.tinynn import TinyNN

class Model(TinyNN):
    def __init__(self):
        super().__init__('temporary-value-model', 'v1',
            {'state': {'dtype': 'float32', 'shape': [1]},
             'candidate_output': {'dtype': 'float32', 'shape': [1]}},
            {'goodness': {'dtype': 'float32', 'shape': [1]}})
        self.layer = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.08)
    def forward(self, inputs):
        features = torch.cat([inputs['state'], inputs['candidate_output']], dim=-1)
        return {'goodness': self.layer(features)}
    def _step(self, batch, train):
        if train: self.optimizer.zero_grad()
        predicted = self.forward(batch['inputs'])['goodness']
        loss = torch.nn.functional.mse_loss(predicted, batch['targets']['goodness'])
        if train:
            loss.backward(); self.optimizer.step()
        return {'loss': float(loss.detach())}
    def training_step(self, batch): return self._step(batch, True)
    def evaluation_step(self, batch): return self._step(batch, False)

def create_tnn(): return Model()
""",
        encoding="utf-8",
    )
    return path


def write_actor_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "model.py"
    path.write_text(
        """import torch
from eve.dock.tinynn import TinyNN

class Model(TinyNN):
    def __init__(self):
        super().__init__('generic-actor', 'v1',
            {'context': {'dtype': 'float32', 'shape': [1]}},
            {'action': {'dtype': 'float32', 'shape': [1]}})
        self.value = torch.nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.2)
    def forward(self, inputs):
        return {'action': self.value.expand(inputs['context'].shape[0], 1)}
    def training_step(self, batch):
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self.forward(batch['inputs'])['action'], batch['targets']['action'])
        loss.backward(); self.optimizer.step()
        return {'loss': float(loss.detach())}
    def evaluation_step(self, batch):
        loss = torch.nn.functional.mse_loss(self.forward(batch['inputs'])['action'], batch['targets']['action'])
        return {'loss': float(loss.detach()), 'goodness': 0.8}

def create_tnn(): return Model()
""",
        encoding="utf-8",
    )
    return path


def test_safe_value_expression_is_small_finite_and_clipped():
    assert _evaluate_goodness_expression(
        "clip((hit * 2 - errors) / max(latency, 1), -1, 1) if allowed > 0 else -1",
        {"hit": 1, "errors": 0.2, "latency": 2, "allowed": 1},
        ["hit", "errors", "latency", "allowed"],
    ) == pytest.approx(0.9)
    assert _evaluate_goodness_expression(
        "min(2, max(-2, abs(delta)))", {"delta": -3}, ["delta"]
    ) == 1.0
    for expression in (
        "__import__('os')",
        "(1).__class__",
        "open('secret')",
        "[x for x in range(2)]",
    ):
        with pytest.raises((NameError, TypeError, ValueError)):
            _evaluate_goodness_expression(expression, {}, [])
    with pytest.raises(KeyError):
        _evaluate_goodness_expression("missing", {}, ["missing"])
    with pytest.raises(ValueError):
        _evaluate_goodness_expression("x", {"x": float("nan")}, ["x"])
    with pytest.raises(ValueError):
        _evaluate_goodness_expression("1e309", {}, [])


def test_goodness_records_are_independent_clipped_and_linked(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    target_payload = {
        "experience_version": 2,
        "status": "observed",
        "task": {},
        "state": {},
        "teacher": {},
        "action": {},
        "output": {},
        "environment": {"environment_event_id": "", "facts": []},
        "timestamps": {"started_at_ns": 1, "finished_at_ns": 1},
        "goodness_memory_ids": [],
        "immutable": True,
    }
    target_id = memory.create(target_payload, "experience")
    definition_id = memory.create(
        {
            "value_definition_version": 1,
            "definition_id": "definition-1",
            "value_version": "values-v1",
            "scope": {"target_kind": "experience"},
            "goal": "compare legal candidates",
            "inputs": [{"name": "external_score", "required": True}],
            "mode": "teacher_direct",
            "function": None,
            "anchors": {"negative": -1, "neutral": 0, "positive": 1},
            "constraints": ["permissions remain binding"],
            "created_by": "human",
            "evidence_memory_ids": [target_id],
            "created_at_ns": 1,
        },
        "value_definition",
    )
    state = create_runtime_state()
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path / "run")
    raw = protocol_reply()
    raw["goodness_update"] = {"score": 0.1, "confidence": 0.2}
    raw["goodness_records"] = [
        {
            "target": {"kind": "experience", "id": target_id},
            "score": 4,
            "confidence": 2,
            "value_basis": {
                "value_version": "values-v1",
                "scope": {"target_kind": "experience"},
                "anchors": {"negative": -1, "neutral": 0, "positive": 1},
            },
            "method": {
                "type": "teacher_direct",
                "producer": "local_llm",
                "definition_memory_id": definition_id,
            },
            "facts": [{"name": "external_score", "value": 7}],
            "reason": "teacher interpretation",
            "evidence_memory_ids": [target_id],
        }
    ]
    request = {"request_id": "r1", "kind": "goodness_evaluation", "message": ""}
    clean = core._coerce_llm_result(raw, request)
    assert clean["goodness_update"]["score"] == 0.1
    assert clean["goodness_records"][0]["score"] == 1.0
    assert clean["goodness_records"][0]["confidence"] == 1.0
    core._apply_llm_result(request, clean)
    core._apply_llm_result({**request, "request_id": "r2"}, clean)
    memory.start_writer()
    memory.flush()
    records = memory.search(payload_type="goodness_record")
    assert len(records) == 2
    assert records[0] != records[1]
    assert memory.read(target_id) == target_payload
    assert state["myself"]["goodness"]["latest_record_memory_id"] == records[-1]
    assert state["blackboard"]["latest_goodness"]["value"]["target"]["id"] == target_id
    assert any(target_id in event.memory_ids for event in memory.events.values())
    invalid = protocol_reply()
    invalid["goodness_records"] = [{**raw["goodness_records"][0], "evidence_memory_ids": ["missing"]}]
    with pytest.raises(KeyError, match="unknown goodness evidence"):
        core._coerce_llm_result(invalid, request)
    memory.stop_writer()


def test_experience_v2_has_facts_and_experience_v1_is_rejected(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience = {
        "experience_version": 2,
        "status": "valued",
        "task": {},
        "state": {},
        "teacher": {},
        "action": {},
        "output": {},
        "environment": {
            "environment_event_id": "",
            "facts": [{"name": "reward", "value": 3, "source": "external"}],
        },
        "timestamps": {"started_at_ns": 1, "finished_at_ns": 2},
        "goodness_memory_ids": ["g1", "g2"],
        "training_sample": {"inputs": {"x": [1]}, "targets": {"y": [2]}},
    }
    v2_id = memory.create(experience, "experience")
    v1_id = memory.create(
        {
            "experience_version": 1,
            "training_sample": {"inputs": {"x": [2]}, "targets": {"y": [3]}},
        },
        "experience",
    )
    trainer = Trainer(memory, workspace_root=tmp_path / "dock")
    loaded = trainer._load_samples([v2_id])
    assert len(loaded) == 1
    with pytest.raises(ValueError, match="only Experience v2"):
        trainer._load_samples([v1_id])
    with pytest.raises(ValueError, match="unsupported experience_version"):
        memory.record_experience({**experience, "experience_version": 1})
    assert "reward" not in experience["environment"]
    assert experience["goodness_memory_ids"] == ["g1", "g2"]


def test_value_definition_and_goodness_request_are_explicit_memory_operations(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    target_id = memory.create({"episode": "facts"}, "episode")
    core = CoreLoop(
        InputBuffer(), memory, state=create_runtime_state(), log_dir=tmp_path / "run"
    )
    definition_id = core.create_value_definition(
        {
            "value_version": "values-v2",
            "scope": {"task_id": "task", "target_kind": "episode"},
            "goal": "prefer legal completion",
            "inputs": [{"name": "completion", "required": True}],
            "mode": "generated_function",
            "function": {"expression": "completion", "variables": ["completion"]},
            "anchors": {"negative": -1, "neutral": 0, "positive": 1},
            "constraints": ["permission gate remains binding"],
            "created_by": "human",
            "evidence_memory_ids": [target_id],
        }
    )
    request_id = core.request_goodness_evaluation(
        target_id,
        target_kind="episode",
        value_definition_memory_id=definition_id,
        evidence_memory_ids=[target_id],
    )
    queued = core._llm_requests.get_nowait()
    assert queued["request_id"] == request_id
    assert queued["kind"] == "goodness_evaluation"
    assert queued["target"] == {"kind": "episode", "id": target_id}
    assert memory.get_record(definition_id).payload_type == "value_definition"


def test_generated_function_creates_record_and_never_uses_missing_fact_default(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    definition_id = memory.create(
        {
            "value_definition_version": 1,
            "definition_id": "definition-function",
            "value_version": "values-function-v1",
            "scope": {"target_kind": "candidate_output"},
            "goal": "interpret supplied facts",
            "inputs": [
                {"name": "quality", "required": True},
                {"name": "delay", "required": True},
            ],
            "mode": "generated_function",
            "function": {
                "expression": "clip(quality - delay, -1, 1)",
                "variables": ["quality", "delay"],
            },
            "anchors": {"negative": -1, "neutral": 0, "positive": 1},
            "constraints": [],
            "created_by": "local_llm",
            "evidence_memory_ids": [],
            "created_at_ns": 1,
        },
        "value_definition",
    )
    candidate_id = memory.create(
        {
            "state_id": "state",
            "candidate_id": "candidate",
            "state": {"context": 0.0},
            "candidate_output": {"action": 0.5},
            "value_version": "values-function-v1",
            "facts": [
                {"name": "quality", "value": 0.9},
                {"name": "delay", "value": 0.2},
            ],
        },
        "candidate_goodness_sample",
    )
    trainer = Trainer(memory, workspace_root=tmp_path / "dock")
    order = TrainingOrder(
        order_id="function-order",
        target_tnn_id="actor",
        value_definition_ids=[definition_id],
        goodness_data=[candidate_id],
    )
    loaded = trainer._load_goodness_candidates(order)
    assert loaded[0]["teacher_goodness"] == pytest.approx(0.7)
    record = memory.read(loaded[0]["goodness_memory_id"])
    assert record["method"]["type"] == "generated_function"
    assert {item["name"] for item in record["facts"]} == {"quality", "delay"}
    broken_id = memory.create(
        {
            **memory.read(candidate_id),
            "candidate_id": "broken",
            "facts": [{"name": "quality", "value": 0.9}],
        },
        "candidate_goodness_sample",
    )
    with pytest.raises(KeyError, match="missing required"):
        trainer._load_goodness_candidates(
            TrainingOrder(
                order_id="broken",
                target_tnn_id="actor",
                value_definition_ids=[definition_id],
                goodness_data=[broken_id],
            )
        )


def test_blocked_candidate_becomes_evaluable_experience_not_positive_feedback(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    memory.start_writer()
    state = create_runtime_state()
    state["myself"]["current_task"] = "bounded action"
    state["blackboard"]["latest_permission_result"] = {
        "value": {"allowed": False, "blocked_atoms": ["mouse.left_click"]}
    }
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path / "run")
    action = {
        "candidate_id": "blocked-candidate",
        "action_type": "mouse",
        "payload": {"action": "click", "x": 1, "y": 2},
        "generated_at_ns": 10,
    }
    core._remember_chain(
        action,
        {
            "candidate_id": "blocked-candidate",
            "action_id": "blocked-candidate",
            "executed": False,
            "simulated": False,
            "blocked": True,
            "reason": "permission denied",
            "finished_at_ns": 20,
        },
    )
    memory.flush()
    experience_ids = memory.search(payload_type="experience")
    assert len(experience_ids) == 1
    experience = memory.read(experience_ids[0])
    assert experience["status"] == "awaiting_goodness"
    assert experience["output"]["executed"] is False
    facts = {item["name"]: item["value"] for item in experience["environment"]["facts"]}
    assert facts["action_allowed"] is False
    assert facts["blocked_reason"] == "permission denied"
    assert "blocked-candidate" not in state["pending_experiences"]
    with pytest.raises(KeyError, match="unknown pending action"):
        core.submit_environment_feedback(
            {
                "candidate_id": "blocked-candidate",
                "action_id": "blocked-candidate",
                "executed_at_ns": 20,
                "environment_event_id": "none",
                "reward": 1,
            }
        )
    memory.stop_writer()


def test_temporary_qnn_ranks_candidates_generates_actor_and_is_deleted(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    goodness_ids = []
    for candidate_id, teacher_goodness in (("A", 0.8), ("B", -0.4), ("C", 0.2)):
        candidate_memory_id = memory.create(
            {
                "state_id": "same-state",
                "candidate_id": candidate_id,
                "state": {"context": 0.0},
                "candidate_output": {"action": teacher_goodness},
                "actor_sample": {
                    "inputs": {"context": [0.0]},
                    "targets": {"action": [teacher_goodness]},
                },
            },
            "candidate_goodness_sample",
        )
        goodness_ids.append(
            memory.record_goodness(
                {
                    "goodness_version": 1,
                    "record_id": f"teacher-{candidate_id}",
                    "target": {"kind": "candidate_output", "id": candidate_memory_id},
                    "score": teacher_goodness,
                    "confidence": 1.0,
                    "value_basis": {
                        "value_version": "values-v1",
                        "scope": {},
                        "anchors": {"negative": -1, "neutral": 0, "positive": 1},
                    },
                    "method": {
                        "type": "teacher_direct",
                        "producer": "synthetic_teacher",
                        "definition_memory_id": "",
                        "qnn_job_id": "",
                    },
                    "facts": [],
                    "reason": "synthetic ordering evidence",
                    "evidence_memory_ids": [candidate_memory_id],
                    "created_at_ns": 1,
                },
                related_memory_ids=[candidate_memory_id],
            )
        )
    evaluation_id = memory.create(
        {"inputs": {"context": [0.0]}, "targets": {"action": [0.8]}},
        "training_sample",
    )
    workspace = tmp_path / "dock" / "workspace"
    trainer = Trainer(memory, workspace_root=workspace)
    result = trainer.process_order(
        TrainingOrder(
            order_id="value-ranked-actor",
            target_tnn_id="generic-actor",
            model_path=str(write_actor_model(tmp_path / "actor")),
            goodness_data=goodness_ids,
            evaluation_data=[evaluation_id],
            regression_data=[evaluation_id],
            minimum_samples=1,
            epochs=4,
            acceptance={"min_goodness": 0.7, "min_regression_goodness": 0.7},
            qnn_stage={
                "enabled": True,
                "model_path": str(write_qnn_model(tmp_path / "qnn")),
                "factory": "create_tnn",
                "epochs": 180,
                "batch_size": 3,
                "input_mapping": {"state": ["context"], "candidate": ["action"]},
                "target_field": "goodness",
                "group_by": "state_id",
                "top_k": 1,
                "delete_after_training": True,
            },
        )
    )
    assert result.success and result.accepted
    qnn = result.metrics["qnn"]
    assert qnn["selected_candidate_ids"] == ["A"]
    assert qnn["ranking_consistency"] == 1.0
    assert qnn["cleanup_complete"] is True
    assert qnn["registered_as_tnn"] is False
    assert not (Path(result.artifact_path) / "_temporary_qnn").exists()
    assert memory.resolve_tnn_artifact("generic-actor")["tnn_id"] == "generic-actor"
    assert all(item["tnn_id"] != "temporary-value-model" for item in memory.list_tnn_artifacts())
    training = Path(result.artifact_path, "training.json").read_text(encoding="utf-8")
    assert '"candidate_id": "A"' in training
    assert '"teacher_sources"' in training


def test_goodness_aggregation_and_only_first_version_acceptance_remain():
    assert Trainer._average(
        [{"goodness": 0.9}, {"goodness": -0.2}], "minimum"
    )["goodness"] == -0.2
    assert Trainer._accept(
        {"min_goodness": 0.7, "min_regression_goodness": 0.6},
        {"goodness": 0.8},
        {"goodness": 0.7},
    ) == (True, None)
    accepted, reason = Trainer._accept({"max_loss": 1.0}, {"loss": 0.5}, {})
    assert accepted is False
    assert reason == "acceptance schema is not first-version goodness"


def test_formal_eve_has_no_runtime_qnn_or_task_specific_value_formula():
    root = Path(__file__).parents[1]
    assert not (root / "eve" / "core" / "qnn.py").exists()
    formal = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("eve/core/loop.py", "eve/dock/trainer.py", "eve/memory/memorizer.py")
    )
    for forbidden in ("red_circle", "blue_triangle", "red_blue_game", "stage_1"):
        assert forbidden not in formal
    for removed_compatibility in (
        "legacy_memorizer",
        "legacy_catalog_path",
        'value.get("world_update"',
        'value.get("myself_update"',
        'value.get("memory_candidates"',
        'value.get("training_order"',
        '"max_loss"',
    ):
        assert removed_compatibility not in formal
