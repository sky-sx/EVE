from __future__ import annotations

import time

import numpy as np
import pytest

from eve.core.loop import CoreLoop, create_runtime_state
from eve.dock.trainer import Trainer
from eve.input.buffer import InputBuffer, ScreenFrame
from eve.memory.memorizer import Memorizer


def wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def protocol(**updates):
    value = {
        "protocol_version": 2,
        "reply": "",
        "thinking_summary": "visible summary",
        "world_interpretation_update": {},
        "myself_cognition_update": {},
        "goodness_update": {},
        "goodness_records": [],
        "blackboard_updates": [],
        "active_tnn": [],
        "memory_actions": [],
        "action_candidates": [],
        "tool_requests": [],
        "training_proposal": None,
        "training_materialization": None,
        "observation_completion": None,
    }
    value.update(updates)
    return value


def frame(frame_id: int) -> ScreenFrame:
    return ScreenFrame(
        frame_id=frame_id,
        captured_at_ns=time.monotonic_ns(),
        slot=0,
        image=np.full((12, 16, 4), frame_id, dtype=np.uint8),
    )


ACTOR_SOURCE = """import torch
from eve.dock.tinynn import TinyNN

class Model(TinyNN):
    def __init__(self):
        super().__init__('learned-actor', 'v1',
            {'features': {'dtype': 'float32', 'shape': [2]}},
            {'prediction': {'dtype': 'float32', 'shape': [1]}})
        self.layer = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    def forward(self, inputs):
        return {'prediction': self.layer(inputs['features'])}
    def _step(self, batch, train):
        if train:
            self.optimizer.zero_grad()
        prediction = self.forward(batch['inputs'])['prediction']
        loss = torch.nn.functional.mse_loss(prediction, batch['targets']['prediction'])
        if train:
            loss.backward()
            self.optimizer.step()
        return {'loss': float(loss.detach()), 'goodness': 0.9}
    def training_step(self, batch):
        return self._step(batch, True)
    def evaluation_step(self, batch):
        return self._step(batch, False)

def create_tnn():
    return Model()
"""


def test_protocol_v2_cleans_bounded_actions_and_rejects_unknown_tool(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    buffer = InputBuffer()
    current = frame(1)
    buffer.store("screen", current, timestamp_ns=current.captured_at_ns)
    core = CoreLoop(buffer, memory, state=create_runtime_state(), log_dir=tmp_path)
    raw = protocol(
        reply="ok",
        action_candidates=[
            {
                "candidate_id": "move-1",
                "action_type": "mouse",
                "payload": {"action": "moveTo", "x": 3, "y": 4, "duration": 0.2},
                "reason_summary": "slow exploration",
                "evidence_memory_ids": [],
                "valid_for_ms": 1000,
                "expected_observation": {
                    "what_may_change": ["cursor position"],
                    "observation_delay_ms": 50,
                },
            }
        ],
    )
    clean = core._coerce_llm_result(
        raw, {"request_id": "r", "kind": "user", "message": "move"}
    )
    assert clean["action_candidates"][0]["payload"]["x"] == 3
    with pytest.raises(ValueError, match="unknown LLM tool"):
        core._coerce_llm_result(
            {
                **raw,
                "tool_requests": [
                    {
                        "request_id": "bad",
                        "tool": "shell",
                        "prompt": "x",
                        "reason_summary": "x",
                        "required_fields": [],
                        "evidence_memory_ids": [],
                    }
                ],
            },
            {"request_id": "r", "kind": "user", "message": "move"},
        )


def test_llm_vlm_tool_freezes_frame_and_continues_same_self_queue(tmp_path, monkeypatch):
    calls = []

    def llm(context):
        calls.append(context)
        if context["trigger_kind"] == "user":
            return protocol(
                reply="正在查看",
                tool_requests=[
                    {
                        "request_id": "vision-1",
                        "tool": "visual_interpretation",
                        "prompt": "identify visible object",
                        "reason_summary": "visual facts are insufficient",
                        "required_fields": ["objects"],
                        "evidence_memory_ids": [],
                    }
                ],
            )
        return protocol()

    buffer = InputBuffer()
    first = frame(1)
    buffer.store("screen", first, timestamp_ns=first.captured_at_ns)
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        local_llm_backend=llm,
        vlm_backend=lambda _request: {
            "objects": [{"class": "object", "bbox": [1, 1, 5, 5]}]
        },
    )
    monkeypatch.setattr(core, "_maybe_enqueue_autonomous", lambda: None)
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("what is visible")
        wait_until(lambda: any(item["trigger_kind"] == "vlm_result" for item in calls))
        continuation = next(
            item["continuation"] for item in calls
            if item["trigger_kind"] == "vlm_result"
        )
        assert continuation["tool_request_id"] == "vision-1"
        assert continuation["reference_frame_id"] == 1
        assert continuation["result_memory_id"]
        assert "may have changed" in continuation["reference_frame_warning"]
        assert state["autonomy_status"]["current_vlm_request_id"] is None
    finally:
        core.stop()
        memory.stop_writer()


def test_slow_action_observes_before_after_and_completes_valued_experience(
    tmp_path, monkeypatch
):
    calls = []

    def llm(context):
        calls.append(context)
        if context["trigger_kind"] == "user":
            return protocol(
                reply="尝试点击",
                action_candidates=[
                    {
                        "candidate_id": "click-1",
                        "action_type": "mouse",
                        "payload": {"action": "click", "button": "left", "x": 4, "y": 5},
                        "reason_summary": "bounded slow attempt",
                        "evidence_memory_ids": [],
                        "valid_for_ms": 2000,
                        "expected_observation": {
                            "what_may_change": ["visible object"],
                            "observation_delay_ms": 80,
                        },
                    }
                ],
            )
        if context["trigger_kind"] == "action_observation":
            bundle_id = context["continuation"]["observation_bundle_memory_id"]
            return protocol(
                goodness_records=[
                    {
                        "record_id": "observed-value-1",
                        "target": {"kind": "memory", "id": bundle_id},
                        "score": 0.4,
                        "confidence": 0.7,
                        "value_basis": {
                            "value_version": "task-values-v1",
                            "scope": {"task": "visible change"},
                            "anchors": {"negative": -1, "neutral": 0, "positive": 1},
                        },
                        "method": {"type": "teacher_direct", "producer": "local_llm"},
                        "facts": [{"name": "visible_change", "value": True}],
                        "reason": "visible evidence changed",
                        "evidence_memory_ids": [bundle_id],
                    }
                ],
                observation_completion={
                    "observation_bundle_memory_id": bundle_id,
                    "inferred_facts": [
                        {
                            "name": "visible_object_changed",
                            "value": True,
                            "source": "self_observation",
                            "source_id": bundle_id,
                            "confidence": 0.8,
                        }
                    ],
                    "goodness_record_ids": ["observed-value-1"],
                },
            )
        return protocol()

    buffer = InputBuffer()
    before = frame(1)
    buffer.store("screen", before, timestamp_ns=before.captured_at_ns)
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state(output_mode="mock")
    state["permissions"]["mouse"]["move"] = True
    state["permissions"]["mouse"]["left_click"] = True
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        local_llm_backend=llm,
        vlm_backend=lambda _request: {
            "objects": [{"class": "changed", "bbox": [2, 2, 7, 7]}]
        },
    )
    monkeypatch.setattr(core, "_maybe_enqueue_autonomous", lambda: None)
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("try the visible task")
        wait_until(lambda: isinstance(state.get("latest_output"), dict))
        after = frame(2)
        buffer.store("screen", after, timestamp_ns=after.captured_at_ns)
        wait_until(lambda: bool(memory.search(payload_type="experience")))
        memory.flush()
        bundle_id, bundle = next(
            (memory_id, memory.read(memory_id))
            for memory_id in memory.search(payload_type="observation_bundle")
            if (memory.read(memory_id) or {}).get("observation_bundle_version") == 1
        )
        assert bundle["observation_bundle_version"] == 1
        assert bundle["candidate_id"] == "click-1"
        assert bundle["before"]["frame_id"] == 1
        assert bundle["after"]["frame_id"] == 2
        assert bundle["after"]["vlm_result_memory_id"]
        experience = memory.read(memory.search(payload_type="experience")[-1])
        assert experience["status"] == "valued"
        assert experience["environment"]["source"] == "self_observed_environment"
        assert "environment_event_id" not in experience["environment"]
        assert experience["goodness_memory_ids"]
        assert memory.search(payload_type="goodness_record")
    finally:
        core.stop()
        memory.stop_writer()


def test_blocked_llm_candidate_has_no_executed_action_id(tmp_path, monkeypatch):
    def llm(context):
        if context["trigger_kind"] == "user":
            return protocol(
                reply="blocked attempt",
                action_candidates=[
                    {
                        "candidate_id": "blocked-1",
                        "action_type": "mouse",
                        "payload": {"action": "click", "button": "left", "x": 1, "y": 1},
                        "reason_summary": "candidate only",
                        "evidence_memory_ids": [],
                        "valid_for_ms": 1000,
                        "expected_observation": {
                            "what_may_change": [], "observation_delay_ms": 50
                        },
                    }
                ],
            )
        return protocol()

    buffer = InputBuffer()
    current = frame(1)
    buffer.store("screen", current, timestamp_ns=current.captured_at_ns)
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state(output_mode="mock")
    core = CoreLoop(
        buffer, memory, state=state, log_dir=tmp_path,
        local_llm_backend=llm,
    )
    monkeypatch.setattr(core, "_maybe_enqueue_autonomous", lambda: None)
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("attempt")
        wait_until(lambda: bool(memory.search(payload_type="experience")))
        memory.flush()
        result = memory.read(memory.search(payload_type="output_result")[-1])
        assert result["blocked"] is True
        assert "action_id" not in result
        assert result["required_permissions"] == ["mouse.move", "mouse.left_click"]
        assert not state["pending_observations"]
    finally:
        core.stop()
        memory.stop_writer()


def test_training_proposal_materializes_repairs_trains_and_loads_actor(
    tmp_path, monkeypatch
):
    memory = Memorizer(tmp_path / "memory")
    sample_ids = [
        memory.create(
            {
                "inputs": {"features": [float(index), 1.0]},
                "targets": {"prediction": [float(index + 1)]},
            },
            "training_sample",
        )
        for index in range(4)
    ]
    calls = []

    proposal = {
        "proposal_id": "proposal-1",
        "target_tnn": {
            "tnn_id": "learned-actor",
            "role": "generic visible-state actor",
            "input_schema": {"features": {"dtype": "float32", "shape": [2]}},
            "output_schema": {"prediction": {"dtype": "float32", "shape": [1]}},
            "runtime_frequency_hz": 2.0,
            "time_horizon_ms": 500.0,
            "upstream_inputs": ["blackboard:features"],
            "downstream_outputs": ["prediction"],
        },
        "evidence": {
            "experience_memory_ids": [],
            "goodness_memory_ids": [],
            "reason_summary": "repeated generic prediction",
        },
        "teacher_plan": {"mode": "experience", "prompt": "distill", "label_semantics": "prediction"},
        "qnn_plan": {"needed": False, "input_semantics": "none", "output_semantics": "goodness"},
        "training_plan": {
            "data_query": {}, "evaluation_query": {}, "regression_query": {},
            "acceptance": {"min_goodness": 0.5, "min_regression_goodness": 0.5},
        },
    }

    def llm(context):
        calls.append(context)
        if context["trigger_kind"] == "user":
            return protocol(reply="proposal", training_proposal=proposal)
        if context["trigger_kind"] == "training_materialization":
            attempt = context["materialization_attempt"]
            source = "def broken(:\n" if attempt == 1 else ACTOR_SOURCE
            return protocol(
                training_materialization={
                    "proposal_id": "proposal-1",
                    "files": [{"relative_path": "actor/model.py", "content": source}],
                    "qnn_files": [],
                    "training_order": {
                        "order_id": f"learn-order-{attempt}",
                        "target_tnn_id": "learned-actor",
                        "version": "v1",
                        "training_data": sample_ids,
                        "evaluation_data": sample_ids,
                        "regression_data": sample_ids,
                        "minimum_samples": 2,
                        "batch_size": 2,
                        "epochs": 1,
                        "acceptance": {"min_goodness": 0.5, "min_regression_goodness": 0.5},
                        "runtime": {
                            "input_refs": {"features": "blackboard:features"},
                            "run_frequency_hz": 2.0,
                            "output_ttl_ns": 500000000,
                        },
                    },
                }
            )
        return protocol()

    trainer = Trainer(memory, workspace_root=tmp_path / "dock" / "workspace")
    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(), memory, state=state, log_dir=tmp_path,
        local_llm_backend=llm, trainer=trainer,
    )
    monkeypatch.setattr(core, "_maybe_enqueue_autonomous", lambda: None)
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("learn this repeated operation")
        wait_until(lambda: "learned-actor" in state["loaded_tnn"], timeout_s=8.0)
        assert [
            item["materialization_attempt"] for item in calls
            if item["trigger_kind"] == "training_materialization"
        ] == [1, 2]
        assert memory.search(payload_type="training_materialization_failure")
        assert memory.search(payload_type="generated_tnn_source")
        assert memory.search(payload_type="source_safety_report")
        assert state["autonomy_status"]["latest_source_check"]["status"] == "passed"
        assert state["autonomy_status"]["materialization_status"] == "submitted_to_dock"
        assert (tmp_path / "dock" / "workspace" / "learn-order-2" / "model.py").is_file()
        assert not list((tmp_path / "dock" / "workspace").rglob("qnn/model.py"))
    finally:
        core.stop()
        memory.stop_writer()


def test_materialization_rejects_path_escape_and_dangerous_source(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    trainer = Trainer(memory, workspace_root=tmp_path / "workspace")
    order = {
        "order_id": "safe-order",
        "target_tnn_id": "learned-actor",
        "training_data": ["x"],
        "evaluation_data": ["x"],
        "regression_data": ["x"],
        "acceptance": {"min_goodness": 0, "min_regression_goodness": 0},
    }
    with pytest.raises(ValueError, match="must be actor/model.py"):
        trainer.materialize_training(
            {
                "proposal_id": "p",
                "files": [{"relative_path": "../model.py", "content": ACTOR_SOURCE}],
                "qnn_files": [],
                "training_order": order,
            },
            attempt=1,
        )
    dangerous = ACTOR_SOURCE.replace("import torch", "import os")
    with pytest.raises(ImportError, match="forbidden import"):
        trainer.materialize_training(
            {
                "proposal_id": "p",
                "files": [{"relative_path": "actor/model.py", "content": dangerous}],
                "qnn_files": [],
                "training_order": order,
            },
            attempt=1,
        )
    assert not (tmp_path / "workspace" / "safe-order").exists()


def test_materialization_stops_after_three_structured_failures(tmp_path, monkeypatch):
    proposal = {
        "proposal_id": "bounded-failure",
        "target_tnn": {
            "tnn_id": "never-created",
            "role": "generic",
            "input_schema": {},
            "output_schema": {},
            "runtime_frequency_hz": 1.0,
            "time_horizon_ms": 100.0,
            "upstream_inputs": [],
            "downstream_outputs": [],
        },
        "evidence": {
            "experience_memory_ids": [], "goodness_memory_ids": [],
            "reason_summary": "bounded failure test",
        },
        "teacher_plan": {"mode": "experience", "prompt": "x", "label_semantics": "x"},
        "qnn_plan": {"needed": False, "input_semantics": "x", "output_semantics": "goodness"},
        "training_plan": {
            "data_query": {}, "evaluation_query": {}, "regression_query": {},
            "acceptance": {"min_goodness": 0, "min_regression_goodness": 0},
        },
    }
    attempts = []

    def llm(context):
        if context["trigger_kind"] == "user":
            return protocol(reply="proposal", training_proposal=proposal)
        if context["trigger_kind"] == "training_materialization":
            attempts.append(context["materialization_attempt"])
            return protocol(
                training_materialization={
                    "proposal_id": "bounded-failure",
                    "files": [
                        {"relative_path": "actor/model.py", "content": "def broken(:\n"}
                    ],
                    "qnn_files": [],
                    "training_order": {
                        "order_id": f"never-{context['materialization_attempt']}",
                        "target_tnn_id": "never-created",
                    },
                }
            )
        return protocol()

    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(), memory, state=state, log_dir=tmp_path,
        local_llm_backend=llm,
    )
    monkeypatch.setattr(core, "_maybe_enqueue_autonomous", lambda: None)
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("materialize with bounded correction")
        wait_until(lambda: state["autonomy_status"]["materialization_status"] == "failed")
        memory.flush()
        assert attempts == [1, 2, 3]
        assert len(memory.search(payload_type="training_materialization_failure")) == 3
        assert not state["training_orders"]
        assert not state["loaded_tnn"]
    finally:
        core.stop()
        memory.stop_writer()
