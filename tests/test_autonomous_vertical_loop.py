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
        "training_proposal": None,
        "prompt_request": None,
    }
    actions = updates.pop("action_candidates", None)
    value.update(updates)
    if actions is not None:
        value["action_candidates"] = [
            {
                "action_type": item["action_type"],
                "payload": item["payload"],
                "horizon_ms": item.get("valid_for_ms", 1000),
                "reason_summary": item.get("reason_summary", ""),
            }
            for item in actions
        ]
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
    assert clean["action_candidates"][0]["candidate_id"].startswith("llm:")
    with pytest.raises(ValueError, match="available organ"):
        core._coerce_llm_result(
            {
                **raw,
                "prompt_request": "shell",
            },
            {"request_id": "r", "kind": "user", "message": "move"},
        )


def test_llm_vlm_tool_freezes_frame_and_continues_same_self_queue(tmp_path, monkeypatch):
    calls = []

    def llm(context):
        calls.append(context)
        if context["trigger_kind"] == "user":
            return protocol(reply="正在查看", prompt_request="vision")
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
        core._queue_frozen_vlm_request(
            request_id="vision-1",
            prompt="identify visible object",
            origin="llm_tool",
        )
        wait_until(lambda: any(item["trigger_kind"] == "vlm_result" for item in calls))
        continuation = next(
            item["continuation_feedback"] for item in calls
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


def test_one_automatic_vision_call_per_root_task(tmp_path):
    buffer = InputBuffer()
    sample = frame(1)
    buffer.store("screen", sample, timestamp_ns=sample.captured_at_ns)
    state = create_runtime_state()
    state["model_status"]["vlm"]["state"] = "ready"
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(
        buffer, memory, state=state, log_dir=tmp_path,
        vlm_backend=lambda _request: {}, trainer=object(),
    )
    try:
        core._queue_frozen_vlm_request(
            request_id="first", prompt="look", origin="llm_tool",
            root_task_id="root",
        )
        with pytest.raises(RuntimeError, match="Vision call limit"):
            core._queue_frozen_vlm_request(
                request_id="second", prompt="look again", origin="action_observation",
                root_task_id="root",
            )
    finally:
        memory.stop_writer()


def test_action_observation_requires_real_execution_and_valid_before_frame(tmp_path):
    state = create_runtime_state(output_mode="mock")
    state["myself"]["current_task"] = "compare the visible result"
    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), state=state,
        log_dir=tmp_path, trainer=object(),
    )
    action = {
        "candidate_id": "candidate",
        "reason_summary": "compare result",
        "expected_observation": {"observation_delay_ms": 50},
    }
    before = {
        "frame_id": 1,
        "frame_timestamp_ns": time.monotonic_ns(),
        "screen_memory_id": "screen-before",
    }
    core._schedule_action_observation(
        action,
        {"executed": False, "simulated": True},
        before,
        {},
    )
    assert core._observation_requests.empty()
    assert state["pending_observations"] == {}

    core._schedule_action_observation(
        action,
        {"executed": True, "simulated": False},
        before,
        {},
    )
    assert core._observation_requests.qsize() == 1
    assert state["pending_observations"]["candidate"]["state"] == "waiting_after_frame"


def test_failed_action_observation_does_not_queue_text_continuation(
    tmp_path, monkeypatch
):
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(
        InputBuffer(), memory, log_dir=tmp_path,
        trainer=object(),
    )
    try:
        continuations = []
        monkeypatch.setattr(
            core, "_queue_llm_continuation",
            lambda **kwargs: continuations.append(kwargs),
        )
        core._finalize_action_observation(
            {
                "candidate_id": "candidate",
                "before": {"frame_id": 1, "frame_timestamp_ns": 1},
                "after": {"frame_id": 2, "frame_timestamp_ns": 2},
                "result": {
                    "action_id": "action", "started_at_ns": 1,
                    "finished_at_ns": 2,
                },
                "memory_ids": {},
            },
            {
                "status": "failed", "label_status": "failed",
                "result_memory_id": None,
            },
        )
        assert continuations == []
    finally:
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


def test_training_proposal_is_stored_without_hidden_materialization_protocol(
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
        wait_until(lambda: bool(memory.search(payload_type="training_proposal")))
        assert len(calls) == 1
        assert not state["loaded_tnn"]
        assert state["autonomy_status"]["training_proposal_status"] == "stored"
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


def test_materialization_stage_is_not_part_of_first_edition_output(tmp_path, monkeypatch):
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
    def llm(context):
        if context["trigger_kind"] == "user":
            return protocol(reply="proposal", training_proposal=proposal)
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
        wait_until(lambda: bool(memory.search(payload_type="training_proposal")))
        memory.flush()
        assert not state["training_orders"]
        assert not state["loaded_tnn"]
    finally:
        core.stop()
        memory.stop_writer()
