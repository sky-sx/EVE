from __future__ import annotations

import time

import numpy as np

from eve.core.loop import (
    CoreLoop,
    ScreenFrame,
    create_runtime_state,
    register_runtime_tnn,
)
from eve.dock.trainer import (
    qnn_critic_definition,
    Trainer,
    TrainingOrder,
    shape_locator_definition,
)
from eve.input.buffer import CursorState, InputBuffer
from eve.memory.memorizer import Memorizer


def wait_until(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def make_experience(
    memory: Memorizer,
    *,
    center: tuple[float, float],
    target_class: str = "red_circle",
    task_id: str = "red_circle_click",
    action_center: tuple[float, float] | None = None,
    hit: bool = True,
) -> str:
    image = np.zeros((96, 128, 4), dtype=np.uint8)
    image[..., 3] = 255
    x, y = (int(center[0]), int(center[1]))
    image[max(0, y - 8) : y + 8, max(0, x - 8) : x + 8, 2] = 255
    screen_id = memory.create(image, "screen_image")
    action_x, action_y = action_center or center
    return memory.record_experience(
        {
            "experience_version": 1,
            "status": "teacher_labeled",
            "task": {
                "task_id": task_id,
                "instruction": target_class,
                "target_classes": [target_class],
            },
            "state": {
                "screen_memory_id": screen_id,
                "frame_id": 1,
                "frame_timestamp_ns": 1,
                "cursor": {"x": 0, "y": 0},
            },
            "teacher": {
                "type": "vlm",
                "screen_memory_id": screen_id,
                "result_memory_id": None,
                "objects": [
                    {
                        "class": target_class,
                        "class_index": 0 if target_class == "red_circle" else 1,
                        "bbox": [x - 8, y - 8, x + 8, y + 8],
                        "center": [x, y],
                        "confidence": 1.0,
                    }
                ],
                "status": "stale",
            },
            "action": {
                "candidate_id": f"demo_{action_x}_{action_y}",
                "source": "human_demonstration",
                "action_type": "mouse",
                "payload": {
                    "action": "click",
                    "x": action_x,
                    "y": action_y,
                },
            },
            "output": {
                "executed": True,
                "blocked": False,
                "reason": "human_demonstration",
            },
            "environment": {
                "hit": hit,
                "target_id": f"target_{x}_{y}",
                "score_delta": 1 if hit else 0,
                "score_total": 1 if hit else 0,
                "reward": 1.0 if hit else -1.0,
            },
            "timestamps": {"started_at_ns": 1, "finished_at_ns": 2},
        },
        related_memory_ids=[screen_id],
    )


def test_teacher_schema_experience_and_readable_snapshots(tmp_path):
    objects = CoreLoop._parse_teacher_objects(
        {
            "objects": [
                {
                    "class": "red circle",
                    "bbox": [10, 20, 30, 40],
                    "confidence": 0.9,
                }
            ]
        },
        width=100,
        height=80,
    )
    assert objects[0]["class"] == "red_circle"
    assert objects[0]["center"] == [20.0, 30.0]

    memory = Memorizer(tmp_path / "memory")
    experience_id = make_experience(memory, center=(40, 50))
    memory.flush()
    experience = memory.read(experience_id)
    assert experience["environment"]["hit"]
    assert experience["state"]["screen_memory_id"]

    assert memory.force_review()
    wait_until(lambda: memory.review_status()["state"] == "completed")
    assert memory.counts()["mtm"] > 0
    assert memory.counts()["ltm"] == 0
    assert memory.force_review()
    wait_until(lambda: memory.review_status()["state"] == "completed")
    assert memory.counts()["ltm"] > 0

    core = CoreLoop(InputBuffer(), memory, runtime_device="cpu")
    world_path, self_path = core.save_readable_snapshots(tmp_path)
    assert world_path.name == "world.md"
    assert self_path.name == "self.md"
    assert "# EVE self" in self_path.read_text(encoding="utf-8")
    memory.stop_writer()


def test_shape_locator_training_uses_holdout_and_regression_gate(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience_ids = [
        make_experience(memory, center=(20 + index * 10, 30 + index * 5))
        for index in range(6)
    ]
    memory.flush()
    trainer = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    )
    accepted = trainer.process_order(
        TrainingOrder(
            order_id="red-shape-order",
            target_tnn_id="red-circle-locator",
            task_id="red_circle_click",
            training_data=experience_ids[:5],
            regression_data=[experience_ids[5]],
            minimum_samples=4,
            epochs=1,
            acceptance={
                "max_evaluation_loss": 1_000.0,
                "max_regression_loss": 1_000.0,
            },
            definition=shape_locator_definition(version="v1"),
        )
    )
    assert accepted.success, accepted.error
    assert accepted.accepted
    assert "evaluation" in accepted.metrics
    assert "regression" in accepted.metrics

    rejected = trainer.process_order(
        TrainingOrder(
            order_id="red-shape-rejected",
            target_tnn_id="red-circle-candidate",
            task_id="red_circle_click",
            training_data=experience_ids,
            minimum_samples=4,
            epochs=0,
            acceptance={"max_evaluation_loss": -1.0},
            definition=shape_locator_definition(version="v1"),
        )
    )
    assert rejected.success
    assert not rejected.accepted
    assert "max_evaluation_loss exceeded" in rejected.rejection_reason
    memory.stop_writer()


def test_qnn_trains_from_state_action_reward_and_loads(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience_ids = []
    for index in range(6):
        center = (25 + index * 9, 30 + index * 6)
        experience_ids.append(
            make_experience(memory, center=center, action_center=center)
        )
        experience_ids.append(
            make_experience(
                memory,
                center=center,
                action_center=(110 - index * 4, 8 + index),
                hit=False,
            )
        )
    memory.flush()
    trainer = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    )
    result = trainer.process_order(
        TrainingOrder(
            order_id="qnn-order",
            target_tnn_id="action-value-qnn",
            task_id="red_circle_click",
            training_data=experience_ids,
            minimum_samples=8,
            epochs=1,
            acceptance={"max_evaluation_loss": 10.0},
            definition=qnn_critic_definition(version="v1"),
        )
    )
    assert result.success, result.error
    assert result.accepted
    assert "evaluation" in result.metrics

    core = CoreLoop(
        InputBuffer(),
        memory,
        runtime_device="cpu",
    )
    status = core.load_qnn_runtime("action-value-qnn", "v1")
    assert status["state"] == "ready"
    assert status["tnn_id"] == "action-value-qnn"
    assert core.state["loaded_tnn"] == {}
    screen = memory.read(
        memory.read(experience_ids[0])["state"]["screen_memory_id"]
    )
    score = core.state["_qnn_runtime"].score(
        screen,
        {
            "action_type": "mouse",
            "payload": {"action": "click", "x": 25, "y": 30},
        },
    )
    assert -1.0 <= score <= 1.0
    snapshot = tmp_path / "qnn_state.json"
    core.save_snapshot(snapshot)
    core.unload_qnn_runtime()

    restored_state = create_runtime_state()
    restored = CoreLoop(
        InputBuffer(),
        memory,
        state=restored_state,
        runtime_device="cpu",
    )
    assert restored.load_snapshot(snapshot)
    try:
        restored.start()
        assert restored_state["qnn_status"]["tnn_id"] == "action-value-qnn"
        assert restored_state["_qnn_runtime"] is not None
    finally:
        restored.stop()
    memory.stop_writer()


def test_qnn_compares_candidates_before_safegate(tmp_path):
    class FakeQNN:
        tnn_id = "ranking-qnn"
        version = "v1"
        minimum_action_score = 0.0

        @staticmethod
        def score(_screen, action):
            return float(action["payload"]["x"]) / 100.0

    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state(output_mode="mock", allow_mock_actions=True)
    state["cold_started"] = True
    state["_qnn_runtime"] = FakeQNN()
    state["qnn_status"].update(
        {"state": "ready", "tnn_id": "ranking-qnn", "version": "v1"}
    )
    buffer = InputBuffer()
    timestamp = time.monotonic_ns()
    screen = np.zeros((80, 100, 4), dtype=np.uint8)
    frame = ScreenFrame(1, timestamp, 0, screen)
    cursor = CursorState(1, timestamp, 0, 0, 0.0, 0.0, 0.0)
    buffer.store("screen", frame, timestamp_ns=timestamp)
    buffer.store("cursor", cursor, timestamp_ns=timestamp)
    for tnn_id, x in (("low-action", 10), ("high-action", 90)):
        register_runtime_tnn(
            state,
            tnn_id,
            lambda _inputs, tnn_id=tnn_id, x=x: {
                "candidate": {
                    "candidate_id": tnn_id,
                    "action_type": "mouse",
                    "payload": {"action": "click", "x": x, "y": 40},
                }
            },
            inputs={"cursor": "state:cursor"},
            outputs=("candidate",),
            action_output="candidate",
            run_frequency_hz=10,
        )
    core = CoreLoop(buffer, memory, state=state, runtime_device="cpu")
    results = core.step(timestamp + 1_000_000_000)
    assert len(results) == 1
    assert results[0]["action_id"] == "high-action"
    decision = state["qnn_status"]["last_decision"]
    assert decision["selected_candidate_id"] == "high-action"
    assert decision["rejected"][0]["candidate_id"] == "low-action"
    memory.stop_writer()


def test_dock_loads_qnn_and_uses_it_as_candidate_fitness(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    qnn_ids = []
    for index in range(5):
        center = (20 + index * 10, 25 + index * 7)
        qnn_ids.extend(
            [
                make_experience(memory, center=center),
                make_experience(
                    memory,
                    center=center,
                    action_center=(115 - index * 3, 10),
                    hit=False,
                ),
            ]
        )
    memory.flush()
    trainer = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    )
    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(),
        memory,
        state=state,
        log_dir=tmp_path,
        runtime_device="cpu",
        trainer=trainer,
    )
    try:
        core.start()
        core.submit_training_order(
            TrainingOrder(
                order_id="dock-qnn",
                target_tnn_id="dock-action-qnn",
                task_id="red_circle_click",
                training_data=qnn_ids,
                minimum_samples=8,
                epochs=1,
                acceptance={"max_evaluation_loss": 10.0},
                definition=qnn_critic_definition(version="v1"),
            )
        )
        wait_until(
            lambda: state["qnn_status"].get("tnn_id")
            == "dock-action-qnn",
            timeout_s=20.0,
        )
        assert state["training_orders"]["dock-qnn"]["state"] == "qnn_loaded"

        core.submit_training_order(
            TrainingOrder(
                order_id="qnn-fitness-reject",
                target_tnn_id="fitness-red-locator",
                task_id="red_circle_click",
                training_data=qnn_ids[::2],
                fitness_data=qnn_ids[::2],
                minimum_samples=4,
                minimum_qnn_fitness=1.0,
                epochs=0,
                definition=shape_locator_definition(version="v1"),
            )
        )
        wait_until(
            lambda: state["training_orders"]
            .get("qnn-fitness-reject", {})
            .get("state")
            == "candidate_rejected",
            timeout_s=20.0,
        )
        record = state["training_orders"]["qnn-fitness-reject"]
        assert record["metrics"]["qnn_fitness"]["sample_count"] == 5
        assert "minimum_qnn_fitness not reached" in record["rejection_reason"]
        assert "fitness-red-locator" not in state["loaded_tnn"]
    finally:
        core.stop()
        memory.stop_writer()


def test_dock_trains_then_queues_and_loads_shape_tnn(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience_ids = [
        make_experience(memory, center=(20 + index * 8, 25 + index * 6))
        for index in range(5)
    ]
    memory.flush()
    trainer = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    )
    state = create_runtime_state()
    buffer = InputBuffer()
    frame = ScreenFrame(
        frame_id=1,
        captured_at_ns=time.monotonic_ns(),
        slot=0,
        image=np.zeros((96, 128, 4), dtype=np.uint8),
    )
    cursor = CursorState(
        frame_id=1,
        captured_at_ns=frame.captured_at_ns,
        x=0,
        y=0,
        velocity_x=0.0,
        velocity_y=0.0,
        speed=0.0,
    )
    buffer.store("screen", frame, timestamp_ns=frame.captured_at_ns)
    buffer.store("cursor", cursor, timestamp_ns=cursor.captured_at_ns)
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        runtime_device="cpu",
        trainer=trainer,
    )
    try:
        core.start()
        core.submit_training_order(
            TrainingOrder(
                order_id="dock-red",
                target_tnn_id="dock-red-locator",
                task_id="red_circle_click",
                training_data=experience_ids,
                minimum_samples=4,
                epochs=1,
                definition=shape_locator_definition(version="v1"),
            )
        )
        wait_until(lambda: "dock-red-locator" in state["loaded_tnn"], 20.0)
        assert "dock-red-locator" in state["active_tnn"]
        node = state["loaded_tnn"]["dock-red-locator"]
        assert node["inputs"] == {"image": "state:screen"}
        assert node["action_template"]["coordinates"] == "normalized_xy"
    finally:
        core.stop()
        memory.stop_writer()


def test_waiting_training_order_starts_when_experience_arrives(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(),
        memory,
        state=state,
        log_dir=tmp_path,
        runtime_device="cpu",
        trainer=Trainer(
            memory,
            workspace_root=tmp_path / "workspace",
            training_device="cpu",
        ),
    )
    try:
        core.start()
        core.submit_training_order(
            {
                "order_id": "wait-for-red",
                "task_id": "red_circle_click",
                "target_tnn_id": "waiting-red-locator",
                "experience_query": {
                    "task_id": "red_circle_click",
                    "teacher_class": "red_circle",
                    "hit": True,
                },
                "definition_template": "shape_locator",
                "minimum_samples": 1,
                "epochs": 0,
            }
        )
        assert (
            state["training_orders"]["wait-for-red"]["state"]
            == "waiting_for_data"
        )

        make_experience(memory, center=(30, 40))
        memory.flush()

        wait_until(
            lambda: "waiting-red-locator" in state["loaded_tnn"],
            timeout_s=20.0,
        )
        assert state["training_orders"]["wait-for-red"]["sample_ids"]
    finally:
        core.stop()
        memory.stop_writer()


def test_multitask_candidate_rejection_preserves_both_loaded_tnns(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    red_ids = [
        make_experience(memory, center=(20 + index * 7, 30 + index * 4))
        for index in range(6)
    ]
    blue_ids = [
        make_experience(
            memory,
            center=(90 - index * 6, 65 - index * 3),
            target_class="blue_triangle",
            task_id="red_blue_shapes_click",
        )
        for index in range(5)
    ]
    memory.flush()
    trainer = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    )
    for tnn_id, task_id, sample_ids in (
        ("stable-red-locator", "red_circle_click", red_ids),
        ("stable-blue-locator", "red_blue_shapes_click", blue_ids),
    ):
        result = trainer.process_order(
            TrainingOrder(
                order_id=f"initial-{tnn_id}",
                target_tnn_id=tnn_id,
                task_id=task_id,
                training_data=sample_ids,
                epochs=0,
                definition=shape_locator_definition(version="v1"),
            )
        )
        assert result.success

    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(),
        memory,
        state=state,
        log_dir=tmp_path,
        runtime_device="cpu",
        trainer=trainer,
    )
    runtime = shape_locator_definition(version="v1")["structure"]["runtime"]
    for tnn_id in ("stable-red-locator", "stable-blue-locator"):
        core.load_tnn_runtime(
            tnn_id,
            "v1",
            input_refs=runtime["input_refs"],
            action_output=runtime["action_output"],
            action_template=runtime["action_template"],
        )
    try:
        core.start()
        core.submit_training_order(
            TrainingOrder(
                order_id="rejected-red-v2",
                target_tnn_id="stable-red-locator",
                task_id="red_circle_click",
                training_data=red_ids,
                regression_data=red_ids[-2:],
                minimum_samples=4,
                epochs=0,
                acceptance={"max_evaluation_loss": -1.0},
                definition=shape_locator_definition(version="v2"),
            )
        )
        wait_until(
            lambda: state["training_orders"]
            .get("rejected-red-v2", {})
            .get("state")
            == "candidate_rejected",
            timeout_s=20.0,
        )
        assert set(state["loaded_tnn"]) == {
            "stable-red-locator",
            "stable-blue-locator",
        }
        assert state["loaded_tnn"]["stable-red-locator"]["version"] == "v1"
        assert state["loaded_tnn"]["stable-blue-locator"]["version"] == "v1"
    finally:
        core.stop()
        memory.stop_writer()


def test_environment_feedback_completes_pending_experience(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    screen = np.zeros((40, 60, 4), dtype=np.uint8)
    screen_id = memory.create(screen, "screen_image")
    state = create_runtime_state(output_mode="mock", allow_mock_actions=True)
    state["cold_started"] = True
    buffer = InputBuffer()
    timestamp = time.monotonic_ns()
    frame = ScreenFrame(1, timestamp, 0, screen)
    cursor = CursorState(1, timestamp, 5, 6, 0.0, 0.0, 0.0)
    buffer.store("screen", frame, timestamp_ns=timestamp)
    buffer.store("cursor", cursor, timestamp_ns=timestamp)
    state["teacher_visual_result"] = {
        "label_status": "valid",
        "status": "current",
        "screen_memory_id": screen_id,
        "result_memory_id": None,
        "objects": [
            {
                "class": "red_circle",
                "class_index": 0,
                "bbox": [10, 10, 20, 20],
                "center": [15, 15],
            }
        ],
    }
    register_runtime_tnn(
        state,
        "action",
        lambda _inputs: {
            "candidate": {
                "candidate_id": "candidate-1",
                "action_type": "mouse",
                "payload": {"action": "click", "x": 15, "y": 15},
            }
        },
        inputs={"cursor": "state:cursor"},
        outputs=("candidate",),
        action_output="candidate",
        run_frequency_hz=10,
    )
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        runtime_device="cpu",
    )
    core.step(time.monotonic_ns())
    assert "candidate-1" in state["pending_experiences"]
    state["pending_experiences"]["candidate-1"]["action"]["qnn_score"] = 0.25
    experience_id = core.submit_environment_feedback(
        "candidate-1",
        {
            "task_id": "red_circle_click",
            "instruction": "点击红色圆形",
            "target_classes": ["red_circle"],
            "target_id": "red-1",
            "hit": True,
            "score_delta": 1,
            "score_total": 1,
        },
    )
    memory.flush()
    experience = memory.read(experience_id)
    assert experience["environment"]["hit"]
    assert experience["environment"]["reward"] == 1.0
    assert experience["teacher"]["objects"][0]["class"] == "red_circle"
    assert state["qnn_status"]["last_feedback"]["predicted_reward"] == 0.25
    assert state["qnn_status"]["last_feedback"]["actual_reward"] == 1.0
    memory.stop_writer()


def test_llm_self_update_can_propose_restricted_training_order(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience_id = make_experience(memory, center=(30, 40))
    memory.flush()
    state = create_runtime_state()
    core = CoreLoop(
        InputBuffer(),
        memory,
        state=state,
        runtime_device="cpu",
        trainer=Trainer(
            memory,
            workspace_root=tmp_path / "workspace",
            training_device="cpu",
        ),
    )
    core._apply_llm_result(
        {
            "request_id": "self-update",
            "kind": "self_update",
            "message": "点击红色圆形",
            "memory_id": None,
        },
        {
            "reply": "",
            "thinking_summary": "需要形成红圆定位经验",
            "world_update": {},
            "myself_update": {"current_task": "red_circle_click"},
            "blackboard_updates": [],
            "active_tnn": [],
            "memory_candidates": [],
            "training_order": {
                "order_id": "llm-red-order",
                "task_id": "red_circle_click",
                "target_tnn_id": "llm-red-locator",
                "experience_query": {
                    "task_id": "red_circle_click",
                    "teacher_class": "red_circle",
                    "hit": True,
                },
                "definition_template": "shape_locator",
                "minimum_samples": 1,
            },
        },
    )
    assert state["myself"]["current_task"] == "red_circle_click"
    order = state["training_orders"]["llm-red-order"]
    assert order["sample_ids"] == [experience_id]
    memory.stop_writer()


def test_tnn_runtime_configuration_restores_from_snapshot(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    experience_ids = [
        make_experience(memory, center=(25 + index * 5, 35 + index * 4))
        for index in range(4)
    ]
    memory.flush()
    result = Trainer(
        memory,
        workspace_root=tmp_path / "workspace",
        training_device="cpu",
    ).process_order(
        TrainingOrder(
            order_id="restore-order",
            target_tnn_id="restore-locator",
            task_id="red_circle_click",
            training_data=experience_ids,
            epochs=0,
            definition=shape_locator_definition(version="v1"),
        )
    )
    assert result.success
    runtime = shape_locator_definition(version="v1")["structure"]["runtime"]
    first_state = create_runtime_state()
    first = CoreLoop(
        InputBuffer(),
        memory,
        state=first_state,
        runtime_device="cpu",
    )
    first.load_tnn_runtime(
        "restore-locator",
        "v1",
        input_refs=runtime["input_refs"],
        run_frequency_hz=runtime["run_frequency_hz"],
        output_ttl_ns=runtime["output_ttl_ns"],
        action_output=runtime["action_output"],
        action_template=runtime["action_template"],
    )
    snapshot = tmp_path / "state_snapshot.json"
    first.save_snapshot(snapshot)
    first.unload_tnn_runtime("restore-locator")

    restored_state = create_runtime_state()
    restored = CoreLoop(
        InputBuffer(),
        memory,
        state=restored_state,
        runtime_device="cpu",
    )
    assert restored.load_snapshot(snapshot)
    try:
        restored.start()
        assert "restore-locator" in restored_state["loaded_tnn"]
        node = restored_state["loaded_tnn"]["restore-locator"]
        assert node["inputs"] == {"image": "state:screen"}
        assert node["action_template"]["action"] == "click"
    finally:
        restored.stop()
        memory.stop_writer()
