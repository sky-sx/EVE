from __future__ import annotations

import json
import time

import numpy as np

from eve.core.loop import CoreLoop, create_runtime_state
from eve.input.buffer import InputBuffer, ScreenFrame
from eve.memory.memorizer import Memorizer


def wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def protocol_reply(reply: str = "ok") -> dict:
    return {
        "protocol_version": 2,
        "reply": reply,
        "thinking_summary": "visible summary",
        "world_interpretation_update": {},
        "myself_cognition_update": {},
        "goodness_update": {},
        "goodness_records": [],
        "blackboard_updates": [],
        "active_tnn": [],
        "memory_actions": [],
        "training_proposal": None,
    }


def test_user_prompt_contains_current_visual_facts_and_perception_is_code_owned(
    tmp_path,
):
    contexts = []

    def llm(context):
        contexts.append(context)
        result = protocol_reply("屏幕上有一个 object。")
        result["world_interpretation_update"] = {
            "perception": {"forbidden": True},
            "interpretation": {"scene": "desktop"},
        }
        return result

    buffer = InputBuffer()
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        local_llm_backend=llm,
        runtime_visual_backend=lambda _image: {
            "detections": [
                {
                    "bbox": [1, 2, 7, 8],
                    "confidence": 0.9,
                    "class_id": 3,
                    "class_name": "object",
                }
            ]
        },
    )
    memory.start_writer()
    try:
        core.start()
        frame = ScreenFrame(
            frame_id=11,
            captured_at_ns=time.monotonic_ns(),
            slot=0,
            image=np.zeros((10, 10, 4), dtype=np.uint8),
        )
        buffer.store("screen", frame, timestamp_ns=frame.captured_at_ns)
        wait_until(
            lambda: state["world"]["perception"]["visual"].get(
                "reference_frame_id"
            )
            == 11
        )
        core.submit_user_message("屏幕上有什么？")
        wait_until(lambda: bool(state["conversation"]))
        visual = contexts[-1]["world_view"]["perception_facts"]["visual"]
        assert "检测到 object，置信度约 0.90" in contexts[-1]["world_view"][
            "visual_summary"
        ]
        assert visual["detections"][0] == {
            "class_name": "object",
            "confidence": 0.9,
            "bbox": [1.0, 2.0, 7.0, 8.0],
            "center": [4.0, 5.0],
            "region": None,
            "class_id": 3,
        }
        assert "forbidden" not in state["world"]["perception"]
        assert state["world"]["interpretation"]["scene"] == "desktop"
    finally:
        core.stop()
        memory.stop_writer()


def test_protocol_repair_once_and_metrics_are_visible(tmp_path):
    calls = []

    def backend(context):
        calls.append(context)
        return {"reply": ""} if len(calls) == 1 else protocol_reply("repaired")

    state = create_runtime_state()
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(
        InputBuffer(), memory, state=state, log_dir=tmp_path,
        local_llm_backend=backend,
    )
    memory.start_writer()
    try:
        core.start()
        core.submit_user_message("hello")
        wait_until(lambda: bool(state["conversation"]))
        wait_until(
            lambda: state["model_status"]["local_llm"]["success_count"] == 1
        )
        status = state["model_status"]["local_llm"]
        assert state["conversation"][-1]["reply"] == "repaired"
        assert status["attempt_count"] == 2
        assert status["schema_failure_count"] == 1
        assert status["repair_count"] == 1
        assert status["success_count"] == 1
    finally:
        core.stop()
        memory.stop_writer()


def test_snapshot_v2_excludes_transient_state_and_feedback_is_explicit(tmp_path):
    state = create_runtime_state()
    state["cold_started"] = True
    state["world"]["interpretation"] = {"stable": True}
    state["world"]["perception"] = {"visual": {"frame": 9}}
    state["blackboard"]["temporary"] = {"value": "drop"}
    state["resource_status"]["secret"] = "drop"
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path)
    before = dict(state["myself"]["goodness"])
    record = core.feedback("praise")
    assert state["myself"]["goodness"] == before
    assert record["memory_id"]
    path = tmp_path / "state_snapshot.json"
    core.save_snapshot(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["snapshot_version"] == 2
    assert saved["world"] == {
        "interpretation": {"stable": True},
        "uncertainty": {},
        "task_state": {},
    }
    assert "blackboard" not in saved
    assert "resource_status" not in saved
    assert "perception" not in saved["world"]
    memory.flush()
    assert memory.search(payload_type="input_snapshot") == []
    core.stop()
    memory.stop_writer()
