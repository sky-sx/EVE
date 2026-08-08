from __future__ import annotations

import json
import time

from eve.input.buffer import InputBuffer
from eve.main import EVEApplication


def wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("smoke condition did not become true")


def test_full_cold_start_self_memory_vlm_and_normal_shutdown(tmp_path):
    buffer = InputBuffer(
        profile="control",
        capture_options={
            "screen_mode": "synthetic",
            "cursor_mode": "synthetic",
            "keyboard_mode": "synthetic",
            "screen_interval_s": 0.02,
            "cursor_interval_s": 0.02,
            "keyboard_interval_s": 0.02,
        },
    )
    app = EVEApplication(
        profile="control",
        mode="mock",
        run_dir=tmp_path / "run",
        memory_dir=tmp_path / "memory",
        input_buffer=buffer,
        use_default_local_models=False,
    )
    contexts = []

    def llm(context):
        contexts.append(context)
        return {
            "reply": "smoke-ok" if context.get("trigger_kind") == "user" else "",
            "thinking_summary": "smoke self update",
            "world_interpretation_update": {
                "interpretation": {"smoke_seen": True}
            },
            "myself_cognition_update": {
                "current_understanding": "smoke loop is active",
                "current_intention": "continue observing",
            },
            "goodness_update": {},
            "goodness_records": [],
            "blackboard_updates": [],
            "active_tnn": None,
            "memory_actions": [],
            "action_candidates": [],
            "training_proposal": None,
            "prompt_request": None,
        }

    app.core.local_llm_backend = llm
    app.core.vlm_backend = lambda _request: json.dumps(
        {
            "summary": "synthetic frame",
            "verified_detections": [
                {"class": "synthetic_object", "bbox": [1, 1, 8, 8], "confidence": 1.0}
            ],
            "corrections": [],
        }
    )
    app.state["model_config"]["autonomous_interval_s"] = 0.1
    app.memory.create({"experience": "smoke memory payload"}, "experience")
    app.state["myself"]["current_task"] = "smoke memory"

    app.cold_start()
    try:
        wait_until(lambda: bool(contexts))
        app.send_user_message("recall smoke memory")
        wait_until(lambda: len(contexts) >= 2)
        wait_until(lambda: app.buffer.get_latest_screen() is not None)
        app.request_vlm()
        wait_until(lambda: app.state.get("last_visual_interpretation_result") is not None)
        assert any(
            "smoke memory payload" in json.dumps(context, ensure_ascii=False, default=str)
            for context in contexts
        )
        assert app.state["world"]["interpretation"]["smoke_seen"] is True
        assert app.state["myself"]["current_task"] == "smoke memory"
        assert app.state["myself"]["current_understanding"] == "smoke loop is active"
        assert app.state["last_visual_interpretation_result"]["status"] in {
            "current", "stale"
        }
        assert app.state["last_visual_interpretation_result"]["label_status"] == "valid"
    finally:
        app.normal_stop()

    state_dir = tmp_path / "state"
    assert (state_dir / "state_snapshot.json").is_file()
    assert (state_dir / "world.md").is_file()
    assert (state_dir / "self.md").is_file()
    assert app.summary()["threads_stopped"] is True
