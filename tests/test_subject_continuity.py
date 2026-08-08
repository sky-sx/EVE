from __future__ import annotations

import time

from eve.core.loop import CoreLoop, create_runtime_state, register_runtime_tnn
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer


def result(**updates):
    value = {
        "reply": "",
        "thinking_summary": "self update",
        "world_interpretation_update": {},
        "myself_cognition_update": {},
        "goodness_update": {},
        "goodness_records": [],
        "blackboard_updates": [],
        "active_tnn": None,
        "memory_actions": [],
        "action_candidates": [],
        "training_proposal": None,
        "prompt_request": None,
    }
    value.update(updates)
    return value


def wait_until(predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_world_and_self_patches_survive_later_rounds(tmp_path):
    state = create_runtime_state()
    core = CoreLoop(InputBuffer(), Memorizer(tmp_path / "memory"), state=state, trainer=object())
    request = {"request_id": "r1", "kind": "self_update", "message": ""}
    core._apply_llm_result(
        request,
        core._coerce_llm_result(
            result(
                world_interpretation_update={"interpretation": {"A": 1, "B": 2}},
                myself_cognition_update={
                    "current_task": "X",
                    "current_understanding": "Y",
                    "current_focus": ["Z"],
                    "current_intention": "continue",
                },
            ),
            request,
        ),
    )
    core._apply_llm_result(
        {**request, "request_id": "r2"},
        core._coerce_llm_result(
            result(world_interpretation_update={"interpretation": {"A": 3}}),
            {**request, "request_id": "r2"},
        ),
    )
    assert state["world"]["interpretation"] == {"A": 3, "B": 2}
    context = core._llm_context({"request_id": "r3", "kind": "self_update", "message": ""})
    assert context["self_view"]["current_task"] == "X"
    assert context["self_view"]["current_understanding"] == "Y"
    assert context["self_view"]["current_focus"] == ["Z"]
    assert context["self_view"]["current_intention"] == "continue"


def test_null_active_tnn_patch_preserves_loaded_activity(tmp_path):
    state = create_runtime_state()
    register_runtime_tnn(state, "A", lambda _inputs: {}, outputs=(), activate=True)
    register_runtime_tnn(state, "B", lambda _inputs: {}, outputs=(), activate=True)
    core = CoreLoop(InputBuffer(), Memorizer(tmp_path / "memory"), state=state, trainer=object())
    request = {"request_id": "r", "kind": "self_update", "message": ""}
    core._apply_llm_result(request, core._coerce_llm_result(result(), request))
    assert state["active_tnn"] == {"A", "B"}


def test_periodic_self_update_runs_without_an_event(tmp_path):
    calls = []
    state = create_runtime_state()
    state["model_config"]["autonomous_interval_s"] = 0.1
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(
        InputBuffer(), memory, state=state, log_dir=tmp_path,
        local_llm_backend=lambda context: calls.append(context) or result(),
    )
    memory.start_writer()
    try:
        core.start()
        wait_until(lambda: bool(calls))
        assert calls[0]["trigger_kind"] == "self_update"
    finally:
        core.stop()
        memory.stop_writer()


def test_recalled_memory_includes_real_payload(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    memory.create({"experience": "orchid task succeeded"}, "experience")
    state = create_runtime_state()
    state["myself"]["current_task"] = "orchid task"
    core = CoreLoop(InputBuffer(), memory, state=state, trainer=object())
    context = core._llm_context(
        {"request_id": "recall", "kind": "self_update", "message": "orchid"}
    )
    assert any(
        "orchid task succeeded" in str(item.get("payload"))
        for item in context["related_memory"]
    )


def test_model_paths_remain_semantically_independent(tmp_path):
    core = CoreLoop(InputBuffer(), Memorizer(tmp_path / "memory"), trainer=object())
    core.configure_models({"local_llm_path": "text-model", "vlm_path": "vision-model"})
    assert core.state["model_config"]["local_llm_path"] == "text-model"
    assert core.state["model_config"]["vlm_path"] == "vision-model"
    assert core.state["model_config"]["qwen_path"] == "vision-model"
