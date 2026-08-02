from __future__ import annotations

import threading
import time
import json

import numpy as np
import pytest

from eve.core.loop import (
    DEFAULT_LOCAL_LLM_PATH,
    DEFAULT_VLM_PATH,
    DEFAULT_YOLO_PATH,
    MAX_LOADED_TNN,
    CoreLoop,
    check_action_permission,
    create_runtime_state,
    default_permissions,
    register_runtime_tnn,
)
from eve.input.buffer import InputBuffer, ScreenFrame
from eve.main import EVEApplication, EVEControlWindow, _load_qt
from eve.memory.memorizer import Memorizer


def synthetic_buffer(profile: str = "control") -> InputBuffer:
    return InputBuffer(
        profile=profile,
        capture_options={
            "screen_mode": "synthetic",
            "cursor_mode": "synthetic",
            "keyboard_mode": "synthetic",
            "window_mode": "synthetic",
        },
    )


def wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_local_models_have_repo_defaults_and_old_snapshot_is_rejected(
    tmp_path,
):
    runtime = EVEApplication(
        profile="control",
        run_dir=tmp_path,
        input_buffer=synthetic_buffer(),
    )

    assert runtime.state["model_config"]["local_llm_path"] == (
        DEFAULT_LOCAL_LLM_PATH
    )
    assert runtime.state["model_config"]["vlm_path"] == DEFAULT_VLM_PATH
    assert runtime.state["model_config"]["yolo_model_path"] == DEFAULT_YOLO_PATH
    assert runtime.state["model_status"]["local_llm"]["state"] == "configured"
    assert runtime.state["model_status"]["yolo"]["auto_start"] is True
    assert runtime.state["model_status"]["vlm"]["state"] == "configured"

    snapshot = tmp_path / "legacy_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "model_config": {
                    "local_llm_path": "",
                    "vlm_path": "",
                    "yolo_model_path": "",
                    "cloud_enabled": True,
                }
            }
        ),
        encoding="utf-8",
    )
    assert not runtime.core.load_snapshot(snapshot)
    assert runtime.state["model_config"]["local_llm_path"] == (
        DEFAULT_LOCAL_LLM_PATH
    )
    assert runtime.state["model_config"]["vlm_path"] == DEFAULT_VLM_PATH
    assert runtime.state["model_config"]["yolo_model_path"] == DEFAULT_YOLO_PATH
    assert runtime.state["model_config"]["cloud_enabled"] is False

    current_snapshot = tmp_path / "current_snapshot.json"
    runtime.core.save_snapshot(current_snapshot)
    runtime.state["myself"]["current_task"] = "changed"
    assert runtime.core.load_snapshot(current_snapshot)
    assert runtime.state["myself"]["current_task"] == ""


def test_control_gui_has_eight_pages_and_does_not_cold_start(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt = _load_qt()
    gui = qt["QApplication"].instance() or qt["QApplication"]([])
    runtime = EVEApplication(
        profile="control",
        run_dir=tmp_path,
        input_buffer=synthetic_buffer(),
    )
    frame = ScreenFrame(
        frame_id=1,
        captured_at_ns=time.monotonic_ns(),
        slot=0,
        image=np.zeros((8, 8, 4), dtype=np.uint8),
    )
    runtime.buffer.store("screen", frame, timestamp_ns=frame.captured_at_ns)
    window = EVEControlWindow.create(runtime)
    window.show()
    gui.processEvents()

    assert window.tabs.count() == 8
    assert window.screen_view.pixmap() is not None
    assert not window.screen_view.pixmap().isNull()
    assert window.tnn_table.rowCount() == MAX_LOADED_TNN
    assert window.tnn_table.columnCount() == 16
    assert window.loop_graph_view is not None
    assert "debug.jsonl" in window.debug_log_path.text()
    window._set_stable_text(
        window.world_view, "\n".join(f"line {index}" for index in range(200))
    )
    scrollbar = window.world_view.verticalScrollBar()
    scrollbar.setValue(max(1, scrollbar.maximum() // 2))
    previous_scroll = scrollbar.value()
    window._set_stable_text(
        window.world_view,
        "\n".join(f"updated {index}" for index in range(220)),
    )
    assert scrollbar.value() == previous_scroll
    node = register_runtime_tnn(
        runtime.state,
        "frequency-probe",
        lambda _inputs: {},
        outputs=(),
        activate=False,
    )
    node["actual_frequency_hz"] = 12.5
    window.refresh()
    assert window.tnn_table.item(0, 8).text() == "12.5"
    assert not runtime.buffer.capture_running
    assert not runtime.core.running
    assert not runtime.memory.writer_running
    assert all(not value for value in runtime.state["permissions"]["mouse"].values())
    assert all(
        not value for value in runtime.state["permissions"]["keyboard"].values()
    )
    assert not runtime.state["permissions"]["send_text"]
    assert not runtime.state["permissions"]["speak"]

    window.close()
    gui.processEvents()


def test_control_cold_start_pause_emergency_reset_and_clean_stop(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("eve.main._global_escape_pressed", lambda: False)
    runtime = EVEApplication(
        profile="control",
        run_dir=tmp_path,
        input_buffer=synthetic_buffer(),
        use_default_local_models=False,
    )
    try:
        runtime.start(load_smoke_node=False)
        assert runtime.buffer.capture_running
        assert runtime.core.running
        assert runtime.memory.writer_running
        assert "tensor_test_passed" in runtime.state["cuda_status"]

        runtime.core.pause()
        assert runtime.state["paused"]
        runtime.core.resume()
        assert not runtime.state["paused"]
        runtime.emergency("test")
        assert runtime.state["emergency_stop"]
        assert not runtime.state["action_queue"]
        runtime.clear_emergency()
        assert not runtime.state["emergency_stop"]
        assert runtime.state["paused"]
        runtime.core.resume()
        runtime.change_permission("mouse", "move", True)
        runtime.memory.flush()
        assert runtime.memory.search(payload_type="permission_change")
        runtime.state["_cloud_api_key"] = "must-not-be-saved"
    finally:
        runtime.stop()
    assert not runtime.buffer.capture_running
    assert not runtime.core.running
    assert not runtime.memory.writer_running
    snapshot = json.loads(
        (tmp_path / "state_snapshot.json").read_text(encoding="utf-8")
    )
    assert "permissions" not in snapshot
    assert "must-not-be-saved" not in _json_dump(snapshot)
    restarted = EVEApplication(
        profile="control",
        run_dir=tmp_path,
        input_buffer=synthetic_buffer(),
        use_default_local_models=False,
    )
    assert not restarted.state["permissions"]["mouse"]["move"]


def test_llm_protocol_rejects_legacy_or_incomplete_shapes(tmp_path):
    core = CoreLoop(
        InputBuffer(),
        Memorizer(tmp_path / "memory"),
        state=create_runtime_state(),
        log_dir=tmp_path,
    )
    request = {"request_id": "strict", "kind": "user", "message": "hello"}
    legacy = {
        "protocol_version": 2,
        "reply": "old",
        "thinking_summary": "visible",
        "world_update": {},
        "myself_update": {},
        "goodness_update": {},
        "blackboard_updates": [],
        "active_tnn": [],
        "memory_actions": [],
        "training_order": None,
    }
    with pytest.raises(ValueError, match="fields mismatch"):
        core._coerce_llm_result(legacy, request)

    strict = {
        "protocol_version": 2,
        "reply": "current",
        "thinking_summary": "visible",
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
    assert core._coerce_llm_result(strict, request)["reply"] == "current"
    with pytest.raises(TypeError, match="myself_cognition_update"):
        core._coerce_llm_result(
            {**strict, "myself_cognition_update": "invalid"}, request
        )


def test_autonomous_self_update_repeats_without_user_messages(tmp_path, monkeypatch):
    calls = []

    def backend(context):
        calls.append(context)
        count = len(calls)
        return {
            "protocol_version": 2,
            "reply": "",
            "thinking_summary": f"visible self update {count}",
            "world_interpretation_update": {
                "interpretation": {"update_count": count}
            },
            "myself_cognition_update": {"last_self_tick": count},
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

    state = create_runtime_state()
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(
        InputBuffer(),
        memory,
        state=state,
        log_dir=tmp_path,
        local_llm_backend=backend,
    )
    monkeypatch.setattr(core, "_autonomous_interval_s", lambda: 0.05)
    memory.start_writer()
    try:
        core.start()
        wait_until(
            lambda: state["model_status"]["local_llm"]["success_count"] >= 3
        )
        assert state["world"]["interpretation"]["update_count"] >= 3
        assert state["myself"]["cognition"]["last_self_tick"] >= 3
        assert state["model_status"]["local_llm"]["failure_count"] == 0
        assert state["conversation"] == []
        wait_until(
            lambda: state["node_status"]
            .get("self_update_loop", {})
            .get("actual_hz", 0)
            > 0
        )
        assert state["node_status"]["self_update_loop"]["actual_hz"] > 0
        assert state["loop_graph"]["edges"]
        wait_until(
            lambda: '"channel": "llm_protocol_output"'
            in (tmp_path / "debug.jsonl").read_text(encoding="utf-8")
        )
        channels = {
            json.loads(line)["channel"]
            for line in (tmp_path / "debug.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        }
        assert {"llm_protocol_output", "loop_snapshot"} <= channels
    finally:
        core.stop()
        memory.stop_writer()


def test_real_llm_user_request_uses_protocol_v2(tmp_path, monkeypatch):
    state = create_runtime_state()
    memory = Memorizer(tmp_path / "memory")
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path)
    monkeypatch.setattr(
        core,
        "_generate_local_llm",
        lambda context: json.dumps(
            {
                "protocol_version": 2,
                "reply": f"直接回复：{context['user_message']}",
                "thinking_summary": "",
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
            },
            ensure_ascii=False,
        ),
    )
    state["model_status"]["local_llm"]["state"] = "ready"
    memory.start_writer()
    try:
        request = {
            "request_id": "chat_test",
            "kind": "user",
            "message": "你好",
            "requested_at_ns": time.monotonic_ns(),
            "memory_id": None,
        }
        core._process_llm_request(request)
        assert state["conversation"][-1]["reply"] == "直接回复：你好"
        assert state["model_status"]["local_llm"]["state"] == "ready"
        assert "latest_llm_result" in state["blackboard"]
    finally:
        memory.stop_writer()


def test_vlm_stale_result_is_bound_but_not_current(tmp_path):
    started = threading.Event()
    release = threading.Event()
    captured_request = {}

    def backend(request):
        captured_request.update(request)
        started.set()
        release.wait(2.0)
        return "frame analysis"

    buffer = InputBuffer()
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        vlm_backend=backend,
    )
    memory.start_writer()
    try:
        core.start()
        wait_until(lambda: state["model_status"]["vlm"]["state"] == "ready")
        first = ScreenFrame(
            frame_id=1,
            captured_at_ns=time.monotonic_ns(),
            slot=0,
            image=np.zeros((8, 8, 4), dtype=np.uint8),
        )
        buffer.store("screen", first, timestamp_ns=first.captured_at_ns)
        state["visual_result"] = {
            "source": "yolo",
            "model": "test-yolo",
            "reference_frame_id": 1,
            "reference_frame_timestamp_ns": first.captured_at_ns,
            "completed_at_ns": time.monotonic_ns(),
            "detections": [{"class_name": "object"}],
            "detection_count": 1,
        }
        core.submit_visual_request()
        assert started.wait(2.0)
        assert captured_request["runtime_visual_result"]["model"] == "test-yolo"
        second = ScreenFrame(
            frame_id=2,
            captured_at_ns=time.monotonic_ns(),
            slot=1,
            image=np.ones((8, 8, 4), dtype=np.uint8),
        )
        buffer.store("screen", second, timestamp_ns=second.captured_at_ns)
        release.set()
        wait_until(
            lambda: state.get("last_visual_interpretation_result") is not None
        )

        result = state["last_visual_interpretation_result"]
        assert result["reference_frame_id"] == 1
        assert result["status"] == "stale"
        assert (
            result["reviewed_runtime_visual"][
                "detection_count"
            ]
            == 1
        )
        assert state["visual_interpretation_result"] is None
        assert state["visual_result"]["source"] == "yolo"
    finally:
        release.set()
        core.stop()
        memory.stop_writer()


def test_yolo_runtime_visual_writes_blackboard_without_vlm(tmp_path):
    buffer = InputBuffer()
    memory = Memorizer(tmp_path / "memory")
    state = create_runtime_state()

    def backend(_image):
        return {
            "detections": [
                {
                    "bbox": [1.0, 2.0, 7.0, 8.0],
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "object",
                }
            ],
            "inference_time_ms": 1.5,
        }

    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        runtime_visual_backend=backend,
    )
    memory.start_writer()
    try:
        core.start()
        wait_until(lambda: state["model_status"]["yolo"]["state"] == "ready")
        frame = ScreenFrame(
            frame_id=7,
            captured_at_ns=time.monotonic_ns(),
            slot=0,
            image=np.zeros((8, 8, 4), dtype=np.uint8),
        )
        buffer.store("screen", frame, timestamp_ns=frame.captured_at_ns)
        wait_until(lambda: state["visual_result"] is not None)

        result = state["visual_result"]
        assert result["source"] == "yolo"
        assert result["reference_frame_id"] == 7
        assert result["detection_count"] == 1
        assert state["blackboard"]["current_visual_result"]["producer"] == "yolo"
        assert state["visual_interpretation_result"] is None
        visual = state["world"]["perception"]["visual"]
        assert visual["reference_frame_id"] == 7
        assert visual["detections"][0]["class_name"] == "object"

        request_id = core.submit_runtime_visual_analysis()
        wait_until(
            lambda: state["visual_result"].get("request_id") == request_id
        )
        requested = state["visual_result"]
        assert requested["reference_frame_id"] == 7
        assert requested["requested_at_ns"] <= requested["completed_at_ns"]
        memory.flush()
        result_ids = memory.search(payload_type="runtime_visual_result")
        assert result_ids
        assert memory.read(result_ids[-1])["request_id"] == request_id
    finally:
        core.stop()
        memory.stop_writer()


def test_visual_tnn_result_has_frame_binding_and_io_summaries(tmp_path):
    import torch

    state = create_runtime_state()
    state["cold_started"] = True
    buffer = InputBuffer()
    frame = ScreenFrame(
        frame_id=17,
        captured_at_ns=time.monotonic_ns(),
        slot=0,
        image=np.zeros((4, 6, 4), dtype=np.uint8),
    )
    buffer.store("screen", frame, timestamp_ns=frame.captured_at_ns)
    model = torch.nn.Linear(2, 3)
    register_runtime_tnn(
        state,
        "visual_tnn",
        lambda _inputs: {"detections": [{"class_name": "shape"}]},
        inputs={"screen": "state:screen"},
        outputs=("detections",),
        run_frequency_hz=10.0,
        model=model,
    )
    core = CoreLoop(buffer, Memorizer(tmp_path / "memory"), state=state)

    core.step(now_ns=time.monotonic_ns())
    wait_until(lambda: core._tnn_futures["visual_tnn"].done())
    core.step(now_ns=time.monotonic_ns())

    result = state["visual_result"]
    node = state["loaded_tnn"]["visual_tnn"]
    assert result["model"] == "visual_tnn"
    assert result["reference_frame_id"] == 17
    assert result["reference_frame_timestamp_ns"] == frame.captured_at_ns
    assert result["requested_at_ns"] <= result["completed_at_ns"]
    assert node["last_input_summary"]["screen"]["shape"] == [4, 6, 4]
    assert node["last_output_summary"]["detections"]["length"] == 1
    assert node["last_output_at_ns"] == result["completed_at_ns"]
    core._update_resources(time.monotonic_ns())
    expected_bytes = sum(
        value.nelement() * value.element_size()
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
    )
    assert state["resource_status"]["tnn_summary"]["total_memory"] == (
        expected_bytes
    )
    core.stop()


def test_atomic_permissions_and_tnn_five_slot_limit():
    state = create_runtime_state(output_mode="mock")
    state["cold_started"] = True
    state["permissions"] = default_permissions(False)
    move_click = {
        "action_type": "mouse",
        "payload": {"action": "click", "x": 10, "y": 20, "button": "left"},
    }
    state["permissions"]["mouse"]["move"] = True
    decision = check_action_permission(state, move_click)
    assert not decision["allowed"]
    assert decision["blocked_atoms"] == ["mouse.left_click"]

    hotkey = {
        "action_type": "keyboard",
        "payload": {"action": "hotkey", "keys": ["CTRL", "SHIFT", "S"]},
    }
    state["permissions"]["keyboard"]["CTRL"] = True
    state["permissions"]["keyboard"]["SHIFT"] = True
    assert not check_action_permission(state, hotkey)["allowed"]
    state["permissions"]["keyboard"]["S"] = True
    assert check_action_permission(state, hotkey)["allowed"]

    unicode_text = {
        "action_type": "keyboard",
        "payload": {"action": "write", "method": "unicode", "text": "你好"},
    }
    assert not check_action_permission(state, unicode_text)["allowed"]
    for atom in ("CTRL", "V"):
        state["permissions"]["keyboard"][atom] = True
    state["permissions"]["send_text"] = True
    assert check_action_permission(state, unicode_text)["allowed"]

    tnn_state = create_runtime_state()
    for index in range(MAX_LOADED_TNN):
        register_runtime_tnn(
            tnn_state,
            f"tnn-{index}",
            lambda _inputs: {},
            activate=False,
        )
    before = set(tnn_state["loaded_tnn"])
    try:
        register_runtime_tnn(
            tnn_state, "sixth", lambda _inputs: {}, activate=False
        )
    except RuntimeError as exc:
        assert "maximum loaded TNN count is 5" in str(exc)
    else:
        raise AssertionError("sixth TNN was not rejected")
    assert set(tnn_state["loaded_tnn"]) == before


def _json_dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=repr)


def test_keyboard_activity_excludes_window_metadata_and_memory_views(tmp_path):
    buffer = InputBuffer(
        profile="control",
        capture_options={
            "screen_mode": "synthetic",
            "cursor_mode": "synthetic",
            "keyboard_mode": "synthetic",
            "synthetic_active_key_count": 1,
        },
    )
    try:
        buffer.start_capture()
        wait_until(lambda: buffer.latest("keyboard_activity") is not None)
        keyboard = buffer.latest("keyboard_activity").value
        assert keyboard["active"] and keyboard["active_key_count"] == 1
        assert buffer.latest("active_window") is None
        assert "active_window" not in buffer.get_state()
        assert buffer.human_takeover_until_ns > time.monotonic_ns()
    finally:
        buffer.close()

    memory = Memorizer(tmp_path / "memory-views")
    ids = [memory.create({"index": index}, "view-item") for index in range(12)]
    memory.load_to_mtm(ids[0])
    memory.persist_to_ltm(ids[0])
    assert ids[0] in memory.stm
    assert ids[0] in memory.mtm
    assert ids[0] in memory.ltm
    assert memory.counts()["catalog"] == 12
