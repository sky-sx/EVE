from __future__ import annotations

import threading
import time
import json

import numpy as np

from eve.core.loop import (
    DEFAULT_LOCAL_LLM_PATH,
    DEFAULT_VLM_PATH,
    MAX_LOADED_TNN,
    CoreLoop,
    create_runtime_state,
    register_runtime_tnn,
)
from eve.core.safegate import check, default_permissions
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


def test_local_models_have_repo_defaults_and_legacy_empty_snapshot_is_migrated(
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
    assert runtime.state["model_status"]["local_llm"]["state"] == "configured"
    assert runtime.state["model_status"]["vlm"]["state"] == "configured"

    snapshot = tmp_path / "legacy_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "model_config": {
                    "local_llm_path": "",
                    "vlm_path": "",
                    "cloud_enabled": True,
                }
            }
        ),
        encoding="utf-8",
    )
    assert runtime.core.load_snapshot(snapshot)
    assert runtime.state["model_config"]["local_llm_path"] == (
        DEFAULT_LOCAL_LLM_PATH
    )
    assert runtime.state["model_config"]["vlm_path"] == DEFAULT_VLM_PATH
    assert runtime.state["model_config"]["cloud_enabled"] is True


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
    )
    assert not restarted.state["permissions"]["mouse"]["move"]


def test_llm_queue_valid_json_updates_and_invalid_result_is_atomic(tmp_path):
    calls = []

    def backend(context):
        calls.append(context)
        if len(calls) == 1:
            return {
                "reply": "ok",
                "thinking_summary": "visible summary",
                "world_update": {"room": "desktop"},
                "myself_update": {"current_task": "test"},
                "blackboard_updates": [{"key": "answer", "value": 42}],
                "active_tnn": [],
                "memory_candidates": [{"candidate": "remember"}],
            }
        return {
            "reply": "bad",
            "thinking_summary": "bad",
            "world_update": {"must_not_apply": True},
            "myself_update": {},
            "blackboard_updates": [{"value": "missing key"}],
            "active_tnn": [],
            "memory_candidates": [],
        }

    state = create_runtime_state()
    memory = Memorizer(tmp_path / "memory")
    buffer = InputBuffer()
    core = CoreLoop(
        buffer,
        memory,
        state=state,
        log_dir=tmp_path,
        local_llm_backend=backend,
    )
    memory.start_writer()
    try:
        core.start()
        wait_until(lambda: state["model_status"]["local_llm"]["state"] == "ready")
        core.submit_user_message("hello")
        wait_until(lambda: len(state["conversation"]) == 1)
        assert state["world"]["room"] == "desktop"
        assert state["blackboard"]["answer"]["value"] == 42

        before_world = dict(state["world"])
        core.submit_user_message("invalid")
        wait_until(lambda: state["model_status"]["local_llm"]["state"] == "error")
        assert state["world"] == before_world
        assert "must_not_apply" not in state["world"]
    finally:
        core.stop()
        memory.stop_writer()


def test_vlm_stale_result_is_bound_but_not_current(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def backend(_request):
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
        core.submit_visual_request()
        assert started.wait(2.0)
        second = ScreenFrame(
            frame_id=2,
            captured_at_ns=time.monotonic_ns(),
            slot=1,
            image=np.ones((8, 8, 4), dtype=np.uint8),
        )
        buffer.store("screen", second, timestamp_ns=second.captured_at_ns)
        release.set()
        wait_until(lambda: state.get("last_visual_result") is not None)

        assert state["last_visual_result"]["reference_frame_id"] == 1
        assert state["last_visual_result"]["status"] == "stale"
        assert state["visual_result"] is None
    finally:
        release.set()
        core.stop()
        memory.stop_writer()


def test_atomic_permissions_and_tnn_five_slot_limit():
    state = create_runtime_state(output_mode="mock")
    state["cold_started"] = True
    state["permissions"] = default_permissions(False)
    move_click = {
        "action_type": "mouse",
        "payload": {"action": "click", "x": 10, "y": 20, "button": "left"},
    }
    state["permissions"]["mouse"]["move"] = True
    decision = check(state, move_click)
    assert not decision["allowed"]
    assert decision["blocked_atoms"] == ["mouse.left_click"]

    hotkey = {
        "action_type": "keyboard",
        "payload": {"action": "hotkey", "keys": ["CTRL", "SHIFT", "S"]},
    }
    state["permissions"]["keyboard"]["CTRL"] = True
    state["permissions"]["keyboard"]["SHIFT"] = True
    assert not check(state, hotkey)["allowed"]
    state["permissions"]["keyboard"]["S"] = True
    assert check(state, hotkey)["allowed"]

    unicode_text = {
        "action_type": "keyboard",
        "payload": {"action": "write", "method": "unicode", "text": "你好"},
    }
    assert not check(state, unicode_text)["allowed"]
    for atom in ("CTRL", "V"):
        state["permissions"]["keyboard"][atom] = True
    state["permissions"]["send_text"] = True
    assert check(state, unicode_text)["allowed"]

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


def test_keyboard_activity_active_window_and_memory_review(tmp_path):
    buffer = InputBuffer(
        profile="control",
        capture_options={
            "screen_mode": "synthetic",
            "cursor_mode": "synthetic",
            "keyboard_mode": "synthetic",
            "window_mode": "synthetic",
            "synthetic_active_key_count": 1,
            "synthetic_window_title": "EVE Test Window",
            "synthetic_window_process": "eve-test.exe",
        },
    )
    try:
        buffer.start_capture()
        wait_until(lambda: buffer.latest("keyboard_activity") is not None)
        wait_until(lambda: buffer.latest("active_window") is not None)
        keyboard = buffer.latest("keyboard_activity").value
        window = buffer.latest("active_window").value
        assert keyboard["active"] and keyboard["active_key_count"] == 1
        assert window["title"] == "EVE Test Window"
        assert window["process"] == "eve-test.exe"
        assert buffer.human_takeover_until_ns > time.monotonic_ns()
    finally:
        buffer.close()

    memory = Memorizer(tmp_path / "review-memory")
    for index in range(12):
        memory.create({"index": index}, "review-item")
    assert memory.force_review()
    wait_until(lambda: memory.review_status()["state"] == "completed")
    review = memory.review_status()
    assert review["processed"] == 12
    assert review["remaining"] == 0
    assert review["eta_s"] == 0.0
    assert memory.counts()["mtm"] == 12
