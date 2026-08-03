from __future__ import annotations

import json
import threading
import time

from eve.core.loop import CoreLoop, create_runtime_state, register_runtime_tnn
from eve.input.buffer import InputBuffer
from eve.main import EVEApplication
from eve.memory.memorizer import Memorizer


def test_tnn_reference_to_action_is_consumed_once(tmp_path):
    state = create_runtime_state(output_mode="mock", allow_mock_actions=True)
    state["cold_started"] = True
    buffer = InputBuffer()
    buffer.store("cursor", (4, 9))
    memory = Memorizer(tmp_path / "memory")
    register_runtime_tnn(
        state,
        "a_upstream",
        lambda inputs: {
            "target": {"x": inputs["cursor"][0], "y": inputs["cursor"][1]}
        },
        inputs={"cursor": "state:cursor"},
        outputs=("target",),
        run_frequency_hz=10,
    )
    register_runtime_tnn(
        state,
        "b_action",
        lambda inputs: {
            "action_candidate": {
                "candidate_id": "only-once",
                "action_type": "mouse",
                "payload": {"action": "moveTo", **inputs["target"]},
            }
        },
        inputs={"target": "tnn:a_upstream.target"},
        outputs=("action_candidate",),
        run_frequency_hz=10,
        action_output="action_candidate",
    )
    loop = CoreLoop(buffer, memory, state=state, log_dir=tmp_path)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and state["latest_output"] is None:
        loop.step()
        time.sleep(0.01)

    assert state["latest_output"]["candidate_id"] == "only-once"
    assert state["latest_output"]["simulated"]
    assert not state["latest_output"]["executed"]
    assert state["consumed_action_ids"] == {"only-once"}
    assert len(state["memory_ids"]) == 3
    memory.flush()
    assert {
            memory.get_record(item).payload_type for item in state["memory_ids"]
    } == {"action_candidate", "permission_result", "output_result"}
    loop.stop()
    memory.stop_writer()


def test_tnn_exception_is_structured_and_visible(tmp_path):
    state = create_runtime_state()
    state["cold_started"] = True
    buffer = InputBuffer()
    buffer.store("cursor", (1, 2))
    memory = Memorizer(tmp_path / "memory")

    def broken(_inputs):
        raise RuntimeError("inference exploded")

    register_runtime_tnn(
        state,
        "broken",
        broken,
        inputs={"cursor": "state:cursor"},
        outputs=("never",),
        run_frequency_hz=10,
    )
    loop = CoreLoop(buffer, memory, state=state, log_dir=tmp_path)

    loop.step(time.monotonic_ns())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and state["tnn_status"]["broken"] != "failed":
        loop.step()
        time.sleep(0.01)
    assert state["latest_error"]["loop_node"] == "tnn:broken"
    assert state["latest_error"]["exception_type"] == "RuntimeError"
    assert state["tnn_status"]["broken"] == "failed"
    records = [
        json.loads(line)
        for line in (tmp_path / "eve.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event"] == "runtime_error" for item in records)
    loop.stop()
    memory.stop_writer()


def test_formal_entry_starts_child_capture_and_stops_cleanly(tmp_path):
    before_threads = {thread.name for thread in threading.enumerate()}
    app = EVEApplication(
        profile="smoke",
        run_dir=tmp_path,
        memory_dir=tmp_path / "memory",
        allow_mock_actions=True,
    )
    app.start(load_smoke_node=True)
    capture_pid = app.buffer.capture_process_id
    time.sleep(0.25)
    app.stop()

    after_threads = {thread.name for thread in threading.enumerate()}
    assert capture_pid is not None
    assert app.state["latest_output"]["simulated"]
    assert not app.state["latest_output"]["executed"]
    assert not app.core.running and not app.buffer.capture_running
    assert not ({"eve-core", "eve-input-ipc"} & (after_threads - before_threads))
