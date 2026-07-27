from __future__ import annotations

import json
import threading
import time

from eve.core.loop import CoreLoop
from eve.core.tnn import (
    SourceRef,
    TNNDescriptor,
    load_tnn,
    run_node,
)
from eve.input.buffer import InputBuffer
from eve.main import EVEApplication
from eve.memory.memorizer import Memorizer
from eve.state import OutputMode, RuntimeState


class UpstreamNode:
    descriptor = TNNDescriptor(
        tnn_id="upstream",
        inputs={"cursor": SourceRef("state:cursor")},
        outputs=("target",),
        run_frequency_hz=10.0,
    )

    def run(self, inputs):
        return {"target": {"x": inputs["cursor"][0], "y": inputs["cursor"][1]}}


class ActionNode:
    descriptor = TNNDescriptor(
        tnn_id="action",
        inputs={"target": SourceRef("tnn:upstream.target")},
        outputs=("action_candidate",),
        run_frequency_hz=10.0,
        action_output="action_candidate",
    )

    def run(self, inputs):
        return {
            "action_candidate": {
                "action_id": "only-once",
                "kind": "mouse",
                "payload": {"action": "moveTo", **inputs["target"]},
            }
        }


class BrokenNode:
    descriptor = TNNDescriptor(
        tnn_id="broken",
        inputs={"cursor": SourceRef("state:cursor")},
        outputs=("never",),
        run_frequency_hz=10.0,
    )

    def run(self, inputs):
        raise RuntimeError("inference exploded")


def test_tnn_source_ref_to_action_and_consume_once(tmp_path):
    state = RuntimeState(
        cold_started=True,
        output_mode=OutputMode.MOCK,
        mouse_allowed=True,
    )
    buffer = InputBuffer()
    buffer.store("cursor", (4, 9))
    memory = Memorizer(tmp_path / "memory")
    load_tnn(state, UpstreamNode())
    load_tnn(state, ActionNode())
    now = time.monotonic_ns()

    assert run_node(state, buffer, "upstream", now) == {
        "target": {"x": 4, "y": 9}
    }
    run_node(state, buffer, "action", now)
    loop = CoreLoop(state, buffer, memory, log_dir=tmp_path)
    first = loop.step(now + 200_000_000)
    second = loop.step(now + 400_000_000)

    assert len(first) == 1
    assert first[0].simulated and not first[0].executed
    assert second == []
    assert state.consumed_action_ids == {"only-once"}
    assert len(state.memory_ids) == 3
    memory.flush()
    assert {memory.get_unit(mid).payload_type for mid in state.memory_ids} == {
        "action_candidate",
        "safegate_result",
        "output_result",
    }
    assert all(memory.read(mid) is not None for mid in state.memory_ids)
    memory.stop_writer()


def test_tnn_exception_is_structured_and_pauses_node(tmp_path):
    state = RuntimeState(cold_started=True, output_mode=OutputMode.MOCK)
    buffer = InputBuffer()
    buffer.store("cursor", (1, 2))
    load_tnn(state, BrokenNode())
    memory = Memorizer(tmp_path / "memory")
    loop = CoreLoop(state, buffer, memory, log_dir=tmp_path)

    assert loop.step(time.monotonic_ns()) == []
    assert state.latest_error is not None
    assert state.latest_error.loop_node == "tnn:broken"
    assert state.latest_error.exception_type == "RuntimeError"
    assert state.latest_error.recovery_action == "node_paused_no_output"
    assert "broken" not in state.myself.active_tnn
    records = [
        json.loads(line)
        for line in (tmp_path / "eve.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event"] == "runtime_error" for item in records)
    memory.stop_writer()


def test_formal_entry_starts_and_stops_without_real_output(tmp_path):
    before = {thread.name for thread in threading.enumerate()}
    app = EVEApplication(mode=OutputMode.MOCK, run_dir=tmp_path)
    app.start()
    time.sleep(0.25)
    app.stop()

    after = {thread.name for thread in threading.enumerate()}
    assert app.state.latest_output is not None
    assert app.state.latest_output.simulated
    assert not app.state.latest_output.executed
    assert not app.core.running and not app.capture.running
    assert not ({"eve-core", "eve-capture"} & (after - before))
