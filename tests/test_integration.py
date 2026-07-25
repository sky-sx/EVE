"""Integration tests combining multiple EVE modules."""

import time
import json
from pathlib import Path

import torch

from eve.state import (
    RuntimeState, ActionCandidate, ActionKind, OutputMode,
    SafegateResult, OutputResult, WorldState, MyselfState, Blackboard,
)
from eve.core.safegate import check, emergency_stop, reset_emergency
from eve.core.runtime_state import RuntimeStateManager
from eve.config import EVEConfig
from eve.core.tnn_base import TNNDescriptor, DummyTNN
from eve.core.tnn_store import TNNStore
from eve.core.graph import TNNGraph
from eve.core.hormones import HormoneManager
from eve.memory.memorizer import Memorizer
from eve.memory.catalog import Catalog
from eve.memory.indexes import IndexManager
from eve.memory.event import EventManager
from eve.memory.retrieval import Retriever, RetrievalRequest
from eve.input.buffer import InputBuffer
from eve.input.schemas import TimedSample
from eve.input.capture import CaptureManager


# ── Full Action Flow ──────────────────────────────────────────

def test_full_action_flow() -> None:
    """RuntimeState → ActionCandidate → Safegate → Output→ Result"""
    state = RuntimeState(
        cold_started=True,
        output_mode=OutputMode.MOCK,
        mouse_allowed=True,
    )
    action = ActionCandidate(
        action_id="act_1",
        kind=ActionKind.MOUSE,
        payload={"x": 100, "y": 200},
    )
    result = check(state, action)
    assert result.allowed is True
    assert result.reason == "ok"

    # Blocked action
    state2 = RuntimeState(cold_started=False)
    action2 = ActionCandidate(action_id="act_2", kind=ActionKind.MOUSE, payload={})
    result2 = check(state2, action2)
    assert result2.allowed is False
    assert result2.reason == "not_cold_started"


# ── Input Buffer ──────────────────────────────────────────────

def test_input_to_state() -> None:
    buf = InputBuffer()
    sample = buf.store("cursor", (100, 200))
    assert isinstance(sample, TimedSample)
    assert sample.kind == "cursor"
    assert sample.value == (100, 200)

    latest = buf.latest("cursor")
    assert latest is not None
    assert latest.value == (100, 200)

    # Range within time window
    now = time.monotonic_ns()
    rng = buf.range("cursor", now - 1_000_000_000, now + 1_000_000_000)
    assert len(rng) >= 1
    assert rng[0].value == (100, 200)


# ── TNN Creation → Save → Load Roundtrip ──────────────────────

def test_tnn_creation_save_load(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="int_tnn", outputs=["out"])
    tnn = DummyTNN(desc, input_dim=64, hidden_dim=32, output_dim=16)
    x = torch.randn(1, 64)
    result1 = tnn({"out": x})["out"]

    save_dir = tmp_path / "int_save"
    tnn.save(save_dir)

    store = TNNStore(tmp_path / "int_store")
    store.register(
        tnn_id="int_tnn",
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )

    loaded = store.load_tnn("int_tnn", device="cpu")
    assert loaded is not None
    loaded.eval()
    with torch.no_grad():
        result2 = loaded({"out": x})["out"]
    assert torch.allclose(result1, result2)


# ── Memory Create → Read → Delete ─────────────────────────────

def test_memory_create_read_delete(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    mid = m.create({"answer": 42}, payload_type="json")
    data = m.read(mid)
    assert data == {"answer": 42}
    m.delete(mid)
    assert m.read(mid) is None


# ── Graph Node Lifecycle ──────────────────────────────────────

def test_graph_node_lifecycle(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="lifecycle", outputs=["out"], run_frequency_hz=10.0)
    tnn = DummyTNN(desc)
    save_dir = tmp_path / "src_lc"
    tnn.save(save_dir)
    store = TNNStore(tmp_path / "store_lc")
    store.register("lifecycle", desc, str(save_dir / "weights.pt"), "1")

    graph = TNNGraph(store)
    assert graph.add_node("lifecycle") is True
    assert graph.has_node("lifecycle") is True

    # Run node
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 64, device=device)
    out = graph.run_node("lifecycle", {"out": x})
    assert out is not None

    # Pause
    graph.pause_node("lifecycle")
    node = graph.get_node("lifecycle")
    assert node.status == "paused"

    # Resume
    graph.resume_node("lifecycle")
    node2 = graph.get_node("lifecycle")
    assert node2.status == "active"

    # Remove
    assert graph.remove_node("lifecycle") is True
    assert graph.has_node("lifecycle") is False


# ── Hormone Affects Interval ──────────────────────────────────

def test_hormone_affects_interval() -> None:
    hm = HormoneManager()
    interval_default = hm.compute_llm_interval(min_s=10.0, max_s=20.0)

    # Update hormones toward high stress
    hm.apply_event("failure")
    hm.apply_event("failure")
    hm.apply_event("resource_pressure")
    interval_stressed = hm.compute_llm_interval(min_s=10.0, max_s=20.0)

    # Stressed interval should be shorter
    assert interval_stressed < interval_default


# ── Stop and Restore ──────────────────────────────────────────

def test_stop_and_restore(tmp_path: Path) -> None:
    config = EVEConfig.default()
    rsm = RuntimeStateManager(config)
    rsm.world.scene = "browser window"
    rsm.world.sub_scene = "github"
    rsm.myself.what_im_thinking = "I should help the user"
    rsm.myself.current_task = "browsing"

    snapshot_dir = tmp_path / "snap"
    rsm.save_snapshot(snapshot_dir)

    # Load into a new manager
    config2 = EVEConfig.default()
    rsm2 = RuntimeStateManager(config2)
    ok = rsm2.load_snapshot(snapshot_dir)
    assert ok is True
    assert rsm2.world.scene == "browser window"
    assert rsm2.world.sub_scene == "github"
    assert rsm2.myself.what_im_thinking == "I should help the user"
    assert rsm2.myself.current_task == "browsing"
