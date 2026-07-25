"""Tests for TNNGraph, TNNOutputCache, GraphTrace."""

import time
import torch
from pathlib import Path

from eve.core.graph import (
    TNNGraph, TNNOutputCache, CachedOutput,
    GraphNode, GraphEdge, GraphTrace, GraphStateSnapshot,
)
from eve.core.tnn_base import TNNDescriptor, SourceRef, DummyTNN
from eve.core.tnn_store import TNNStore


def _setup_store_and_register(
    tmp_path: Path, tnn_id: str, outputs: list[str] | None = None,
) -> TNNStore:
    """Helper: save a DummyTNN, register in store, return store."""
    outputs = outputs or ["out"]
    desc = TNNDescriptor(tnn_id=tnn_id, outputs=outputs, run_frequency_hz=10.0)
    tnn = DummyTNN(desc)
    save_dir = tmp_path / f"src_{tnn_id}"
    tnn.save(save_dir)
    store = TNNStore(tmp_path / "store")
    store.register(
        tnn_id=tnn_id,
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )
    return store


# ── TNNOutputCache ────────────────────────────────────────────

def test_output_cache_ttl() -> None:
    cache = TNNOutputCache()
    cache.put("tnn1", "out", torch.tensor([1.0]), ttl_ns=1_000_000)  # 1ms TTL
    val = cache.get("tnn1", "out")
    assert val is not None

    # Wait for expiry (100ms > 1ms TTL, ample margin)
    time.sleep(0.15)
    expired = cache.get("tnn1", "out")
    assert expired is None


def test_output_cache_invalidate() -> None:
    cache = TNNOutputCache()
    cache.put("tnn1", "out_a", torch.tensor([1.0]))
    cache.put("tnn1", "out_b", torch.tensor([2.0]))
    cache.put("tnn2", "out_x", torch.tensor([3.0]))

    cache.invalidate("tnn1")
    assert cache.get("tnn1", "out_a") is None
    assert cache.get("tnn1", "out_b") is None
    # tnn2 should still be there
    assert cache.get("tnn2", "out_x") is not None


# ── Node / Edge Management ────────────────────────────────────

def test_add_remove_node(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "node_a")
    graph = TNNGraph(store)

    assert graph.add_node("node_a") is True
    assert graph.has_node("node_a") is True
    assert graph.get_node("node_a") is not None

    assert graph.remove_node("node_a") is True
    assert graph.has_node("node_a") is False
    assert graph.remove_node("node_a") is False  # already removed


def test_add_edge(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "src_node")
    _setup_store_and_register(tmp_path, "dst_node")
    # Rebuild store so both are in it
    desc_a = TNNDescriptor(tnn_id="src_node", outputs=["out"], run_frequency_hz=10.0)
    desc_b = TNNDescriptor(tnn_id="dst_node", outputs=["out"], run_frequency_hz=10.0)
    tnn_a = DummyTNN(desc_a)
    tnn_b = DummyTNN(desc_b)
    d_a = tmp_path / "src_src_node"
    d_b = tmp_path / "src_dst_node"
    tnn_a.save(d_a)
    tnn_b.save(d_b)
    store2 = TNNStore(tmp_path / "store2")
    store2.register("src_node", desc_a, str(d_a / "weights.pt"), "1")
    store2.register("dst_node", desc_b, str(d_b / "weights.pt"), "1")

    graph = TNNGraph(store2)
    graph.add_node("src_node")
    graph.add_node("dst_node")

    graph.add_edge("src_node", "dst_node", "out", "tnn:src_node.out")
    downstream = graph.get_downstream("src_node")
    assert "dst_node" in downstream
    upstream = graph.get_upstream("dst_node")
    assert "src_node" in upstream


# ── Build Edges from Descriptors ──────────────────────────────

def test_build_edges_from_descriptors(tmp_path: Path) -> None:
    # detector has output "bbox"
    desc_det = TNNDescriptor(
        tnn_id="detector", outputs=["bbox"], run_frequency_hz=10.0,
    )
    # tracker has input SourceRef pointing to detector.bbox
    desc_trk = TNNDescriptor(
        tnn_id="tracker", outputs=["track"],
        inputs=[SourceRef(source_type="tnn_output", source_id="tnn:detector.bbox")],
        run_frequency_hz=10.0,
    )
    tnn_d = DummyTNN(desc_det)
    tnn_t = DummyTNN(desc_trk)
    d_d = tmp_path / "src_det"
    d_t = tmp_path / "src_trk"
    tnn_d.save(d_d)
    tnn_t.save(d_t)

    store = TNNStore(tmp_path / "store")
    store.register("detector", desc_det, str(d_d / "weights.pt"), "1")
    store.register("tracker", desc_trk, str(d_t / "weights.pt"), "1")

    graph = TNNGraph(store)
    graph.add_node("detector")
    graph.add_node("tracker")
    graph.build_edges_from_descriptors()

    downstream = graph.get_downstream("detector")
    assert "tracker" in downstream
    upstream = graph.get_upstream("tracker")
    assert "detector" in upstream


# ── Sync Active Set ───────────────────────────────────────────

def test_sync_active_set(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "keep_me")
    _setup_store_and_register(tmp_path, "load_me")
    # Rebuild store with both
    desc_k = TNNDescriptor(tnn_id="keep_me", outputs=["out"], run_frequency_hz=10.0)
    desc_l = TNNDescriptor(tnn_id="load_me", outputs=["out"], run_frequency_hz=10.0)
    tnn_k = DummyTNN(desc_k)
    tnn_l = DummyTNN(desc_l)
    d_k = tmp_path / "src_keep"
    d_l = tmp_path / "src_load"
    tnn_k.save(d_k)
    tnn_l.save(d_l)
    store2 = TNNStore(tmp_path / "store_sync")
    store2.register("keep_me", desc_k, str(d_k / "weights.pt"), "1")
    store2.register("load_me", desc_l, str(d_l / "weights.pt"), "1")

    graph = TNNGraph(store2)
    graph.add_node("keep_me")
    assert graph.has_node("keep_me")

    diff = graph.sync_active_set(["keep_me", "load_me"])
    assert "load_me" in diff["loaded"]
    assert "keep_me" in diff["kept"]
    assert len(diff["unloaded"]) == 0


# ── Schedule ──────────────────────────────────────────────────

def test_schedule_by_frequency(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "sched_node")
    graph = TNNGraph(store)
    graph.add_node("sched_node")

    now = 0
    scheduled = graph.schedule(now_ns=now)
    # Node has frequency 10 Hz → interval = 100_000_000 ns
    assert "sched_node" in scheduled

    # Second call at the same time should not schedule again
    scheduled2 = graph.schedule(now_ns=now + 50_000_000)
    assert "sched_node" not in scheduled2

    # After full interval, should schedule again
    scheduled3 = graph.schedule(now_ns=now + 100_000_000)
    assert "sched_node" in scheduled3


# ── Run Node / Cached Output ──────────────────────────────────

def test_run_node_caches_output(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "runner")
    graph = TNNGraph(store)
    graph.add_node("runner")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 64, device=device)
    output = graph.run_node("runner", {"out": x})
    assert output is not None
    assert "out" in output

    # Output should be cached
    cached = graph.get_output("runner", "out")
    assert cached is not None


# ── Pause / Resume ────────────────────────────────────────────

def test_pause_resume_node(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "pause_node")
    graph = TNNGraph(store)
    graph.add_node("pause_node")

    graph.pause_node("pause_node")
    node = graph.get_node("pause_node")
    assert node is not None
    assert node.status == "paused"

    # Schedule should not include paused node
    scheduled = graph.schedule(now_ns=0)
    assert "pause_node" not in scheduled

    graph.resume_node("pause_node")
    node2 = graph.get_node("pause_node")
    assert node2 is not None
    assert node2.status == "active"

    scheduled2 = graph.schedule(now_ns=0)
    assert "pause_node" in scheduled2


# ── GraphTrace ────────────────────────────────────────────────

def test_graph_trace_record(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "trace_node")
    graph = TNNGraph(store)
    graph.add_node("trace_node")

    tracer = GraphTrace()
    tracer.record(graph)

    recent = tracer.get_recent(5)
    assert len(recent) == 1
    assert isinstance(recent[0], GraphStateSnapshot)
    assert "trace_node" in recent[0].active_nodes


# ── Stats ─────────────────────────────────────────────────────

def test_graph_stats(tmp_path: Path) -> None:
    store = _setup_store_and_register(tmp_path, "stat_a")
    _setup_store_and_register(tmp_path, "stat_b")
    desc_a = TNNDescriptor(tnn_id="stat_a", outputs=["out"], run_frequency_hz=10.0)
    desc_b = TNNDescriptor(tnn_id="stat_b", outputs=["out"], run_frequency_hz=10.0)
    tnn_a = DummyTNN(desc_a)
    tnn_b = DummyTNN(desc_b)
    d_a = tmp_path / "src_sa"
    d_b = tmp_path / "src_sb"
    tnn_a.save(d_a)
    tnn_b.save(d_b)
    store2 = TNNStore(tmp_path / "store_stats")
    store2.register("stat_a", desc_a, str(d_a / "weights.pt"), "1")
    store2.register("stat_b", desc_b, str(d_b / "weights.pt"), "1")

    graph = TNNGraph(store2)
    graph.add_node("stat_a")
    graph.add_node("stat_b")
    graph.add_edge("stat_a", "stat_b", "out", "tnn:stat_a.out")

    s = graph.stats()
    assert s["node_count"] == 2
    assert s["edge_count"] == 1
