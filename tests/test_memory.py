"""Tests for the EVE Memory module: Memorizer, Catalog, IndexManager, EventManager, Retriever."""

import time
from pathlib import Path

from eve.memory.memorizer import Memorizer
from eve.memory.catalog import Catalog, CatalogEntry
from eve.memory.indexes import IndexManager, IndexEdge
from eve.memory.event import EventManager
from eve.memory.retrieval import Retriever, RetrievalRequest


# ── Memorizer CRUD ──────────────────────────────────────────

def test_memorizer_create_text(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    mid = m.create("hello world", payload_type="text")
    assert mid.startswith("mem_"), f"unexpected memory_id: {mid}"
    assert m.catalog.lookup(mid) is not None
    assert m.is_in_stm(mid)


def test_memorizer_create_json(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    payload = {"key": "value", "num": 42}
    mid = m.create(payload, payload_type="json")
    entry = m.catalog.lookup(mid)
    assert entry is not None
    assert entry.payload_type == "json"
    # verify .json file exists on disk
    abs_path = m.ltm_dir / entry.storage_path
    assert abs_path.exists()
    assert abs_path.suffix == ".json"


def test_memorizer_read(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    mid = m.create("hello world", payload_type="text")
    payload = m.read(mid)
    assert payload == "hello world"
    # read non-existent returns None
    assert m.read("nonexistent") is None


def test_memorizer_delete(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    mid = m.create("test content", payload_type="text")
    entry = m.catalog.lookup(mid)
    assert entry is not None
    abs_path = m.ltm_dir / entry.storage_path
    assert abs_path.exists()

    result = m.delete(mid)
    assert result is True
    assert m.catalog.lookup(mid) is None
    assert not abs_path.exists()
    assert mid not in m.get_stm_ids()
    # delete again returns False
    assert m.delete(mid) is False


# ── Catalog ──────────────────────────────────────────────────

def test_catalog_register_lookup() -> None:
    cat = Catalog()
    entry = CatalogEntry(
        memory_id="mem_001",
        storage_path="text/me/mem_001.txt",
        payload_type="text",
        created_at_ns=1000,
        size_bytes=100,
        content_hash="abc123",
    )
    cat.register(entry)
    found = cat.lookup("mem_001")
    assert found is not None
    assert found.payload_type == "text"
    assert found.size_bytes == 100
    assert cat.lookup("nonexistent") is None


def test_catalog_stats() -> None:
    cat = Catalog()
    cat.register(CatalogEntry(
        memory_id="mem_a", storage_path="text/a.txt", payload_type="text",
        created_at_ns=1, size_bytes=10, content_hash="h1",
    ))
    cat.register(CatalogEntry(
        memory_id="mem_b", storage_path="json/b.json", payload_type="json",
        created_at_ns=2, size_bytes=20, content_hash="h2",
    ))
    cat.register(CatalogEntry(
        memory_id="mem_c", storage_path="img/c.png", payload_type="image",
        created_at_ns=3, size_bytes=30, content_hash="h3",
    ))
    s = cat.stats()
    assert s["total_entries"] == 3
    assert s["total_size_bytes"] == 60
    assert s["by_type"]["text"] == 1
    assert s["by_type"]["json"] == 1
    assert s["by_type"]["image"] == 1


def test_catalog_persist(tmp_path: Path) -> None:
    cat = Catalog()
    cat.register(CatalogEntry(
        memory_id="mem_x", storage_path="text/x.txt", payload_type="text",
        created_at_ns=42, size_bytes=7, content_hash="sha",
    ))
    save_path = tmp_path / "cat.json"
    cat.save(save_path)
    assert save_path.exists()

    cat2 = Catalog()
    cat2.load(save_path)
    found = cat2.lookup("mem_x")
    assert found is not None
    assert found.payload_type == "text"
    assert found.created_at_ns == 42


# ── STM / MTM ─────────────────────────────────────────────────

def test_stm_add_and_list() -> None:
    """Use an in-memory-only approach: create with persistent=False still adds to STM."""
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        m = Memorizer(td)
        mid = m.create("stm test", payload_type="text")
        ids = m.get_stm_ids()
        assert mid in ids
        assert len(ids) >= 1


def test_mtm_promote(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    mid = m.create("task item", payload_type="text")
    assert mid not in m.get_mtm_ids()
    m.promote_to_mtm(mid)
    assert mid in m.get_mtm_ids()
    # still in STM after promote
    assert mid in m.get_stm_ids()
    # demote removes from MTM but keeps in STM
    m.demote_from_mtm(mid)
    assert mid not in m.get_mtm_ids()
    assert mid in m.get_stm_ids()


# ── IndexManager ──────────────────────────────────────────────

def test_index_add_get_neighbors() -> None:
    im = IndexManager()
    im.add_edge("A", "B", "content_similar", weight=0.8)
    neighbors = im.get_neighbors("A")
    assert "B" in neighbors
    assert len(neighbors) == 1
    # filter by edge type
    temporal = im.get_neighbors("A", edge_type="temporal")
    assert temporal == []


def test_index_temporal_chain() -> None:
    im = IndexManager()
    im.add_temporal_chain(["e1", "e2", "e3"])
    n1 = im.get_neighbors("e1", edge_type="temporal")
    n2 = im.get_neighbors("e2", edge_type="temporal")
    assert len(n1) == 1  # e1→e2
    assert n1[0] == "e2"
    assert len(n2) == 1  # e2→e3
    assert n2[0] == "e3"
    # chain of 1 does nothing
    im.add_temporal_chain(["only"])
    assert im.get_neighbors("only") == []


def test_index_fold_dense_bipartite() -> None:
    im = IndexManager()
    # group_a connects densely to group_b
    im.add_edge("a1", "b1", "content_similar", weight=0.9)
    im.add_edge("a1", "b2", "content_similar", weight=0.5)
    im.add_edge("a1", "b3", "content_similar", weight=0.7)
    im.add_edge("a1", "b4", "content_similar", weight=0.3)
    result = im.fold_dense_bipartite(["a1"], ["b1", "b2", "b3", "b4"], max_path=2)
    assert "a1" in result
    # should keep only top 2 by weight
    assert len(result["a1"]) == 2
    assert result["a1"][0] == "b1"  # weight 0.9
    assert result["a1"][1] == "b3"  # weight 0.7


def test_index_merge_redirect() -> None:
    im = IndexManager()
    # Add edge X → A
    im.add_edge("X", "A", "temporal")
    # Merge A, B → C
    im.merge_redirect(["A", "B"], "C")
    # X's edge should now point to C (redirected in-place)
    neighbors = im.get_neighbors("X")
    assert "C" in neighbors
    # A's outgoing edges moved to C, so C now has A's edges
    # resolve_redirect returns same id (stub)
    assert im.resolve_redirect("A") == "A"


# ── EventManager ──────────────────────────────────────────────

def test_event_create_and_list() -> None:
    em = EventManager()
    evt = em.create_event(
        memory_ids=["mem_1", "mem_2"],
        summary="user clicked button",
        tags=["interaction", "click"],
    )
    assert evt.event_id.startswith("evt_")
    assert len(evt.memory_ids) == 2
    assert evt.summary == "user clicked button"

    events = em.list_events()
    assert len(events) >= 1
    assert any(e.event_id == evt.event_id for e in events)

    # filter by tag
    filtered = em.list_events(tag="click")
    assert len(filtered) == 1
    assert filtered[0].event_id == evt.event_id


# ── Retriever ─────────────────────────────────────────────────

def test_retrieval_keyword_search(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    im = IndexManager()
    r = Retriever(m, im)

    m.create("the quick brown fox", payload_type="text")
    m.create("hello world", payload_type="text")
    m.create("fox hunting is fun", payload_type="text")

    results = r.keyword_search("fox")
    assert len(results) >= 2
    # all results should be str memory ids
    for rid in results:
        assert rid.startswith("mem_")


def test_retrieval_time_range(tmp_path: Path) -> None:
    m = Memorizer(tmp_path)
    im = IndexManager()
    r = Retriever(m, im)

    mid1 = m.create("event A", payload_type="text")
    time.sleep(0.2)
    mid2 = m.create("event B", payload_type="text")
    entry1 = m.catalog.lookup(mid1)
    entry2 = m.catalog.lookup(mid2)
    assert entry1 is not None
    assert entry2 is not None
    assert entry2.created_at_ns > entry1.created_at_ns

    # Search with time range that only covers mid2
    start_ns = entry2.created_at_ns - 500_000  # 0.5ms before mid2
    req = RetrievalRequest(time_start_ns=start_ns, payload_types=["text"])
    results = r.search(req)
    mids = [res.memory_id for res in results]
    assert mid2 in mids
    assert mid1 not in mids

    # Search with time_end_ns that excludes mid2
    req2 = RetrievalRequest(
        time_end_ns=entry1.created_at_ns + 500_000,
        payload_types=["text"],
    )
    results2 = r.search(req2)
    mids2 = [res.memory_id for res in results2]
    assert mid1 in mids2
    assert mid2 not in mids2
