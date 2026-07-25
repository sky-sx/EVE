"""
Tests for InputBuffer (eve/input/buffer.py).
"""
import threading
import time

import pytest

from eve.input.buffer import InputBuffer


# ── store + latest ───────────────────────────────────────

def test_store_and_latest():
    """Store a sample then retrieve via latest."""
    buf = InputBuffer()
    sample = buf.store("screen", "frame_1")
    latest = buf.latest("screen")
    assert latest is not None
    assert latest.value == "frame_1"
    assert latest.kind == "screen"
    assert latest.timestamp_ns == sample.timestamp_ns


def test_latest_returns_most_recent():
    """latest returns the most recently stored sample."""
    buf = InputBuffer()
    buf.store("cursor", (0, 0))
    buf.store("cursor", (100, 100))
    buf.store("cursor", (200, 200))
    latest = buf.latest("cursor")
    assert latest.value == (200, 200)


def test_empty_latest():
    """latest on empty buffer returns None."""
    buf = InputBuffer()
    assert buf.latest("nonexistent") is None


# ── range filtering ──────────────────────────────────────

def test_range_filtering():
    """range returns correct subset by time."""
    buf = InputBuffer(retention_ns=10_000_000_000)
    # Store samples with generous time gaps for Windows timer resolution
    s1 = buf.store("cursor", (0, 0))
    time.sleep(0.05)
    s2 = buf.store("cursor", (100, 100))
    time.sleep(0.05)
    s3 = buf.store("cursor", (200, 200))

    # Use end_ns far in the future to avoid timer resolution issues
    far_future = s3.timestamp_ns + 1_000_000_000  # +1 second
    results = buf.range("cursor", s1.timestamp_ns + 1, far_future)
    indices = [r.index for r in results]
    assert s2.index in indices
    assert s3.index in indices


def test_range_with_end_ns():
    """range with explicit end_ns excludes later samples."""
    buf = InputBuffer()
    s1 = buf.store("cursor", (0, 0))
    time.sleep(0.02)
    s2 = buf.store("cursor", (100, 100))
    time.sleep(0.02)
    s3 = buf.store("cursor", (200, 200))

    # Range between s1 and s3 (exclusive on end)
    results = buf.range("cursor", s1.timestamp_ns + 1, s3.timestamp_ns)
    assert len(results) >= 1
    # s3 should not be included (end_ns is exclusive)
    assert s3.index not in [r.index for r in results]


def test_range_empty_kind():
    """range on unknown kind returns empty list."""
    buf = InputBuffer()
    assert buf.range("unknown", 0) == []


# ── snapshot ─────────────────────────────────────────────

def test_snapshot_by_duration():
    """snapshot returns correct time window for all kinds."""
    buf = InputBuffer(retention_ns=5_000_000_000)
    now = time.monotonic_ns()
    buf.store("screen", "frame_A")
    buf.store("cursor", (10, 10))
    time.sleep(0.01)

    snap = buf.snapshot(duration_ns=1_000_000_000)
    assert "screen" in snap
    assert "cursor" in snap
    assert len(snap["screen"]) >= 1
    assert len(snap["cursor"]) >= 1


def test_snapshot_empty():
    """snapshot on empty buffer returns empty dict."""
    buf = InputBuffer()
    assert buf.snapshot() == {}


# ── count ────────────────────────────────────────────────

def test_count():
    """count returns correct number of samples."""
    buf = InputBuffer()
    assert buf.count("screen") == 0
    buf.store("screen", "f1")
    buf.store("screen", "f2")
    buf.store("screen", "f3")
    assert buf.count("screen") == 3
    # Different kind unaffected
    assert buf.count("cursor") == 0


def test_count_unknown_kind():
    """count on unknown kind returns 0."""
    buf = InputBuffer()
    assert buf.count("nonexistent") == 0


# ── eviction ─────────────────────────────────────────────

def test_eviction():
    """Old samples are auto-evicted beyond retention window."""
    short_ns = 50_000_000  # 50ms retention
    buf = InputBuffer(retention_ns=short_ns)
    buf.store("screen", "old_frame")
    assert buf.count("screen") == 1
    time.sleep(0.1)  # Wait past retention
    # Store a new sample to trigger eviction
    buf.store("screen", "new_frame")
    # Old sample should be gone
    assert buf.count("screen") == 1


def test_eviction_only_affects_expired():
    """Eviction only removes samples of the kind being stored."""
    short_ns = 50_000_000
    buf = InputBuffer(retention_ns=short_ns)
    buf.store("screen", "old_screen")
    buf.store("cursor", "old_cursor")
    time.sleep(0.1)
    buf.store("screen", "new_screen")
    # screen has eviction triggered
    assert buf.count("screen") == 1
    # cursor still has the old sample (eviction not triggered for it)
    assert buf.count("cursor") == 1


# ── thread safety ────────────────────────────────────────

def test_thread_safety():
    """Concurrent stores from multiple threads don't crash."""
    buf = InputBuffer()

    def store_batch():
        for i in range(100):
            buf.store("cursor", (i, i))

    threads = [threading.Thread(target=store_batch) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert buf.count("cursor") == 1000
    # Verify data integrity: latest should exist
    assert buf.latest("cursor") is not None


# ── multiple kinds ───────────────────────────────────────

def test_multiple_kinds():
    """Different kinds stored separately."""
    buf = InputBuffer()
    buf.store("screen", "frame_1")
    buf.store("screen", "frame_2")
    buf.store("cursor", (50, 50))
    buf.store("audio", "beep")

    assert buf.count("screen") == 2
    assert buf.count("cursor") == 1
    assert buf.count("audio") == 1
    assert set(buf.kinds) == {"screen", "cursor", "audio"}

    # Latest per kind
    assert buf.latest("screen").value == "frame_2"
    assert buf.latest("cursor").value == (50, 50)
    assert buf.latest("audio").value == "beep"


# ── index increment ──────────────────────────────────────

def test_index_increments():
    """Each stored sample has an incrementing index."""
    buf = InputBuffer()
    s1 = buf.store("cursor", (0, 0))
    s2 = buf.store("cursor", (1, 1))
    s3 = buf.store("cursor", (2, 2))
    assert s1.index < s2.index < s3.index
    assert s1.index >= 1


# ── kinds property ───────────────────────────────────────

def test_kinds_returns_current_kinds():
    """kinds property returns correct list."""
    buf = InputBuffer()
    assert buf.kinds == []
    buf.store("screen", "f")
    assert buf.kinds == ["screen"]
    buf.store("cursor", (0, 0))
    assert set(buf.kinds) == {"screen", "cursor"}
