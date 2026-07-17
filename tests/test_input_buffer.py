"""Phase 2 InputBuffer 测试。

覆盖：
- latest/range/snapshot 接口
- 线程安全
- 自动淘汰旧样本
- count/kinds
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from eve.input.buffer import InputBuffer
from eve.input.schemas import TimedSample


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def buffer() -> InputBuffer:
    """默认 2 秒保留的 buffer。"""
    return InputBuffer(retention_ns=2_000_000_000)


@pytest.fixture
def short_buffer() -> InputBuffer:
    """短保留期 buffer，用于测试淘汰。"""
    return InputBuffer(retention_ns=50_000_000)  # 50ms


# ── store + latest ────────────────────────────────────────


def test_store_and_latest(buffer: InputBuffer):
    """store 后 latest 返回最新样本。"""
    buffer.store("cursor", (100, 200))
    buffer.store("cursor", (300, 400))
    latest = buffer.latest("cursor")
    assert latest is not None
    assert latest.value == (300, 400)
    assert latest.kind == "cursor"
    assert latest.timestamp_ns > 0
    assert latest.index > 0


def test_store_multiple_kinds(buffer: InputBuffer):
    """不同 kind 独立存储。"""
    buffer.store("cursor", (1, 2))
    buffer.store("screen", b"fake_frame")
    assert buffer.latest("cursor") is not None
    assert buffer.latest("screen") is not None
    assert buffer.latest("cursor").value == (1, 2)
    assert buffer.latest("screen").value == b"fake_frame"


def test_latest_empty_returns_none(buffer: InputBuffer):
    """空 buffer latest 返回 None。"""
    assert buffer.latest("nonexistent") is None


def test_index_monotonic(buffer: InputBuffer):
    """index 单调递增。"""
    s1 = buffer.store("cursor", (1, 2))
    s2 = buffer.store("cursor", (3, 4))
    s3 = buffer.store("screen", b"x")
    assert s1.index < s2.index < s3.index


# ── range ────────────────────────────────────────────────


def test_range_filters_by_time(buffer: InputBuffer):
    """range 只返回时间范围内的样本。"""
    t0 = time.monotonic_ns()
    buffer.store("cursor", (1, 1))
    time.sleep(0.02)
    t_mid = time.monotonic_ns()
    buffer.store("cursor", (2, 2))
    time.sleep(0.02)
    t_end = time.monotonic_ns()
    buffer.store("cursor", (3, 3))

    # range 只取中间一条
    results = buffer.range("cursor", t_mid, t_end)
    assert len(results) == 1
    assert results[0].value == (2, 2)


def test_range_open_end(buffer: InputBuffer):
    """end_ns=None 时取到现在。"""
    buffer.store("cursor", (1, 1))
    time.sleep(0.02)
    # 用存储后拿到的样本索引来获取其时间戳作为 range 起点
    s1 = buffer.store("cursor", (2, 2))
    time.sleep(0.01)
    buffer.store("cursor", (3, 3))
    results = buffer.range("cursor", s1.timestamp_ns)
    assert len(results) >= 1


def test_range_empty(buffer: InputBuffer):
    """无匹配时返回空列表。"""
    buffer.store("cursor", (1, 1))
    future = time.monotonic_ns() + 10_000_000_000
    results = buffer.range("cursor", future)
    assert results == []


# ── snapshot ─────────────────────────────────────────────


def test_snapshot(buffer: InputBuffer):
    """snapshot 返回所有 kind 的最近样本。"""
    buffer.store("cursor", (1, 1))
    buffer.store("screen", b"f1")
    time.sleep(0.01)
    buffer.store("cursor", (2, 2))
    buffer.store("screen", b"f2")

    snap = buffer.snapshot(duration_ns=1_000_000_000)
    assert "cursor" in snap
    assert "screen" in snap
    assert len(snap["cursor"]) >= 2
    assert len(snap["screen"]) >= 2


def test_snapshot_short_duration(buffer: InputBuffer):
    """短 duration 只返回近期样本。"""
    buffer.store("cursor", (1, 1))
    time.sleep(0.1)
    buffer.store("cursor", (2, 2))

    # 10ms duration 只能拿到最后一条
    snap = buffer.snapshot(duration_ns=10_000_000)
    assert len(snap.get("cursor", [])) <= 1


# ── 淘汰 ──────────────────────────────────────────────


def test_eviction(short_buffer: InputBuffer):
    """旧样本应被自动淘汰。"""
    short_buffer.store("cursor", (1, 1))
    # 等待超过保留期
    time.sleep(0.06)
    # 存储新样本触发淘汰
    short_buffer.store("cursor", (2, 2))
    # 旧样本已被淘汰
    assert short_buffer.count("cursor") == 1
    assert short_buffer.latest("cursor").value == (2, 2)


def test_eviction_only_affects_old_kind(short_buffer: InputBuffer):
    """淘汰只影响对应 kind。"""
    short_buffer.store("cursor", (1, 1))
    time.sleep(0.06)
    short_buffer.store("screen", b"f1")
    assert short_buffer.count("cursor") == 1  # 被新 cursor store 触发淘汰
    assert short_buffer.count("screen") == 1


# ── count + kinds ──────────────────────────────────────


def test_count(buffer: InputBuffer):
    """count 返回正确数量。"""
    assert buffer.count("cursor") == 0
    buffer.store("cursor", (1, 1))
    buffer.store("cursor", (2, 2))
    assert buffer.count("cursor") == 2


def test_kinds(buffer: InputBuffer):
    """kinds 返回所有种类。"""
    buffer.store("cursor", (1, 1))
    buffer.store("screen", b"f")
    assert set(buffer.kinds) == {"cursor", "screen"}


# ── 线程安全 ──────────────────────────────────────────


def test_concurrent_store(buffer: InputBuffer):
    """多线程 store 不应崩溃或丢数据。"""
    errors: list[Exception] = []

    def _writer(kind: str, base: int, count: int):
        try:
            for i in range(count):
                buffer.store(kind, (base + i, base + i))
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_writer, args=("cursor", 0, 50)),
        threading.Thread(target=_writer, args=("screen", 100, 50)),
        threading.Thread(target=_writer, args=("cursor", 200, 50)),
        threading.Thread(target=_writer, args=("screen", 300, 50)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"线程错误: {errors}"
    assert buffer.count("cursor") == 100
    assert buffer.count("screen") == 100


# ── 不导入 src.eve ────────────────────────────────────


def test_no_src_eve_import() -> None:
    import sys
    assert "src.eve" not in sys.modules
