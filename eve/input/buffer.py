"""
EVE 输入 buffer — 统一时间轴上的一秒输入缓存。

提供：
- buffer.store("screen", frame)
- buffer.store("cursor", (x, y))
- buffer.latest("screen") -> TimedSample | None
- buffer.latest("cursor") -> TimedSample | None
- buffer.range("screen", start_ns, end_ns) -> list[TimedSample]
- buffer.snapshot(duration_ns=1_000_000_000) -> dict[str, list[TimedSample]]

线程安全。自动淘汰超出保留时间的旧样本。
"""
from __future__ import annotations

import threading
import time
from typing import Any

from eve.input.schemas import TimedSample

_DEFAULT_RETENTION_NS = 2_000_000_000  # 默认保留 2 秒


class InputBuffer:
    """统一输入缓冲区。"""

    def __init__(self, retention_ns: int = _DEFAULT_RETENTION_NS):
        self._retention_ns = retention_ns
        self._lock = threading.Lock()
        self._index_counter = 0
        # kind -> list[TimedSample]
        self._samples: dict[str, list[TimedSample]] = {}

    def store(self, kind: str, value: Any) -> TimedSample:
        """存入一个样本。"""
        now_ns = time.monotonic_ns()
        with self._lock:
            self._index_counter += 1
            sample = TimedSample(
                timestamp_ns=now_ns,
                kind=kind,
                value=value,
                index=self._index_counter,
            )
            if kind not in self._samples:
                self._samples[kind] = []
            self._samples[kind].append(sample)
            self._evict(kind)
        return sample

    def latest(self, kind: str) -> TimedSample | None:
        """获取最新样本。"""
        with self._lock:
            samples = self._samples.get(kind, [])
            return samples[-1] if samples else None

    def range(
        self, kind: str, start_ns: int, end_ns: int | None = None
    ) -> list[TimedSample]:
        """获取 [start_ns, end_ns) 时间范围内的样本。end_ns=None 表示到现在。"""
        if end_ns is None:
            end_ns = time.monotonic_ns()
        with self._lock:
            samples = self._samples.get(kind, [])
            return [s for s in samples if start_ns <= s.timestamp_ns < end_ns]

    def snapshot(
        self, duration_ns: int = 1_000_000_000
    ) -> dict[str, list[TimedSample]]:
        """获取最近 duration_ns 内所有种类的样本快照。"""
        now_ns = time.monotonic_ns()
        start_ns = now_ns - duration_ns
        result: dict[str, list[TimedSample]] = {}
        with self._lock:
            for kind in list(self._samples.keys()):
                result[kind] = [
                    s for s in self._samples[kind]
                    if s.timestamp_ns >= start_ns
                ]
        return result

    def count(self, kind: str) -> int:
        """某类样本当前数量。"""
        with self._lock:
            return len(self._samples.get(kind, []))

    @property
    def kinds(self) -> list[str]:
        """当前所有样本种类。"""
        with self._lock:
            return list(self._samples.keys())

    def _evict(self, kind: str) -> None:
        """淘汰超出保留时间的旧样本。"""
        cutoff = time.monotonic_ns() - self._retention_ns
        samples = self._samples.get(kind, [])
        while samples and samples[0].timestamp_ns < cutoff:
            samples.pop(0)
