"""Thread-safe recent input window on a monotonic timeline."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimedSample:
    timestamp_ns: int
    kind: str
    value: Any
    index: int


class InputBuffer:
    def __init__(self, retention_ns: int = 2_000_000_000):
        if retention_ns <= 0:
            raise ValueError("retention_ns must be positive")
        self.retention_ns = retention_ns
        self._samples: dict[str, deque[TimedSample]] = {}
        self._index = 0
        self._lock = threading.Lock()

    def store(
        self, kind: str, value: Any, *, timestamp_ns: int | None = None
    ) -> TimedSample:
        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        with self._lock:
            self._index += 1
            sample = TimedSample(timestamp_ns, kind, value, self._index)
            items = self._samples.setdefault(kind, deque())
            if items and timestamp_ns < items[-1].timestamp_ns:
                raise ValueError("input timestamps must be monotonic per kind")
            items.append(sample)
            cutoff = timestamp_ns - self.retention_ns
            while items and items[0].timestamp_ns < cutoff:
                items.popleft()
            return sample

    def latest(self, kind: str) -> TimedSample | None:
        with self._lock:
            items = self._samples.get(kind)
            return items[-1] if items else None

    def range(
        self, kind: str, start_ns: int, end_ns: int | None = None
    ) -> list[TimedSample]:
        end_ns = time.monotonic_ns() if end_ns is None else end_ns
        with self._lock:
            return [
                sample
                for sample in self._samples.get(kind, ())
                if start_ns <= sample.timestamp_ns < end_ns
            ]

    def snapshot(self, duration_ns: int = 1_000_000_000) -> dict[str, list[TimedSample]]:
        now_ns = time.monotonic_ns()
        with self._lock:
            return {
                kind: [
                    sample
                    for sample in items
                    if sample.timestamp_ns >= now_ns - duration_ns
                ]
                for kind, items in self._samples.items()
            }

    def count(self, kind: str) -> int:
        with self._lock:
            return len(self._samples.get(kind, ()))

    @property
    def kinds(self) -> list[str]:
        with self._lock:
            return sorted(self._samples)
