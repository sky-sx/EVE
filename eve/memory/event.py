"""Event: 一组 MemoryID 的语义分组。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Event:
    event_id: str
    memory_ids: list[str]       # Related MemoryIDs
    summary: str                # Text description
    tags: list[str] = field(default_factory=list)
    start_ns: int = 0
    end_ns: int = 0
    created_at_ns: int = 0

    def __post_init__(self) -> None:
        if self.created_at_ns == 0:
            self.created_at_ns = time.monotonic_ns()


class EventManager:
    """Event 生命周期管理。"""

    def __init__(self) -> None:
        self._events: dict[str, Event] = {}

    def create_event(
        self,
        memory_ids: list[str],
        summary: str,
        tags: list[str] | None = None,
    ) -> Event:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        event = Event(
            event_id=event_id,
            memory_ids=list(memory_ids),
            summary=summary,
            tags=list(tags) if tags else [],
        )
        self._events[event_id] = event
        return event

    def get_event(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def list_events(self, tag: str | None = None, limit: int = 50) -> list[Event]:
        events = list(self._events.values())
        if tag is not None:
            events = [e for e in events if tag in e.tags]
        events.sort(key=lambda e: e.created_at_ns, reverse=True)
        return events[:limit]

    def add_memory_to_event(self, event_id: str, memory_id: str) -> bool:
        event = self._events.get(event_id)
        if event is None:
            return False
        if memory_id not in event.memory_ids:
            event.memory_ids.append(memory_id)
        return True

    def save(self, path: Path) -> None:
        data = {
            eid: asdict(event) for eid, event in self._events.items()
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._events = {}
        for eid, d in data.items():
            self._events[eid] = Event(**d)
