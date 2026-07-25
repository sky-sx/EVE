"""Minimal immutable Memory storage with one authoritative catalog."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryUnit:
    memory_id: str
    payload_type: str
    created_at_ns: int
    storage_path: str
    content_hash: str


@dataclass(frozen=True)
class Event:
    event_id: str
    memory_ids: tuple[str, ...]
    created_at_ns: int
    description: str = ""


class Memorizer:
    """Owns Catalog, STM/MTM ID views, payload files and minimal retrieval."""

    def __init__(self, base_dir: str | Path, stm_limit: int = 1000) -> None:
        self.base_dir = Path(base_dir)
        self.payload_dir = self.base_dir / "payloads"
        self.catalog_path = self.base_dir / "catalog.json"
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.stm_limit = stm_limit
        self.catalog: dict[str, MemoryUnit] = {}
        self.stm: list[str] = []
        self.mtm: set[str] = set()
        self.events: dict[str, Event] = {}
        self._lock = threading.RLock()
        self.load_catalog()

    def create(self, payload: Any, payload_type: str = "json") -> str:
        created_at_ns = time.time_ns()
        memory_id = f"mem_{created_at_ns}_{uuid.uuid4().hex[:8]}"
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=repr
        ).encode("utf-8")
        path = self.payload_dir / f"{memory_id}.json"
        path.write_bytes(encoded)
        unit = MemoryUnit(
            memory_id=memory_id,
            payload_type=payload_type,
            created_at_ns=created_at_ns,
            storage_path=str(path.relative_to(self.base_dir)),
            content_hash=hashlib.sha256(encoded).hexdigest(),
        )
        with self._lock:
            self.catalog[memory_id] = unit
            self.stm.append(memory_id)
            self.stm = self.stm[-self.stm_limit :]
            self.save_catalog()
        return memory_id

    def read(self, memory_id: str) -> Any | None:
        with self._lock:
            unit = self.catalog.get(memory_id)
        if unit is None:
            return None
        path = self.base_dir / unit.storage_path
        if not path.exists():
            raise FileNotFoundError(f"catalog payload missing: {memory_id}")
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != unit.content_hash:
            raise ValueError(f"immutable payload hash mismatch: {memory_id}")
        return json.loads(encoded.decode("utf-8"))

    def get_unit(self, memory_id: str) -> MemoryUnit | None:
        return self.catalog.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            unit = self.catalog.get(memory_id)
            if unit is None:
                return False
            path = (self.base_dir / unit.storage_path).resolve()
            payload_root = self.payload_dir.resolve()
            if payload_root not in path.parents:
                raise ValueError("catalog path escapes payload directory")
            if path.exists():
                path.unlink()
            self.catalog.pop(memory_id)
            self.stm = [item for item in self.stm if item != memory_id]
            self.mtm.discard(memory_id)
            self.save_catalog()
            return True

    def promote_to_mtm(self, memory_id: str) -> None:
        if memory_id not in self.catalog:
            raise KeyError(memory_id)
        self.mtm.add(memory_id)

    def create_event(self, memory_ids: list[str], description: str = "") -> Event:
        missing = set(memory_ids) - set(self.catalog)
        if missing:
            raise KeyError(f"unknown MemoryID(s): {sorted(missing)}")
        event = Event(
            event_id=f"event_{time.time_ns()}_{uuid.uuid4().hex[:6]}",
            memory_ids=tuple(memory_ids),
            created_at_ns=time.time_ns(),
            description=description,
        )
        self.events[event.event_id] = event
        return event

    def search(
        self,
        *,
        payload_type: str | None = None,
        keyword: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[str]:
        results: list[str] = []
        for unit in sorted(self.catalog.values(), key=lambda item: item.created_at_ns):
            if payload_type is not None and unit.payload_type != payload_type:
                continue
            if start_ns is not None and unit.created_at_ns < start_ns:
                continue
            if end_ns is not None and unit.created_at_ns >= end_ns:
                continue
            if keyword is not None:
                payload_text = json.dumps(self.read(unit.memory_id), ensure_ascii=False)
                if keyword.casefold() not in payload_text.casefold():
                    continue
            results.append(unit.memory_id)
        return results

    def save_catalog(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "units": {key: asdict(value) for key, value in self.catalog.items()},
            "stm": self.stm,
            "mtm": sorted(self.mtm),
        }
        self.catalog_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.catalog = {
            key: MemoryUnit(**value) for key, value in data.get("units", {}).items()
        }
        self.stm = [item for item in data.get("stm", []) if item in self.catalog]
        self.mtm = {item for item in data.get("mtm", []) if item in self.catalog}
