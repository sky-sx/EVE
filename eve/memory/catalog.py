"""Catalog: MemoryID → LTM 对象映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CatalogEntry:
    memory_id: str
    storage_path: str         # relative path under LTM/
    payload_type: str         # "text" | "image" | "audio" | "json" | "numpy" | "other"
    created_at_ns: int
    size_bytes: int
    content_hash: str         # SHA256 of payload
    persistent: bool = True   # False = only in STM cache
    resident: bool = True     # False = unloaded from memory


class Catalog:
    """MemoryID → LTM 对象映射。"""

    def __init__(self) -> None:
        self._entries: dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry) -> None:
        self._entries[entry.memory_id] = entry

    def lookup(self, memory_id: str) -> CatalogEntry | None:
        return self._entries.get(memory_id)

    def list_by_type(self, payload_type: str) -> list[str]:
        return [
            mid for mid, e in self._entries.items()
            if e.payload_type == payload_type
        ]

    def list_recent(self, limit: int = 100) -> list[str]:
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at_ns,
            reverse=True,
        )
        return [e.memory_id for e in sorted_entries[:limit]]

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._entries:
            del self._entries[memory_id]
            return True
        return False

    def save(self, path: Path) -> None:
        data = {
            mid: asdict(entry) for mid, entry in self._entries.items()
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._entries = {}
        for mid, d in data.items():
            self._entries[mid] = CatalogEntry(**d)

    def stats(self) -> dict:
        type_counts: dict[str, int] = {}
        total_size = 0
        total_entries = len(self._entries)
        for entry in self._entries.values():
            type_counts[entry.payload_type] = type_counts.get(entry.payload_type, 0) + 1
            total_size += entry.size_bytes
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "by_type": type_counts,
        }
