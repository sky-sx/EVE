"""Immutable payload storage with incremental catalog and an async writer."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class MemoryUnit:
    memory_id: str
    payload_type: str
    created_at_ns: int
    storage_path: str
    content_hash: str
    size_bytes: int = 0


@dataclass(frozen=True)
class Event:
    event_id: str
    started_at_ns: int
    ended_at_ns: int
    memory_ids: tuple[str, ...]
    summary: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MemoryWriteRequest:
    memory_id: str
    payload: Any
    payload_type: str
    priority: str
    requested_at_ns: int


class Memorizer:
    """Own payload files, one append-only catalog, and one bounded writer."""

    def __init__(
        self,
        base_dir: str | Path,
        stm_limit: int = 1000,
        *,
        queue_capacity: int = 256,
        writer_error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.base_dir = Path(base_dir)
        self.payload_dir = self.base_dir / "payloads"
        self.tnn_dir = self.base_dir / "TNNweights"
        self.catalog_path = self.base_dir / "catalog.jsonl"
        self.event_catalog_path = self.base_dir / "events.jsonl"
        self.legacy_catalog_path = self.base_dir / "catalog.json"
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.tnn_dir.mkdir(parents=True, exist_ok=True)
        self.stm_limit = stm_limit
        self.queue_capacity = queue_capacity
        self.catalog: dict[str, MemoryUnit] = {}
        self.stm: list[str] = []
        self.mtm: set[str] = set()
        self.ltm: set[str] = set()
        self.events: dict[str, Event] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[MemoryWriteRequest] = deque()
        self._writer_thread: threading.Thread | None = None
        self._writer_stop = False
        self._writer_busy = False
        self._writer_current_id: str | None = None
        self._writer_error_callback = writer_error_callback
        self.last_writer_error: Exception | None = None
        self._enqueued_count = 0
        self._written_count = 0
        self._dropped_count = 0
        self._failed_count = 0
        self._promotion_thread: threading.Thread | None = None
        self._promotion_stop = threading.Event()
        self._promotion_status: dict[str, Any] = {
            "state": "idle",
            "total": 0,
            "processed": 0,
            "remaining": 0,
            "eta_s": None,
            "last_result": None,
            "last_error": None,
        }
        self.load_catalog()
        self._load_events()

    @property
    def writer_running(self) -> bool:
        return (
            self._writer_thread is not None
            and self._writer_thread.is_alive()
        )

    def start_writer(self) -> None:
        if self.writer_running:
            return
        with self._condition:
            self._writer_stop = False
            self.last_writer_error = None
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="eve-memory-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def stop_writer(self, timeout_s: float = 3.0, *, flush: bool = True) -> None:
        self.stop_promotion(timeout_s)
        failure: Exception | None = None
        if flush:
            try:
                self.flush(timeout_s)
            except Exception as exc:
                failure = exc
        with self._condition:
            self._writer_stop = True
            self._condition.notify_all()
        thread = self._writer_thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError("memory writer did not stop")
        self._writer_thread = None
        if failure is not None:
            raise failure

    def enqueue(
        self,
        payload: Any,
        payload_type: str = "json",
        *,
        priority: str = "normal",
    ) -> str | None:
        """Queue a write without doing payload encoding or disk I/O."""
        if priority not in {"low", "normal", "critical"}:
            raise ValueError("priority must be low, normal, or critical")
        if not self.writer_running:
            self.start_writer()
        memory_id = self._new_memory_id()
        request = MemoryWriteRequest(
            memory_id=memory_id,
            payload=payload,
            payload_type=payload_type,
            priority=priority,
            requested_at_ns=time.time_ns(),
        )
        overflow_error: Exception | None = None
        with self._condition:
            if len(self._queue) >= self.queue_capacity:
                low_index = next(
                    (
                        index
                        for index, queued in enumerate(self._queue)
                        if queued.priority == "low"
                    ),
                    None,
                )
                if low_index is not None:
                    items = list(self._queue)
                    items.pop(low_index)
                    self._queue = deque(items)
                    self._dropped_count += 1
                elif priority == "critical":
                    self._dropped_count += 1
                    overflow_error = RuntimeError(
                        "memory queue full; critical write was rejected"
                    )
                    self.last_writer_error = overflow_error
                else:
                    self._dropped_count += 1
                    return None
            if overflow_error is None:
                self._queue.append(request)
                self._enqueued_count += 1
                self._condition.notify()
        if overflow_error is not None:
            if self._writer_error_callback is not None:
                self._writer_error_callback(overflow_error)
            return None
        return memory_id

    def flush(self, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._queue or self._writer_busy:
                if self.last_writer_error is not None:
                    raise RuntimeError(
                        f"memory writer failed: {self.last_writer_error}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("memory writer flush timed out")
                self._condition.wait(remaining)
        if self.last_writer_error is not None:
            raise RuntimeError(f"memory writer failed: {self.last_writer_error}")

    def writer_stats(self) -> dict[str, int | str | None]:
        with self._condition:
            return {
                "enqueued": self._enqueued_count,
                "written": self._written_count,
                "dropped": self._dropped_count,
                "failed": self._failed_count,
                "queue_depth": len(self._queue),
                "last_error": (
                    str(self.last_writer_error)
                    if self.last_writer_error is not None
                    else None
                ),
            }

    def create(self, payload: Any, payload_type: str = "json") -> str:
        """Synchronous creation retained for cold paths and training data."""
        memory_id = self._new_memory_id()
        self._create_with_id(memory_id, payload, payload_type)
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
        if path.suffix == ".npy":
            import numpy as np

            return np.load(path, allow_pickle=False)
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
            self.ltm.discard(memory_id)
            self._append_catalog({"op": "delete", "memory_id": memory_id})
            return True

    def promote_to_mtm(self, memory_id: str) -> None:
        with self._lock:
            if memory_id not in self.catalog:
                raise KeyError(memory_id)
            self._promote_to_mtm_locked(memory_id)

    def promote_to_ltm(self, memory_id: str) -> None:
        with self._lock:
            if memory_id not in self.catalog:
                raise KeyError(memory_id)
            if memory_id not in self.mtm:
                raise ValueError("memory must pass through MTM before LTM")
            self.mtm.discard(memory_id)
            self.ltm.add(memory_id)
            self._append_catalog({"op": "promote_ltm", "memory_id": memory_id})

    def _promote_to_mtm_locked(self, memory_id: str) -> None:
        self.stm = [item for item in self.stm if item != memory_id]
        self.mtm.add(memory_id)
        self._append_catalog({"op": "promote", "memory_id": memory_id})

    def _append_to_stm_locked(self, memory_id: str) -> None:
        self.stm.append(memory_id)
        while len(self.stm) > self.stm_limit:
            self._promote_to_mtm_locked(self.stm[0])

    @property
    def ltm_count(self) -> int:
        return len(self.ltm)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "stm": len(self.stm),
                "mtm": len(self.mtm),
                "ltm": len(self.ltm),
                "events": len(self.events),
            }

    def tier_ids(self, tier: str) -> list[str]:
        with self._lock:
            if tier == "stm":
                values = list(self.stm)
            elif tier == "mtm":
                values = list(self.mtm)
            elif tier == "ltm":
                values = list(self.ltm)
            else:
                raise ValueError("tier must be stm, mtm, or ltm")
            return sorted(
                values,
                key=lambda memory_id: self.catalog[memory_id].created_at_ns,
            )

    def create_event(
        self,
        memory_ids: list[str],
        summary: str = "",
        tags: list[str] | tuple[str, ...] = (),
        *,
        started_at_ns: int | None = None,
        ended_at_ns: int | None = None,
    ) -> Event:
        with self._condition:
            pending = {request.memory_id for request in self._queue}
            if self._writer_current_id is not None:
                pending.add(self._writer_current_id)
        missing = set(memory_ids) - set(self.catalog) - pending
        if missing:
            raise KeyError(f"unknown MemoryID(s): {sorted(missing)}")
        now_ns = time.time_ns()
        event = Event(
            event_id=f"event_{now_ns}_{uuid.uuid4().hex[:8]}",
            started_at_ns=int(started_at_ns or now_ns),
            ended_at_ns=int(ended_at_ns or now_ns),
            memory_ids=tuple(memory_ids),
            summary=str(summary),
            tags=tuple(str(tag) for tag in tags),
        )
        with self._lock:
            self.events[event.event_id] = event
            with self.event_catalog_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(asdict(event), ensure_ascii=False) + "\n"
                )
                handle.flush()
        return event

    def record_experience(
        self,
        experience: dict[str, Any],
        *,
        related_memory_ids: list[str] | tuple[str, ...] = (),
    ) -> str:
        required = {
            "experience_version",
            "task",
            "state",
            "teacher",
            "action",
            "output",
            "environment",
            "timestamps",
        }
        missing = required - set(experience)
        if missing:
            raise ValueError(f"incomplete experience: {sorted(missing)}")
        memory_id = self.enqueue(
            experience, "experience", priority="critical"
        )
        if memory_id is None:
            raise RuntimeError("experience write was rejected")
        related = [
            item
            for item in (*related_memory_ids, memory_id)
            if item
        ]
        self.create_event(
            related,
            summary=str(
                experience.get("task", {}).get("instruction", "experience")
            ),
            tags=[
                "experience",
                str(experience.get("task", {}).get("task_id", "")),
                str(experience.get("status", "")),
            ],
            started_at_ns=int(
                experience.get("timestamps", {}).get(
                    "started_at_ns", time.time_ns()
                )
            ),
            ended_at_ns=int(
                experience.get("timestamps", {}).get(
                    "finished_at_ns", time.time_ns()
                )
            ),
        )
        return memory_id

    def read_event(self, event_id: str) -> Event | None:
        return self.events.get(event_id)

    def latest_event(self) -> Event | None:
        return max(
            self.events.values(),
            key=lambda item: item.ended_at_ns,
            default=None,
        )

    def force_promotion(self) -> bool:
        if (
            self._promotion_thread is not None
            and self._promotion_thread.is_alive()
        ):
            return False
        self._promotion_stop.clear()
        self._promotion_thread = threading.Thread(
            target=self._promotion_loop,
            name="eve-memory-promotion",
        )
        self._promotion_thread.start()
        return True

    def stop_promotion(self, timeout_s: float = 3.0) -> None:
        self._promotion_stop.set()
        thread = self._promotion_thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeError("memory promotion did not stop")
        self._promotion_thread = None

    @property
    def promotion_running(self) -> bool:
        return bool(
            self._promotion_thread is not None
            and self._promotion_thread.is_alive()
        )

    def promotion_status(self) -> dict[str, Any]:
        return dict(self._promotion_status)

    def _promotion_loop(self) -> None:
        started = time.monotonic()
        to_ltm = list(self.mtm)
        to_mtm = list(self.stm)
        candidates = [("ltm", item) for item in to_ltm] + [
            ("mtm", item) for item in to_mtm
        ]
        total = len(candidates)
        self._promotion_status.update(
            {
                "state": "running",
                "total": total,
                "processed": 0,
                "remaining": total,
                "eta_s": None,
                "last_error": None,
            }
        )
        try:
            promoted_mtm = 0
            promoted_ltm = 0
            for index, (target, memory_id) in enumerate(candidates, start=1):
                if self._promotion_stop.is_set():
                    self._promotion_status["state"] = "cancelled"
                    return
                if target == "ltm":
                    self.promote_to_ltm(memory_id)
                    promoted_ltm += 1
                else:
                    self.promote_to_mtm(memory_id)
                    promoted_mtm += 1
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = index / elapsed
                remaining = total - index
                self._promotion_status.update(
                    {
                        "processed": index,
                        "remaining": remaining,
                        "eta_s": remaining / rate if rate > 0 else None,
                    }
                )
            self._promotion_status.update(
                {
                    "state": "completed",
                    "eta_s": 0.0,
                    "last_result": {
                        "processed": total,
                        "promoted_mtm": promoted_mtm,
                        "promoted_ltm": promoted_ltm,
                        "finished_at_ns": time.time_ns(),
                    },
                }
            )
        except Exception as exc:
            self._promotion_status.update(
                {
                    "state": "error",
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            )

    def search(
        self,
        *,
        payload_type: str | None = None,
        keyword: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[str]:
        results: list[str] = []
        with self._lock:
            units = list(self.catalog.values())
        for unit in sorted(units, key=lambda item: item.created_at_ns):
            if payload_type is not None and unit.payload_type != payload_type:
                continue
            if start_ns is not None and unit.created_at_ns < start_ns:
                continue
            if end_ns is not None and unit.created_at_ns >= end_ns:
                continue
            if keyword is not None:
                payload = self.read(unit.memory_id)
                if not isinstance(payload, (dict, list, str, int, float, bool)):
                    continue
                payload_text = json.dumps(payload, ensure_ascii=False)
                if keyword.casefold() not in payload_text.casefold():
                    continue
            results.append(unit.memory_id)
        return results

    @staticmethod
    def _artifact_component(value: str, field_name: str) -> str:
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"invalid {field_name}: {value!r}")
        return value

    def store_tnn_artifact(
        self,
        source_directory: str,
        tnn_id: str,
        version: str,
    ) -> str:
        required = {
            "model.py",
            "weights.pt",
            "structure.json",
            "description.json",
            "training.json",
        }
        source = Path(source_directory).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"TNN artifact directory does not exist: {source}")
        missing = sorted(name for name in required if not (source / name).is_file())
        if missing:
            raise ValueError(f"incomplete TNN artifact; missing: {missing}")
        safe_tnn_id = self._artifact_component(str(tnn_id), "tnn_id")
        safe_version = self._artifact_component(str(version), "version")
        destination = self.tnn_dir / safe_tnn_id / safe_version
        if destination.exists():
            raise FileExistsError(
                f"TNN artifact already exists: {safe_tnn_id}/{safe_version}"
            )
        destination.mkdir(parents=True, exist_ok=False)
        for name in required:
            shutil.copy2(source / name, destination / name)
        return self.create(
            {
                "tnn_id": safe_tnn_id,
                "version": safe_version,
                "artifact_path": str(destination.resolve()),
            },
            payload_type="tnn_artifact",
        )

    def resolve_tnn_artifact(
        self,
        tnn_id: str,
        version: str | None = None,
    ) -> dict[str, str]:
        matches: list[tuple[MemoryUnit, dict[str, Any]]] = []
        direct = self.get_unit(tnn_id)
        with self._lock:
            catalog_values = list(self.catalog.values())
        candidates = (
            [direct]
            if direct is not None and direct.payload_type == "tnn_artifact"
            else catalog_values
        )
        for unit in candidates:
            if unit is None or unit.payload_type != "tnn_artifact":
                continue
            descriptor = self.read(unit.memory_id)
            if not isinstance(descriptor, dict):
                continue
            if (
                unit.memory_id == tnn_id or descriptor.get("tnn_id") == tnn_id
            ) and (version is None or descriptor.get("version") == version):
                matches.append((unit, descriptor))
        if not matches:
            suffix = f" version {version}" if version is not None else ""
            raise KeyError(f"unknown TNN artifact: {tnn_id}{suffix}")
        unit, descriptor = max(matches, key=lambda item: item[0].created_at_ns)
        artifact = Path(descriptor["artifact_path"]).resolve()
        required_paths = {
            "model_path": artifact / "model.py",
            "weights_path": artifact / "weights.pt",
            "description_path": artifact / "description.json",
            "structure_path": artifact / "structure.json",
            "training_path": artifact / "training.json",
        }
        missing = [str(path) for path in required_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cataloged TNN artifact is incomplete: {missing}")
        return {
            "memory_id": unit.memory_id,
            "tnn_id": str(descriptor["tnn_id"]),
            "version": str(descriptor["version"]),
            "artifact_path": str(artifact),
            **{name: str(path) for name, path in required_paths.items()},
        }

    def list_tnn_artifacts(self) -> list[dict[str, str]]:
        artifacts = []
        with self._lock:
            units = list(self.catalog.values())
        for unit in units:
            if unit.payload_type != "tnn_artifact":
                continue
            descriptor = self.read(unit.memory_id)
            if isinstance(descriptor, dict):
                artifacts.append(
                    {
                        "memory_id": unit.memory_id,
                        "tnn_id": str(descriptor.get("tnn_id", "")),
                        "version": str(descriptor.get("version", "")),
                        "artifact_path": str(descriptor.get("artifact_path", "")),
                    }
                )
        return sorted(
            artifacts, key=lambda item: (item["tnn_id"], item["version"])
        )

    def save_catalog(self) -> None:
        """Compatibility no-op: catalog mutations are already appended."""

    def load_catalog(self) -> None:
        if self.legacy_catalog_path.exists():
            data = json.loads(self.legacy_catalog_path.read_text(encoding="utf-8"))
            self.catalog = {
                key: MemoryUnit(**value)
                for key, value in data.get("units", {}).items()
            }
            self.stm = [
                item for item in data.get("stm", []) if item in self.catalog
            ]
            self.mtm = {
                item for item in data.get("mtm", []) if item in self.catalog
            }
            self.ltm = {
                item for item in data.get("ltm", []) if item in self.catalog
            }
            self.mtm.difference_update(self.ltm)
            self.stm = [
                item for item in self.stm
                if item not in self.mtm and item not in self.ltm
            ]
        if not self.catalog_path.exists():
            return
        for line in self.catalog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            operation = record.get("op")
            if operation == "create":
                unit = MemoryUnit(**record["unit"])
                self.catalog[unit.memory_id] = unit
                self.stm.append(unit.memory_id)
            elif operation == "delete":
                memory_id = record["memory_id"]
                self.catalog.pop(memory_id, None)
                self.stm = [item for item in self.stm if item != memory_id]
                self.mtm.discard(memory_id)
                self.ltm.discard(memory_id)
            elif operation == "promote" and record["memory_id"] in self.catalog:
                self.stm = [
                    item for item in self.stm if item != record["memory_id"]
                ]
                self.mtm.add(record["memory_id"])
            elif (
                operation == "promote_ltm"
                and record["memory_id"] in self.catalog
            ):
                self.stm = [
                    item for item in self.stm if item != record["memory_id"]
                ]
                self.mtm.discard(record["memory_id"])
                self.ltm.add(record["memory_id"])
        assigned = set(self.stm) | self.mtm | self.ltm
        self.mtm.update(set(self.catalog) - assigned)

    def _load_events(self) -> None:
        if not self.event_catalog_path.exists():
            return
        for line in self.event_catalog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            data["memory_ids"] = tuple(data.get("memory_ids", ()))
            data["tags"] = tuple(data.get("tags", ()))
            event = Event(**data)
            self.events[event.event_id] = event

    def _writer_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._writer_stop:
                    self._condition.wait()
                if self._writer_stop and not self._queue:
                    return
                request = self._queue.popleft()
                self._writer_busy = True
                self._writer_current_id = request.memory_id
            try:
                self._create_with_id(
                    request.memory_id,
                    request.payload,
                    request.payload_type,
                )
                with self._condition:
                    self._written_count += 1
            except Exception as exc:
                with self._condition:
                    self._failed_count += 1
                    self.last_writer_error = exc
                    self._writer_stop = True
                if self._writer_error_callback is not None:
                    self._writer_error_callback(exc)
            finally:
                with self._condition:
                    self._writer_busy = False
                    self._writer_current_id = None
                    self._condition.notify_all()

    def _create_with_id(
        self,
        memory_id: str,
        payload: Any,
        payload_type: str,
    ) -> None:
        created_at_ns = time.time_ns()
        path, encoded = self._write_payload(memory_id, payload)
        unit = MemoryUnit(
            memory_id=memory_id,
            payload_type=payload_type,
            created_at_ns=created_at_ns,
            storage_path=str(path.relative_to(self.base_dir)),
            content_hash=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )
        with self._lock:
            self.catalog[memory_id] = unit
            self._append_catalog({"op": "create", "unit": asdict(unit)})
            self._append_to_stm_locked(memory_id)

    def _write_payload(self, memory_id: str, payload: Any) -> tuple[Path, bytes]:
        array = self._as_array(payload)
        if array is not None:
            import io
            import numpy as np

            handle = io.BytesIO()
            np.save(handle, array, allow_pickle=False)
            encoded = handle.getvalue()
            path = self.payload_dir / f"{memory_id}.npy"
        else:
            normalized = self._json_value(payload)
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            path = self.payload_dir / f"{memory_id}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
        return path, encoded

    @staticmethod
    def _as_array(payload: Any) -> Any | None:
        try:
            import numpy as np

            if isinstance(payload, np.ndarray):
                return payload
        except ImportError:
            pass
        try:
            import torch

            if isinstance(payload, torch.Tensor):
                return payload.detach().cpu().numpy()
        except ImportError:
            pass
        return None

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return cls._json_value(asdict(value))
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if cls._as_array(value) is not None:
            raise TypeError(
                "arrays nested inside JSON payloads must be stored separately"
            )
        raise TypeError(f"payload is not JSON serializable: {type(value).__name__}")

    def _append_catalog(self, record: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.catalog_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    @staticmethod
    def _new_memory_id() -> str:
        return f"mem_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
