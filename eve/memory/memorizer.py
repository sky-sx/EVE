"""Memorizer: EVE Memory 总管理器。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from eve.memory.catalog import Catalog, CatalogEntry


class Memorizer:
    """EVE Memory 总管理器。

    STM = 最近 MemoryID 集合（最多 1000 个）。
    MTM = 当前任务工作集 MemoryID。
    LTM = 持久化 payload 存储。
    """

    _STM_CAP = 1000
    _EXT_MAP: dict[str, str] = {
        "text": ".txt",
        "json": ".json",
        "numpy": ".npy",
        "image": ".png",
        "audio": ".wav",
        "other": ".bin",
    }

    def __init__(self, base_dir: str | Path) -> None:
        self.catalog = Catalog()
        self.stm: list[str] = []
        self.mtm: list[str] = []
        self.base_dir = Path(base_dir)
        self.ltm_dir = self.base_dir / "LTM"
        self.ltm_dir.mkdir(parents=True, exist_ok=True)
        self._index_counter = 0

    # ── CRUD ───────────────────────────────────────────────

    def create(
        self,
        payload: Any,
        payload_type: str,
        persistent: bool = True,
    ) -> str:
        """创建 MemoryUnit，返回 memory_id。

        payload 类型处理：
        - text/str → .txt
        - json/dict → .json
        - numpy ndarray (image) → .png
        - numpy ndarray (numpy) → .npy
        """
        self._index_counter += 1
        now_ns = time.monotonic_ns()
        date_str = datetime.now().strftime("%Y%m%d")
        ts_part = now_ns % 1_000_000_000
        memory_id = f"mem_{self._index_counter:06d}_{date_str}_{ts_part:09d}"

        ext = self._EXT_MAP.get(payload_type, ".bin")
        first2 = memory_id[:2]
        rel_dir = Path(payload_type) / first2
        rel_path = str(rel_dir / f"{memory_id}{ext}")

        abs_dir = self.ltm_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        abs_path = self.ltm_dir / rel_path

        content_bytes: bytes
        if payload_type == "text":
            content = payload if isinstance(payload, str) else str(payload)
            content_bytes = content.encode("utf-8")
            abs_path.write_bytes(content_bytes)
        elif payload_type == "json":
            content_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            abs_path.write_bytes(content_bytes)
        elif payload_type == "numpy":
            import numpy as np
            arr = np.asarray(payload)
            np.save(str(abs_path), arr)
            content_bytes = abs_path.read_bytes()
        elif payload_type == "image":
            import numpy as np
            arr = np.asarray(payload)
            try:
                import cv2
                cv2.imwrite(str(abs_path), arr)
            except ImportError:
                from PIL import Image
                if arr.ndim == 2:
                    img = Image.fromarray(arr, mode="L")
                elif arr.shape[2] == 4:
                    img = Image.fromarray(arr, mode="RGBA")
                else:
                    img = Image.fromarray(arr, mode="RGB")
                img.save(str(abs_path))
            content_bytes = abs_path.read_bytes()
        elif payload_type == "audio":
            content_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
            abs_path.write_bytes(content_bytes)
        else:
            content_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
            abs_path.write_bytes(content_bytes)

        content_hash = hashlib.sha256(content_bytes).hexdigest()

        entry = CatalogEntry(
            memory_id=memory_id,
            storage_path=rel_path,
            payload_type=payload_type,
            created_at_ns=now_ns,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
            persistent=persistent,
            resident=True,
        )
        self.catalog.register(entry)
        self.add_to_stm(memory_id)
        return memory_id

    def read(self, memory_id: str) -> Any | None:
        """读取 payload。"""
        entry = self.catalog.lookup(memory_id)
        if entry is None:
            return None
        abs_path = self.ltm_dir / entry.storage_path
        if not abs_path.exists():
            return None

        ptype = entry.payload_type
        if ptype == "text":
            return abs_path.read_text(encoding="utf-8")
        elif ptype == "json":
            return json.loads(abs_path.read_text(encoding="utf-8"))
        elif ptype == "numpy":
            import numpy as np
            return np.load(str(abs_path))
        elif ptype == "image":
            try:
                import cv2
                return cv2.imread(str(abs_path))
            except ImportError:
                from PIL import Image
                import numpy as np
                img = Image.open(str(abs_path))
                return np.array(img)
        elif ptype == "audio":
            return abs_path.read_bytes()
        else:
            return abs_path.read_bytes()

    def delete(self, memory_id: str) -> bool:
        """删除 LTM 文件和 catalog 条目，同时从 STM/MTM 中移除。"""
        entry = self.catalog.lookup(memory_id)
        if entry is None:
            return False
        abs_path = self.ltm_dir / entry.storage_path
        try:
            if abs_path.exists():
                abs_path.unlink()
        except OSError:
            pass
        self.catalog.delete(memory_id)
        self._remove_from_list(self.stm, memory_id)
        self._remove_from_list(self.mtm, memory_id)
        return True

    # ── STM ────────────────────────────────────────────────

    def add_to_stm(self, memory_id: str) -> None:
        """添加到 STM，超过上限时移除最旧的。"""
        self._remove_from_list(self.stm, memory_id)
        self.stm.append(memory_id)
        if len(self.stm) > self._STM_CAP:
            self.stm = self.stm[-self._STM_CAP:]

    def is_in_stm(self, memory_id: str) -> bool:
        return memory_id in self.stm

    def get_stm_ids(self) -> list[str]:
        return list(self.stm)

    # ── MTM ────────────────────────────────────────────────

    def promote_to_mtm(self, memory_id: str) -> None:
        """从 STM 提升到 MTM（当前任务工作集）。"""
        if memory_id not in self.mtm:
            self.mtm.append(memory_id)
        # 保持在 STM 中（MTM 是 STM 的子集视角）

    def demote_from_mtm(self, memory_id: str) -> None:
        self._remove_from_list(self.mtm, memory_id)

    def is_in_mtm(self, memory_id: str) -> bool:
        return memory_id in self.mtm

    def get_mtm_ids(self) -> list[str]:
        return list(self.mtm)

    # ── Persistence ────────────────────────────────────────

    def stats(self) -> dict:
        cat_stats = self.catalog.stats()
        return {
            "stm_count": len(self.stm),
            "mtm_count": len(self.mtm),
            "ltm_count": cat_stats["total_entries"],
            "total_size_bytes": cat_stats["total_size_bytes"],
        }

    def save_catalog(self) -> None:
        path = self.base_dir / "catalog.json"
        self.catalog.save(path)

    def load_catalog(self) -> None:
        path = self.base_dir / "catalog.json"
        self.catalog.load(path)

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _remove_from_list(lst: list[str], item: str) -> None:
        try:
            lst.remove(item)
        except ValueError:
            pass
