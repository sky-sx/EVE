"""
EVE TNN Store — TNN 本地存储与索引。

管理 TNN 的版本化权重、描述符和网络结构信息。
所有 TNN 权重文件存放在 {base_dir}/TNNweights/ 下，
按 {tnn_id}/v{version}/ 组织。
"""
from __future__ import annotations

import json
import shutil
import time
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eve.core.tnn_base import TNNBase, TNNDescriptor, _descriptor_from_dict


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class TNNStoreEntry:
    """TNN 存储条目。"""
    tnn_id: str
    descriptor_path: str
    weights_path: str
    structure_path: str    # 指向 descriptor.json（结构信息集成其中）
    version: int
    status: str            # "available" | "loading" | "loaded" | "error" | "unavailable"
    loaded_at_ns: int = 0
    error_message: str = ""


# ── TNN Store ─────────────────────────────────────────────


class TNNStore:
    """TNN 本地存储与索引。

    目录结构：
      {base_dir}/TNNweights/{tnn_id}/v{version}/
          descriptor.json
          weights.pt
          structure.json
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.weights_dir = self.base_dir / "TNNweights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, TNNStoreEntry] = {}
        self._loaded_instances: dict[str, TNNBase] = {}
        self._scan()

    # ── 公共 API ───────────────────────────────────────────

    def register(
        self,
        tnn_id: str,
        descriptor: TNNDescriptor,
        weights_path: str,
        structure_version: str,
    ) -> TNNStoreEntry:
        """注册 TNN 到存储。

        将 descriptor 写入存储目录，复制 weights 文件，
        记录 structure_version。
        """
        from dataclasses import asdict

        version_dir = self._version_dir(tnn_id, descriptor.version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # 保存 descriptor
        desc_dict = asdict(descriptor)
        desc_dict["tnn_class"] = desc_dict.get("tnn_class", "")
        desc_path = version_dir / "descriptor.json"
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(desc_dict, f, indent=2, ensure_ascii=False)

        # 复制 weights
        dst_weights = version_dir / "weights.pt"
        src_weights = Path(weights_path)
        if src_weights != dst_weights:
            shutil.copy2(src_weights, dst_weights)

        # 保存结构版本信息
        structure_path = version_dir / "structure.json"
        structure_info = {"structure_version": structure_version}
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(structure_info, f, indent=2, ensure_ascii=False)

        entry = TNNStoreEntry(
            tnn_id=tnn_id,
            descriptor_path=str(desc_path),
            weights_path=str(dst_weights),
            structure_path=str(structure_path),
            version=descriptor.version,
            status="available",
        )
        self._entries[tnn_id] = entry
        return entry

    def list_available(self) -> list[str]:
        """列出所有已注册的 TNN ID。"""
        return list(self._entries.keys())

    def list_loaded(self) -> list[str]:
        """列出当前已加载到内存的 TNN ID。"""
        return list(self._loaded_instances.keys())

    def get_descriptor(self, tnn_id: str) -> TNNDescriptor | None:
        """获取 TNN 的描述符（从磁盘读取，保证最新）。"""
        entry = self._entries.get(tnn_id)
        if entry is None:
            return None
        desc_path = Path(entry.descriptor_path)
        if not desc_path.exists():
            return None
        with open(desc_path, "r", encoding="utf-8") as f:
            desc_dict = json.load(f)
        return _descriptor_from_dict(desc_dict)

    def load_tnn(self, tnn_id: str, device: str = "cuda") -> TNNBase | None:
        """加载 TNN 权重并返回实例。

        如果已加载则直接返回缓存实例。
        加载成功后将 status 更新为 "loaded"。
        """
        # 已加载则返回缓存
        if tnn_id in self._loaded_instances:
            return self._loaded_instances[tnn_id]

        entry = self._entries.get(tnn_id)
        if entry is None:
            return None

        try:
            entry.status = "loading"

            # 读取 descriptor 以确定 TNN 类
            desc_path = Path(entry.descriptor_path)
            with open(desc_path, "r", encoding="utf-8") as f:
                desc_dict = json.load(f)

            tnn_class_path = desc_dict.get("tnn_class") or "eve.core.tnn_base.DummyTNN"

            # 动态导入 TNN 类
            module_path, class_name = tnn_class_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            target_cls = getattr(mod, class_name)

            # 使用类方法加载
            version_dir = Path(entry.weights_path).parent
            instance = target_cls.load(str(version_dir))
            instance.to(device)
            instance.eval()

            self._loaded_instances[tnn_id] = instance
            entry.status = "loaded"
            entry.loaded_at_ns = time.monotonic_ns()
            entry.error_message = ""
            return instance

        except Exception as e:
            entry.status = "error"
            entry.error_message = str(e)
            return None

    def unload_tnn(self, tnn_id: str) -> None:
        """从内存卸载 TNN，保留磁盘文件。"""
        self._loaded_instances.pop(tnn_id, None)
        entry = self._entries.get(tnn_id)
        if entry and entry.status == "loaded":
            entry.status = "available"
            entry.loaded_at_ns = 0

    def save_new_version(self, tnn_id: str, tnn: TNNBase, new_version: int) -> str:
        """保存 TNN 新版本，返回版本目录路径。"""
        version_dir = self._version_dir(tnn_id, new_version)
        version_dir.mkdir(parents=True, exist_ok=True)
        tnn.save(version_dir)

        # 保存结构版本信息
        structure_path = version_dir / "structure.json"
        structure_info = {"structure_version": str(new_version)}
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(structure_info, f, indent=2, ensure_ascii=False)

        entry = TNNStoreEntry(
            tnn_id=tnn_id,
            descriptor_path=str(version_dir / "descriptor.json"),
            weights_path=str(version_dir / "weights.pt"),
            structure_path=str(structure_path),
            version=new_version,
            status="available",
        )
        self._entries[tnn_id] = entry
        return str(version_dir)

    def rollback(self, tnn_id: str, target_version: int) -> bool:
        """回滚到指定版本。

        切换到目标版本的 descriptor 和 weights。
        如果目标版本文件不存在则返回 False。
        """
        target_dir = self._version_dir(tnn_id, target_version)
        desc_path = target_dir / "descriptor.json"
        weights_path = target_dir / "weights.pt"
        structure_path = target_dir / "structure.json"

        if not desc_path.exists() or not weights_path.exists():
            return False

        # 卸载当前版本
        self.unload_tnn(tnn_id)

        entry = TNNStoreEntry(
            tnn_id=tnn_id,
            descriptor_path=str(desc_path),
            weights_path=str(weights_path),
            structure_path=str(structure_path) if structure_path.exists() else str(desc_path),
            version=target_version,
            status="available",
        )
        self._entries[tnn_id] = entry
        return True

    def mark_unavailable(self, tnn_id: str, reason: str) -> None:
        """将 TNN 标记为不可用，同时从内存卸载。"""
        entry = self._entries.get(tnn_id)
        if entry:
            self.unload_tnn(tnn_id)
            entry.status = "unavailable"
            entry.error_message = reason

    def stats(self) -> dict[str, Any]:
        """返回存储统计信息。"""
        status_counts: dict[str, int] = {}
        for e in self._entries.values():
            status_counts[e.status] = status_counts.get(e.status, 0) + 1

        return {
            "total_entries": len(self._entries),
            "loaded_instances": len(self._loaded_instances),
            "by_status": status_counts,
            "entries": {
                tid: {"version": e.version, "status": e.status}
                for tid, e in self._entries.items()
            },
        }

    # ── 内部方法 ───────────────────────────────────────────

    def _version_dir(self, tnn_id: str, version: int) -> Path:
        """获取指定 TNN 版本的目录路径。"""
        return self.weights_dir / tnn_id / f"v{version}"

    def _scan(self) -> None:
        """扫描 TNNweights/ 目录，索引已存在的 TNN 文件。

        对每个 tnn_id 只注册最新发现的版本。
        """
        if not self.weights_dir.exists():
            return

        for tnn_dir in sorted(self.weights_dir.iterdir()):
            if not tnn_dir.is_dir():
                continue
            tnn_id = tnn_dir.name

            # 按版本号降序排列，取最新有效版本
            version_dirs = sorted(
                [d for d in tnn_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
                key=lambda d: self._parse_version(d.name),
                reverse=True,
            )

            for version_dir in version_dirs:
                desc_path = version_dir / "descriptor.json"
                weights_path = version_dir / "weights.pt"
                if desc_path.exists() and weights_path.exists():
                    version = self._parse_version(version_dir.name)
                    structure_path = version_dir / "structure.json"
                    entry = TNNStoreEntry(
                        tnn_id=tnn_id,
                        descriptor_path=str(desc_path),
                        weights_path=str(weights_path),
                        structure_path=str(structure_path) if structure_path.exists() else str(desc_path),
                        version=version,
                        status="available",
                    )
                    self._entries[tnn_id] = entry
                    break  # 只取最新版本

    @staticmethod
    def _parse_version(dir_name: str) -> int:
        """从 'v{N}' 目录名解析版本号。"""
        try:
            return int(dir_name[1:])
        except (ValueError, IndexError):
            return 0
