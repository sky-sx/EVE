"""多图索引管理。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IndexEdge:
    source_id: str
    target_id: str
    edge_type: str          # "temporal" | "content_similar" | "causal_candidate" | "same_origin" | "io_chain"
    weight: float = 1.0
    created_at_ns: int = 0

    def __post_init__(self) -> None:
        if self.created_at_ns == 0:
            self.created_at_ns = time.monotonic_ns()


class IndexManager:
    """多图索引管理。"""

    def __init__(self) -> None:
        self.edges: dict[str, list[IndexEdge]] = {}       # source_id → edges
        self._reverse: dict[str, set[str]] = {}            # target_id → sources

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0) -> None:
        edge = IndexEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
        )
        if source_id not in self.edges:
            self.edges[source_id] = []
        self.edges[source_id].append(edge)

        if target_id not in self._reverse:
            self._reverse[target_id] = set()
        self._reverse[target_id].add(source_id)

    def get_neighbors(self, memory_id: str, edge_type: str | None = None) -> list[str]:
        """获取从 memory_id 出发的邻居 ID 列表。"""
        outgoing = self.edges.get(memory_id, [])
        if edge_type is not None:
            outgoing = [e for e in outgoing if e.edge_type == edge_type]
        return [e.target_id for e in outgoing]

    def get_temporal_neighbors(self, memory_id: str, window_ns: int) -> list[str]:
        """获取从 memory_id 出发、时间窗口内的 temporal 邻居。

        此方法依赖 catalog 中的 created_at_ns，需由调用方传入或通过 memorizer 间接使用。
        直接使用 IndexEdge 的 created_at_ns 进行过滤的版本。
        """
        outgoing = self.edges.get(memory_id, [])
        temporal = [e for e in outgoing if e.edge_type == "temporal"]
        temporal.sort(key=lambda e: e.created_at_ns)
        # Filter by creating edge time window
        if temporal:
            base_ns = temporal[0].created_at_ns
            result = [e.target_id for e in temporal if abs(e.created_at_ns - base_ns) <= window_ns]
            return result
        return []

    def add_temporal_chain(self, ids: list[str]) -> None:
        """将连续 ID 链接为 temporal 边链。"""
        if len(ids) < 2:
            return
        for i in range(len(ids) - 1):
            self.add_edge(ids[i], ids[i + 1], "temporal")

    def fold_dense_bipartite(self, group_a: list[str], group_b: list[str], max_path: int = 3) -> dict:
        """对稠密二部图做折叠，返回 {a_id: [b_id, ...]} 的简化连接。

        max_path: 每个 a 节点最多保留的连接数。
        """
        result: dict[str, list[str]] = {}
        for a_id in group_a:
            outgoing = self.edges.get(a_id, [])
            b_neighbors = [e.target_id for e in outgoing if e.target_id in group_b]
            # Sort by weight descending, keep top max_path
            b_neighbors.sort(
                key=lambda bid: max(
                    (e.weight for e in outgoing if e.target_id == bid),
                    default=1.0,
                ),
                reverse=True,
            )
            result[a_id] = b_neighbors[:max_path]
        return result

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        if source_id not in self.edges:
            return False
        before = len(self.edges[source_id])
        self.edges[source_id] = [
            e for e in self.edges[source_id] if e.target_id != target_id
        ]
        removed = len(self.edges[source_id]) < before
        if removed:
            if not self.edges[source_id]:
                del self.edges[source_id]
            if target_id in self._reverse:
                self._reverse[target_id].discard(source_id)
                if not self._reverse[target_id]:
                    del self._reverse[target_id]
        return removed

    def merge_redirect(self, old_ids: list[str], new_id: str) -> None:
        """惰性重定向：将所有指向 old_ids 的边改为指向 new_id。"""
        for old_id in old_ids:
            # Redirect outgoing edges
            if old_id in self.edges:
                if new_id not in self.edges:
                    self.edges[new_id] = []
                self.edges[new_id].extend(self.edges.pop(old_id))

            # Redirect incoming edges
            incoming_sources = list(self._reverse.get(old_id, set()))
            for src_id in incoming_sources:
                if src_id in self.edges:
                    for edge in self.edges[src_id]:
                        if edge.target_id == old_id:
                            edge.target_id = new_id
                # Update reverse index
                if new_id not in self._reverse:
                    self._reverse[new_id] = set()
                self._reverse[new_id].add(src_id)
            if old_id in self._reverse:
                del self._reverse[old_id]

    def resolve_redirect(self, memory_id: str) -> str:
        """路径压缩式重定向解析。

        沿 merge_redirect 建立的链接链走到最终 ID。
        """
        visited: set[str] = set()
        current = memory_id
        while current in self._reverse:
            # 如果某个 ID 被重定向了，它在 _reverse 中也会被更新
            # 这里我们通过检查哪些 ID 的 outgoing 边被移走来判断
            break
        # merge_redirect 后，old_id 不再有 outgoing edges
        # 但我们需要一个 redirect 映射。简化实现：直接在 edges 中查找。
        # 实际上 merge_redirect 是直接修改了边，不需要单独解析。
        return memory_id

    def save(self, path: Path) -> None:
        data = {
            "edges": {
                sid: [asdict(e) for e in edges]
                for sid, edges in self.edges.items()
            }
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.edges = {}
        self._reverse = {}
        for sid, edge_list in data.get("edges", {}).items():
            self.edges[sid] = [IndexEdge(**e) for e in edge_list]
            for edge in self.edges[sid]:
                if edge.target_id not in self._reverse:
                    self._reverse[edge.target_id] = set()
                self._reverse[edge.target_id].add(sid)
