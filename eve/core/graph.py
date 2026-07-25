"""EVE 动态 TNN 运行图。"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── TNNOutputCache ──────────────────────────────────────

@dataclass
class CachedOutput:
    tnn_id: str
    output_name: str
    value: Any  # dict[str, torch.Tensor] or other
    timestamp_ns: int
    ttl_ns: int  # 0 = no expiry
    version: int


class TNNOutputCache:
    """TNN 输出缓存，支持 TTL 过期。"""

    def __init__(self):
        self._cache: dict[str, dict[str, CachedOutput]] = {}  # tnn_id → {output_name → cached}
        self._lock = threading.Lock()
        self._version_counter: dict[str, int] = {}  # tnn_id → version

    def put(
        self, tnn_id: str, output_name: str, value: Any, ttl_ns: int = 100_000_000
    ) -> None:
        """存入一条输出记录。"""
        now_ns = time.monotonic_ns()
        with self._lock:
            if tnn_id not in self._version_counter:
                self._version_counter[tnn_id] = 0
            self._version_counter[tnn_id] += 1
            version = self._version_counter[tnn_id]
            if tnn_id not in self._cache:
                self._cache[tnn_id] = {}
            self._cache[tnn_id][output_name] = CachedOutput(
                tnn_id=tnn_id,
                output_name=output_name,
                value=value,
                timestamp_ns=now_ns,
                ttl_ns=ttl_ns,
                version=version,
            )

    def get(self, tnn_id: str, output_name: str) -> Any | None:
        """获取一条输出，已过期返回 None。"""
        with self._lock:
            node_cache = self._cache.get(tnn_id, {})
            cached = node_cache.get(output_name)
            if cached is None:
                return None
            if self._is_expired(cached):
                del node_cache[output_name]
                return None
            return cached.value

    def get_all(self, tnn_id: str) -> dict[str, Any]:
        """获取某节点所有未过期输出。"""
        result: dict[str, Any] = {}
        with self._lock:
            node_cache = self._cache.get(tnn_id, {})
            expired: list[str] = []
            for name, cached in node_cache.items():
                if self._is_expired(cached):
                    expired.append(name)
                else:
                    result[name] = cached.value
            for name in expired:
                del node_cache[name]
        return result

    def invalidate(self, tnn_id: str) -> None:
        """清除某节点的所有输出缓存。"""
        with self._lock:
            self._cache.pop(tnn_id, None)
            self._version_counter.pop(tnn_id, None)

    def clear_expired(self) -> int:
        """清除所有过期条目，返回清除数量。"""
        removed = 0
        with self._lock:
            for tnn_id in list(self._cache.keys()):
                node_cache = self._cache[tnn_id]
                expired_names = [
                    name
                    for name, c in node_cache.items()
                    if self._is_expired(c)
                ]
                for name in expired_names:
                    del node_cache[name]
                    removed += 1
                if not node_cache:
                    del self._cache[tnn_id]
        return removed

    def _is_expired(self, cached: CachedOutput) -> bool:
        if cached.ttl_ns == 0:
            return False
        return time.monotonic_ns() - cached.timestamp_ns > cached.ttl_ns


# ── GraphNode ───────────────────────────────────────────

@dataclass
class GraphNode:
    tnn_id: str
    descriptor: Any  # TNNDescriptor
    run_frequency_hz: float = 10.0  # 0 = event-driven
    trigger_condition: str = ""
    last_run_ns: int = 0
    run_count: int = 0
    total_run_time_ns: int = 0
    status: str = "active"  # "active" | "paused" | "error"
    error_message: str = ""
    next_scheduled_ns: int = 0


@dataclass
class GraphEdge:
    from_node: str  # source tnn_id
    to_node: str  # target tnn_id
    output_field: str  # which output of source
    input_source_id: str  # which SourceRef of target this maps to
    created_at_ns: int = 0

    def __post_init__(self):
        if self.created_at_ns == 0:
            self.created_at_ns = time.monotonic_ns()


# ── GraphTrace ──────────────────────────────────────────

@dataclass
class GraphStateSnapshot:
    """某一时刻的图状态快照。"""
    timestamp_ns: int
    active_nodes: list[str]
    active_edges: list[tuple[str, str, str]]  # (from, to, output_field)
    node_frequencies: dict[str, float]
    node_errors: dict[str, str]


class GraphTrace:
    """运行轨迹记录器。"""

    def __init__(self, max_entries: int = 1000):
        self.snapshots: list[GraphStateSnapshot] = []
        self._max = max_entries

    def record(self, graph: TNNGraph) -> None:
        """记录当前图状态快照。"""
        active_edges: list[tuple[str, str, str]] = []
        for from_node, edges in graph._edges.items():
            for edge in edges:
                active_edges.append((edge.from_node, edge.to_node, edge.output_field))

        node_frequencies = {
            n.tnn_id: n.run_frequency_hz for n in graph._nodes.values()
        }
        node_errors = {
            n.tnn_id: n.error_message
            for n in graph._nodes.values()
            if n.error_message
        }

        snapshot = GraphStateSnapshot(
            timestamp_ns=time.monotonic_ns(),
            active_nodes=list(graph._nodes.keys()),
            active_edges=active_edges,
            node_frequencies=node_frequencies,
            node_errors=node_errors,
        )
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self._max:
            self.snapshots = self.snapshots[-self._max:]

    def get_recent(self, count: int = 10) -> list[GraphStateSnapshot]:
        """获取最近的 N 条快照。"""
        return self.snapshots[-count:] if self.snapshots else []

    def save(self, path: Path) -> None:
        """保存轨迹到 JSON 文件。"""
        data = []
        for s in self.snapshots:
            data.append({
                "timestamp_ns": s.timestamp_ns,
                "active_nodes": s.active_nodes,
                "active_edges": [
                    list(e) for e in s.active_edges
                ],
                "node_frequencies": s.node_frequencies,
                "node_errors": s.node_errors,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """从 JSON 文件加载轨迹。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.snapshots.clear()
        for entry in data:
            self.snapshots.append(GraphStateSnapshot(
                timestamp_ns=entry["timestamp_ns"],
                active_nodes=entry["active_nodes"],
                active_edges=[
                    tuple(e) for e in entry.get("active_edges", [])
                ],
                node_frequencies=entry.get("node_frequencies", {}),
                node_errors=entry.get("node_errors", {}),
            ))


# ── TNNGraph ────────────────────────────────────────────

class TNNGraph:
    """动态 TNN 运行图。

    负责：
    - 管理已加载 TNN 节点
    - 维护节点间的数据承接边
    - 异频调度（频率驱动 + 事件驱动）
    - TNN 输出缓存
    - 运行轨迹记录
    - active_tnn 集合差分

    不负责：
    - TNN 训练（Dock 的事）
    - 决定当前加载哪些 TNN（LLM 的事，但接受结果）
    """

    def __init__(self, tnn_store: Any, output_cache: TNNOutputCache | None = None):
        self._store = tnn_store  # TNNStore instance
        self._cache = output_cache or TNNOutputCache()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, list[GraphEdge]] = {}  # from_node → edges
        self._tracer = GraphTrace()
        self._lock = threading.Lock()
        self._running = False

    # ── 节点管理 ──

    def add_node(self, tnn_id: str) -> bool:
        """从 TNNStore 加载 TNN 并添加为图节点。"""
        with self._lock:
            if tnn_id in self._nodes:
                return True  # already present

            descriptor = self._store.get_descriptor(tnn_id)
            if descriptor is None:
                return False

            instance = self._store.load_tnn(tnn_id)
            if instance is None:
                return False

            self._nodes[tnn_id] = GraphNode(
                tnn_id=tnn_id,
                descriptor=descriptor,
                run_frequency_hz=descriptor.run_frequency_hz,
                trigger_condition=descriptor.trigger_condition,
                status="active",
            )
            return True

    def remove_node(self, tnn_id: str) -> bool:
        """移除节点，清理其输出缓存和边。"""
        with self._lock:
            if tnn_id not in self._nodes:
                return False

            del self._nodes[tnn_id]

            # 清理以该节点为起点的边
            self._edges.pop(tnn_id, None)

            # 清理以该节点为终点的边
            for from_node in list(self._edges.keys()):
                self._edges[from_node] = [
                    e for e in self._edges[from_node] if e.to_node != tnn_id
                ]
                if not self._edges[from_node]:
                    del self._edges[from_node]

            self._cache.invalidate(tnn_id)
            self._store.unload_tnn(tnn_id)
            return True

    def get_node(self, tnn_id: str) -> GraphNode | None:
        return self._nodes.get(tnn_id)

    def list_nodes(self) -> list[str]:
        return list(self._nodes.keys())

    def has_node(self, tnn_id: str) -> bool:
        return tnn_id in self._nodes

    # ── 边管理 ──

    def add_edge(
        self, from_tnn: str, to_tnn: str, output_field: str, input_source_id: str
    ) -> None:
        """添加一条数据承接边。"""
        with self._lock:
            if from_tnn not in self._nodes or to_tnn not in self._nodes:
                return
            if from_tnn not in self._edges:
                self._edges[from_tnn] = []
            # 避免重复边
            for existing in self._edges[from_tnn]:
                if (
                    existing.to_node == to_tnn
                    and existing.output_field == output_field
                    and existing.input_source_id == input_source_id
                ):
                    return
            self._edges[from_tnn].append(
                GraphEdge(
                    from_node=from_tnn,
                    to_node=to_tnn,
                    output_field=output_field,
                    input_source_id=input_source_id,
                )
            )

    def remove_edge(self, from_tnn: str, to_tnn: str) -> bool:
        """移除从 from_tnn 到 to_tnn 的所有边。"""
        with self._lock:
            if from_tnn not in self._edges:
                return False
            before = len(self._edges[from_tnn])
            self._edges[from_tnn] = [
                e for e in self._edges[from_tnn] if e.to_node != to_tnn
            ]
            after = len(self._edges[from_tnn])
            if not self._edges[from_tnn]:
                del self._edges[from_tnn]
            return before != after

    def get_downstream(self, tnn_id: str) -> list[str]:
        """获取下游节点 ID 列表。"""
        down: list[str] = []
        for edge in self._edges.get(tnn_id, []):
            if edge.to_node not in down:
                down.append(edge.to_node)
        return down

    def get_upstream(self, tnn_id: str) -> list[str]:
        """获取上游节点 ID 列表。"""
        up: list[str] = []
        for from_node, edges in self._edges.items():
            for edge in edges:
                if edge.to_node == tnn_id and from_node not in up:
                    up.append(from_node)
        return up

    def build_edges_from_descriptors(self) -> None:
        """根据已加载 TNN 的 SourceRef 自动建立边。

        对于每个节点的 SourceRef，如果 source_type 为 "tnn_output"，
        则从对应的 TNN 输出建立边。
        """
        with self._lock:
            for tnn_id, node in self._nodes.items():
                descriptor = node.descriptor
                for src_ref in descriptor.inputs:
                    if src_ref.source_type != "tnn_output":
                        continue
                    # source_id 格式: "tnn:<other_tnn_id>.<output_field>"
                    # 比如 "tnn:detector.target_position"
                    sid = src_ref.source_id
                    if not sid.startswith("tnn:"):
                        continue
                    # 移除 "tnn:" 前缀
                    rest = sid[4:]
                    if "." not in rest:
                        continue
                    parts = rest.split(".", 1)
                    source_tnn_id = parts[0]
                    output_field = parts[1]

                    if source_tnn_id not in self._nodes:
                        continue

                    self.add_edge(source_tnn_id, tnn_id, output_field, src_ref.source_id)

    # ── 集合差分 ──

    def sync_active_set(self, active_tnn_ids: list[str]) -> dict:
        """根据 LLM 输出的 active_tnn 集合，加载/卸载/保持。

        Returns: {"loaded": [...], "unloaded": [...], "kept": [...], "errors": [...]}
        """
        active_set = set(active_tnn_ids)
        current_set = set(self._nodes.keys())

        to_load = active_set - current_set
        to_unload = current_set - active_set
        kept = active_set & current_set

        loaded: list[str] = []
        unloaded: list[str] = []
        errors: list[str] = []

        for tnn_id in to_load:
            if self.add_node(tnn_id):
                loaded.append(tnn_id)
            else:
                errors.append(tnn_id)

        for tnn_id in to_unload:
            if self.remove_node(tnn_id):
                unloaded.append(tnn_id)

        return {
            "loaded": loaded,
            "unloaded": unloaded,
            "kept": list(kept),
            "errors": errors,
        }

    # ── 调度 ──

    def schedule(self, now_ns: int | None = None) -> list[str]:
        """返回应该在 now_ns 运行的 TNN ID 列表。

        考虑频率调度和事件触发。
        """
        if now_ns is None:
            now_ns = time.monotonic_ns()

        scheduled: list[str] = []
        with self._lock:
            for node in self._nodes.values():
                if node.status != "active":
                    continue
                if node.run_frequency_hz == 0:
                    continue  # event-driven, not scheduled here
                interval_ns = int(1e9 / node.run_frequency_hz)
                if now_ns >= node.next_scheduled_ns:
                    scheduled.append(node.tnn_id)
                    node.next_scheduled_ns = now_ns + interval_ns
        return scheduled

    def run_node(self, tnn_id: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """运行单个 TNN 节点，缓存输出。"""
        node = self._nodes.get(tnn_id)
        if node is None or node.status != "active":
            return None

        start_ns = time.monotonic_ns()
        try:
            instance = self._store.load_tnn(tnn_id)
            if instance is None:
                node.status = "error"
                node.error_message = "TNN instance not found"
                return None

            output = instance.forward(inputs)
            if output is None:
                return None

            # 缓存每个命名的输出
            for output_name, value in output.items():
                self._cache.put(tnn_id, output_name, value)

            with self._lock:
                node.last_run_ns = time.monotonic_ns()
                node.run_count += 1
                node.total_run_time_ns += time.monotonic_ns() - start_ns
                if node.status == "error":
                    node.status = "active"
                    node.error_message = ""

            return output

        except Exception as e:
            with self._lock:
                node.status = "error"
                node.error_message = str(e)
            return None

    def run_event_driven(
        self, tnn_id: str, event_type: str, event_data: Any
    ) -> dict[str, Any] | None:
        """触发事件驱动的 TNN。"""
        node = self._nodes.get(tnn_id)
        if node is None or node.status != "active":
            return None
        if node.run_frequency_hz != 0:
            return None  # not event-driven
        return self.run_node(tnn_id, {"event_type": event_type, "event_data": event_data})

    # ── 输出访问 ──

    def get_output(self, tnn_id: str, output_name: str) -> Any | None:
        """从缓存读取 TNN 输出。"""
        return self._cache.get(tnn_id, output_name)

    # ── 状态 ──

    def tick(self) -> None:
        """主 tick：调度到期节点、清理过期缓存、记录轨迹。"""
        if not self._running:
            return
        self._cache.clear_expired()
        self._tracer.record(self)

    def pause_node(self, tnn_id: str) -> None:
        """暂停节点调度。"""
        node = self._nodes.get(tnn_id)
        if node:
            node.status = "paused"

    def resume_node(self, tnn_id: str) -> None:
        """恢复节点调度。"""
        node = self._nodes.get(tnn_id)
        if node:
            node.status = "active"
            node.error_message = ""

    def get_active_subgraph(self) -> dict:
        """返回当前活动子图快照。"""
        with self._lock:
            nodes = list(self._nodes.keys())
            edges: list[dict] = []
            for from_node, edge_list in self._edges.items():
                for edge in edge_list:
                    edges.append({
                        "from": edge.from_node,
                        "to": edge.to_node,
                        "output_field": edge.output_field,
                        "input_source_id": edge.input_source_id,
                    })
            return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        """统计信息：节点数、边数、各节点频率、运行次数、最近错误。"""
        with self._lock:
            total_edges = sum(len(el) for el in self._edges.values())
            node_stats: dict[str, dict] = {}
            errors: list[dict] = []
            for node in self._nodes.values():
                node_stats[node.tnn_id] = {
                    "frequency_hz": node.run_frequency_hz,
                    "run_count": node.run_count,
                    "last_run_ns": node.last_run_ns,
                    "status": node.status,
                }
                if node.error_message:
                    errors.append({
                        "tnn_id": node.tnn_id,
                        "error": node.error_message,
                    })
            return {
                "node_count": len(self._nodes),
                "edge_count": total_edges,
                "nodes": node_stats,
                "recent_errors": errors,
                "running": self._running,
            }

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
