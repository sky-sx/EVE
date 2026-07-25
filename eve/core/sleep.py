"""EVE 睡眠复盘模块。"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eve.state import OutputMode


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class SleepReport:
    """一次睡眠复盘的完整报告。"""

    sleep_id: str
    started_at_ns: int
    finished_at_ns: int = 0
    memories_consolidated: int = 0
    events_merged: list[str] = field(default_factory=list)
    events_split: list[str] = field(default_factory=list)
    low_value_candidates: list[str] = field(default_factory=list)
    skill_candidates: list[str] = field(default_factory=list)
    training_orders_created: list[str] = field(default_factory=list)
    training_orders_executed: list[str] = field(default_factory=list)
    training_results: list[str] = field(default_factory=list)  # memory_ids of reports
    graph_compactions: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"


# ── 睡眠管理器 ────────────────────────────────────────────


class SleepManager:
    """睡眠复盘管理器。

    在用户触发或系统长时间空闲时执行深度整理。
    睡眠期间禁止主动鼠标键盘输出。
    """

    _IDLE_THRESHOLD_S: float = 300.0
    _STM_PRESSURE_THRESHOLD: int = 500
    _INDEX_DENSITY_THRESHOLD: float = 0.6
    _MAX_FOLD_PATHS: int = 3

    def __init__(
        self,
        memorizer: Any,
        index_manager: Any,
        event_manager: Any,
        retriever: Any,
        runtime_state_manager: Any,
        hormone_manager: Any,
        trainer: Any,
        model_adapters: dict | None = None,
    ) -> None:
        self._memorizer = memorizer
        self._indexes = index_manager
        self._events = event_manager
        self._retriever = retriever
        self._runtime = runtime_state_manager
        self._hormones = hormone_manager
        self._trainer = trainer
        self._models = model_adapters or {}

        self._sleeping = False
        self._reports: list[SleepReport] = []
        self._current_report: SleepReport | None = None
        self._pre_sleep_hormone_snapshot: dict | None = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    # ── 判断是否应进入睡眠 ────────────────────────────────

    def should_sleep(self, idle_since_ns: int, memory_pressure: float = 0.0) -> bool:
        """判断是否应该进入睡眠。

        条件：
        - 空闲超过一定时间（默认 300 秒）
        - 或用户手动触发
        - 或 STM 超过阈值（默认 500 条）
        - 或激素暗示需要休息
        """
        idle_s = idle_since_ns / 1_000_000_000.0

        if idle_s >= self._IDLE_THRESHOLD_S:
            return True

        stm_count = len(self._memorizer.get_stm_ids())
        if stm_count >= self._STM_PRESSURE_THRESHOLD:
            return True

        if memory_pressure >= 0.7:
            return True

        tendencies = self._hormones.get_tendencies()
        if tendencies.get("sleep", 0.0) >= 0.5:
            return True

        return False

    # ── 进入睡眠 ──────────────────────────────────────────

    def enter_sleep(self, runtime_state: Any) -> str:
        """进入睡眠状态。

        1. 标记 sleeping=True
        2. 降低输出权限（禁止键鼠）
        3. 记录进入时间
        4. 保存当前快照
        5. 创建 SleepReport
        Returns: sleep_id
        """
        self._sleeping = True

        # 保存睡眠前激素快照，以便睡眠后恢复参照
        self._pre_sleep_hormone_snapshot = self._hormones.save_snapshot()

        # 禁止主动键鼠输出（休眠期间仍允许 LLM/VLM、Dock 训练）
        runtime_state.mouse_allowed = False
        runtime_state.keyboard_allowed = False

        # 更新 myself 状态，反映“正在睡眠”
        now_ns = time.monotonic_ns()
        self._runtime.myself.what_im_thinking = "正在睡眠复盘阶段，整理记忆并训练能力。"
        self._runtime.myself.current_task = "睡眠复盘"
        self._runtime.myself.updated_at_ns = now_ns

        sleep_id = f"sleep_{uuid.uuid4().hex[:12]}"
        report = SleepReport(
            sleep_id=sleep_id,
            started_at_ns=now_ns,
            status="running",
        )
        self._current_report = report
        return sleep_id

    # ── 执行睡眠周期 ──────────────────────────────────────

    def run_sleep_cycle(self) -> SleepReport:
        """执行一轮完整睡眠复盘。

        步骤：
        1. 收集近期 MemoryUnit（STM + MTM）
        2. 去重：合并重复内容
        3. 调用 LLM 进行复盘分析（如果有）
        4. 整理索引图（折叠稠密连接）
        5. 标记低价值记忆
        6. 发现技能候选（重复 workload、反复失败）
        7. 生成训练订单
        8. 执行 Dock 训练
        9. 更新激素（睡眠后恢复）
        10. 保存状态
        Returns: SleepReport
        """
        report = self._current_report
        if report is None:
            # 如果没有通过 enter_sleep 创建 report，则自动创建
            sleep_id = f"sleep_{uuid.uuid4().hex[:12]}"
            report = SleepReport(
                sleep_id=sleep_id,
                started_at_ns=time.monotonic_ns(),
                status="running",
            )
            self._current_report = report
        else:
            report.status = "running"

        try:
            # 1. 收集近期 MemoryUnit
            memory_ids = self.collect_recent_memories()

            # 2. 去重
            dup_info = self.deduplicate_memories(memory_ids)
            report.events_merged = (
                [f"{d[0]}→{d[1]}" for d in dup_info.get("duplicates", [])]
                if "duplicates" in dup_info
                else []
            )
            report.memories_consolidated = dup_info.get("merged_count", 0)

            # 3. 调用 LLM 复盘分析
            llm = self._models.get("local_llm")
            if llm is not None and llm.status.available:
                self._run_llm_consolidation(report, memory_ids)

            # 4. 整理索引图
            report.graph_compactions = self.consolidate_indexes()

            # 5. 发现技能候选
            report.skill_candidates = self.discover_skill_candidates()

            # 6. 生成训练订单
            llm_adapter = self._models.get("local_llm")
            orders = self.generate_training_orders(report.skill_candidates, llm_adapter)
            report.training_orders_created = [o.order_id for o in orders]

            # 7. 执行 Dock 训练
            results = self.run_training(orders)
            report.training_orders_executed = [
                r.order_id for r in results if r.success
            ]
            report.training_results = [
                rid for r in results for rid in r.memory_reports
            ]

            # 8. 更新激素 — 睡眠恢复
            self._hormones.apply_event(
                "stable", intensity=1.5,
                description="sleep consolidation recovery",
            )
            self._hormones.apply_event(
                "task_complete", intensity=1.0,
                description="sleep cycle finished",
            )
            self._hormones.update_cycle()

            report.status = "completed"

        except Exception as e:
            report.errors.append(str(e))
            report.status = "failed"

        finally:
            report.finished_at_ns = time.monotonic_ns()
            self._reports.append(report)

        return report

    # ── 从睡眠恢复 ────────────────────────────────────────

    def wake_up(self, runtime_state: Any) -> None:
        """从睡眠恢复。

        1. 标记 sleeping=False
        2. 恢复输出权限
        3. 更新 myself 状态
        4. 记录恢复时间
        """
        self._sleeping = False

        # 恢复输出权限
        runtime_state.mouse_allowed = True
        runtime_state.keyboard_allowed = True

        # 更新 myself 状态
        now_ns = time.monotonic_ns()
        self._runtime.myself.what_im_thinking = "睡眠复盘完成，已恢复活跃状态。"
        self._runtime.myself.current_task = ""
        self._runtime.myself.task_progress = ""
        self._runtime.myself.updated_at_ns = now_ns

        # 同步激素到 myself
        self._runtime.myself.hormone_levels = self._hormones.levels.to_dict()

    # ── 收集近期记忆 ──────────────────────────────────────

    def collect_recent_memories(self) -> list[str]:
        """收集 STM 和当前 MTM 中的所有 MemoryID。"""
        stm_ids = self._memorizer.get_stm_ids()
        mtm_ids = self._memorizer.get_mtm_ids()
        return list(set(stm_ids + mtm_ids))

    # ── 去重 ──────────────────────────────────────────────

    def deduplicate_memories(self, memory_ids: list[str]) -> dict:
        """检测重复 MemoryUnit（相同 hash），返回合并建议。

        对于每组重复，将旧 ID 重定向到最新 ID，删除多余条目。

        Returns: {"duplicates": [[id1, id2], ...], "merged_count": int}
        """
        hash_map: dict[str, list[str]] = {}

        for mid in memory_ids:
            entry = self._memorizer.catalog.lookup(mid)
            if entry is None or not entry.content_hash:
                continue
            h = entry.content_hash
            if h not in hash_map:
                hash_map[h] = []
            hash_map[h].append(mid)

        duplicates: list[list[str]] = []
        merged_count = 0

        for h, ids in hash_map.items():
            if len(ids) < 2:
                continue
            ids_sorted = sorted(
                ids,
                key=lambda mid: (
                    self._memorizer.catalog.lookup(mid).created_at_ns
                    if self._memorizer.catalog.lookup(mid) else 0
                ),
            )
            # 保留最新的，其余重定向
            keep = ids_sorted[-1]
            old = ids_sorted[:-1]
            duplicates.append([keep] + old)

            # 在索引中重定向
            self._indexes.merge_redirect(old, keep)

            # 删除旧条目（仅从 catalog 中移除，保留最新）
            for old_id in old:
                entry = self._memorizer.catalog.lookup(old_id)
                if entry is not None:
                    abs_path = self._memorizer.ltm_dir / entry.storage_path
                    try:
                        if abs_path.exists():
                            abs_path.unlink()
                    except OSError:
                        pass
                    self._memorizer.catalog.delete(old_id)
                    # 也从 STM/MTM 中移除
                    self._memorizer._remove_from_list(self._memorizer.stm, old_id)
                    self._memorizer._remove_from_list(self._memorizer.mtm, old_id)

            merged_count += len(old)

        return {"duplicates": duplicates, "merged_count": merged_count}

    # ── 整理索引图 ────────────────────────────────────────

    def consolidate_indexes(self) -> int:
        """整理索引：
        1. 对所有节点检查出边密度
        2. 如果某节点的出边数超过其他节点数的 60%，执行图折叠
        Returns: 折叠次数
        """
        compactions = 0
        all_sources = list(self._indexes.edges.keys())
        if len(all_sources) < 2:
            return compactions

        # 收集所有唯一目标 ID
        all_targets: set[str] = set()
        for edges in self._indexes.edges.values():
            for e in edges:
                all_targets.add(e.target_id)
        all_nodes = list(set(all_sources) | all_targets)

        if len(all_nodes) < 2:
            return compactions

        for src in all_sources:
            edges = self._indexes.edges.get(src, [])
            if not edges:
                continue
            targets = list(set(e.target_id for e in edges))
            other_nodes = [n for n in all_nodes if n != src]
            if len(other_nodes) == 0:
                continue
            density = len(targets) / len(other_nodes)
            if density >= self._INDEX_DENSITY_THRESHOLD:
                self._indexes.fold_dense_bipartite(
                    [src], targets, max_path=self._MAX_FOLD_PATHS,
                )
                compactions += 1

        return compactions

    # ── 发现技能候选 ──────────────────────────────────────

    def discover_skill_candidates(self) -> list[str]:
        """发现技能候选。

        启发式规则：
        - 近期频繁产生相同类型的 LLM 查询
        - 反复失败的动作
        - 用户重复纠正
        - 可复用的局部判断

        Returns: 建议的训练技能描述列表
        """
        candidates: list[str] = []

        # 1. 在近期记忆中检索失败模式
        failure_ids = self._retriever.keyword_search("failure", top_k=20)
        failure_ids += self._retriever.keyword_search("error", top_k=20)
        failure_ids += self._retriever.keyword_search("失败", top_k=20)
        failure_ids = list(set(failure_ids))

        failure_texts: list[str] = []
        for mid in failure_ids[:10]:
            payload = self._memorizer.read(mid)
            if payload is not None:
                text = str(payload)[:500]
                failure_texts.append(text)

        # 2. 检索重复执行模式
        retry_ids = self._retriever.keyword_search("retry", top_k=10)
        repeat_ids = self._retriever.keyword_search("重复", top_k=10)

        # 3. 检索 LLM 调用模式
        llm_call_ids = self._retriever.keyword_search("LLM", top_k=10)
        llm_call_ids += self._retriever.keyword_search("infer", top_k=10)

        # 4. 发现重复 workload
        # 按 payload_type 统计高频类型
        type_counts: dict[str, int] = {}
        for mid in self._memorizer.catalog._entries:
            entry = self._memorizer.catalog.lookup(mid)
            if entry is not None:
                t = entry.payload_type
                type_counts[t] = type_counts.get(t, 0) + 1

        for ptype, count in type_counts.items():
            if count >= 50:
                candidates.append(
                    f"learn to categorize {ptype} inputs (high frequency: {count})"
                )

        # 5. 反复失败 → 技能候选
        if len(failure_ids) >= 5:
            # 提取失败的关键语境
            keywords: dict[str, int] = {}
            for text in failure_texts:
                for word in ["screenshot", "window", "detect", "classify",
                             "parse", "click", "type", "navigate", "visual",
                             "yolo", "ocr", "screen", "element"]:
                    if word in text.lower():
                        keywords[word] = keywords.get(word, 0) + 1
            for kw, cnt in keywords.items():
                if cnt >= 3:
                    candidates.append(
                        f"learn to handle {kw} better (failed {cnt} times)"
                    )

        # 6. 用户纠正模式
        correction_ids = self._retriever.keyword_search("correct", top_k=10)
        correction_ids += self._retriever.keyword_search("纠正", top_k=10)
        correction_ids += self._retriever.keyword_search("fix", top_k=10)
        if len(correction_ids) >= 3:
            candidates.append(
                "learn from user corrections to reduce repeated fix patterns"
            )

        # 7. 高频率 LLM 调用 → 可能值得本地化
        if len(llm_call_ids) >= 10:
            candidates.append(
                "train a local fast classifier to reduce LLM call frequency"
            )

        # 去重
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    # ── 生成训练订单 ──────────────────────────────────────

    def generate_training_orders(
        self, skill_candidates: list[str], llm_adapter: Any = None,
    ) -> list[Any]:
        """为技能候选生成训练订单。

        如果有 LLM，使用 training_order_prompt。
        否则使用启发式生成简单订单。

        Returns: List of TrainingOrder
        """
        from eve.dock.order import TrainingOrder

        orders: list[TrainingOrder] = []

        if not skill_candidates:
            return orders

        # 尝试使用 LLM 生成订单
        if llm_adapter is not None and llm_adapter.status.loaded:
            orders = self._generate_orders_with_llm(skill_candidates, llm_adapter)

        # LLM 不可用或失败 → 启发式生成
        if not orders:
            orders = self._generate_orders_heuristic(skill_candidates)

        return orders

    def _generate_orders_with_llm(
        self, skill_candidates: list[str], llm_adapter: Any,
    ) -> list[Any]:
        """使用 LLM 分析技能候选并生成训练订单。"""
        from eve.dock.order import TrainingOrder

        try:
            from eve.core.prompts import training_order_prompt

            # 收集上下文
            recent_failures = ""
            failure_ids = self._retriever.keyword_search("failure", top_k=5)
            for mid in failure_ids[:3]:
                payload = self._memorizer.read(mid)
                if payload is not None:
                    recent_failures += str(payload)[:300] + "\n---\n"

            if not recent_failures:
                recent_failures = "暂无近期失败记录。"

            candidates_text = "\n".join(
                f"- {c}" for c in skill_candidates
            )
            hormone_summary = self._hormones.levels.summary()
            available_tnn = ", ".join(
                self._runtime.myself.available_tnn_summary or ["none"]
            )

            prompt = f"""你是 EVE 的训练机会分析师。

请基于以下技能候选生成具体的训练订单。

【技能候选】
{candidates_text}

【激素状态】
{hormone_summary}

【已有 TNN】
{available_tnn}

请以 JSON 格式输出训练订单列表：
{{
  "orders": [
    {{
      "purpose": "训练目的",
      "teacher": "rule 或 local_llm",
      "structure_hint": "mlp 或 cnn",
      "priority": "low 或 medium",
      "data_keywords": ["用于搜索训练数据的关键词"]
    }}
  ]
}}

只输出 JSON，不要额外解释。"""

            result = llm_adapter.infer(prompt)
            if result.success and result.structured:
                parsed = result.structured
                order_dicts = parsed.get("orders", [])
                orders: list[TrainingOrder] = []
                for od in order_dicts[:5]:
                    purpose = od.get("purpose", "auto-skill")
                    teacher = od.get("teacher", "rule")
                    structure = od.get("structure_hint", "mlp")
                    priority = od.get("priority", "low")
                    data_kws = od.get("data_keywords", [purpose])

                    # 搜索训练数据
                    data_ids: list[str] = []
                    for kw in (data_kws if isinstance(data_kws, list) else [str(data_kws)]):
                        data_ids += self._retriever.keyword_search(kw, top_k=5)
                    data_ids = list(set(data_ids))

                    tnn_id = f"tnn_skill_{uuid.uuid4().hex[:8]}"
                    order = TrainingOrder(
                        order_id=f"order_{uuid.uuid4().hex[:12]}",
                        target_tnn_id=tnn_id,
                        purpose=purpose,
                        training_data=data_ids,
                        teacher=teacher if teacher in ("rule", "local_llm") else "rule",
                        structure_hint=structure if structure in ("mlp", "cnn", "conv_small") else "mlp",
                        priority=priority if priority in ("low", "medium", "high") else "low",
                    )
                    orders.append(order)
                return orders

        except Exception:
            pass

        return []

    def _generate_orders_heuristic(
        self, skill_candidates: list[str],
    ) -> list[Any]:
        """启发式生成简单训练订单（无 LLM 时使用）。"""
        from eve.dock.order import TrainingOrder

        orders: list[TrainingOrder] = []

        for i, candidate in enumerate(skill_candidates[:3]):
            # 从描述中提取关键词用于搜索训练数据
            words = candidate.lower().split()
            data_kw = next((w for w in words if len(w) > 3), "learn")

            data_ids: list[str] = self._retriever.keyword_search(data_kw, top_k=5)
            data_ids += self._retriever.keyword_search("visual", top_k=5)
            data_ids += self._retriever.keyword_search("screen", top_k=5)
            data_ids = list(set(data_ids))

            # 选择结构：包含 "classify" 或 "visual" → cnn，否则 → mlp
            structure = "mlp"
            if any(kw in candidate.lower() for kw in
                   ("classify", "visual", "detect", "image", "screen", "cnn")):
                structure = "cnn"

            tnn_id = f"tnn_skill_{uuid.uuid4().hex[:8]}"
            order = TrainingOrder(
                order_id=f"order_{uuid.uuid4().hex[:12]}",
                target_tnn_id=tnn_id,
                purpose=candidate[:200],
                training_data=data_ids,
                teacher="rule",
                structure_hint=structure,
                priority="low",
            )
            orders.append(order)

        return orders

    # ── 执行训练 ──────────────────────────────────────────

    def run_training(self, orders: list[Any]) -> list[Any]:
        """执行训练队列。

        将订单加入 Trainer 队列，然后逐条执行。

        Returns: List of TrainingResult
        """
        results: list[Any] = []

        if not orders:
            return results

        # 入队
        for order in orders:
            self._trainer.enqueue(order)

        # 逐条处理
        while self._trainer.has_pending():
            try:
                result = self._trainer.process_one(self._models)
                results.append(result)
            except Exception:
                # 单条失败不阻断整体流程
                continue

        return results

    # ── LLM 复盘分析 ──────────────────────────────────────

    def _run_llm_consolidation(
        self, report: SleepReport, memory_ids: list[str],
    ) -> None:
        """调用 LLM 执行睡眠复盘分析。"""
        llm = self._models.get("local_llm")
        if llm is None or not llm.status.loaded:
            return

        try:
            from eve.core.prompts import sleep_consolidation_prompt

            # 构建上下文
            today_summary = self._build_today_summary(memory_ids)
            failures = self._build_failure_summary()
            successes = self._build_success_summary()
            hormone_history = self._build_hormone_history()
            memory_stats = json.dumps(self._memorizer.stats(), ensure_ascii=False)
            training_queue = str(self._trainer.get_queue_size())

            prompt = sleep_consolidation_prompt(
                today_summary=today_summary,
                failures=failures,
                successes=successes,
                hormone_history=hormone_history,
                memory_stats=memory_stats,
                training_queue=training_queue,
            )

            result = llm.infer(prompt)
            if not result.success or not result.structured:
                return

            parsed = result.structured
            mc = parsed.get("memory_consolidation", {})

            if isinstance(mc, dict):
                events_merge = mc.get("events_to_merge", [])
                if isinstance(events_merge, list):
                    report.events_merged.extend(
                        str(e) for e in events_merge
                    )

                events_split = mc.get("events_to_split", [])
                if isinstance(events_split, list):
                    report.events_split.extend(
                        str(e) for e in events_split
                    )

                low_val = mc.get("low_value_candidates", [])
                if isinstance(low_val, list):
                    report.low_value_candidates.extend(
                        str(lv) for lv in low_val
                    )

            llm_skills = parsed.get("skill_candidates", [])
            if isinstance(llm_skills, list):
                for s in llm_skills:
                    if isinstance(s, str) and s not in report.skill_candidates:
                        report.skill_candidates.append(s)

        except Exception as e:
            report.errors.append(f"LLM consolidation: {e}")

    # ── 辅助方法 ──────────────────────────────────────────

    def _build_today_summary(self, memory_ids: list[str]) -> str:
        """构建今日摘要文本。"""
        lines: list[str] = []
        recent_ids = sorted(
            memory_ids,
            key=lambda mid: (
                self._memorizer.catalog.lookup(mid).created_at_ns
                if self._memorizer.catalog.lookup(mid) else 0
            ),
            reverse=True,
        )[:20]

        for mid in recent_ids:
            entry = self._memorizer.catalog.lookup(mid)
            if entry is None:
                continue
            payload = self._memorizer.read(mid)
            preview = (
                str(payload)[:100].replace("\n", " ")
                if payload is not None
                else "(binary)"
            )
            lines.append(
                f"[{entry.payload_type}] {mid}: {preview}"
            )

        return "\n".join(lines) if lines else "暂无今日记录。"

    def _build_failure_summary(self) -> str:
        """构建失败摘要。"""
        failure_ids = self._retriever.keyword_search("failure", top_k=10)
        failure_ids += self._retriever.keyword_search("error", top_k=10)
        failure_ids = list(set(failure_ids))

        lines: list[str] = []
        for mid in failure_ids[:5]:
            payload = self._memorizer.read(mid)
            if payload is not None:
                lines.append(f"[{mid}]: {str(payload)[:200]}")

        return "\n---\n".join(lines) if lines else "暂无失败记录。"

    def _build_success_summary(self) -> str:
        """构建成功摘要。"""
        success_ids = self._retriever.keyword_search("success", top_k=10)
        success_ids += self._retriever.keyword_search("成功", top_k=10)
        success_ids = list(set(success_ids))

        lines: list[str] = []
        for mid in success_ids[:5]:
            payload = self._memorizer.read(mid)
            if payload is not None:
                lines.append(f"[{mid}]: {str(payload)[:200]}")

        return "\n---\n".join(lines) if lines else "暂无成功记录。"

    def _build_hormone_history(self) -> str:
        """构建激素变化历程摘要。"""
        events = self._hormones.get_recent_events(20)
        if not events:
            return "暂无激素变化记录。"

        lines: list[str] = []
        for e in events:
            desc = e.description or ""
            lines.append(
                f"[{e.source}] {e.hormone}: {e.delta:+.3f} {desc}"
            )

        return "\n".join(lines)

    # ── 查询接口 ──────────────────────────────────────────

    def get_last_report(self) -> SleepReport | None:
        return self._reports[-1] if self._reports else None

    def get_all_reports(self) -> list[SleepReport]:
        return list(self._reports)

    # ── 持久化 ────────────────────────────────────────────

    def save_sleep_state(self, path: Path) -> None:
        """保存睡眠复盘状态。"""
        path.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "sleeping": self._sleeping,
            "reports": [],
            "pre_sleep_hormone_snapshot": self._pre_sleep_hormone_snapshot,
        }

        for r in self._reports:
            data["reports"].append({
                "sleep_id": r.sleep_id,
                "started_at_ns": r.started_at_ns,
                "finished_at_ns": r.finished_at_ns,
                "memories_consolidated": r.memories_consolidated,
                "events_merged": r.events_merged,
                "events_split": r.events_split,
                "low_value_candidates": r.low_value_candidates,
                "skill_candidates": r.skill_candidates,
                "training_orders_created": r.training_orders_created,
                "training_orders_executed": r.training_orders_executed,
                "training_results": r.training_results,
                "graph_compactions": r.graph_compactions,
                "errors": r.errors,
                "status": r.status,
            })

        report_path = path / "sleep_state.json"
        report_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
