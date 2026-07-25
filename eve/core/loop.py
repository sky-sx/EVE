"""
EVE 多循环运行时 — 多频率、多职责的异步循环架构。

循环：
- 实时捕获循环：持续采集屏幕和光标
- 缓冲更新循环：更新 InputBuffer 中的 state
- TNNGraph tick：调度在期 TNN 节点，清理过期缓存
- 本地 LLM 大循环：慢速认知更新（world/myself/active_tnn）
- 动作执行循环：从 blackboard 拉取动作，经 Safegate 输出
- 记忆写入循环：将模块产出写入 Memory
- 激素状态循环：逐轮更新激素和节律
- Dock 训练消费：处理训练队列
- 资源监控循环：监控硬件资源
- 状态持久化：定期快照

同时保留原有的 run_once() 和 log_event()。
"""
from __future__ import annotations

import json
import time
import threading
import traceback
from pathlib import Path
from typing import Any

from eve.core import safegate
from eve.output import keyboard, mouse, speak
from eve.state import (
    ActionCandidate, ActionKind, OutputMode, OutputResult, RuntimeState,
)


# ── 原有的 run_once / log_event（保持不变）─────────────────────

def run_once(
    state: RuntimeState,
    action: ActionCandidate,
    log_dir: str | Path = "runs/phase1",
) -> OutputResult:
    """单次运行闭环。"""
    state.pending_action = action
    try:
        result = safegate.check(state, action)
        output: OutputResult
        if result.allowed:
            output = _dispatch_output(action, state.output_mode)
        else:
            output = OutputResult(
                action_id=action.action_id,
                kind=action.kind.value,
                mode=state.output_mode.value,
                started_at_ns=time.monotonic_ns(),
                finished_at_ns=time.monotonic_ns(),
                executed=False,
                simulated=False,
                blocked=True,
                reason=f"safegate_{result.reason}",
            )
        state.latest_output = output
        _log_event(log_dir, {
            "event": "run_once",
            "action": {
                "action_id": action.action_id,
                "kind": action.kind.value,
                "payload": action.payload,
                "origin": action.origin,
                "created_at_ns": action.created_at_ns,
                "valid_until_ns": action.valid_until_ns,
            },
            "safegate": {
                "allowed": result.allowed,
                "reason": result.reason,
                "checked_at_ns": result.checked_at_ns,
            },
            "output": {
                "executed": output.executed,
                "simulated": output.simulated,
                "blocked": output.blocked,
                "started_at_ns": output.started_at_ns,
                "finished_at_ns": output.finished_at_ns,
                "reason": output.reason,
                "payload": output.payload,
            },
        })
        return output
    except Exception:
        _log_event(log_dir, {
            "event": "runtime_error",
            "action_id": action.action_id,
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        state.pending_action = None


def log_event(
    state: RuntimeState | None,
    event_type: str,
    log_dir: str | Path = "runs/phase1",
    **extra,
) -> None:
    """记录一条日志事件。"""
    entry = {"event": event_type, **extra}
    _log_event(log_dir, entry)


# ── 多循环运行器 ──────────────────────────────────────────


class EVELoops:
    """EVE 多循环运行管理器。

    管理多个不同频率的运行循环/任务，提供启停和监控。
    """

    def __init__(
        self,
        state: RuntimeState,
        runtime_mgr,
        config,
        capture,
        buffer,
        graph,
        hormones,
        memorizer,
        tnn_store,
        trainer,
        sleep_mgr,
        model_adapters: dict | None = None,
        llm_adapter=None,
    ):
        self.state = state
        self.runtime = runtime_mgr
        self.config = config
        self.capture = capture
        self.buffer = buffer
        self.graph = graph
        self.hormones = hormones
        self.memorizer = memorizer
        self.tnn_store = tnn_store
        self.trainer = trainer
        self.sleep_mgr = sleep_mgr
        self.models = model_adapters or {}
        self.llm = llm_adapter

        self._threads: list[threading.Thread] = []
        self._running = False
        self._last_llm_ns: int = 0
        self._last_hormone_ns: int = 0
        self._last_snapshot_ns: int = 0
        self._loop_stats: dict[str, dict] = {}

    @property
    def running(self) -> bool:
        return self._running

    def start_all(self) -> None:
        """启动所有后台循环。"""
        if self._running:
            return
        self._running = True

        # 循环列表（名称, 目标函数, 间隔秒, 是否daemon）
        loops = [
            ("capture", self._loop_capture, 0, True),
            ("graph_tick", self._loop_graph_tick, 0.05, True),
            ("action", self._loop_action, 0.01, True),
            ("memory_write", self._loop_memory_write, 0.5, True),
            ("llm_slow", self._loop_llm_slow, 0, True),
            ("hormone", self._loop_hormone, 1.0, True),
            ("dock", self._loop_dock, 1.0, True),
            ("snapshot", self._loop_snapshot, 60.0, True),
        ]

        for name, target, interval, daemon in loops:
            t = threading.Thread(
                target=self._wrap_loop(name, target, interval),
                daemon=daemon,
                name=f"eve-{name}",
            )
            t.start()
            self._threads.append(t)

        log_event(self.state, "loops_started", loop_count=len(loops))

    def stop_all(self) -> None:
        """停止所有循环。"""
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()
        log_event(self.state, "loops_stopped")

    def stats(self) -> dict:
        """各循环统计。"""
        return dict(self._loop_stats)

    # ── 循环实现 ─────────────────────────────────────────

    def _wrap_loop(self, name: str, target, interval_s: float):
        """包装循环以处理异常和统计。"""
        def wrapper():
            count = 0
            total_time_ns = 0
            while self._running:
                t0 = time.monotonic_ns()
                try:
                    target()
                except Exception:
                    log_event(self.state, "loop_error", loop=name,
                               error=traceback.format_exc())
                elapsed_ns = time.monotonic_ns() - t0
                count += 1
                total_time_ns += elapsed_ns
                self._loop_stats[name] = {
                    "count": count,
                    "avg_time_ms": (total_time_ns / max(count, 1)) / 1e6,
                    "total_time_s": total_time_ns / 1e9,
                }
                if interval_s > 0:
                    sleep_s = interval_s - (elapsed_ns / 1e9)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        return wrapper

    def _loop_capture(self) -> None:
        """捕获循环：保持 capture 运行（CaptureManager 自身有线程）。"""
        # CaptureManager manages its own threads; this loop monitors
        if not self.capture.running and self.state.cold_started:
            pass  # Already started in cold_start

    def _loop_graph_tick(self) -> None:
        """TNNGraph tick：调度在期节点，清理过期缓存。"""
        if not self.graph.running or not self.state.cold_started:
            return
        try:
            self.graph.tick()
            now_ns = time.monotonic_ns()
            scheduled = self.graph.schedule(now_ns)
            for tnn_id in scheduled:
                try:
                    # Build inputs from SourceRef
                    inputs = self._resolve_tnn_inputs(tnn_id)
                    if inputs is not None:
                        outputs = self.graph.run_node(tnn_id, inputs)
                        if outputs:
                            # Write to blackboard
                            for name, value in outputs.items():
                                from eve.state import TimedEntry
                                import uuid
                                entry = TimedEntry(
                                    entry_id=f"tnn_{tnn_id}_{uuid.uuid4().hex[:8]}",
                                    kind=f"tnn_output.{tnn_id}.{name}",
                                    producer=tnn_id,
                                    produced_at_ns=now_ns,
                                    valid_until_ns=now_ns + 500_000_000,  # 500ms TTL
                                    payload=value,
                                )
                                self.runtime.blackboard.write(entry)
                except Exception as e:
                    self.graph.pause_node(tnn_id)
                    log_event(self.state, "tnn_run_error", tnn_id=tnn_id, error=str(e))
        except Exception:
            pass

    def _loop_action(self) -> None:
        """动作执行循环：从 blackboard 拉取动作候选，经 Safegate 输出。"""
        if not self.state.cold_started:
            return
        # Check blackboard for action candidates
        from eve.state import TimedEntry
        try:
            entries = self.runtime.blackboard.read("action_candidate")
            for entry in entries:
                payload = entry.payload
                if isinstance(payload, dict):
                    action = ActionCandidate(
                        action_id=entry.entry_id,
                        kind=ActionKind(payload.get("kind", "mouse")),
                        payload=payload.get("payload", {}),
                        origin=entry.producer,
                        created_at_ns=entry.produced_at_ns,
                        valid_until_ns=entry.valid_until_ns,
                    )
                    result = run_once(self.state, action)
                    # Write result back
                    result_entry = TimedEntry(
                        entry_id=f"result_{entry.entry_id}",
                        kind="action_result",
                        producer="action_loop",
                        produced_at_ns=time.monotonic_ns(),
                        valid_until_ns=0,
                        payload={
                            "action_id": result.action_id,
                            "executed": result.executed,
                            "simulated": result.simulated,
                            "blocked": result.blocked,
                            "reason": result.reason,
                        },
                    )
                    self.runtime.blackboard.write(result_entry)
        except Exception:
            pass

    def _loop_memory_write(self) -> None:
        """记忆写入循环：将 blackboard 中的候选结果写入 Memory。"""
        if not self.state.cold_started:
            return
        try:
            candidates = self.runtime.blackboard.read("memory_candidate")
            for entry in candidates:
                payload = entry.payload
                payload_type = "json" if isinstance(payload, dict) else "text"
                memory_id = self.memorizer.create(str(payload), payload_type)
                self.memorizer.add_to_stm(memory_id)
        except Exception:
            pass

    def _loop_llm_slow(self) -> None:
        """本地 LLM 大循环：慢速认知更新。

        根据激素计算间隔，到期时：
        1. 调用 LLM 更新 world/myself
        2. 获取 active_tnn 集合
        3. 同步到 TNNGraph
        """
        if not self.state.cold_started or self.llm is None:
            return

        now_ns = time.monotonic_ns()
        interval_ns = int(self.hormones.compute_llm_interval() * 1e9)

        if now_ns - self._last_llm_ns < interval_ns:
            return

        self._last_llm_ns = now_ns

        if not self.llm.status.loaded:
            return

        try:
            # 1. Build world update prompt
            w_prompt = _build_world_update_context(self.runtime, self.hormones, self.buffer)
            result = self.llm.infer(w_prompt)
            if result.success and result.structured:
                self.runtime.update_from_llm_world(result.structured)
                log_event(self.state, "llm_world_updated")

            # 2. Build myself update prompt
            m_prompt = _build_myself_update_context(self.runtime, self.hormones, self.graph)
            result = self.llm.infer(m_prompt)
            if result.success and result.structured:
                self.runtime.update_from_llm_myself(result.structured)
                log_event(self.state, "llm_myself_updated")

            # 3. Get active_tnn
            available = self.tnn_store.list_available()
            loaded = self.graph.list_nodes()
            a_prompt = _build_active_tnn_context(
                self.runtime, available, loaded, self.hormones
            )
            result = self.llm.infer(a_prompt)
            if result.success and result.structured:
                active = result.structured.get("active_tnn", [])
                diff = self.graph.sync_active_set(active)
                log_event(self.state, "llm_active_tnn", diff=diff)

        except Exception as e:
            log_event(self.state, "llm_slow_error", error=str(e))

    def _loop_hormone(self) -> None:
        """激素和节律更新循环。"""
        if not self.state.cold_started:
            return
        self.hormones.update_cycle()
        # Record hormone state in myself
        self.runtime.myself.hormone_levels = self.hormones.levels.to_dict()
        self.runtime.myself.tendencies = self.hormones.get_tendencies()

    def _loop_dock(self) -> None:
        """Dock 训练队列消费循环。"""
        if not self.state.cold_started:
            return
        if self.trainer.has_pending():
            try:
                result = self.trainer.process_one(self.models)
                log_event(self.state, "dock_training_complete",
                           order_id=result.order_id, success=result.success)
            except Exception as e:
                log_event(self.state, "dock_training_error", error=str(e))

    def _loop_snapshot(self) -> None:
        """定期状态持久化。"""
        if not self.state.cold_started:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_snapshot_ns > 300_000_000_000:  # 5 minutes
            self._last_snapshot_ns = now_ns
            try:
                snapshot_path = self.config.snapshot_dir / "latest"
                self.runtime.save_snapshot(snapshot_path)
                log_event(self.state, "periodic_snapshot")
            except Exception as e:
                log_event(self.state, "snapshot_error", error=str(e))

    def _resolve_tnn_inputs(self, tnn_id: str) -> dict[str, Any] | None:
        """根据 TNN 的 SourceRef 解析输入。"""
        desc = self.tnn_store.get_descriptor(tnn_id)
        if desc is None:
            return None

        inputs = {}
        for ref in desc.inputs:
            try:
                if ref.source_type == "state" and ref.source_id == "screen":
                    sample = self.buffer.latest("screen")
                    if sample:
                        inputs[ref.source_id] = sample.value
                elif ref.source_type == "state" and ref.source_id == "cursor":
                    sample = self.buffer.latest("cursor")
                    if sample:
                        inputs[ref.source_id] = sample.value
                elif ref.source_type == "world":
                    val = getattr(self.runtime.world, ref.source_id, None)
                    if val:
                        inputs[ref.source_id] = val
                elif ref.source_type == "myself":
                    val = getattr(self.runtime.myself, ref.source_id, None)
                    if val:
                        inputs[ref.source_id] = val
                elif ref.source_type == "blackboard":
                    entries = self.runtime.blackboard.read(ref.source_id)
                    if entries:
                        inputs[ref.source_id] = entries[-1].payload
                elif ref.source_type == "tnn_output":
                    # ref.source_id format: "tnn:other_tnn.output_field"
                    parts = ref.source_id.split(":")
                    if len(parts) == 2:
                        other_tnn, field = parts[0], parts[1]
                        cached = self.graph.get_output(other_tnn, field)
                        if cached is not None:
                            inputs[ref.source_id] = cached
            except Exception:
                continue

        return inputs if inputs else None


# ── Prompt 构建辅助函数 ─────────────────────────────────


def _build_world_update_context(runtime, hormones, buffer) -> str:
    """构建 LLM world 更新上下文。"""
    return f"""【当前世界认知】scene={runtime.world.scene}, sub={runtime.world.sub_scene}
【自身状态】task={runtime.myself.current_task}, thinking={runtime.myself.what_im_thinking}
【激素】{hormones.levels.summary()}
【缓冲】screens={buffer.count('screen')}, cursors={buffer.count('cursor')}"""


def _build_myself_update_context(runtime, hormones, graph) -> str:
    """构建 LLM myself 更新上下文。"""
    return f"""【世界】scene={runtime.world.scene}
【自身】task={runtime.myself.current_task}, progress={runtime.myself.task_progress}
【激素】{hormones.levels.summary()}
【已加载TNN】{graph.list_nodes()}"""


def _build_active_tnn_context(runtime, available, loaded, hormones) -> str:
    """构建 LLM active_tnn 选择上下文。"""
    return f"""【世界】{runtime.world.scene}, {runtime.world.sub_scene}
【自身】{runtime.myself.current_task}
【可用TNN】{available}
【已加载TNN】{loaded}
【激素】{hormones.levels.summary()}"""


# ── 内部 ──────────────────────────────────────────────────

_RUN_ID: str = ""


def _dispatch_output(action: ActionCandidate, mode: OutputMode) -> OutputResult:
    if action.kind == ActionKind.MOUSE:
        return mouse.execute(action.action_id, action.payload, mode)
    elif action.kind == ActionKind.KEYBOARD:
        return keyboard.execute(action.action_id, action.payload, mode)
    elif action.kind == ActionKind.SPEAK:
        return speak.execute(action.action_id, action.payload, mode)
    else:
        return OutputResult(
            action_id=action.action_id,
            kind=action.kind.value,
            mode=mode.value,
            started_at_ns=time.monotonic_ns(),
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason=f"unknown_kind_{action.kind.value}",
        )


def _ensure_run_id(log_dir: str | Path) -> str:
    global _RUN_ID
    if not _RUN_ID:
        _RUN_ID = time.strftime("%Y%m%d_%H%M%S_") + f"{time.monotonic_ns()}"
    return _RUN_ID


def _log_event(log_dir: str | Path, entry: dict) -> None:
    run_id = _ensure_run_id(log_dir)
    log_path = Path(log_dir) / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("timestamp_ns", time.monotonic_ns())
    entry.setdefault("run_id", run_id)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
