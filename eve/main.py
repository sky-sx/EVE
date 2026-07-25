"""EVE 主入口 — 启动、控制与监视。"""
from __future__ import annotations
import time
import threading
from pathlib import Path

from eve.config import EVEConfig
from eve.state import RuntimeState, OutputMode
from eve.core.safegate import check, report_human_activity, emergency_stop, detect_human_cursor_activity
from eve.core.loop import run_once, log_event
from eve.core.runtime_state import RuntimeStateManager
from eve.core.hormones import HormoneManager
from eve.core.graph import TNNGraph, TNNOutputCache
from eve.core.tnn_store import TNNStore
from eve.core.model_adapter import LLMAdapter, VLMAdapter, YOLOAdapter
from eve.core.prompts import (world_update_prompt, myself_update_prompt,
                               active_tnn_selection_prompt, training_order_prompt,
                               sleep_consolidation_prompt)
from eve.core.sleep import SleepManager
from eve.input.capture import CaptureManager
from eve.input.buffer import InputBuffer
from eve.output import mouse, keyboard, speak
from eve.memory.memorizer import Memorizer
from eve.memory.indexes import IndexManager
from eve.memory.event import EventManager
from eve.memory.retrieval import Retriever
from eve.dock.trainer import Trainer
from eve.dock.order import TrainingOrder


def main():
    """EVE 主函数。

    启动流程：
    1. 加载配置
    2. 初始化所有模块
    3. 绑定 GUI
    4. 冷启动等待
    5. 启动多循环
    6. 运行直到退出
    """
    print("=" * 60)
    print("EVE — Digital Life Research System")
    print("=" * 60)

    # 1. Load config
    config = EVEConfig.default()
    print(f"[INIT] Config loaded: {config.runs_dir}")

    # 2. Create runtime state
    state = RuntimeState()
    print("[INIT] RuntimeState created (default disabled)")

    # 3. Initialize input
    buffer = InputBuffer(retention_ns=2_000_000_000)
    capture = CaptureManager(buffer, monitor_index=1, screen_fps=config.screen_fps)
    print("[INIT] InputBuffer + CaptureManager ready")

    # 4. Initialize runtime state manager
    runtime_mgr = RuntimeStateManager(config)
    print("[INIT] RuntimeStateManager ready")

    # 5. Initialize Memory
    memorizer = Memorizer(config.memory_dir)
    memorizer.load_catalog()
    index_mgr = IndexManager()
    event_mgr = EventManager()
    retriever = Retriever(memorizer, index_mgr)
    print(f"[INIT] Memory ready (STM {len(memorizer.get_stm_ids())}, "
          f"MTM {len(memorizer.get_mtm_ids())})")

    # 6. Initialize TNN Store
    tnn_store = TNNStore(config.tnn_weights_dir)
    print(f"[INIT] TNN Store ready ({len(tnn_store.list_available())} TNNs)")

    # 7. Initialize TNNGraph
    output_cache = TNNOutputCache()
    graph = TNNGraph(tnn_store, output_cache)
    print("[INIT] TNNGraph ready")

    # 8. Initialize Hormones
    hormones = HormoneManager()
    print("[INIT] HormoneManager ready")

    # 9. Initialize model adapters (optional, may fail gracefully)
    llm = LLMAdapter()
    vlm = VLMAdapter()
    yolo = YOLOAdapter()
    model_adapters = {"llm": llm, "vlm": vlm, "yolo": yolo}

    if llm.detect():
        print(f"[INIT] LLM detected at: {llm.model_path}")
    else:
        print("[INIT] LLM: not available")
    if yolo.detect():
        print("[INIT] YOLO detected")

    # 10. Initialize Dock
    trainer = Trainer(tnn_store, memorizer)
    print("[INIT] Dock Trainer ready")

    # 11. Initialize Sleep Manager
    sleep_mgr = SleepManager(memorizer, index_mgr, event_mgr, retriever,
                              runtime_mgr, hormones, trainer, model_adapters)
    print("[INIT] SleepManager ready")

    # 12. Try to load previous snapshot
    snapshot_path = config.snapshot_dir
    snap_dir = Path(snapshot_path) / "latest"
    if snap_dir.exists():
        try:
            runtime_mgr.load_snapshot(snap_dir)
            print("[INIT] Previous snapshot loaded")
        except Exception as e:
            print(f"[INIT] Failed to load snapshot: {e}")
    else:
        print("[INIT] No previous snapshot found")

    # 13. Initialize GUI
    from eve.gui.control_panel import ControlPanel
    gui = ControlPanel(
        on_cold_start=lambda: _cold_start(state, capture, graph, gui),
        on_emergency_stop=lambda: _do_emergency_stop(state, gui),
        on_praise=lambda: _do_praise(state, hormones, gui),
        on_criticize=lambda: _do_criticize(state, hormones, gui),
        on_force_sleep=lambda: _do_force_sleep(state, sleep_mgr, gui),
        on_manual_save=lambda: _do_save_snapshot(runtime_mgr, config, gui),
    )
    gui.bind_state(state, runtime_mgr.world, runtime_mgr.myself,
                    runtime_mgr.blackboard, hormones, graph, tnn_store,
                    memorizer, buffer, trainer, config)

    # 14. Start GUI in main thread (it will block)
    print("[READY] GUI starting. Press 'Cold Start' to begin.")
    print("        Press Esc for emergency stop.")
    gui.start()

    # 15. Cleanup on exit
    print("[SHUTDOWN] Saving state...")
    _do_save_snapshot(runtime_mgr, config, gui)
    if capture.running:
        capture.stop()
    print("[SHUTDOWN] EVE stopped.")


def _cold_start(state, capture, graph, gui):
    """冷启动：开始捕获、启用图。"""
    state.cold_started = True
    capture.start()
    graph.start()
    log_event(state, "cold_start")
    print("[EVE] Cold start complete. Capture running.")
    gui.add_log("Cold start complete")


def _do_emergency_stop(state, gui):
    """触发紧急停止。"""
    emergency_stop(state)
    log_event(state, "emergency_stop")
    print("[EVE] EMERGENCY STOP triggered!")
    gui.add_log("EMERGENCY STOP triggered!")


def _do_praise(state, hormones, gui):
    """用户表扬 EVE。"""
    hormones.apply_event("user_praise", intensity=1.0, description="User praised EVE")
    log_event(state, "user_praise")
    print("[EVE] User praised EVE")
    gui.add_log("User praise applied")


def _do_criticize(state, hormones, gui):
    """用户批评 EVE。"""
    hormones.apply_event("user_critique", intensity=1.0, description="User criticized EVE")
    log_event(state, "user_critique")
    print("[EVE] User criticized EVE")
    gui.add_log("User critique applied")


def _do_force_sleep(state, sleep_mgr, gui):
    """强制触发睡眠周期。"""
    if not sleep_mgr.is_sleeping:
        gui.add_log("Entering forced sleep...")
        sleep_mgr.enter_sleep(state)
        result = sleep_mgr.run_sleep_cycle()
        sleep_mgr.wake_up(state)
        gui.add_log(f"Forced sleep cycle completed: {result}")
    else:
        gui.add_log("Already sleeping, skipping force sleep")


def _do_save_snapshot(runtime_mgr, config, gui):
    """保存当前状态快照。"""
    snapshot_path = Path(config.snapshot_dir) / "latest"
    try:
        runtime_mgr.save_snapshot(snapshot_path)
        print(f"[EVE] Snapshot saved to {snapshot_path}")
        gui.add_log(f"Snapshot saved to {snapshot_path}")
    except Exception as e:
        print(f"[EVE] Snapshot save failed: {e}")
        gui.add_log(f"Snapshot save failed: {e}")


if __name__ == "__main__":
    main()
