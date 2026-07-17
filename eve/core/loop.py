"""
EVE Phase 1 最小运行循环。

流程：
1. 检查运行状态
2. 把 ActionCandidate 交给 Safegate
3. 若允许 → 交给对应 mock output
4. 若阻断 → 形成 OutputResult
5. 写一行 JSONL 日志
6. 返回结果

不使用复杂事件总线、插件系统、Manager 或多 Agent。
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from eve.core import safegate
from eve.output import keyboard, mouse, speak
from eve.state import ActionCandidate, ActionKind, OutputMode, OutputResult, RuntimeState


def run_once(
    state: RuntimeState,
    action: ActionCandidate,
    log_dir: str | Path = "runs/phase1",
) -> OutputResult:
    """单次运行闭环。"""
    state.pending_action = action
    try:
        # 1. Safegate 判定
        result = safegate.check(state, action)

        # 2. 路由到对应输出
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

        # 3. 写 JSONL 日志（嵌套结构）
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
        # 异常时形成 runtime_error 日志
        _log_event(log_dir, {
            "event": "runtime_error",
            "action_id": action.action_id,
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        state.pending_action = None


def log_event(
    state: RuntimeState,
    event_type: str,
    log_dir: str | Path = "runs/phase1",
    **extra,
) -> None:
    """记录一条日志事件。"""
    entry = {"event": event_type, **extra}
    _log_event(log_dir, entry)


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
