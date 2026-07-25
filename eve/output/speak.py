"""
EVE 语音输出 — 支持 disabled / mock / real 模式。

REAL 模式接口已就绪但尚未实现 TTS 后端。
"""
from __future__ import annotations

import time
from typing import Any

from eve.state import OutputMode, OutputResult


def execute(
    action_id: str,
    payload: dict[str, Any],
    mode: OutputMode,
) -> OutputResult:
    """执行语音动作。"""
    started_ns = time.monotonic_ns()

    if mode == OutputMode.DISABLED:
        return OutputResult(
            action_id=action_id,
            kind="speak",
            mode=mode.value,
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason="output_disabled",
        )

    if mode == OutputMode.MOCK:
        return OutputResult(
            action_id=action_id,
            kind="speak",
            mode=mode.value,
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=True,
            blocked=False,
            reason="mock_ok",
            payload=dict(payload),
        )

    # REAL 模式：接口就绪，TTS 后端尚未实现
    return _execute_real(action_id, payload, started_ns)


def _execute_real(
    action_id: str,
    payload: dict[str, Any],
    started_ns: int,
) -> OutputResult:
    return OutputResult(
        action_id=action_id,
        kind="speak",
        mode="real",
        started_at_ns=started_ns,
        finished_at_ns=time.monotonic_ns(),
        executed=False,
        simulated=False,
        blocked=True,
        reason="real_unimplemented_no_tts_backend",
        payload={"text": payload.get("text", ""), "error": "TTS backend not implemented"},
    )
