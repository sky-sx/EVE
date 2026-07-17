"""
EVE 键盘输出 — 仅支持 disabled / mock 模式。

不调用 keyboard.press/write、SendInput 或任何真实系统控制 API。
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
    """执行键盘动作。"""
    started_ns = time.monotonic_ns()

    if mode == OutputMode.DISABLED:
        return OutputResult(
            action_id=action_id,
            kind="keyboard",
            mode=mode.value,
            started_at_ns=started_ns,
            finished_at_ns=time.monotonic_ns(),
            executed=False,
            simulated=False,
            blocked=True,
            reason="output_disabled",
        )

    return OutputResult(
        action_id=action_id,
        kind="keyboard",
        mode=mode.value,
        started_at_ns=started_ns,
        finished_at_ns=time.monotonic_ns(),
        executed=False,
        simulated=True,
        blocked=False,
        reason="mock_ok",
        payload=dict(payload),
    )
