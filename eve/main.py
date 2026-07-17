"""
EVE Phase 1 安全实验入口。

默认：
- 不启动屏幕捕获
- 不加载 YOLO
- 不加载 DeepSeek/Qwen
- 不开启真实输出
- 不读取 Memory
- 不启动 Dock

功能：
1. 创建 RuntimeState
2. cold start
3. 生成一条假 ActionCandidate
4. 运行一次 mock 闭环
5. 打印 OutputResult
6. 触发或模拟 Esc 急停
7. 安全退出
"""
from __future__ import annotations

import time
import uuid

from eve.core import loop, safegate
from eve.state import ActionCandidate, ActionKind, OutputMode, RuntimeState


def main() -> None:
    print("=" * 50)
    print("EVE Phase 1 — 最小 Mock 运行闭环")
    print("=" * 50)
    print()

    # 1. 创建 RuntimeState（默认 disabled）
    state = RuntimeState()
    loop.log_event(state, "runtime_start")
    print("[EVE] RuntimeState 已创建（默认 disabled）")
    print()

    # 2. cold start
    state.cold_started = True
    loop.log_event(state, "cold_start")
    print("[EVE] Cold start 完成")
    print()

    # 3. 场景 A: disabled 模式 — 动作应被阻断
    print("── 场景 A: 默认 disabled — 动作应被阻断 ──")
    action_a = ActionCandidate(
        action_id=f"action_{uuid.uuid4().hex[:8]}",
        kind=ActionKind.MOUSE,
        payload={"x": 100, "y": 200, "button": "left"},
        origin="test_main",
    )
    result_a = loop.run_once(state, action_a)
    print(f"  Safegate 结果: allowed={result_a.executed}, blocked={result_a.blocked}, reason={result_a.reason}")
    print()

    # 4. 场景 B: 切换到 mock + 授权鼠标
    print("── 场景 B: mock 模式 + mouse 授权 — 动作应执行 ──")
    state.output_mode = OutputMode.MOCK
    state.mouse_allowed = True
    action_b = ActionCandidate(
        action_id=f"action_{uuid.uuid4().hex[:8]}",
        kind=ActionKind.MOUSE,
        payload={"x": 300, "y": 400, "button": "right"},
        origin="test_main",
    )
    result_b = loop.run_once(state, action_b)
    print(f"  Safegate 结果: allowed={result_b.executed}, blocked={result_b.blocked}, reason={result_b.reason}")
    print(f"  payload={result_b.payload}")
    print()

    # 5. 场景 C: 模拟人类接管 → 冻结键鼠
    print("── 场景 C: 人类接管冻结 5 秒 ──")
    safegate.report_human_activity(state)
    loop.log_event(state, "human_takeover")
    print(f"  blocked_until_ns 已设置")
    action_c = ActionCandidate(
        action_id=f"action_{uuid.uuid4().hex[:8]}",
        kind=ActionKind.MOUSE,
        payload={"x": 500, "y": 600},
        origin="test_main",
    )
    result_c = loop.run_once(state, action_c)
    print(f"  Safegate 结果: allowed={result_c.executed}, blocked={result_c.blocked}, reason={result_c.reason}")
    print()

    # 6. 场景 D: emergency stop
    print("── 场景 D: emergency stop ──")
    safegate.emergency_stop(state)
    loop.log_event(state, "emergency_stop")
    action_d = ActionCandidate(
        action_id=f"action_{uuid.uuid4().hex[:8]}",
        kind=ActionKind.SPEAK,
        payload={"text": "hello"},
        origin="test_main",
    )
    result_d = loop.run_once(state, action_d)
    print(f"  Safegate 结果: allowed={result_d.executed}, blocked={result_d.blocked}, reason={result_d.reason}")
    print()

    # 7. 输出完整 state 摘要
    print("── 运行时状态摘要 ──")
    print(f"  cold_started:     {state.cold_started}")
    print(f"  emergency_stopped:  {state.emergency_stopped}")
    print(f"  output_mode:        {state.output_mode.value}")
    print(f"  mouse_allowed:      {state.mouse_allowed}")
    print(f"  keyboard_allowed:   {state.keyboard_allowed}")
    print(f"  speak_allowed:      {state.speak_allowed}")
    print(f"  blocked_until_ns:   {state.blocked_until_ns}")
    print()

    # 8. 安全退出
    loop.log_event(state, "runtime_stop")
    print("=" * 50)
    print("EVE Phase 1 实验完成，安全退出。")
    print(f"日志写入: runs/phase1/")
    print("=" * 50)


if __name__ == "__main__":
    main()
