"""Phase 1 Runtime Loop 测试。

覆盖：
- 完整 run_once
- 阻断也产生 OutputResult
- mock 不调用真实输出 API
- emergency stop 后 pending 和新动作被拒绝
- 日志可解析
- 停止后没有存活线程
- 不导入 src.eve
- 异常时 pending_action 被清空
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from eve.core import loop, safegate
from eve.state import ActionCandidate, ActionKind, OutputMode, OutputResult, RuntimeState


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """临时日志目录。"""
    return tmp_path / "phase1"


@pytest.fixture
def state() -> RuntimeState:
    return RuntimeState()


@pytest.fixture
def ready_state() -> RuntimeState:
    s = RuntimeState()
    s.cold_started = True
    s.output_mode = OutputMode.MOCK
    s.mouse_allowed = True
    s.keyboard_allowed = True
    s.speak_allowed = True
    return s


# ── 完整 run_once ────────────────────────────────────────


def test_run_once_disabled(state: RuntimeState, tmp_log_dir: Path):
    """disabled 模式下 run_once 应产生阻断结果。"""
    action = ActionCandidate(
        action_id="test-001",
        kind=ActionKind.MOUSE,
        payload={"x": 10, "y": 20},
        origin="test",
    )
    result = loop.run_once(state, action, log_dir=tmp_log_dir)
    assert isinstance(result, OutputResult)
    assert result.executed is False
    assert result.simulated is False
    assert result.blocked is True
    assert "safegate_output_disabled" in result.reason or "not_cold_started" in result.reason


def test_run_once_mock(ready_state: RuntimeState, tmp_log_dir: Path):
    """mock 授权模式下 run_once 应产生模拟结果 (executed=False, simulated=True)。"""
    action = ActionCandidate(
        action_id="test-002",
        kind=ActionKind.MOUSE,
        payload={"x": 50, "y": 60},
        origin="test",
    )
    result = loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    assert isinstance(result, OutputResult)
    assert result.executed is False
    assert result.simulated is True
    assert result.blocked is False
    assert result.reason == "mock_ok"
    assert result.payload == {"x": 50, "y": 60}


def test_run_once_keyboard(ready_state: RuntimeState, tmp_log_dir: Path):
    """键盘 mock — simulated=True, executed=False。"""
    action = ActionCandidate(
        action_id="test-key-001",
        kind=ActionKind.KEYBOARD,
        payload={"key": "enter"},
        origin="test",
    )
    result = loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    assert result.executed is False
    assert result.simulated is True
    assert result.kind == "keyboard"


def test_run_once_speak(ready_state: RuntimeState, tmp_log_dir: Path):
    """语音 mock — simulated=True, executed=False。"""
    action = ActionCandidate(
        action_id="test-speak-001",
        kind=ActionKind.SPEAK,
        payload={"text": "hello world"},
        origin="test",
    )
    result = loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    assert result.executed is False
    assert result.simulated is True
    assert result.kind == "speak"


# ── 阻断也产生 OutputResult ──────────────────────────────


def test_blocked_produces_result(state: RuntimeState, tmp_log_dir: Path):
    """被阻断的请求也应该产生有效的 OutputResult。"""
    action = ActionCandidate(
        action_id="test-blocked-001",
        kind=ActionKind.MOUSE,
        payload={},
        origin="test",
    )
    result = loop.run_once(state, action, log_dir=tmp_log_dir)
    assert isinstance(result, OutputResult)
    assert result.executed is False
    assert result.simulated is False
    assert result.blocked is True
    assert result.finished_at_ns >= result.started_at_ns


# ── mock 不调用真实输出 API ────────────────────────────────


def test_mouse_does_not_import_pyautogui_output() -> None:
    """验证 mouse.execute 不导入 pyautogui 控制函数。"""
    import inspect
    import eve.output.mouse as m

    source = inspect.getsource(m.execute)
    assert "pyautogui.click" not in source
    assert "pyautogui.moveTo" not in source
    assert "SendInput" not in source


def test_keyboard_does_not_call_real_api() -> None:
    """验证 keyboard.execute 不调用真实 API。"""
    import inspect
    import eve.output.keyboard as k

    source = inspect.getsource(k.execute)
    assert "keyboard.press" not in source
    assert "keyboard.write" not in source
    assert "SendInput" not in source


def test_speak_does_not_call_tts() -> None:
    """验证 speak.execute 不调用 TTS。"""
    import inspect
    import eve.output.speak as s

    source = inspect.getsource(s.execute)
    assert "sounddevice" not in source
    assert "pyttsx3" not in source
    assert "tts" not in source.lower().replace("_", "").replace("-", "")


# ── emergency stop 后拒绝 ────────────────────────────────


def test_emergency_stop_rejects_pending(ready_state: RuntimeState, tmp_log_dir: Path):
    """emergency stop 后 pending 动作应被拒绝。"""
    safegate.emergency_stop(ready_state)
    action = ActionCandidate(
        action_id="test-emergency-001",
        kind=ActionKind.MOUSE,
        payload={},
        origin="test",
    )
    result = loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    assert result.executed is False
    assert "emergency_stopped" in result.reason


def test_emergency_stop_rejects_new_action(ready_state: RuntimeState, tmp_log_dir: Path):
    """emergency stop 后新动作也应被拒绝。"""
    safegate.emergency_stop(ready_state)
    action = ActionCandidate(
        action_id="test-emergency-002",
        kind=ActionKind.SPEAK,
        payload={},
        origin="test",
    )
    result = loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    assert result.executed is False
    assert "emergency_stopped" in result.reason


# ── 日志可解析 ──────────────────────────────────────────


def test_log_is_valid_jsonl(ready_state: RuntimeState, tmp_log_dir: Path):
    """日志文件应包含有效 JSONL 嵌套结构。"""
    action = ActionCandidate(
        action_id="test-log-001",
        kind=ActionKind.MOUSE,
        payload={"x": 1},
        origin="test",
    )
    loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    loop.log_event(ready_state, "custom_event", log_dir=tmp_log_dir, key="val")

    log_files = list(tmp_log_dir.glob("*.jsonl"))
    assert len(log_files) == 1, f"应该有 1 个日志文件，实际: {log_files}"
    lines = log_files[0].read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 2

    run_once_line = None
    for line in lines:
        data = json.loads(line)
        assert "event" in data
        assert "timestamp_ns" in data
        assert "run_id" in data
        if data["event"] == "run_once":
            run_once_line = data

    # 验证嵌套结构
    assert run_once_line is not None, "应有一条 run_once 事件"
    assert "action" in run_once_line
    assert "safegate" in run_once_line
    assert "output" in run_once_line
    assert run_once_line["action"]["action_id"] == "test-log-001"
    assert run_once_line["output"]["simulated"] is True
    assert run_once_line["output"]["executed"] is False


# ── 停止后无存活线程 ────────────────────────────────────


def test_no_lingering_threads(ready_state: RuntimeState, tmp_log_dir: Path):
    """run_once 是同步函数，执行后不应产生后台线程。"""
    before = threading.active_count()
    action = ActionCandidate(
        action_id="test-thread-001",
        kind=ActionKind.MOUSE,
        payload={"x": 1},
        origin="test",
    )
    loop.run_once(ready_state, action, log_dir=tmp_log_dir)
    after = threading.active_count()
    assert after <= before, f"run_once 后线程数不应增加: before={before} after={after}"


# ── 异常时 pending_action 被清空 ──────────────────────────


def test_pending_action_cleared_on_exception(
    ready_state: RuntimeState, monkeypatch, tmp_log_dir: Path
):
    """即使输出过程抛异常，pending_action 也应该在 finally 中清空。"""
    # 注入一个会抛异常的 output 函数
    def _explode(*_a, **_kw):
        raise RuntimeError("injected error")

    monkeypatch.setattr(
        "eve.output.mouse.execute", _explode
    )
    action = ActionCandidate(
        action_id="test-explode-001",
        kind=ActionKind.MOUSE,
        payload={},
        origin="test",
    )
    with pytest.raises(RuntimeError, match="injected error"):
        loop.run_once(ready_state, action, log_dir=tmp_log_dir)

    # pending_action 必须在 finally 中清空
    assert ready_state.pending_action is None


# ── 不导入 src.eve ─────────────────────────────────────


def test_no_src_eve_import() -> None:
    """确保 runtime loop 测试模块不导入 src.eve。"""
    import sys
    assert "src.eve" not in sys.modules, "test_runtime_loop 不应导入 src.eve"
