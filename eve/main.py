"""Safe process entry point; Main controls Input only through InputBuffer."""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from eve.core.loop import (
    DEFAULT_LOCAL_LLM_PATH,
    DEFAULT_VLM_PATH,
    DEFAULT_YOLO_PATH,
    CoreLoop,
    create_runtime_state,
    log_event,
)
from eve.core.safegate import emergency_stop, reset_emergency
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer
from eve.output import keyboard, mouse, speak


class EVEApplication:
    """Own one runtime and enforce its shutdown order."""

    def __init__(
        self,
        *,
        profile: str = "smoke",
        mode: str | None = None,
        run_dir: str | Path = "runs",
        memory_dir: str | Path | None = None,
        input_buffer: InputBuffer | None = None,
        tnn_id: str | None = None,
        allow_mock_actions: bool = False,
        use_default_local_models: bool = True,
    ) -> None:
        if profile not in {"smoke", "observe", "control"}:
            raise ValueError(f"unsupported runtime profile: {profile}")
        normalized_mode = getattr(mode, "value", mode) or (
            "real" if profile == "control" else "mock"
        )
        if profile != "control" and normalized_mode == "real":
            raise ValueError("real output is only available in control profile")
        self.profile = profile
        self.run_dir = Path(run_dir)
        self.buffer = input_buffer or InputBuffer(profile=profile)
        self.state = create_runtime_state(
            output_mode=normalized_mode,
            allow_mock_actions=allow_mock_actions,
        )
        self._critical_event = threading.Event()
        self._stop_requested = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._pending_permission_records: list[dict[str, Any]] = []
        self.started_at_ns = 0
        self.finished_at_ns = 0
        self.exit_reason = "not_started"
        self.memory = Memorizer(
            Path(memory_dir) if memory_dir is not None else self.run_dir / "memory",
            writer_error_callback=self._memory_error,
        )
        self.core = CoreLoop(
            self.buffer,
            self.memory,
            state=self.state,
            log_dir=self.run_dir,
            tnn_id=tnn_id,
            smoke_node=tnn_id is None,
        )
        self.core.load_snapshot(self.run_dir / "state_snapshot.json")
        if profile == "control" and use_default_local_models:
            model_config = self.state["model_config"]
            model_config["local_llm_path"] = (
                model_config.get("local_llm_path") or DEFAULT_LOCAL_LLM_PATH
            )
            model_config["vlm_path"] = (
                model_config.get("vlm_path") or DEFAULT_VLM_PATH
            )
            model_config["yolo_model_path"] = (
                model_config.get("yolo_model_path") or DEFAULT_YOLO_PATH
            )
            for name, path in (
                ("local_llm", model_config["local_llm_path"]),
                ("vlm", model_config["vlm_path"]),
                ("yolo", model_config["yolo_model_path"]),
            ):
                self.state["model_status"][name].update(
                    {
                        "state": "configured",
                        "path": path,
                        "quantization": (
                            "4bit-nf4-required"
                            if name in {"local_llm", "vlm"}
                            else "native-cuda"
                        ),
                    }
                )

    @property
    def running(self) -> bool:
        return (
            self.state["cold_started"]
            and self.core.running
            and self.buffer.capture_running
        )

    @property
    def critical_failure(self) -> bool:
        return self._critical_event.is_set() or self.core.failed

    def start(self, *, load_smoke_node: bool = True) -> None:
        if self.running:
            return
        if self.buffer.closed:
            raise RuntimeError("runtime was closed; create a new EVEApplication")
        self.core.smoke_node = load_smoke_node and self.core.tnn_id is None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_ns = time.monotonic_ns()
        self.exit_reason = "running"
        self._critical_event.clear()
        self._stop_requested.clear()
        self.memory.start_writer()
        mouse.reset_stop()
        keyboard.reset_stop()
        speak.reset_stop()
        self.state["resource_status"].update(
            {"memory_writer": "running", "capture": "starting", "core": "starting"}
        )
        try:
            self.buffer.start_capture()
            self.state["resource_status"]["capture"] = "running"
            self.core.start()
            for record in self._pending_permission_records:
                self.memory.enqueue(
                    record, "permission_change", priority="critical"
                )
            self._pending_permission_records.clear()
            self.state["resource_status"]["core"] = "running"
            self._watch_thread = threading.Thread(
                target=self._watch_stop, name="eve-stop-watch"
            )
            self._watch_thread.start()
            log_event(
                self.run_dir,
                "runtime_started",
                profile=self.profile,
                output_mode=self.state["output_mode"],
                permissions=self.state["permissions"],
                tnn_id=self.core.tnn_id or "smoke_rule",
                capture_pid=self.buffer.capture_process_id,
            )
        except Exception:
            self.exit_reason = "startup_error"
            self._critical_event.set()
            self._stop_requested.set()
            try:
                self.stop()
            except Exception as cleanup_exc:
                self.state["resource_status"]["startup_cleanup_error"] = (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise

    def wait(self, duration_s: float | None = None) -> bool:
        deadline = time.monotonic() + duration_s if duration_s is not None else None
        while not self._stop_requested.wait(0.02):
            if deadline is not None and time.monotonic() >= deadline:
                self.exit_reason = "duration_elapsed"
                self._stop_requested.set()
                break
        if self.core.failed:
            self._critical_event.set()
            self.exit_reason = "core_error"
        return not self.critical_failure

    def request_stop(self, reason: str = "requested") -> None:
        if self.exit_reason == "running":
            self.exit_reason = reason
        self._stop_requested.set()

    def stop(self) -> None:
        """Stop Loop and Memory first, then Buffer closes Capture and shared memory."""
        self._stop_requested.set()
        failures: list[Exception] = []
        try:
            self.core.save_snapshot(self.run_dir / "state_snapshot.json")
        except Exception as exc:
            failures.append(exc)
        for stop_output in (mouse.stop_all, keyboard.stop_all, speak.stop_all):
            try:
                stop_output()
            except Exception as exc:
                failures.append(exc)
        for stop in (self.core.stop, self.memory.stop_writer, self.buffer.close):
            try:
                stop()
            except Exception as exc:
                failures.append(exc)
        self.state["resource_status"].update(
            {
                "core": "stopped",
                "memory_writer": "stopped",
                "capture": "stopped",
                "input_buffer": "closed",
            }
        )
        watch = self._watch_thread
        if watch is not None and watch is not threading.current_thread():
            watch.join(3.0)
            if watch.is_alive():
                failures.append(RuntimeError("stop watch thread did not stop"))
        self._watch_thread = None
        self.finished_at_ns = time.monotonic_ns()
        if self.exit_reason == "running":
            self.exit_reason = "stopped"
        if failures:
            self._critical_event.set()
            self.exit_reason = "shutdown_error"
            raise RuntimeError(
                "shutdown failure(s): " + "; ".join(str(item) for item in failures)
            )
        log_event(self.run_dir, "shutdown_complete", reason=self.exit_reason)

    def emergency(self, reason: str = "user_emergency_stop") -> None:
        emergency_stop(self.state)
        self.state["lifecycle"]["escape_triggered_at_ns"] = time.monotonic_ns()
        self.state["lifecycle"]["reason"] = reason
        self.core.cancel_generation()
        for stop_output in (mouse.stop_all, keyboard.stop_all, speak.stop_all):
            try:
                stop_output()
            except Exception as exc:
                self.state["latest_error"] = {
                    "timestamp_ns": time.time_ns(),
                    "loop_node": "emergency_output_stop",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
        log_event(self.run_dir, "emergency_stop", source=reason)

    def clear_emergency(self) -> None:
        reset_emergency(self.state)
        mouse.reset_stop()
        keyboard.reset_stop()
        speak.reset_stop()
        log_event(self.run_dir, "emergency_reset", source="gui")

    def change_permission(
        self, group: str, name: str | None, enabled: bool
    ) -> None:
        permissions = self.state["permissions"]
        if name is None:
            if group not in {"send_text", "speak"}:
                raise KeyError(group)
            permissions[group] = bool(enabled)
            atom = group
        else:
            values = permissions.get(group)
            if not isinstance(values, dict) or name not in values:
                raise KeyError(f"{group}.{name}")
            values[name] = bool(enabled)
            atom = f"{group}.{name}"
        record = {
            "atom": atom,
            "enabled": bool(enabled),
            "timestamp_ns": time.time_ns(),
        }
        log_event(self.run_dir, "permission_changed", **record)
        if self.memory.writer_running:
            self.memory.enqueue(
                record, "permission_change", priority="critical"
            )
        else:
            self._pending_permission_records.append(record)

    def summary(self) -> dict[str, Any]:
        finished_ns = self.finished_at_ns or time.monotonic_ns()
        duration_s = (
            max(0, finished_ns - self.started_at_ns) / 1_000_000_000
            if self.started_at_ns else 0.0
        )
        capture = self.buffer.capture_stats()
        core = self.core.stats()
        memory = self.memory.writer_stats()
        stats = self.state["runtime_stats"]
        latest_error = self.state["latest_error"]
        return {
            "profile": self.profile,
            "duration_s": duration_s,
            "screen_fps": capture["screen_fps"],
            "cursor_hz": capture["cursor_hz"],
            "screen_latency_ms": capture["screen_average_latency_ms"],
            "cursor_latency_ms": capture["cursor_average_latency_ms"],
            "core_loop_hz": core["loop_hz"],
            "tnn_invocations": core["tnn_invocations"],
            "tnn_device": self.state["resource_status"].get("tnn_device"),
            "safegate_allowed": stats.get("safegate_allowed", 0),
            "safegate_blocked": stats.get("safegate_blocked", 0),
            "mock_outputs": stats.get("mock_outputs", 0),
            "real_output_calls": 0,
            "memory_written": memory["written"],
            "memory_dropped": memory["dropped"],
            "memory_failed": memory["failed"],
            "critical_error": (
                latest_error.get("message") if latest_error else memory["last_error"]
            ),
            "exit_reason": self.exit_reason,
            "threads_stopped": not (
                self.buffer.capture_running
                or self.core.running
                or self.memory.writer_running
                or (
                    self._watch_thread is not None
                    and self._watch_thread.is_alive()
                )
            ),
            "capture_process_stopped": not self.buffer.capture_running,
            "log": str(self.run_dir / "eve.jsonl"),
        }

    def _watch_stop(self) -> None:
        while not self._stop_requested.wait(0.01):
            if _global_escape_pressed():
                self.emergency("global_escape")
                self.exit_reason = "escape_key"
                if self.profile != "control":
                    self._stop_requested.set()
                    return
                while _global_escape_pressed() and not self._stop_requested.wait(0.05):
                    pass
            capture_error = self.buffer.capture_error
            if capture_error is not None:
                self.state["latest_error"] = capture_error
                self._critical_event.set()
                self.exit_reason = "capture_error"
                self._stop_requested.set()
                return
            if not self.buffer.capture_running:
                self.state["latest_error"] = {
                    "loop_node": "capture_process",
                    "exception_type": "ProcessExit",
                    "message": "Capture exited unexpectedly",
                }
                self._critical_event.set()
                self.exit_reason = "capture_error"
                self._stop_requested.set()
                return
            if self.core.failed:
                self._critical_event.set()
                self.exit_reason = "core_error"
                self._stop_requested.set()
                return

    def _memory_error(self, error: Exception) -> None:
        self.state["latest_error"] = {
            "timestamp_ns": time.time_ns(),
            "loop_node": "memory_writer",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "recovery_action": "runtime_stopped_for_memory_integrity",
        }
        self._critical_event.set()
        self.exit_reason = "memory_error"
        self._stop_requested.set()


def _global_escape_pressed() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except (AttributeError, OSError):
        return False


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=repr)


def _conversation_text(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        user = str(item.get("user", "")).strip()
        reply = str(item.get("reply", "")).strip()
        if user:
            lines.append(f"用户：{user}")
        if reply:
            lines.append(f"EVE：{reply}")
        if user or reply:
            lines.append("")
    return "\n".join(lines).strip() or "暂无对话"


def _format_ns(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "-"
    if number <= 0:
        return "-"
    return f"{number} ns"


def _load_qt() -> dict[str, Any]:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    return locals()


def _qimage_format(qimage: Any, name: str) -> Any:
    """Resolve QImage formats across PySide6's legacy and scoped enums."""
    value = getattr(qimage, name, None)
    if value is not None:
        return value
    scoped = getattr(qimage, "Format", None)
    if scoped is not None:
        value = getattr(scoped, name, None)
        if value is not None:
            return value
    raise RuntimeError(f"PySide6 QImage does not support {name}")


class EVEControlWindow:
    """Thin facade that creates the actual Qt window without a GUI business layer."""

    @staticmethod
    def create(
        application: EVEApplication,
        *,
        application_factory: Any | None = None,
    ) -> Any:
        qt = _load_qt()
        QMainWindow = qt["QMainWindow"]

        class Window(QMainWindow):
            def __init__(self) -> None:
                super().__init__()
                self.application = application
                self.application_factory = application_factory
                self._operation_lock = threading.Lock()
                self._operation_thread: threading.Thread | None = None
                self._background_error: str | None = None
                self._closing = False
                self._keyboard_checks: dict[str, Any] = {}
                self._mouse_checks: dict[str, Any] = {}
                self.setWindowTitle("EVE Runtime Control")
                self.resize(1380, 900)
                self._build_ui(qt)
                self._timer = qt["QTimer"](self)
                self._timer.timeout.connect(self.refresh)
                self._timer.start(200)
                shortcut = qt["QShortcut"](qt["QKeySequence"]("Esc"), self)
                shortcut.activated.connect(
                    lambda: self._emergency("gui_escape")
                )
                self._escape_shortcut = shortcut
                self.refresh()

            def _build_ui(self, qt: dict[str, Any]) -> None:
                QWidget = qt["QWidget"]
                QVBoxLayout = qt["QVBoxLayout"]
                QTabWidget = qt["QTabWidget"]
                root = QWidget()
                layout = QVBoxLayout(root)
                self.banner = qt["QLabel"](
                    "GUI 已启动；Capture/Core/模型尚未冷启动。真实权限全部关闭。"
                )
                self.banner.setWordWrap(True)
                layout.addWidget(self.banner)
                self.tabs = QTabWidget()
                layout.addWidget(self.tabs)
                self.setCentralWidget(root)
                builders = (
                    ("实时视觉", self._build_visual_page),
                    ("文本与认知", self._build_text_page),
                    ("资源与节点", self._build_resource_page),
                    ("冷启动与急停", self._build_lifecycle_page),
                    ("Memory / Blackboard", self._build_memory_page),
                    ("权限与倾向", self._build_permission_page),
                    ("设置 / 激素 / 反馈", self._build_settings_page),
                    ("TNN", self._build_tnn_page),
                )
                for title, builder in builders:
                    page = QWidget()
                    builder(page, qt)
                    self.tabs.addTab(page, title)

            @staticmethod
            def _readonly(qt: dict[str, Any], height: int = 150) -> Any:
                editor = qt["QPlainTextEdit"]()
                editor.setReadOnly(True)
                editor.setMinimumHeight(height)
                return editor

            def _build_visual_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                row = qt["QHBoxLayout"]()
                self.screen_view = qt["QLabel"]("暂无屏幕帧")
                self.screen_view.setAlignment(qt["Qt"].AlignCenter)
                self.screen_view.setMinimumSize(720, 405)
                self.screen_view.setStyleSheet("background:#111;color:#bbb;")
                row.addWidget(self.screen_view, 3)
                right = qt["QVBoxLayout"]()
                self.visual_metrics = qt["QLabel"]("未启动")
                self.visual_metrics.setWordWrap(True)
                self.visual_result = self._readonly(qt, 170)
                self.teacher_visual_result = self._readonly(qt, 130)
                analyze = qt["QPushButton"]("YOLO/TNN 分析当前帧")
                analyze.clicked.connect(self._request_visual)
                teacher = qt["QPushButton"]("VLM 教师复核当前帧")
                teacher.clicked.connect(self._request_teacher_review)
                right.addWidget(self.visual_metrics)
                right.addWidget(analyze)
                right.addWidget(qt["QLabel"]("运行时视觉结果"))
                right.addWidget(self.visual_result)
                right.addWidget(teacher)
                right.addWidget(qt["QLabel"]("VLM 教师结果"))
                right.addWidget(self.teacher_visual_result)
                row.addLayout(right, 2)
                layout.addLayout(row)
                note = qt["QLabel"](
                    "目光聚焦、注意点和显著性坐标：未实现（不使用鼠标或屏幕中心伪造）。"
                )
                note.setWordWrap(True)
                layout.addWidget(note)

            def _build_text_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                row = qt["QHBoxLayout"]()
                self.message_input = qt["QLineEdit"]()
                self.message_input.setPlaceholderText("输入给本地 LLM 的消息")
                self.message_input.returnPressed.connect(self._send_message)
                send = qt["QPushButton"]("发送")
                send.clicked.connect(self._send_message)
                cancel = qt["QPushButton"]("停止当前生成")
                cancel.clicked.connect(
                    lambda: self.application.core.cancel_generation()
                )
                row.addWidget(self.message_input)
                row.addWidget(send)
                row.addWidget(cancel)
                layout.addLayout(row)
                self.conversation_view = self._readonly(qt, 260)
                layout.addWidget(self.conversation_view)
                grid = qt["QGridLayout"]()
                self.thinking_view = self._readonly(qt, 100)
                self.world_view = self._readonly(qt, 180)
                self.myself_view = self._readonly(qt, 180)
                grid.addWidget(qt["QLabel"]("可见思想摘要"), 0, 0)
                grid.addWidget(self.thinking_view, 1, 0)
                grid.addWidget(qt["QLabel"]("world"), 0, 1)
                grid.addWidget(self.world_view, 1, 1)
                grid.addWidget(qt["QLabel"]("myself / 当前任务"), 0, 2)
                grid.addWidget(self.myself_view, 1, 2)
                layout.addLayout(grid)

            def _build_resource_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                self.resource_summary = qt["QLabel"]("未冷启动")
                self.resource_summary.setWordWrap(True)
                layout.addWidget(self.resource_summary)
                self.node_table = qt["QTableWidget"](0, 7)
                self.node_table.setHorizontalHeaderLabels(
                    [
                        "节点", "状态", "实际 Hz", "最近运行",
                        "最近耗时 ms", "平均耗时 ms", "错误",
                    ]
                )
                layout.addWidget(self.node_table)

            def _build_lifecycle_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                row = qt["QHBoxLayout"]()
                buttons = (
                    ("冷启动", self._cold_start),
                    ("暂停", lambda: self.application.core.pause("gui_pause")),
                    ("恢复", self._resume),
                    ("正常停机", self._normal_stop),
                    ("急停", lambda: self._emergency("gui_button")),
                    ("显式解除急停", self._clear_emergency),
                )
                for label, callback in buttons:
                    button = qt["QPushButton"](label)
                    button.clicked.connect(callback)
                    row.addWidget(button)
                layout.addLayout(row)
                self.lifecycle_status = qt["QLabel"]()
                self.lifecycle_status.setWordWrap(True)
                layout.addWidget(self.lifecycle_status)
                layout.addStretch()

            def _build_memory_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                row = qt["QHBoxLayout"]()
                self.memory_counts = qt["QLabel"]("STM 0 / MTM 0 / LTM 0")
                review = qt["QPushButton"]("强制睡眠整理")
                review.clicked.connect(self._force_review)
                row.addWidget(self.memory_counts)
                row.addWidget(review)
                layout.addLayout(row)
                self.review_progress = qt["QProgressBar"]()
                layout.addWidget(self.review_progress)
                self.review_label = qt["QLabel"]("整理：空闲")
                layout.addWidget(self.review_label)
                self.memory_view = self._readonly(qt, 170)
                self.blackboard_view = self._readonly(qt, 280)
                layout.addWidget(qt["QLabel"]("最近 MemoryUnit / Event"))
                layout.addWidget(self.memory_view)
                layout.addWidget(qt["QLabel"]("Blackboard 有效条目"))
                layout.addWidget(self.blackboard_view)

            def _build_permission_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QHBoxLayout"](page)
                mouse_group = qt["QGroupBox"]("鼠标原子权限（全部默认关闭）")
                mouse_layout = qt["QVBoxLayout"](mouse_group)
                for atom in self.application.state["permissions"]["mouse"]:
                    check = qt["QCheckBox"](atom)
                    check.toggled.connect(
                        lambda value, name=atom: self._permission_changed(
                            "mouse", name, value
                        )
                    )
                    self._mouse_checks[atom] = check
                    mouse_layout.addWidget(check)
                text_check = qt["QCheckBox"]("send_text")
                text_check.toggled.connect(
                    lambda value: self._permission_changed(
                        "send_text", None, value
                    )
                )
                speak_check = qt["QCheckBox"]("speak")
                speak_check.toggled.connect(
                    lambda value: self._permission_changed(
                        "speak", None, value
                    )
                )
                self.send_text_check = text_check
                self.speak_check = speak_check
                mouse_layout.addWidget(text_check)
                mouse_layout.addWidget(speak_check)
                mouse_layout.addStretch()
                layout.addWidget(mouse_group, 1)

                key_group = qt["QGroupBox"]("键盘逐键权限")
                key_layout = qt["QVBoxLayout"](key_group)
                self.key_search = qt["QLineEdit"]()
                self.key_search.setPlaceholderText("搜索按键")
                self.key_search.textChanged.connect(self._filter_keys)
                key_layout.addWidget(self.key_search)
                scroll = qt["QScrollArea"]()
                scroll.setWidgetResizable(True)
                key_body = qt["QWidget"]()
                key_body_layout = qt["QGridLayout"](key_body)
                for index, key in enumerate(
                    self.application.state["permissions"]["keyboard"]
                ):
                    check = qt["QCheckBox"](key)
                    check.toggled.connect(
                        lambda value, name=key: self._permission_changed(
                            "keyboard", name, value
                        )
                    )
                    self._keyboard_checks[key] = check
                    key_body_layout.addWidget(check, index // 6, index % 6)
                scroll.setWidget(key_body)
                key_layout.addWidget(scroll)
                layout.addWidget(key_group, 3)
                tendency_group = qt["QGroupBox"]("行为倾向（不能越过权限）")
                tendency_layout = qt["QVBoxLayout"](tendency_group)
                self.tendency_view = self._readonly(qt, 420)
                tendency_layout.addWidget(self.tendency_view)
                layout.addWidget(tendency_group, 2)

            def _build_settings_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                form = qt["QFormLayout"]()
                config = self.application.state["model_config"]
                self.local_path = qt["QLineEdit"](config["local_llm_path"])
                self.vlm_path = qt["QLineEdit"](config["vlm_path"])
                self.yolo_path = qt["QLineEdit"](config["yolo_model_path"])
                self.cloud_url = qt["QLineEdit"](config["cloud_base_url"])
                self.cloud_model = qt["QLineEdit"](config["cloud_model"])
                self.cloud_key = qt["QLineEdit"]()
                self.cloud_key.setEchoMode(
                    qt["QLineEdit"].EchoMode.Password
                )
                self.cloud_enabled = qt["QCheckBox"]()
                self.cloud_enabled.setChecked(bool(config["cloud_enabled"]))
                form.addRow("本地 LLM 路径（强制 4-bit NF4）", self.local_path)
                form.addRow("VLM 教师路径（按需 4-bit NF4）", self.vlm_path)
                form.addRow("YOLO 运行时视觉路径", self.yolo_path)
                form.addRow("云端 base_url", self.cloud_url)
                form.addRow("云端 model", self.cloud_model)
                form.addRow("API key（仅内存，不保存）", self.cloud_key)
                form.addRow("启用云端", self.cloud_enabled)
                layout.addLayout(form)
                apply_button = qt["QPushButton"]("应用设置（模型路径下次冷启动加载）")
                apply_button.clicked.connect(self._apply_settings)
                layout.addWidget(apply_button)
                self.model_status_view = self._readonly(qt, 150)
                self.hormone_view = self._readonly(qt, 180)
                layout.addWidget(self.model_status_view)
                layout.addWidget(self.hormone_view)
                row = qt["QHBoxLayout"]()
                praise = qt["QPushButton"]("表扬")
                praise.clicked.connect(lambda: self._feedback("praise"))
                criticism = qt["QPushButton"]("批评")
                criticism.clicked.connect(lambda: self._feedback("criticism"))
                row.addWidget(praise)
                row.addWidget(criticism)
                layout.addLayout(row)

            def _build_tnn_page(self, page: Any, qt: dict[str, Any]) -> None:
                layout = qt["QVBoxLayout"](page)
                self.tnn_summary = qt["QLabel"]("最大加载数：5")
                layout.addWidget(self.tnn_summary)
                row = qt["QHBoxLayout"]()
                self.tnn_id_input = qt["QLineEdit"]()
                self.tnn_id_input.setPlaceholderText("已存在 TNN 的 tnn_id 或 MemoryID")
                load = qt["QPushButton"]("加载")
                load.clicked.connect(self._load_tnn)
                unload = qt["QPushButton"]("卸载")
                unload.clicked.connect(self._unload_tnn)
                activate = qt["QPushButton"]("激活")
                activate.clicked.connect(self._activate_tnn)
                pause = qt["QPushButton"]("暂停")
                pause.clicked.connect(self._pause_tnn)
                row.addWidget(self.tnn_id_input)
                for button in (load, unload, activate, pause):
                    row.addWidget(button)
                layout.addLayout(row)
                self.tnn_table = qt["QTableWidget"](5, 12)
                self.tnn_table.setHorizontalHeaderLabels(
                    [
                        "Slot", "tnn_id", "版本", "用途", "状态", "设备",
                        "精度", "频率", "TTL", "最近耗时", "平均耗时", "错误",
                    ]
                )
                layout.addWidget(self.tnn_table)
                self.tnn_detail = self._readonly(qt, 180)
                layout.addWidget(self.tnn_detail)

            def _run_operation(self, operation: Any) -> None:
                if self._operation_thread and self._operation_thread.is_alive():
                    self._background_error = "已有生命周期操作正在进行"
                    return

                def run() -> None:
                    try:
                        with self._operation_lock:
                            operation()
                    except Exception as exc:
                        self._background_error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                self._operation_thread = threading.Thread(
                    target=run, name="eve-gui-lifecycle"
                )
                self._operation_thread.start()

            def _cold_start(self) -> None:
                if self.application.running:
                    return
                if self.application.buffer.closed:
                    if self.application_factory is None:
                        self._background_error = "运行时已关闭，无法重建"
                        return
                    self.application = self.application_factory()
                self._run_operation(
                    lambda: self.application.start(load_smoke_node=False)
                )

            def _normal_stop(self) -> None:
                if not (
                    self.application.core.running
                    or self.application.buffer.capture_running
                    or self.application.memory.writer_running
                ):
                    return
                self.application.exit_reason = "gui_normal_stop"
                self._run_operation(self.application.stop)

            def _resume(self) -> None:
                try:
                    self.application.core.resume()
                except Exception as exc:
                    self._background_error = str(exc)

            def _emergency(self, source: str) -> None:
                self.application.emergency(source)

            def _clear_emergency(self) -> None:
                self.application.clear_emergency()

            def _send_message(self) -> None:
                text = self.message_input.text().strip()
                if not text:
                    return
                try:
                    self.application.core.submit_user_message(text)
                    self.message_input.clear()
                except Exception as exc:
                    self._background_error = str(exc)

            def _request_visual(self) -> None:
                try:
                    self.application.core.submit_runtime_visual_analysis()
                except Exception as exc:
                    self._background_error = str(exc)

            def _request_teacher_review(self) -> None:
                try:
                    self.application.core.submit_teacher_review()
                except Exception as exc:
                    self._background_error = str(exc)

            def _permission_changed(
                self, group: str, name: str | None, value: bool
            ) -> None:
                try:
                    self.application.change_permission(group, name, value)
                except Exception as exc:
                    self._background_error = str(exc)

            def _filter_keys(self, text: str) -> None:
                needle = text.strip().upper()
                for key, check in self._keyboard_checks.items():
                    check.setVisible(not needle or needle in key)

            def _apply_settings(self) -> None:
                try:
                    self.application.core.configure_models(
                        {
                            "local_llm_path": self.local_path.text().strip(),
                            "vlm_path": self.vlm_path.text().strip(),
                            "yolo_model_path": self.yolo_path.text().strip(),
                            "cloud_base_url": self.cloud_url.text().strip(),
                            "cloud_model": self.cloud_model.text().strip(),
                            "cloud_enabled": self.cloud_enabled.isChecked(),
                        }
                    )
                    self.application.state["_cloud_api_key"] = (
                        self.cloud_key.text().strip()
                    )
                    self.cloud_key.clear()
                except Exception as exc:
                    self._background_error = str(exc)

            def _feedback(self, kind: str) -> None:
                try:
                    self.application.core.feedback(kind)
                except Exception as exc:
                    self._background_error = str(exc)

            def _force_review(self) -> None:
                if not self.application.state["cold_started"]:
                    self._background_error = "冷启动后才能执行 Memory 整理"
                    return
                if not self.application.memory.force_review():
                    self._background_error = "Memory 整理已在运行"

            def _load_tnn(self) -> None:
                tnn_id = self.tnn_id_input.text().strip()
                if not tnn_id:
                    return
                try:
                    self.application.core.request_tnn_load(tnn_id)
                except Exception as exc:
                    self._background_error = str(exc)

            def _unload_tnn(self) -> None:
                tnn_id = self.tnn_id_input.text().strip()
                if tnn_id:
                    self.application.core.request_tnn_unload(tnn_id)

            def _activate_tnn(self) -> None:
                tnn_id = self.tnn_id_input.text().strip()
                if tnn_id and not self.application.core.activate_tnn(tnn_id):
                    self._background_error = "TNN 激活失败；请查看 TNN 错误"

            def _pause_tnn(self) -> None:
                tnn_id = self.tnn_id_input.text().strip()
                if tnn_id and not self.application.core.pause_tnn(tnn_id):
                    self._background_error = "TNN 暂停失败；请查看 TNN 错误"

            def refresh(self) -> None:
                state = self.application.state
                lifecycle = state["lifecycle"]
                if self._background_error:
                    self.banner.setText(f"错误：{self._background_error}")
                    self._background_error = None
                else:
                    self.banner.setText(
                        f"生命周期：{lifecycle['state']} | "
                        f"急停：{state['emergency_stop']} | "
                        f"输出：{state['output_mode']} | "
                        "本地模型量化策略：强制 4-bit NF4"
                    )
                self._refresh_visual(state)
                self._refresh_text(state)
                self._refresh_resources(state)
                self._refresh_lifecycle(state)
                self._refresh_memory(state)
                self._refresh_permissions(state)
                self._refresh_settings(state)
                self._refresh_tnn(state)

            def _refresh_visual(self, state: dict[str, Any]) -> None:
                qt = _load_qt()
                sample = self.application.buffer.get_latest_screen()
                cursor = self.application.buffer.get_latest_cursor()
                stats = self.application.buffer.capture_stats()
                frame_id = "-"
                frame_time = "-"
                if sample is not None:
                    frame = sample.value
                    frame_id = str(frame.frame_id)
                    frame_time = _format_ns(frame.captured_at_ns)
                    image = frame.image
                    height, width = image.shape[:2]
                    if image.shape[2] == 4:
                        qimage = qt["QImage"](
                            image.data,
                            width,
                            height,
                            int(image.strides[0]),
                            # MSS frames are BGRA. On little-endian Windows,
                            # ARGB32 has the same in-memory byte order.
                            _qimage_format(qt["QImage"], "Format_ARGB32"),
                        ).copy()
                    else:
                        qimage = qt["QImage"](
                            image.data,
                            width,
                            height,
                            int(image.strides[0]),
                            _qimage_format(qt["QImage"], "Format_RGB888"),
                        ).copy()
                    pixmap = qt["QPixmap"].fromImage(qimage)
                    self.screen_view.setPixmap(
                        pixmap.scaled(
                            self.screen_view.size(),
                            qt["Qt"].KeepAspectRatio,
                            qt["Qt"].SmoothTransformation,
                        )
                    )
                cursor_text = "-"
                if cursor is not None:
                    value = cursor.value
                    cursor_text = f"({value.x}, {value.y})"
                self.visual_metrics.setText(
                    f"cursor {cursor_text}\nFPS {stats['screen_fps']:.2f}\n"
                    f"capture latency {stats['screen_average_latency_ms']:.3f} ms\n"
                    f"dropped {stats.get('dropped_screen_frames', 0)}\n"
                    f"frame_id {frame_id}\nframe_time {frame_time}"
                )
                result = state.get("visual_result")
                self.visual_result.setPlainText(
                    _json_text(result) if result else "暂无 YOLO/TNN 结果"
                )
                teacher_result = state.get("teacher_visual_result")
                self.teacher_visual_result.setPlainText(
                    _json_text(teacher_result)
                    if teacher_result
                    else "尚未请求 VLM 教师复核"
                )

            def _refresh_text(self, state: dict[str, Any]) -> None:
                self.conversation_view.setPlainText(
                    _conversation_text(state["conversation"][-30:])
                )
                self.thinking_view.setPlainText(
                    str(state["myself"].get("what_im_thinking", ""))
                    or "暂无可见思想摘要"
                )
                self.world_view.setPlainText(_json_text(state["world"]))
                self.myself_view.setPlainText(_json_text(state["myself"]))

            def _refresh_resources(self, state: dict[str, Any]) -> None:
                resources = state["resource_status"]
                cuda = state.get("cuda_status", {})
                self.resource_summary.setText(
                    f"CPU {resources.get('cpu_percent', '未采样')}% | "
                    f"RAM {resources.get('system_memory_percent', '未采样')}% | "
                    f"GPU utilization {resources.get('gpu_utilization', '不可用')}% | "
                    f"GPU allocated {resources.get('gpu_memory_allocated', '未采样')} | "
                    f"peak {resources.get('gpu_memory_peak', '未采样')}\n"
                    f"CUDA available={cuda.get('available', '未验证')} | "
                    f"GPU={cuda.get('device_name', '未验证')} | "
                    f"RTX 5080={cuda.get('is_rtx_5080', '未验证')} | "
                    f"CC={cuda.get('compute_capability', '未验证')} | "
                    f"torch CUDA={cuda.get('torch_cuda_version', '未验证')} | "
                    f"tensor test={cuda.get('tensor_test_passed', '未验证')}"
                )
                required = [
                    "capture", "buffer", "core", "local_llm", "yolo", "vlm",
                    "cloud_llm", "memory_writer", "memory_review", "safegate",
                    "mouse_output", "keyboard_output", "speak_output", "dock",
                ]
                nodes = state["node_status"]
                names = required + [
                    name for name in nodes if name not in required
                ]
                self.node_table.setRowCount(len(names))
                qt = _load_qt()
                for row, name in enumerate(names):
                    node = nodes.get(name, {"state": "未启动"})
                    values = (
                        name,
                        node.get("state", "未启动"),
                        node.get("actual_hz", "-"),
                        _format_ns(node.get("last_run_ns")),
                        node.get("last_duration_ms", "-"),
                        node.get("average_duration_ms", "-"),
                        node.get("last_error") or "",
                    )
                    for column, value in enumerate(values):
                        self.node_table.setItem(
                            row, column, qt["QTableWidgetItem"](str(value))
                        )

            def _refresh_lifecycle(self, state: dict[str, Any]) -> None:
                lifecycle = state["lifecycle"]
                self.lifecycle_status.setText(
                    f"state: {lifecycle['state']}\n"
                    f"changed_at: {_format_ns(lifecycle['changed_at_ns'])}\n"
                    f"reason: {lifecycle['reason']}\n"
                    f"Esc/emergency_at: "
                    f"{_format_ns(lifecycle.get('escape_triggered_at_ns'))}\n"
                    f"Capture: {self.application.buffer.capture_health()}"
                )

            def _refresh_memory(self, state: dict[str, Any]) -> None:
                memory = self.application.memory
                counts = memory.counts()
                self.memory_counts.setText(
                    f"STM {counts['stm']} | MTM {counts['mtm']} | "
                    f"LTM {counts['ltm']} | Event {counts['events']}"
                )
                review = memory.review_status()
                total = int(review.get("total", 0))
                processed = int(review.get("processed", 0))
                self.review_progress.setMaximum(max(1, total))
                self.review_progress.setValue(processed)
                eta = review.get("eta_s")
                eta_text = "正在计算" if eta is None else f"{eta:.2f}s"
                self.review_label.setText(
                    f"{review.get('state')} | {processed}/{total} | ETA {eta_text} | "
                    f"结果 {review.get('last_result')} | 错误 {review.get('last_error')}"
                )
                latest_id = memory.stm[-1] if memory.stm else None
                latest_unit = memory.get_unit(latest_id) if latest_id else None
                self.memory_view.setPlainText(
                    _json_text(
                        {
                            "latest_memory": (
                                latest_unit.__dict__ if latest_unit else None
                            ),
                            "latest_event": (
                                memory.latest_event().__dict__
                                if memory.latest_event()
                                else None
                            ),
                        }
                    )
                )
                self.blackboard_view.setPlainText(
                    _json_text(state["blackboard"])
                )

            def _refresh_permissions(self, state: dict[str, Any]) -> None:
                permissions = state["permissions"]
                for name, check in self._mouse_checks.items():
                    check.blockSignals(True)
                    check.setChecked(bool(permissions["mouse"][name]))
                    check.blockSignals(False)
                for name, check in self._keyboard_checks.items():
                    check.blockSignals(True)
                    check.setChecked(bool(permissions["keyboard"][name]))
                    check.blockSignals(False)
                for name, check in (
                    ("send_text", self.send_text_check),
                    ("speak", self.speak_check),
                ):
                    check.blockSignals(True)
                    check.setChecked(bool(permissions[name]))
                    check.blockSignals(False)
                self.tendency_view.setPlainText(
                    _json_text(state["myself"]["tendencies"])
                )

            def _refresh_settings(self, state: dict[str, Any]) -> None:
                self.model_status_view.setPlainText(
                    _json_text(state["model_status"])
                )
                self.hormone_view.setPlainText(
                    _json_text(
                        {
                            "current": state["myself"]["hormones"],
                            "last_change": state["myself"].get(
                                "last_hormone_change"
                            ),
                            "last_feedback": state.get("last_feedback"),
                        }
                    )
                )

            def _refresh_tnn(self, state: dict[str, Any]) -> None:
                nodes = list(state["loaded_tnn"].values())
                summary = state["resource_status"].get("tnn_summary", {})
                self.tnn_summary.setText(
                    f"最大加载数 5 | loaded {len(nodes)} | "
                    f"active {len(state['active_tnn'])} | 剩余 {5-len(nodes)} | "
                    f"显存 {summary.get('gpu_memory', 0)} | "
                    f"总推理耗时 {summary.get('total_inference_ms', 0)} ms | "
                    f"最近错误 {state['resource_status'].get('last_tnn_error')}"
                )
                qt = _load_qt()
                for row in range(5):
                    node = nodes[row] if row < len(nodes) else {}
                    values = (
                        f"Slot {row + 1}",
                        node.get("tnn_id", ""),
                        node.get("version", ""),
                        node.get("description", ""),
                        node.get("status", "empty"),
                        node.get("device", ""),
                        node.get("precision", ""),
                        node.get("run_frequency_hz", ""),
                        node.get("output_ttl_ns", ""),
                        node.get("last_duration_ms", ""),
                        node.get("average_duration_ms", ""),
                        node.get("last_error", ""),
                    )
                    for column, value in enumerate(values):
                        self.tnn_table.setItem(
                            row, column, qt["QTableWidgetItem"](str(value))
                        )
                connections = []
                for node in nodes:
                    for input_name, source in node.get("inputs", {}).items():
                        connections.append(
                            {
                                "downstream": node.get("tnn_id"),
                                "input": input_name,
                                "source_ref": source,
                            }
                        )
                self.tnn_detail.setPlainText(
                    _json_text(
                        {
                            "connections_from_core_source_refs": connections,
                            "slots": [
                                {
                                    "tnn_id": node.get("tnn_id"),
                                    "input_schema": node.get("input_schema"),
                                    "output_schema": node.get("output_schema"),
                                    "last_output_summary": node.get(
                                        "last_output_summary"
                                    ),
                                }
                                for node in nodes
                            ],
                        }
                    )
                )

            def closeEvent(self, event: Any) -> None:
                self._closing = True
                self._timer.stop()
                thread = self._operation_thread
                if thread is not None and thread.is_alive():
                    thread.join(10.0)
                try:
                    with self._operation_lock:
                        if (
                            self.application.core.running
                            or self.application.buffer.capture_running
                            or self.application.memory.writer_running
                        ):
                            self.application.exit_reason = "gui_closed"
                            self.application.stop()
                except Exception as exc:
                    qt["QMessageBox"].critical(
                        self, "EVE shutdown error", str(exc)
                    )
                event.accept()

        return Window()


def run_control_gui(
    *,
    run_dir: str | Path,
    memory_dir: str | Path | None,
    tnn_id: str | None,
    duration_s: float | None = None,
) -> int:
    qt = _load_qt()
    QApplication = qt["QApplication"]
    qt_application = QApplication.instance() or QApplication(sys.argv[:1])

    def factory() -> EVEApplication:
        return EVEApplication(
            profile="control",
            mode="real",
            run_dir=run_dir,
            memory_dir=memory_dir,
            tnn_id=tnn_id,
            allow_mock_actions=False,
        )

    window = EVEControlWindow.create(factory(), application_factory=factory)
    window.show()
    if duration_s is not None:
        qt["QTimer"].singleShot(
            max(1, int(duration_s * 1000)),
            window.close,
        )
    return int(qt_application.exec())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the safe EVE integration")
    parser.add_argument(
        "--profile", choices=("smoke", "observe", "control"), default="smoke"
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--tnn-id")
    parser.add_argument("--memory-dir")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    duration = args.duration
    if args.profile == "control":
        return run_control_gui(
            run_dir=args.run_dir,
            memory_dir=args.memory_dir,
            tnn_id=args.tnn_id,
            duration_s=duration,
        )
    if args.profile == "smoke" and duration is None:
        duration = 1.0
    application = EVEApplication(
        profile=args.profile,
        run_dir=args.run_dir,
        memory_dir=args.memory_dir,
        tnn_id=args.tnn_id,
        allow_mock_actions=False,
    )
    exit_code = 0
    try:
        application.start()
        if not application.wait(duration):
            exit_code = 1
    except KeyboardInterrupt:
        emergency_stop(application.state)
        application.exit_reason = "keyboard_interrupt"
    except Exception as exc:
        exit_code = 1
        if application.state["latest_error"] is None:
            application.state["latest_error"] = {
                "timestamp_ns": time.time_ns(),
                "loop_node": "main",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "recovery_action": "startup_or_runtime_aborted",
            }
    finally:
        try:
            application.stop()
        except Exception:
            exit_code = 1
    print(json.dumps(application.summary(), ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
