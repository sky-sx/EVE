from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

import eve.main as main_module
from eve.core.tnn import SourceRef, TNNDescriptor, load_tnn
from eve.input.buffer import InputBuffer
from eve.input.capture import Capture
from eve.main import EVEApplication, main
from eve.memory.memorizer import Memorizer


def test_buffer_one_second_window_capacity_concurrency_and_close():
    buffer = InputBuffer(retention_ns=1_000_000_000, max_samples_per_kind=8)
    now_ns = time.monotonic_ns()
    buffer.store("cursor", (0, 0), timestamp_ns=now_ns - 900_000_000)
    for index in range(12):
        buffer.store(
            "cursor",
            (index, index),
            timestamp_ns=now_ns - 100_000_000 + index,
        )

    assert len(buffer.get_state()["cursor"]) == 8
    assert buffer.get_latest_cursor().value == (11, 11)

    failures = []

    def producer():
        try:
            for index in range(100):
                buffer.store("screen", {"index": index})
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=producer)
    thread.start()
    while thread.is_alive():
        buffer.get_state()
    thread.join()
    assert failures == []
    assert buffer.count("screen") <= 8

    buffer.close()
    with pytest.raises(RuntimeError, match="closed"):
        buffer.store("cursor", (99, 99))


def test_async_memory_flush_numpy_and_incremental_catalog(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    memory.start_writer()
    started = time.perf_counter()
    memory_id = memory.enqueue(
        {"candidate": "a1"},
        "action_candidate",
        priority="critical",
    )
    enqueue_latency = time.perf_counter() - started
    assert memory_id is not None
    assert enqueue_latency < 0.05
    memory.flush()
    assert memory.read(memory_id) == {"candidate": "a1"}

    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image_id = memory.create(image, "screen_image")
    restored = memory.read(image_id)
    assert np.array_equal(restored, image)
    assert memory.get_unit(image_id).storage_path.endswith(".npy")
    records = [
        json.loads(line)
        for line in memory.catalog_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(record["op"] == "create" for record in records) == 2
    memory.stop_writer()


def test_memory_overflow_drops_low_before_critical(tmp_path, monkeypatch):
    errors = []
    memory = Memorizer(
        tmp_path / "memory",
        queue_capacity=2,
        writer_error_callback=errors.append,
    )
    release = threading.Event()
    original = memory._create_with_id

    def slow_create(*args):
        release.wait(1.0)
        return original(*args)

    monkeypatch.setattr(memory, "_create_with_id", slow_create)
    blocker = memory.enqueue({"n": 0}, "blocker", priority="normal")
    deadline = time.monotonic() + 1
    while not memory._writer_busy and time.monotonic() < deadline:
        time.sleep(0.001)
    memory.enqueue({"n": 1}, "snapshot", priority="low")
    memory.enqueue({"n": 2}, "snapshot", priority="low")
    critical = memory.enqueue({"n": 3}, "output_result", priority="critical")
    release.set()
    memory.flush()

    assert blocker is not None and critical is not None
    assert memory.read(critical) == {"n": 3}
    assert memory.writer_stats()["dropped"] == 1
    assert errors == []
    memory.stop_writer()


def test_profiles_use_mock_reject_control_and_complete_chain(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    monkeypatch.setattr(
        "eve.output.mouse._execute_real",
        lambda *args, **kwargs: pytest.fail("real mouse backend was called"),
    )
    monkeypatch.setattr(
        "eve.output.keyboard._execute_real",
        lambda *args, **kwargs: pytest.fail("real keyboard backend was called"),
    )
    monkeypatch.setattr(
        "eve.output.speak._execute_real",
        lambda *args, **kwargs: pytest.fail("real speak backend was called"),
    )
    assert main(["--profile", "control"]) == 2
    assert "control profile is not enabled" in capsys.readouterr().err
    assert main(
        [
            "--profile",
            "observe",
            "--duration",
            "0.1",
            "--tnn-id",
            "missing-critical-tnn",
            "--run-dir",
            str(tmp_path / "missing-tnn"),
        ]
    ) == 1

    buffer = InputBuffer()
    capture = Capture(
        buffer,
        screen_fps=30,
        cursor_hz=60,
        screen_reader=lambda: np.zeros((4, 4, 4), dtype=np.uint8),
        cursor_reader=lambda: (12, 34),
    )
    application = EVEApplication(
        profile="observe",
        run_dir=tmp_path,
        capture=capture,
        allow_mock_actions=False,
    )
    application.start()
    assert application.wait(0.25)
    application.stop()
    summary = application.summary()

    assert summary["profile"] == "observe"
    assert summary["real_output_calls"] == 0
    assert summary["safegate_blocked"] == 1
    assert summary["threads_stopped"]
    assert application.state.latest_output.blocked
    assert {
        "latest_input_summary",
        "latest_tnn_output",
        "latest_action_candidate",
        "latest_safegate_result",
        "latest_output_feedback",
    } <= set(application.state.blackboard)
    assert application.state.active_tnn == {"smoke_rule"}
    assert set(application.state.loaded_tnn) == {"smoke_rule"}


def test_smoke_cli_and_global_escape_have_clean_exit_codes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    assert main(
        [
            "--profile",
            "smoke",
            "--duration",
            "0.1",
            "--run-dir",
            str(tmp_path / "cli-smoke"),
        ]
    ) == 0

    pressed = iter((False, True))
    monkeypatch.setattr(
        main_module,
        "_global_escape_pressed",
        lambda: next(pressed, True),
    )
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path / "escape",
        allow_mock_actions=False,
    )
    application.start()
    started = time.monotonic()
    assert application.wait(1.0)
    application.stop()
    assert application.state.emergency_stopped
    assert application.exit_reason == "escape_key"
    assert time.monotonic() - started < 3.0
    assert application.summary()["threads_stopped"]


class BrokenRuntimeNode:
    descriptor = TNNDescriptor(
        tnn_id="critical-broken",
        inputs={"cursor": SourceRef("state:cursor")},
        outputs=("unused",),
        run_frequency_hz=60.0,
    )

    def run(self, inputs):
        raise RuntimeError("critical node failed")


def test_critical_core_error_propagates_and_all_threads_stop(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    buffer = InputBuffer()
    capture = Capture(
        buffer,
        screen_reader=lambda: {"synthetic": True},
        cursor_reader=lambda: (1, 2),
    )
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path,
        capture=capture,
    )
    application.state.active_tnn.add("critical-broken")
    load_tnn(application.state, BrokenRuntimeNode(), activate=False)
    application.start(load_smoke_node=False)

    assert not application.wait(1.0)
    application.stop()
    assert application.state.latest_error is not None
    assert application.state.latest_error.loop_node == "tnn:critical-broken"
    assert application.summary()["threads_stopped"]
    assert not any(
        thread.name.startswith("eve-")
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
    )


def test_memory_writer_failure_becomes_critical_runtime_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    buffer = InputBuffer()
    capture = Capture(
        buffer,
        screen_reader=lambda: {"synthetic": True},
        cursor_reader=lambda: (1, 2),
    )
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path,
        capture=capture,
        allow_mock_actions=False,
    )

    def fail_write(*args, **kwargs):
        raise OSError("memory device unavailable")

    monkeypatch.setattr(application.memory, "_create_with_id", fail_write)
    application.start()
    assert not application.wait(1.0)
    with pytest.raises(RuntimeError, match="shutdown failure"):
        application.stop()
    assert application.critical_failure
    assert application.state.latest_error.loop_node == "memory_writer"
    assert not application.capture.running
    assert not application.core.running
    assert not application.memory.writer_running
