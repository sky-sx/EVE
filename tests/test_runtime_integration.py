from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

import eve.main as main_module
from eve.core.loop import register_runtime_tnn
from eve.input.buffer import InputBuffer
from eve.main import EVEApplication, main
from eve.memory.memorizer import Memorizer


def synthetic_buffer(profile="smoke"):
    return InputBuffer(
        profile=profile,
        capture_options={"screen_mode": "synthetic", "cursor_mode": "synthetic"},
    )


def test_buffer_one_second_window_capacity_concurrency_and_close():
    buffer = InputBuffer(retention_ns=1_000_000_000, max_samples_per_kind=8)
    now_ns = time.monotonic_ns()
    buffer.store("cursor", (0, 0), timestamp_ns=now_ns - 900_000_000)
    for index in range(12):
        buffer.store("cursor", (index, index), timestamp_ns=now_ns - 100_000_000 + index)
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
        {"candidate": "a1"}, "action_candidate", priority="critical"
    )
    assert memory_id is not None
    assert time.perf_counter() - started < 0.05
    memory.flush()
    assert memory.read(memory_id) == {"candidate": "a1"}

    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image_id = memory.create(image, "screen_image")
    assert np.array_equal(memory.read(image_id), image)
    assert memory.get_record(image_id).storage_path.endswith(".npy")
    records = [
        json.loads(line)
        for line in memory.catalog_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(record["op"] == "create" for record in records) == 2
    memory.stop_writer()


def test_memory_overflow_drops_low_before_critical(tmp_path, monkeypatch):
    errors = []
    memory = Memorizer(
        tmp_path / "memory", queue_capacity=2, writer_error_callback=errors.append
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


def test_profiles_are_mock_and_observe_completes_full_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    monkeypatch.setattr(
        "eve.output.mouse._execute_real",
        lambda *args, **kwargs: pytest.fail("real mouse backend was called"),
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert main(
        [
            "--profile", "control", "--duration", "0.05",
            "--run-dir", str(tmp_path / "control-gui"),
        ]
    ) == 0
    assert "control profile is not enabled" not in capsys.readouterr().err

    application = EVEApplication(
        profile="observe",
        run_dir=tmp_path,
        memory_dir=tmp_path / "memory-observe",
        input_buffer=synthetic_buffer("observe"),
        allow_mock_actions=False,
    )
    application.start(load_smoke_node=True)
    assert application.wait(0.25)
    active_while_running = set(application.state["active_tnn"])
    loaded_while_running = set(application.state["loaded_tnn"])
    blackboard = set(application.state["blackboard"])
    application.stop()
    summary = application.summary()

    assert summary["profile"] == "observe"
    assert summary["real_output_calls"] == 0
    assert summary["actions_blocked"] == 1
    assert summary["threads_stopped"] and summary["capture_process_stopped"]
    assert application.state["latest_output"]["blocked"]
    assert {
        "latest_input_summary",
        "latest_tnn_output",
        "latest_action_candidate",
        "latest_permission_result",
        "latest_output_feedback",
    } <= blackboard
    assert active_while_running == {"smoke_rule"}
    assert loaded_while_running == {"smoke_rule"}
    assert application.state["loaded_tnn"] == {}


def test_smoke_cli_escape_and_capture_error_have_clean_exit_codes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    assert main(
        [
            "--profile", "smoke", "--duration", "0.1",
            "--run-dir", str(tmp_path / "cli-smoke"),
            "--memory-dir", str(tmp_path / "memory-cli-smoke"),
        ]
    ) == 0

    pressed = iter((False, True))
    monkeypatch.setattr(
        main_module, "_global_escape_pressed", lambda: next(pressed, True)
    )
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path / "escape",
        memory_dir=tmp_path / "memory-escape",
        input_buffer=synthetic_buffer(),
    )
    application.start()
    started = time.monotonic()
    assert application.wait(1.0)
    application.stop()
    assert application.state["emergency_stop"]
    assert application.exit_reason == "escape_key"
    assert time.monotonic() - started < 3.0
    assert application.summary()["threads_stopped"]

    failing = InputBuffer(
        profile="smoke", capture_options={"screen_mode": "error"}
    )
    monkeypatch.setattr(main_module, "InputBuffer", lambda **_kwargs: failing)
    assert main(
        [
            "--profile", "smoke", "--duration", "0.1",
            "--run-dir", str(tmp_path / "capture-error"),
        ]
    ) == 1
    assert not failing.capture_running


def test_tnn_error_isolated_and_all_workers_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path,
        memory_dir=tmp_path / "memory-tnn-error",
        input_buffer=synthetic_buffer(),
    )

    def broken(_inputs):
        raise RuntimeError("critical node failed")

    register_runtime_tnn(
        application.state,
        "critical-broken",
        broken,
        inputs={"cursor": "state:cursor"},
        outputs=("unused",),
        run_frequency_hz=60,
    )
    application.core.smoke_node = False
    application.start(load_smoke_node=False)

    assert application.wait(0.2)
    assert application.core.running
    assert application.state["tnn_status"]["critical-broken"] == "failed"
    assert "critical-broken" not in application.state["active_tnn"]
    application.stop()
    assert application.state["latest_error"]["loop_node"] == "tnn:critical-broken"
    assert application.summary()["threads_stopped"]
    assert not any(
        thread.name.startswith("eve-")
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
    )


def test_memory_writer_failure_becomes_critical_runtime_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_global_escape_pressed", lambda: False)
    application = EVEApplication(
        profile="smoke",
        run_dir=tmp_path,
        memory_dir=tmp_path / "memory-writer-error",
        input_buffer=synthetic_buffer(),
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("memory device unavailable")

    monkeypatch.setattr(application.memory, "_create_with_id", fail_write)
    application.start()
    application.memory.enqueue({"trigger": True}, "writer_failure", priority="critical")
    assert not application.wait(1.0)
    with pytest.raises(RuntimeError, match="shutdown failure"):
        application.stop()
    assert application.critical_failure
    assert application.state["latest_error"]["loop_node"] == "memory_writer"
    assert not application.buffer.capture_running
    assert not application.core.running
    assert not application.memory.writer_running
