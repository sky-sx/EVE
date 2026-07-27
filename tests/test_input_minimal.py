from __future__ import annotations

import os
import time

import pytest

from eve.input.buffer import InputBuffer


def test_buffer_has_monotonic_latest_range_and_recent_window():
    buffer = InputBuffer(retention_ns=1_000)
    buffer.store("cursor", (1, 1), timestamp_ns=100)
    buffer.store("cursor", (2, 2), timestamp_ns=500)
    buffer.store("cursor", (3, 3), timestamp_ns=1_200)

    assert buffer.latest("cursor").value == (3, 3)
    assert [sample.value for sample in buffer.range("cursor", 400, 1_300)] == [
        (2, 2), (3, 3)
    ]
    with pytest.raises(ValueError):
        buffer.store("cursor", (0, 0), timestamp_ns=1_100)


def test_buffer_owns_independent_capture_process_and_stops_it():
    buffer = InputBuffer(profile="smoke")
    buffer.start_capture()
    child_pid = buffer.capture_process_id
    deadline = time.monotonic() + 2
    while buffer.get_latest_screen() is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert child_pid is not None and child_pid != os.getpid()
    assert buffer.capture_running
    assert buffer.get_state()["latest"]["screen"] is not None
    buffer.close()
    assert not buffer.capture_running


def test_capture_failure_is_propagated_through_buffer():
    buffer = InputBuffer(
        profile="smoke",
        capture_options={"screen_mode": "error"},
    )
    with pytest.raises(RuntimeError, match="capture process initialization failed"):
        buffer.start_capture()
    assert buffer.capture_error is not None
    assert buffer.capture_error["exception_type"] == "OSError"
    assert not buffer.capture_running
    buffer.close()
