from __future__ import annotations

import time

import pytest

from eve.input.buffer import InputBuffer
from eve.input.capture import Capture


def test_buffer_has_monotonic_latest_range_and_recent_window():
    buffer = InputBuffer(retention_ns=1_000)
    buffer.store("cursor", (1, 1), timestamp_ns=100)
    buffer.store("cursor", (2, 2), timestamp_ns=500)
    buffer.store("cursor", (3, 3), timestamp_ns=1_200)

    assert buffer.latest("cursor").value == (3, 3)
    assert [sample.value for sample in buffer.range("cursor", 400, 1_300)] == [
        (2, 2),
        (3, 3),
    ]
    with pytest.raises(ValueError):
        buffer.store("cursor", (0, 0), timestamp_ns=1_100)


def test_capture_failure_is_visible_and_thread_stops():
    errors = []

    def broken_screen():
        raise OSError("capture unavailable")

    capture = Capture(
        InputBuffer(),
        screen_reader=broken_screen,
        cursor_reader=lambda: (0, 0),
        error_callback=errors.append,
    )
    capture.start()
    deadline = time.monotonic() + 1
    while capture.running and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    assert capture.last_error is not None
    assert capture.last_error.exception_type == "OSError"
    assert capture.last_error.recovery_action == "capture_stopped_no_output"
    assert len(errors) == 1
