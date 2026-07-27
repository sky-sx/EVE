"""Speech output boundary; real TTS is intentionally not implemented."""
from __future__ import annotations

import time
from typing import Any


def execute(
    action_id: str, payload: dict[str, Any], mode: str
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    common = {
        "action_id": action_id,
        "kind": "speak",
        "mode": mode,
        "started_at_ns": started_ns,
        "finished_at_ns": time.monotonic_ns(),
        "executed": False,
        "payload": dict(payload),
    }
    if mode == "disabled":
        return {
            **common,
            "simulated": False,
            "blocked": True,
            "reason": "output_disabled",
        }
    if mode == "mock":
        return {
            **common,
            "simulated": True,
            "blocked": False,
            "reason": "mock_ok",
        }
    if mode != "real":
        raise ValueError(f"unknown output mode: {mode}")
    return {
        **common,
        "simulated": False,
        "blocked": True,
        "reason": "real_unimplemented_no_tts_backend",
    }
