"""
Tests for runtime state system (eve/state.py, eve/core/runtime_state.py).
"""
import time
from pathlib import Path

import pytest

from eve.config import EVEConfig
from eve.core.runtime_state import RuntimeStateManager
from eve.state import (
    Blackboard,
    MyselfState,
    RuntimeState,
    TimedEntry,
    WorldState,
)


# ── Blackboard write/read ────────────────────────────────

def test_blackboard_write_read():
    """Write TimedEntry, read by kind."""
    bb = Blackboard()
    now = time.monotonic_ns()
    entry = TimedEntry(
        entry_id="e1",
        kind="test_results",
        producer="producer_a",
        produced_at_ns=now,
        valid_until_ns=now + 10_000_000_000,
        payload={"score": 0.95},
    )
    bb.write(entry)
    results = bb.read("test_results")
    assert len(results) == 1
    assert results[0].entry_id == "e1"
    assert results[0].payload == {"score": 0.95}


def test_blackboard_read_by_producer():
    """read with producer filter returns only matching entries."""
    bb = Blackboard()
    now = time.monotonic_ns()
    bb.write(TimedEntry("e1", "test_results", "a", now, 0, {"v": 1}))
    bb.write(TimedEntry("e2", "test_results", "b", now, 0, {"v": 2}))
    bb.write(TimedEntry("e3", "test_results", "a", now, 0, {"v": 3}))

    results_a = bb.read("test_results", producer="a")
    assert len(results_a) == 2
    assert all(r.producer == "a" for r in results_a)

    results_b = bb.read("test_results", producer="b")
    assert len(results_b) == 1


# ── Blackboard TTL expiry ───────────────────────────────

def test_blackboard_ttl_expiry():
    """Entry with expired valid_until_ns not returned by read."""
    bb = Blackboard()
    past = time.monotonic_ns() - 10_000_000_000
    bb.write(TimedEntry("expired", "test", "p", past, past + 1, "v"))

    results = bb.read("test")
    assert len(results) == 0


def test_blackboard_valid_until_zero_never_expires():
    """valid_until_ns=0 means never expire."""
    bb = Blackboard()
    bb.write(TimedEntry("e1", "test", "p", time.monotonic_ns(), 0, "v"))
    results = bb.read("test")
    assert len(results) == 1


# ── Blackboard clear_expired ─────────────────────────────

def test_blackboard_clear_expired():
    """clear_expired removes only expired entries."""
    bb = Blackboard()
    now = time.monotonic_ns()

    # Non-expired
    bb.write(TimedEntry("e1", "results", "p", now, now + 60_000_000_000, "keep"))
    bb.write(TimedEntry("e2", "results", "p", now, 0, "keep_forever"))
    # Expired
    bb.write(TimedEntry("e3", "results", "p", now - 100, now - 50, "expired"))
    bb.write(TimedEntry("e4", "other", "p", now - 100, now - 50, "expired_other"))

    bb.clear_expired()

    # results kind still exists (non-expired entries)
    remaining = bb.read("results")
    assert len(remaining) == 2
    ids = [r.entry_id for r in remaining]
    assert "e1" in ids
    assert "e2" in ids
    assert "e3" not in ids

    # other kind should be removed entirely (all entries expired)
    assert bb.read("other") == []


# ── Blackboard latest ────────────────────────────────────

def test_blackboard_latest():
    """latest returns the most recent non-expired entry."""
    bb = Blackboard()
    now = time.monotonic_ns()
    bb.write(TimedEntry("e1", "test", "p", now, 0, "first"))
    bb.write(TimedEntry("e2", "test", "p", now + 1, 0, "second"))

    latest = bb.latest("test")
    assert latest is not None
    assert latest.payload == "second"


def test_blackboard_latest_empty():
    """latest on empty kind returns None."""
    bb = Blackboard()
    assert bb.latest("nonexistent") is None


# ── WorldState fields ───────────────────────────────────

def test_world_state_fields():
    """WorldState has all required fields with defaults."""
    ws = WorldState()
    assert ws.scene == ""
    assert ws.sub_scene == ""
    assert ws.active_window == ""
    assert ws.visible_objects == []
    assert ws.detected_text == ""
    assert ws.visual_results == {}
    assert ws.uncertainty == ""
    assert ws.updated_at_ns == 0


def test_world_state_field_assignment():
    """WorldState fields can be assigned and read back."""
    ws = WorldState(scene="desktop", sub_scene="browser")
    ws.active_window = "Chrome"
    ws.visible_objects = ["icon1", "icon2"]
    ws.detected_text = "some text"
    ws.uncertainty = "low"
    ws.updated_at_ns = 123456789

    assert ws.scene == "desktop"
    assert ws.sub_scene == "browser"
    assert ws.active_window == "Chrome"
    assert ws.visible_objects == ["icon1", "icon2"]
    assert ws.detected_text == "some text"
    assert ws.uncertainty == "low"
    assert ws.updated_at_ns == 123456789


# ── MyselfState fields ──────────────────────────────────

def test_myself_state_fields():
    """MyselfState has all required fields with defaults."""
    ms = MyselfState()
    assert ms.what_im_thinking == ""
    assert ms.current_task == ""
    assert ms.task_progress == ""
    assert ms.loaded_tnn == []
    assert ms.available_tnn_summary == []
    assert ms.resource_status == {}
    assert ms.hormone_levels == {}
    assert ms.tendencies == {}
    assert ms.control_summary == ""
    assert ms.updated_at_ns == 0


def test_myself_state_field_assignment():
    """MyselfState fields can be assigned and read back."""
    ms = MyselfState(
        what_im_thinking="what to do next",
        current_task="browse email",
        task_progress="50%",
    )
    ms.loaded_tnn = ["tnn_1", "tnn_2"]
    ms.hormone_levels = {"cortisol": 0.1}
    ms.tendencies = {"explore": 0.8}
    ms.updated_at_ns = 999

    assert ms.what_im_thinking == "what to do next"
    assert ms.current_task == "browse email"
    assert ms.task_progress == "50%"
    assert ms.loaded_tnn == ["tnn_1", "tnn_2"]
    assert ms.hormone_levels == {"cortisol": 0.1}
    assert ms.tendencies == {"explore": 0.8}
    assert ms.updated_at_ns == 999


# ── RuntimeStateManager save/load snapshot ───────────────

def test_save_load_snapshot(tmp_path):
    """save_snapshot + load_snapshot roundtrip."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)

    # Set up world state
    mgr.world.scene = "desktop"
    mgr.world.sub_scene = "editor"
    mgr.world.active_window = "VSCode"
    mgr.world.visible_objects = ["file.py", "test.py"]
    mgr.world.detected_text = "def foo"
    mgr.world.uncertainty = "low"

    # Set up myself state
    mgr.myself.what_im_thinking = "need to debug"
    mgr.myself.current_task = "fix bug"
    mgr.myself.task_progress = "30%"
    mgr.myself.loaded_tnn = ["tnn_a"]
    mgr.myself.control_summary = "active"
    mgr.myself.resource_status = {"cpu": 50}
    mgr.myself.hormone_levels = {"cortisol": 0.2}

    # Set up blackboard
    now = time.monotonic_ns()
    mgr.blackboard.write(TimedEntry("bb1", "test", "p", now, 0, {"data": 1}))

    # Save
    snap_path = tmp_path / "snap"
    mgr.save_snapshot(snap_path)
    assert (snap_path / "world.md").exists()
    assert (snap_path / "self.md").exists()
    assert (snap_path / "blackboard.md").exists()

    # Create new manager and load
    mgr2 = RuntimeStateManager(config)
    success = mgr2.load_snapshot(snap_path)
    assert success is True

    # Verify world restored
    assert mgr2.world.scene == "desktop"
    assert mgr2.world.sub_scene == "editor"
    assert mgr2.world.active_window == "VSCode"
    assert mgr2.world.detected_text == "def foo"
    assert mgr2.world.uncertainty == "low"

    # Verify myself restored
    assert mgr2.myself.what_im_thinking == "need to debug"
    assert mgr2.myself.current_task == "fix bug"
    assert mgr2.myself.task_progress == "30%"
    assert mgr2.myself.control_summary == "active"


def test_load_snapshot_nonexistent():
    """load_snapshot on nonexistent path returns False."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)
    assert mgr.load_snapshot(Path("/nonexistent/path")) is False


def test_load_snapshot_no_world_md(tmp_path):
    """load_snapshot without world.md returns False."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)
    assert mgr.load_snapshot(tmp_path) is False


# ── Update from LLM ─────────────────────────────────────

def test_update_from_llm_world():
    """Structured dict applied to WorldState."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)

    llm_output = {
        "scene": "game",
        "sub_scene": "level_3",
        "active_window": "Game.exe",
        "visible_objects": ["enemy", "coin", "door"],
        "detected_text": "HP: 100",
        "uncertainty": "medium",
        "visual_results": {"enemy_count": 3},
    }
    mgr.update_from_llm_world(llm_output)

    assert mgr.world.scene == "game"
    assert mgr.world.sub_scene == "level_3"
    assert mgr.world.active_window == "Game.exe"
    assert mgr.world.visible_objects == ["enemy", "coin", "door"]
    assert mgr.world.detected_text == "HP: 100"
    assert mgr.world.uncertainty == "medium"
    assert mgr.world.visual_results == {"enemy_count": 3}
    assert mgr.world.updated_at_ns > 0


def test_update_from_llm_world_partial():
    """Partial LLM output only overwrites specified fields."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)
    mgr.world.scene = "original_scene"
    mgr.world.uncertainty = "original_uncertainty"

    mgr.update_from_llm_world({"scene": "new_scene"})

    assert mgr.world.scene == "new_scene"
    assert mgr.world.uncertainty == "original_uncertainty"


def test_update_from_llm_myself():
    """Structured dict applied to MyselfState."""
    config = EVEConfig()
    mgr = RuntimeStateManager(config)

    llm_output = {
        "what_im_thinking": "time to rest",
        "current_task": "waiting",
        "task_progress": "complete",
        "loaded_tnn": ["tnn_x", "tnn_y"],
        "hormone_levels": {"dopamine": 0.7},
        "tendencies": {"rest": 0.9},
        "control_summary": "idle",
    }
    mgr.update_from_llm_myself(llm_output)

    assert mgr.myself.what_im_thinking == "time to rest"
    assert mgr.myself.current_task == "waiting"
    assert mgr.myself.task_progress == "complete"
    assert mgr.myself.loaded_tnn == ["tnn_x", "tnn_y"]
    assert mgr.myself.hormone_levels == {"dopamine": 0.7}
    assert mgr.myself.tendencies == {"rest": 0.9}
    assert mgr.myself.control_summary == "idle"
    assert mgr.myself.updated_at_ns > 0


# ── RuntimeState fields ─────────────────────────────────

def test_runtime_state_defaults():
    """RuntimeState has correct defaults."""
    s = RuntimeState()
    assert s.cold_started is False
    assert s.emergency_stopped is False
    assert s.output_mode.value == "disabled"
    assert s.mouse_allowed is False
    assert s.keyboard_allowed is False
    assert s.speak_allowed is False
    assert s.blocked_until_ns == 0
    assert s.eve_expected_events == set()
    assert s.pending_action is None
    assert s.latest_output is None
    assert s.human_activity_detected_at_ns == 0
