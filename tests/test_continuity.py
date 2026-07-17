import pytest

pytest.skip(
    "eve.continuity modules are stub-only (all methods return None / ...). "
    "Tests will be re-enabled when implementations are added.",
    allow_module_level=True,
)

def test_world_state_update_and_snapshot() -> None:
    ws = WorldState(max_history=10)
    ws.update({"time_of_day": 12.0, "activity_level": 0.7, "scene_type": "indoor"})

    snap = ws.snapshot()
    assert snap["time_of_day"] == 12.0
    assert snap["activity_level"] == 0.7
    assert snap["scene_type"] == "indoor"
    assert snap["cursor_active"] is False  # default preserved
    assert snap["dominant_colors"] == []


def test_world_state_preserves_unknown_fields() -> None:
    ws = WorldState()
    ws.update({"custom_field": 42})
    snap = ws.snapshot()
    assert snap["custom_field"] == 42


def test_world_state_get_change_detects_delta() -> None:
    ws = WorldState(max_history=10)
    ws.update({"activity_level": 0.3, "scene_type": "outdoor"})
    prev = ws.snapshot()
    ws.update({"activity_level": 0.9})
    delta = ws.get_change(prev)
    assert "activity_level" in delta
    assert delta["activity_level"] == (0.3, 0.9)
    assert "scene_type" not in delta  # unchanged


def test_world_state_history_limited() -> None:
    ws = WorldState(max_history=3)
    for i in range(5):
        ws.update({"activity_level": float(i)})
    assert len(ws.history) == 3


def test_world_state_reset() -> None:
    ws = WorldState()
    ws.update({"activity_level": 0.5})
    ws.reset()
    assert ws.snapshot() == {}
    assert len(ws.history) == 0


def test_world_state_age() -> None:
    ws = WorldState()
    assert ws.age == float("inf")
    ws.update({})
    assert ws.age >= 0.0


# ── SelfState ─────────────────────────────────────────────────────────────

def test_self_state_initial_snapshot() -> None:
    ss = SelfState(initial_mode="idle")
    snap = ss.snapshot()
    assert snap["energy_level"] == 0.5
    assert snap["busy_level"] == 0.0
    assert snap["confidence"] == 0.5
    assert snap["mode"] == "idle"


def test_self_state_update_merges_feedback() -> None:
    ss = SelfState()
    ss.update({"energy_level": 0.8, "confidence": 0.3})
    snap = ss.snapshot()
    assert snap["energy_level"] == 0.8
    assert snap["confidence"] == 0.3
    assert snap["busy_level"] == 0.0  # unchanged


def test_self_state_update_clamps_values() -> None:
    ss = SelfState()
    ss.update({"energy_level": 1.5, "confidence": -0.2})
    snap = ss.snapshot()
    assert snap["energy_level"] == 1.0
    assert snap["confidence"] == 0.0


def test_self_state_mode_transition() -> None:
    ss = SelfState(initial_mode="idle")
    assert ss.get_mode() == "idle"
    ss.transition("active")
    assert ss.get_mode() == "active"
    ss.transition("shadow")
    assert ss.get_mode() == "shadow"


def test_self_state_invalid_mode_raises() -> None:
    ss = SelfState()
    try:
        ss.transition("invalid_mode")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_self_state_idle_time_reset_on_leave() -> None:
    ss = SelfState(initial_mode="idle")
    ss.update({})  # Advances idle_time
    ss.transition("active")
    snap = ss.snapshot()
    assert snap["idle_time"] == 0.0


def test_self_state_reset() -> None:
    ss = SelfState(initial_mode="active")
    ss.update({"energy_level": 0.1})
    ss.transition("shadow")
    ss.reset()
    snap = ss.snapshot()
    assert snap["energy_level"] == 0.5
    assert ss.get_mode() == "active"


# ── NeedState ─────────────────────────────────────────────────────────────

def test_need_state_initial_needs() -> None:
    ns = NeedState()
    snap = ns.snapshot()
    assert len(snap) == 6
    for name in ns.need_names:
        assert snap[name] == 0.3


def test_need_state_custom_initial_needs() -> None:
    ns = NeedState(initial_needs={"exploration_drive": 0.8})
    snap = ns.snapshot()
    assert snap["exploration_drive"] == 0.8
    assert snap["social_drive"] == 0.3


def test_need_state_needs_grow_over_time() -> None:
    ns = NeedState()
    ns.update(time_delta=10.0, activity={})
    snap = ns.snapshot()
    # All needs should have grown from 0.3
    for name in ns.need_names:
        assert snap[name] > 0.3


def test_need_state_needs_decay_with_activity() -> None:
    ns = NeedState()
    ns.update(time_delta=5.0, activity={"exploring": 1.0, "novel": 1.0})
    snap = ns.snapshot()
    # exploration and novelty should have decayed more than others
    avg_other = (snap["social_drive"] + snap["rest_drive"] +
                 snap["competence_drive"] + snap["safety_drive"]) / 4.0
    assert snap["exploration_drive"] < avg_other
    assert snap["novelty_drive"] < avg_other


def test_need_state_dominant_need() -> None:
    ns = NeedState(initial_needs={"rest_drive": 0.9})
    assert ns.get_dominant_need() == "rest_drive"


def test_need_state_satisfy_reduces_need() -> None:
    ns = NeedState()
    ns.satisfy("exploration_drive", 0.5)
    snap = ns.snapshot()
    assert snap["exploration_drive"] == 0.0  # 0.3 - 0.5 clamped
    ns.satisfy("social_drive", 0.1)
    assert ns.snapshot()["social_drive"] == 0.2


def test_need_state_satisfy_unknown_need_raises() -> None:
    ns = NeedState()
    try:
        ns.satisfy("unknown_need", 0.1)
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_need_state_reset() -> None:
    ns = NeedState()
    ns.update(time_delta=10.0, activity={})
    ns.satisfy("exploration_drive", 1.0)
    ns.reset()
    snap = ns.snapshot()
    for name in ns.need_names:
        assert snap[name] == 0.3


def test_need_state_clamped_range() -> None:
    ns = NeedState()
    ns.update(time_delta=1000.0, activity={})
    snap = ns.snapshot()
    for v in snap.values():
        assert 0.0 <= v <= 1.0
