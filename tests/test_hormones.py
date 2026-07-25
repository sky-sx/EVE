"""Tests for EVE Hormone System: HormoneManager, HormoneLevels, HormoneEvent."""

import pytest
from eve.core.hormones import HormoneManager, HormoneLevels, HormoneEvent


def test_initial_levels() -> None:
    hm = HormoneManager()
    assert hm.levels.dopamine == 0.5
    assert hm.levels.serotonin == 0.5
    assert hm.levels.norepinephrine == 0.5
    assert hm.levels.oxytocin == 0.5
    assert hm.levels.cortisol == 0.5
    assert hm.levels.acetylcholine == 0.5


def test_apply_success() -> None:
    hm = HormoneManager()
    hm.apply_event("success")
    assert hm.levels.dopamine > 0.5
    assert hm.levels.serotonin > 0.5
    # cortisol should NOT increase on success
    assert hm.levels.cortisol == 0.5


def test_apply_failure() -> None:
    hm = HormoneManager()
    hm.apply_event("failure")
    assert hm.levels.cortisol > 0.5
    assert hm.levels.dopamine < 0.5
    assert hm.levels.serotonin < 0.5


def test_apply_user_praise() -> None:
    hm = HormoneManager()
    hm.apply_event("user_praise")
    assert hm.levels.dopamine > 0.5
    assert hm.levels.oxytocin > 0.5
    assert hm.levels.serotonin > 0.5


def test_update_cycle_recovery() -> None:
    hm = HormoneManager()
    # Push dopamine way up
    hm.levels.dopamine = 0.9
    hm.update_cycle()
    # Should recover toward 0.5 baseline
    assert hm.levels.dopamine < 0.9

    # Push cortisol way down
    hm.levels.cortisol = 0.1
    hm.update_cycle()
    assert hm.levels.cortisol > 0.1


def test_llm_interval_computation() -> None:
    hm = HormoneManager()
    interval = hm.compute_llm_interval(min_s=10.0, max_s=20.0)
    assert 10.0 <= interval <= 20.0
    assert isinstance(interval, float)


def test_high_stress_shorter_interval() -> None:
    hm = HormoneManager()
    # High stress scenario
    hm.levels.cortisol = 0.9
    hm.levels.norepinephrine = 0.9
    hm.levels.dopamine = 0.9
    hm.levels.serotonin = 0.1
    interval_high = hm.compute_llm_interval(min_s=10.0, max_s=20.0)

    # Low stress scenario
    hm2 = HormoneManager()
    hm2.levels.cortisol = 0.1
    hm2.levels.norepinephrine = 0.1
    hm2.levels.dopamine = 0.1
    hm2.levels.serotonin = 0.9
    interval_low = hm2.compute_llm_interval(min_s=10.0, max_s=20.0)

    assert interval_high < interval_low


def test_low_stress_longer_interval() -> None:
    hm = HormoneManager()
    hm.levels.serotonin = 0.9
    hm.levels.cortisol = 0.1
    hm.levels.norepinephrine = 0.1
    interval = hm.compute_llm_interval(min_s=10.0, max_s=20.0)
    assert interval > 15.0  # should be closer to max


def test_tendencies_derivation() -> None:
    hm = HormoneManager()
    tendencies = hm.get_tendencies()
    expected_keys = ["explore", "exploit", "pause", "sleep", "active_output", "think_more", "train"]
    for key in expected_keys:
        assert key in tendencies, f"missing tendency key: {key}"
    for key in expected_keys:
        assert 0.0 <= tendencies[key] <= 1.0


def test_snapshot_restore() -> None:
    hm = HormoneManager()
    hm.apply_event("success")
    hm.apply_event("user_praise")
    snapshot = hm.save_snapshot()

    hm2 = HormoneManager()
    assert hm2.levels.dopamine == 0.5
    hm2.restore_snapshot(snapshot)
    assert hm2.levels.dopamine == hm.levels.dopamine
    assert hm2.levels.oxytocin == hm.levels.oxytocin
    assert hm2.levels.serotonin == hm.levels.serotonin


def test_multiple_events_accumulate() -> None:
    hm = HormoneManager()
    hm.apply_event("success")
    hm.apply_event("success")
    hm.apply_event("success")
    # Each success adds 0.1 dopamine, starting from 0.5
    # Use approx due to floating point
    assert hm.levels.dopamine == pytest.approx(0.8)


def test_levels_clamped() -> None:
    hm = HormoneManager()
    # Apply many success events to push dopamine beyond 1.0
    for _ in range(20):
        hm.apply_event("success")
    assert hm.levels.dopamine <= 1.0
    assert hm.levels.dopamine >= 0.0

    # Apply many failures to push cortisol up and dopamine down
    for _ in range(20):
        hm.apply_event("failure")
    assert hm.levels.cortisol <= 1.0
    assert hm.levels.dopamine >= 0.0
