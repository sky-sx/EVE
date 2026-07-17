"""Smoke-test the inhibition and teacher packages."""
import sys
sys.path.insert(0, "src")

# ---- inhibition ----
from eve.inhibition import PermissionManager, InhibitionGate, UserInterrupt, ActionAudit

pm = PermissionManager()
assert pm.check({"action": "move"}) == (True, "allowed")
assert pm.check({"action": "delete"}) == (False, "high-risk action 'delete' requires explicit grant")
pm.grant("delete")
assert pm.check({"action": "delete"}) == (True, "allowed")
assert pm.is_high_risk({"action": "execute"})
assert not pm.is_high_risk({"action": "move"})
assert pm.get_permissions() == {"move", "click", "type", "speak", "animate", "delete"}
pm.revoke("delete")
assert "delete" not in pm.get_permissions()
print("PermissionManager: OK")

gate = InhibitionGate()
allowed = gate.allow([{"action": "move"}, {"action": "delete"}, {"action": "click"}], pm)
assert len(allowed) == 2
assert gate.get_blocked_count() == 1
gate.block_all()
assert gate.is_blocked()
assert gate.allow([{"action": "move"}], pm) == []
gate.release()
assert not gate.is_blocked()
print("InhibitionGate: OK")

ui = UserInterrupt()
ui.signal("pause")
assert ui.is_paused()
ui.signal("resume")
assert not ui.is_paused()
assert ui.get_last_interrupt()["type"] == "resume"
ui.signal("stop")
assert ui.is_stopped()
assert ui.is_paused()
ui.clear()
assert ui.get_last_interrupt() is None
assert not ui.is_stopped()
assert not ui.is_paused()
print("UserInterrupt: OK")

aa = ActionAudit(capacity=10)
aa.log({"action": "move"}, True, "allowed")
aa.log({"action": "delete"}, False, "blocked")
s = aa.summary()
assert s["total"] == 2
assert s["allowed"] == 1
assert s["blocked"] == 1
assert s["by_type"]["move"] == 1
assert s["by_type"]["delete"] == 1
history = aa.get_history(10)
assert len(history) == 2
blocked = aa.get_blocked_actions()
assert len(blocked) == 1
# export test
import tempfile, os
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
    aa.export(tf.name)
    jsonl_path = tf.name
assert os.path.exists(jsonl_path)
os.unlink(jsonl_path)
print("ActionAudit: OK")

# ---- teacher ----
from eve.teacher import FastTeacher, SlowTeacher, RewardOracle, TeacherOrchestra

ft = FastTeacher()
state = {"slots": {"target": (0.5, 0.5), "cursor": (0.3, 0.3)}}
r = ft.label(state, {"action": "move", "dx": 0.1, "dy": 0.1})
assert r is not None and r["label"] == "approaching_target", r
r = ft.label(state, {"action": "noop"})
assert r is not None and r["label"] == "idle", r
# click off target
r = ft.label(state, {"action": "click"})
assert r is not None and r["label"] == "click_off_target", r
# at target
state_at = {"slots": {"target": (0.5, 0.5), "cursor": (0.51, 0.51)}}
r = ft.label(state_at, {"action": "click"})
assert r is not None and r["label"] == "click_on_target", r
print("FastTeacher: OK")

st = SlowTeacher(processing_delay_s=0.0)
state2 = {"slots": {"target": (0.5, 0.5), "cursor": (0.5, 0.5)}}
r = st.label(state2, {"action": "click"})
assert r["label"] == "click_on_target", r
assert "reasoning" in r
r = st.label(state2, {"action": "noop"})
assert r["label"] == "idle_at_target", r
r = st.label(state2, {"action": "unknown"})
assert r["label"] == "unrecognized", r
print("SlowTeacher: OK")

ro = RewardOracle()
s_before = {"slots": {"target": (0.5, 0.5), "cursor": (0.3, 0.3)}}
s_after = {"slots": {"target": (0.5, 0.5), "cursor": (0.4, 0.4)}}
reward = ro.compute(s_before, {"action": "move", "dx": 0.1, "dy": 0.1}, s_after)
assert reward > 0, reward  # moving closer = positive
components = ro.reward_components(s_before, {"action": "move", "dx": 0.1, "dy": 0.1}, s_after)
assert "proximity" in components
assert "energy_efficiency" in components
assert "collision" in components
assert "idle" in components
# collision test
s_boundary = {"slots": {"target": (0.5, 0.5), "cursor": (0.0, 0.5)}}
comp_coll = ro.reward_components(s_before, {"action": "move", "dx": -0.3, "dy": 0.0}, s_boundary)
assert comp_coll["collision"] < 0, comp_coll
# idle test
comp_idle = ro.reward_components(s_before, {"action": "noop"}, s_before)
assert comp_idle["idle"] < 0, comp_idle
print("RewardOracle: OK")

to = TeacherOrchestra(fast_teacher=ft, slow_teacher=st, reward_oracle=ro)
assert to.request_label(state, {"action": "move", "dx": 0.1, "dy": 0.1})
assert to.get_pending_count() == 1
results = to.process_queue()
assert len(results) == 1
assert "reward" in results[0]
assert results[0]["action"]["action"] == "move"
# fast teacher can handle this, so fast_teacher result should appear
assert results[0]["label"] in ("approaching_target", "moving_away")
to.shutdown()
assert to.get_pending_count() == 0
print("TeacherOrchestra: OK")

print()
print("ALL SMOKE TESTS PASSED")
