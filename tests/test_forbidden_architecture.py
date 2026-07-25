"""
Architecture constraint tests for the EVE project.
Validates that the codebase structure follows the architectural rules.
"""
import re
from pathlib import Path

from eve.memory.memorizer import Memorizer


# ── Directory scanning helpers ───────────────────────────

def _eve_py_files() -> list[Path]:
    """Return all .py files under eve/ excluding yolo26 vendored code."""
    project_root = Path(__file__).parent.parent
    eve_dir = project_root / "eve"
    files = []
    for f in eve_dir.rglob("*.py"):
        rel = str(f.relative_to(eve_dir))
        # Exclude vendored yolo26 code
        if "yolo26" in rel:
            continue
        files.append(f)
    return files


def _eve_filenames() -> list[str]:
    """Return filenames (stem) of eve/ .py files."""
    return sorted(set(p.stem for p in _eve_py_files()))


# ── No router/planner/orchestrator/etc. names ────────────

def test_no_router_planner_names():
    """Scan eve/ filenames: no file should contain 'router', 'planner',
    'orchestrator', 'coordinator', 'engine', or 'registry' as a word."""
    forbidden = ["router", "planner", "orchestrator", "coordinator", "engine", "registry"]

    filenames = _eve_filenames()
    violations = []
    for name in filenames:
        name_lower = name.lower()
        for word in forbidden:
            # Check if the word appears as a distinct word-part
            # We split on underscores and check component equality
            parts = name_lower.split("_")
            if word in parts:
                violations.append(f"{name} contains '{word}'")
                break
            # Also check if the word appears as a sub-component
            # e.g., "routing_engine" should flag
            for part in parts:
                if part == word:
                    violations.append(f"{name} contains '{word}'")
                    break

    assert violations == [], (
        f"Found {len(violations)} filename(s) with forbidden names:\n"
        + "\n".join(violations)
    )


# ── No agent module names ────────────────────────────────

def test_no_agent_module_names():
    """No file named with 'agent' pattern (agent, agents, agentic, etc.)."""
    filenames = _eve_filenames()
    violations = [name for name in filenames if "agent" in name.lower()]
    assert violations == [], (
        f"Found files with 'agent' in name: {violations}"
    )


# ── MemoryUnit minimal ──────────────────────────────────

def test_memory_unit_minimal():
    """MemoryUnit (if it exists as a class) should NOT have
    reward, importance, confidence, or decision fields.
    Currently there is no MemoryUnit class in the codebase --
    the architecture defines MemoryUnit = MemoryID + Payload."""
    import eve.memory as mem

    forbidden_fields = {"reward", "importance", "confidence", "decision", "success", "failure"}

    # Check if there's a class named MemoryUnit anywhere in eve.memory
    memory_unit_cls = getattr(mem, "MemoryUnit", None)
    if memory_unit_cls is not None:
        # If it exists, verify it doesn't have forbidden fields
        import dataclasses
        if dataclasses.is_dataclass(memory_unit_cls):
            field_names = {f.name for f in dataclasses.fields(memory_unit_cls)}
        else:
            field_names = set(vars(memory_unit_cls).keys())
        overlap = field_names & forbidden_fields
        assert overlap == set(), (
            f"MemoryUnit has forbidden fields: {overlap}"
        )
    # If no MemoryUnit class, that's fine -- the architecture says
    # MemoryUnit = MemoryID + Payload, not a standalone class.


def test_memory_catalog_no_decision_fields():
    """CatalogEntry should not have reward/importance/confidence/decision."""
    from eve.memory.catalog import CatalogEntry
    import dataclasses

    forbidden = {"reward", "importance", "confidence", "decision"}
    fields = {f.name for f in dataclasses.fields(CatalogEntry)}
    overlap = fields & forbidden
    assert overlap == set(), f"CatalogEntry has forbidden fields: {overlap}"


# ── STM/MTM are ID lists, not data stores ────────────────

def test_stm_mtm_no_payload_copy():
    """Memorizer.stm and .mtm are list[str] (ID lists), not data stores
    with embedded payload."""
    # Verify by type-checking the attribute type annotations or checking
    # that the lists contain strings (memory IDs), not dicts/objects.

    m = Memorizer(base_dir="/tmp/test_eve_mem")
    # STM should be a list
    assert isinstance(m.stm, list)
    # After creating a memory, STM should contain string IDs
    mem_id = m.create("test payload", "text")
    assert isinstance(m.stm[0], str)
    assert m.stm[0] == mem_id

    # MTM should be a list
    assert isinstance(m.mtm, list)


def test_stm_max_capacity():
    """STM respects max capacity (1000 entries)."""
    m = Memorizer(base_dir="/tmp/test_eve_stm_cap")
    for i in range(1500):
        m.create(f"payload_{i}", "text")
    assert len(m.stm) <= 1000
    # The most recent entries should be kept
    assert len(m.stm) == 1000


def test_mtm_promote_demote():
    """MTM promote/demote works correctly."""
    m = Memorizer(base_dir="/tmp/test_eve_mtm")
    mem_id = m.create("task data", "text")
    assert m.is_in_mtm(mem_id) is False

    m.promote_to_mtm(mem_id)
    assert m.is_in_mtm(mem_id) is True
    assert mem_id in m.get_mtm_ids()

    m.demote_from_mtm(mem_id)
    assert m.is_in_mtm(mem_id) is False


# ── Blackboard is not long-term storage ──────────────────

def test_blackboard_not_long_term_storage():
    """Blackboard entries have TTL semantics (valid_until_ns), not
    meant for long-term storage. This test verifies Blackboard
    exists and has expiry behavior."""
    from eve.state import Blackboard, TimedEntry
    import time

    bb = Blackboard()
    now = time.monotonic_ns()

    # Write entry with short TTL (100 microseconds)
    bb.write(TimedEntry("tmp1", "temp_result", "node_a", now, now + 100_000, "data"))
    # Entry should expire after a short wait
    time.sleep(0.05)
    results = bb.read("temp_result")
    assert len(results) == 0  # Expired


# ── No graph_manager.py or similar ───────────────────────

def test_no_graph_manager_file():
    """No file named graph_manager, runtime_graph, or graph_engine."""
    filenames = _eve_filenames()
    forbidden = {"graph_manager", "runtime_graph", "graph_engine"}
    violations = [n for n in filenames if n in forbidden]
    assert violations == [], f"Found forbidden graph management files: {violations}"


# ── TNN descriptors are simple ───────────────────────────

def test_tnn_store_no_agent_fields():
    """TNN store files should not contain agent/planner/router patterns in
    their filenames."""
    filenames = _eve_filenames()
    # Only check eve/core/* files
    tnn_related = [n for n in filenames if "tnn" in n.lower()]
    agent_patterns = ["agent", "planner", "router", "orchestrator"]
    for name in tnn_related:
        for pat in agent_patterns:
            assert pat not in name.lower(), f"TNN file '{name}' contains '{pat}'"
