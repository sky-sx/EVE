"""Verify no forbidden legacy module names appear inside src/eve."""

from pathlib import Path

FORBIDDEN = (
    "ActionExecutor",
    "CodebookTrainer",
    "ScreenTrainer",
    "OfflineTrainer",
    "ActionController",
    "Mamba ActionController",
    "VisualMonitor",
    "VisualOrchestrator",
    "VoiceOrchestrator",
    "OrchestratorBridge",
    "Planner",
    "TaskManager",
    "ToolExecutor",
    "pyautogui",
)

# LangGraph is checked as a sub-case: "langgraph" (case-insensitive)
FORBIDDEN_LOWER = ("langgraph",)


def _collect_source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_forbidden_names_not_in_src_eve() -> None:
    eve_core = Path("src/eve")
    if not eve_core.exists():
        return  # nothing to scan

    violations: list[str] = []
    for path in _collect_source_files(eve_core):
        if path.name == "__init__.py":
            continue  # doc-only files may mention forbidden names as negations
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in FORBIDDEN:
            if name in text:
                violations.append(f"{path}: contains '{name}'")
        for name in FORBIDDEN_LOWER:
            if name in text.lower():
                violations.append(f"{path}: contains '{name}' (case-insensitive)")

    assert not violations, "\n".join(violations)


def test_forbidden_names_not_in_tests() -> None:
    tests = Path("tests")
    if not tests.exists():
        return

    violations: list[str] = []
    for path in _collect_source_files(tests):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in FORBIDDEN:
            if name in text:
                    # Allow these test files to reference the forbidden names
                    if "test_forbidden_architecture" in str(path):
                        continue
                    if "test_trajectory_dataset" in str(path):
                        continue
                    if "test_runtime_loop" in str(path):
                        continue
                    if "test_safegate" in str(path):
                        continue
                    violations.append(f"{path}: contains '{name}'")

    assert not violations, "\n".join(violations)
