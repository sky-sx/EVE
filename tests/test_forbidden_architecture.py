import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def active_python_files():
    return [
        path
        for path in (ROOT / "eve").rglob("*.py")
        if not any(part in {"deepseek-7b", "qwen", "yolo26"} for part in path.parts)
        and not path.name.startswith("_")
    ]


def test_removed_architecture_is_not_imported_or_defined():
    forbidden = (
        "RuntimeState" + "Manager",
        "Hormone" + "Manager",
        "Sleep" + "Manager",
        "Graph" + "Manager",
        "Runtime" + "Graph",
        "Graph" + "Trace",
        "TNNGraph",
        "TNNOutputCache",
        "screen_" + "capture",
        "cursor_" + "capture",
    )
    violations = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if name in text:
                violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []


def test_no_silent_core_exception_handlers():
    violations = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8")
        if "except Exception:\n            pass" in text or "except:\n            pass" in text:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_removed_runtime_modules_stay_removed():
    assert not (ROOT / "eve" / "core" / "qnn.py").exists()
    assert not (ROOT / "eve" / "core" / "safegate.py").exists()
    assert not (ROOT / "eve" / "state.py").exists()
    assert not (ROOT / "eve" / "core" / "tnn.py").exists()


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_capture_access_is_only_through_buffer():
    assert "eve.input.capture" not in _imports(ROOT / "eve" / "main.py")
    assert "eve.input.capture" not in _imports(ROOT / "eve" / "core" / "loop.py")
    assert "eve.input.buffer" not in _imports(ROOT / "eve" / "input" / "capture.py")
    assert "eve.input.capture" in _imports(ROOT / "eve" / "input" / "buffer.py")


def test_behavior_boundaries_are_kept_generic():
    core_imports = _imports(ROOT / "eve" / "core" / "loop.py")
    dock_imports = _imports(ROOT / "eve" / "dock" / "trainer.py")
    memory_text = (ROOT / "eve" / "memory" / "memorizer.py").read_text(
        encoding="utf-8"
    )
    formal_text = "\n".join(
        path.read_text(encoding="utf-8") for path in active_python_files()
    ).casefold()

    assert "eve.core.qnn" not in core_imports
    assert not any(name.startswith("eve.output") for name in dock_imports)
    assert "register_runtime_tnn" not in memory_text
    assert "load_tnn_runtime" not in memory_text
    forbidden_experiment_tokens = (
        "red_" + "circle",
        "blue_" + "triangle",
        "red_" + "blue",
        "shape_" + "locator",
        "red_" + "ball",
    )
    assert [token for token in forbidden_experiment_tokens if token in formal_text] == []


def test_foreground_window_metadata_is_not_an_eve_perception_source():
    active = [
        ROOT / "eve" / "input" / "capture.py",
        ROOT / "eve" / "input" / "buffer.py",
        ROOT / "eve" / "core" / "loop.py",
    ]
    forbidden = (
        "GetForegroundWindow",
        "GetWindowText",
        "active_window",
        "window_mode",
        "process_name",
    )
    violations = [
        f"{path.relative_to(ROOT)}: {token}"
        for path in active
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert violations == []
