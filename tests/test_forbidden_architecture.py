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


def test_formal_runtime_file_structure_and_removed_modules():
    expected = {
        "eve/main.py",
        "eve/input/capture.py",
        "eve/input/buffer.py",
        "eve/output/keyboard.py",
        "eve/output/mouse.py",
        "eve/output/speak.py",
        "eve/memory/memorizer.py",
        "eve/core/loop.py",
        "eve/core/qnn.py",
        "eve/core/safegate.py",
        "eve/dock/trainer.py",
        "eve/dock/tinynn.py",
    }
    actual = {"eve/main.py"}
    for directory in ("input", "output", "memory", "core", "dock"):
        for path in (ROOT / "eve" / directory).glob("*.py"):
            if path.name != "__init__.py":
                actual.add(path.relative_to(ROOT).as_posix())
    assert actual == expected
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
    assert "eve.input.capture" not in _imports(ROOT / "eve" / "core" / "qnn.py")
    assert "eve.input.capture" not in _imports(ROOT / "eve" / "core" / "safegate.py")
    assert "eve.input.buffer" not in _imports(ROOT / "eve" / "input" / "capture.py")
    assert "eve.input.capture" in _imports(ROOT / "eve" / "input" / "buffer.py")


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
