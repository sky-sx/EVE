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
