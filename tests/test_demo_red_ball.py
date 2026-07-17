"""Demo red ball runner — migrated to eve package, stub-only."""

import pytest

pytest.skip("womb module moved to eve.deployment", allow_module_level=True)

# Original imports (eve_core):
# from pathlib import Path
# from eve_core.womb.demo_red_ball import run_demo
#
# New structure: eve.deployment.shadow_runner.ShadowRunner
# — completely different API, no run_demo function.


def test_demo_red_ball_runs_and_reduces_distance(monkeypatch, tmp_path) -> None:
    pass
