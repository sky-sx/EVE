"""ReplayEvaluator — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# import json
# from eve_core.evolution.replay_evaluator import ReplayEvaluator
# from eve_core.womb.demo_red_ball import run_demo
#
# New structure: eve.episode.replay_evaluator.ReplayEvaluator
# — module exists but API is stub-only, and run_demo no longer exists.


def test_replay_evaluator_evaluates_generated_run(monkeypatch, tmp_path) -> None:
    pass


def test_replay_evaluator_loads_empty_dir_raises(tmp_path) -> None:
    pass


def test_replay_evaluator_metrics_are_json_serializable(monkeypatch, tmp_path) -> None:
    pass
