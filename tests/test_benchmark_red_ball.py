"""Benchmark runner — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# from eve_core.womb.benchmark_red_ball import run_benchmark
#
# New structure: eve.deployment.shadow_runner.ShadowRunner
# — completely different API, no run_benchmark function.


def test_benchmark_produces_per_seed_jsonl_and_summary(monkeypatch, tmp_path) -> None:
    pass


def test_benchmark_success_rate_above_threshold(monkeypatch, tmp_path) -> None:
    pass


def test_benchmark_unsafe_action_total_is_zero(monkeypatch, tmp_path) -> None:
    pass
