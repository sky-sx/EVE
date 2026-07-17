"""Trajectory dataset — migrated to eve package, stub-only."""

import pytest

pytest.skip("migrated to eve package, stub-only", allow_module_level=True)

# Original imports (eve_core):
# import json
# from pathlib import Path
# from eve_core.evolution.trajectory_dataset import (
#     TrajectoryDatasetBuilder,
#     iter_samples,
#     load_samples,
# )
#
# New structure: eve.training.dataset_builder.DatasetBuilder
# — completely different API, no TrajectoryDatasetBuilder.


def test_builder_creates_samples_jsonl(monkeypatch, tmp_path) -> None:
    pass


def test_builder_creates_dataset_summary(monkeypatch, tmp_path) -> None:
    pass


def test_num_samples_greater_than_zero(monkeypatch, tmp_path) -> None:
    pass


def test_samples_contain_hot_state_and_teacher_impulse(monkeypatch, tmp_path) -> None:
    pass


def test_all_teacher_impulses_have_zero_click_type_prob(monkeypatch, tmp_path) -> None:
    pass


def test_unsafe_action_total_is_zero(monkeypatch, tmp_path) -> None:
    pass


def test_dataset_generation_is_deterministic(monkeypatch, tmp_path) -> None:
    pass


def test_load_samples_returns_valid_list(monkeypatch, tmp_path) -> None:
    pass


def test_iter_samples_yields_samples(monkeypatch, tmp_path) -> None:
    pass


def test_no_forbidden_names_in_dataset_code() -> None:
    pass
