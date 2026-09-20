from collections import Counter
import json
import random
import string

import pytest
import torch

from experiments.stage_zero.environment import (
    ACTIONS,
    GLYPHS,
    VisualEnvironment,
    balanced_targets,
    exact_goodness,
)


@pytest.fixture(scope="module", autouse=True)
def _small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_all_letters_render_distinct_centered_real_eye_frames():
    environment = VisualEnvironment()
    assert ACTIONS == tuple(string.ascii_uppercase) + ("MOUSE_LEFT",)
    seen = set()
    top, left = 316, 800
    for target, letter in enumerate(string.ascii_uppercase):
        frame = environment.render(target)
        assert frame.shape == (3, 1080, 1920)
        assert frame.dtype == torch.float32
        assert frame.device.type == "cpu"
        assert torch.equal(frame[0], frame[1]) and torch.equal(frame[1], frame[2])
        cells = frame[0, top : top + 448 : 64, left : left + 320 : 64]
        rows = tuple("".join(str(int(bit)) for bit in row) for row in cells.tolist())
        assert rows == GLYPHS[letter]
        seen.add(rows)
        expected_pixels = sum(row.count("1") for row in GLYPHS[letter]) * 64 * 64
        assert torch.count_nonzero(frame[0]).item() == expected_pixels
        assert frame.min().item() == 0.0 and frame.max().item() == 1.0
        expected_glyph = cells.repeat_interleave(64, 0).repeat_interleave(64, 1)
        assert torch.equal(frame[0, top : top + 448, left : left + 320], expected_glyph)
        actions = torch.zeros(27, dtype=torch.bool)
        actions[target] = True
        assert exact_goodness(target, actions) == 1.0
    assert len(seen) == 26


def test_green_maps_to_mouse_left_and_render_is_deterministic():
    environment = VisualEnvironment()
    green = environment.render(26)
    assert ACTIONS[26] == "MOUSE_LEFT"
    assert green.shape == (3, 1080, 1920)
    assert green.dtype == torch.float32
    assert torch.count_nonzero(green[0]).item() == 0
    assert torch.count_nonzero(green[2]).item() == 0
    assert torch.all(green[1] == 1.0)
    assert torch.equal(environment.render(2), environment.render(2))
    actions = torch.zeros(27, dtype=torch.bool)
    actions[26] = True
    assert exact_goodness(26, actions) == 1.0
    assert exact_goodness(0, actions) == 0.0


def test_environment_config_preserves_complete_stimulus_specification():
    config = json.loads(json.dumps(VisualEnvironment().config()))
    assert config["shape"] == [3, 1080, 1920]
    assert config["glyph_size"] == [448, 320]
    assert config["glyph_top_left"] == [316, 800]
    assert config["pixel_size"] == 64
    assert config["font_table"] == {letter: list(rows) for letter, rows in GLYPHS.items()}
    assert config["actions"] == list(ACTIONS)
    config["font_table"]["A"][0] = "00000"
    assert GLYPHS["A"][0] == "01110"


@pytest.mark.parametrize("target", [0, 2, 25, 26])
def test_exact_goodness_rejects_none_wrong_extra_and_all_actions(target):
    actions = torch.zeros(27, dtype=torch.bool)
    assert exact_goodness(target, actions) == 0.0
    actions[(target + 1) % 27] = True
    assert exact_goodness(target, actions) == 0.0
    actions[target] = True
    assert exact_goodness(target, actions) == 0.0
    actions.fill_(True)
    assert exact_goodness(target, actions) == 0.0
    actions.fill_(False)
    actions[target] = True
    assert exact_goodness(target, actions) == 1.0


@pytest.mark.parametrize("episodes", [0, 1, 26, 27, 28, 81, 100])
def test_balanced_targets_are_reproducible_local_rng_cycles(episodes):
    state = random.getstate()
    targets = balanced_targets(episodes, seed=42)
    assert random.getstate() == state
    assert targets == balanced_targets(episodes, seed=42)
    assert len(targets) == episodes
    counts = Counter(targets)
    values = [counts[target] for target in range(27)]
    assert max(values) - min(values) <= 1
    for offset in range(0, episodes - 26, 27):
        assert sorted(targets[offset : offset + 27]) == list(range(27))
    if episodes >= 27:
        assert targets != balanced_targets(episodes, seed=43)


def test_environment_rejects_malformed_inputs():
    environment = VisualEnvironment()
    for target in (-1, 27):
        with pytest.raises(ValueError):
            environment.render(target)
        with pytest.raises(ValueError):
            exact_goodness(target, torch.zeros(27, dtype=torch.bool))
    with pytest.raises(TypeError):
        environment.render(True)
    with pytest.raises(ValueError):
        exact_goodness(0, torch.zeros(28, dtype=torch.bool))
    with pytest.raises(TypeError):
        exact_goodness(0, torch.zeros(27))
    with pytest.raises(TypeError):
        exact_goodness(0, [False] * 27)
    with pytest.raises(ValueError):
        balanced_targets(-1, seed=42)
    with pytest.raises(TypeError):
        balanced_targets(1.5, seed=42)
