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
    fractional_goodness,
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


@pytest.mark.parametrize("pressed,expected", [
    ((), 0.0),
    ((1,), 0.0),
    (tuple(range(1, 27)), 0.0),
    ((0,), 1.0),
    ((0, 1), 0.5),
    ((0, 1, 2), 1 / 3),
    (tuple(range(27)), 1 / 27),
])
def test_fractional_goodness_cases(pressed, expected):
    actions = torch.zeros(27, dtype=torch.bool)
    actions[list(pressed)] = True
    goodness = fractional_goodness(0, actions)
    assert goodness == expected
    assert 0.0 <= goodness <= 1.0


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
        with pytest.raises(ValueError):
            fractional_goodness(target, torch.zeros(27, dtype=torch.bool))
    with pytest.raises(TypeError):
        environment.render(True)
    with pytest.raises(ValueError):
        exact_goodness(0, torch.zeros(28, dtype=torch.bool))
    with pytest.raises(TypeError):
        exact_goodness(0, torch.zeros(27))
    with pytest.raises(TypeError):
        exact_goodness(0, [False] * 27)
    with pytest.raises(ValueError):
        fractional_goodness(0, torch.zeros(28, dtype=torch.bool))
    with pytest.raises(TypeError):
        fractional_goodness(0, torch.zeros(27))
    with pytest.raises(TypeError):
        fractional_goodness(0, [False] * 27)
    with pytest.raises(ValueError):
        balanced_targets(-1, seed=42)
    with pytest.raises(TypeError):
        balanced_targets(1.5, seed=42)


from experiments.stage_zero.environment import action_quality_bucket, TeacherTable


@pytest.fixture(scope="module")
def teacher_table(tmp_path_factory):
    """Synthetic nonmonotonic values ONLY for wiring tests, never formal runs."""
    path = tmp_path_factory.mktemp("teacher-fixture") / "test-only.json"
    data = {"schema_version": 1, "frozen": True,
            "calibration": {"model": "test-only", "prompt_version": "test-only",
                            "system_prompt": "test-only"},
            "cells": {f"{c},{k}": {"mean": (1 + (k * 7 + c * 11) % 29) / 32,
                       "raw_scores": [(1 + (k * 7 + c * 11) % 29) / 32],
                       "sample_count": 1, "variance": 0.0, "std": 0.0,
                       "model": "test-only", "prompt_version": "test-only"}
                      for c in (0, 1) for k in range(27)}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return TeacherTable.load(path)


def test_all_quality_buckets_and_frozen_table(teacher_table):
    for target in (0, 13, 26):
        for c in (0, 1):
            for k in range(27):
                bits = torch.zeros(27, dtype=torch.bool)
                bits[[i for i in range(27) if i != target][:k]] = True
                bits[target] = bool(c)
                assert action_quality_bucket(target, bits) == (c, k)
                assert exact_goodness(target, bits) == int((c, k) == (1, 0))
                assert 0 < teacher_table.lookup(c, k) < 1
                assert teacher_table.lookup(c, k) != exact_goodness(target, bits)
    with pytest.raises(ValueError):
        action_quality_bucket(0, torch.zeros(1, 27, dtype=torch.bool))
    with pytest.raises(TypeError):
        action_quality_bucket(0, torch.zeros(27))
    with pytest.raises(TypeError):
        action_quality_bucket(0, [False] * 27)
    with pytest.raises(ValueError):
        teacher_table.lookup(1, 27)


@pytest.mark.parametrize("damage", ["missing", "extra", "nan", "inf", "negative", "large", "count", "mean", "std", "unfrozen"])
def test_invalid_table_is_rejected(tmp_path, teacher_table, damage):
    data = json.loads(teacher_table.raw)
    cell = data["cells"]["1,0"]
    if damage == "missing":
        del data["cells"]["0,26"]
    elif damage == "extra":
        data["cells"]["1,27"] = cell
    elif damage in ("nan", "inf", "negative", "large"):
        cell["mean"] = {"nan": float("nan"), "inf": float("inf"), "negative": -0.1, "large": 1.1}[damage]
    elif damage == "count":
        cell["sample_count"] = 2
    elif damage == "mean":
        cell["mean"] = 0.123
    elif damage == "std":
        cell["std"] = 0.5
    else:
        data["frozen"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        TeacherTable.load(path)


def test_calibration_png_and_payload_are_actual_io_only():
    import base64
    import struct
    import zlib
    from experiments.stage_zero.calibrate_teacher import png_bytes, build_messages
    frame = VisualEnvironment().render(2)
    encoded = png_bytes(frame)
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", encoded[16:24]) == (1920, 1080)
    offset, compressed = 8, b""
    while offset < len(encoded):
        size = struct.unpack(">I", encoded[offset:offset+4])[0]
        if encoded[offset+4:offset+8] == b"IDAT":
            compressed += encoded[offset+8:offset+8+size]
        offset += size + 12
    decoded = zlib.decompress(compressed)
    expected = (frame.permute(1, 2, 0) * 255).to(torch.uint8)
    pixels = torch.frombuffer(bytearray(decoded), dtype=torch.uint8).reshape(1080, 5761)
    assert not pixels[:, 0].any()
    assert torch.equal(pixels[:, 1:].reshape(1080, 1920, 3), expected)
    bits = torch.zeros(27, dtype=torch.bool)
    bits[4] = True
    messages = build_messages(encoded, bits)
    content = messages[1]["content"]
    assert base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1]) == encoded
    observed = json.loads(content[1]["text"].split("\n", 1)[1])
    assert observed == {"button_order": list(ACTIONS), "hand_readout": bits.tolist(), "pressed_buttons": ["E"]}
    for forbidden in ("target_id", "correct_pressed", "wrong_count", "exact_match", "reward", "success"):
        assert forbidden not in content[1]["text"]


@pytest.mark.parametrize("text", ["NaN", "inf", "-0.1", "1.1", "0.5 explanation", "```0.5```", "", "1e999", "true", None])
def test_teacher_parser_rejects_invalid_scores(text):
    from experiments.stage_zero.calibrate_teacher import parse_score
    with pytest.raises(ValueError):
        parse_score(text)


def test_calibration_requires_real_key_and_reads_file_only(tmp_path, monkeypatch):
    from experiments.stage_zero.calibrate_teacher import read_api_key, main
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        read_api_key()
    output = tmp_path / "no-table.json"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 1 and not output.exists()
    keyfile = tmp_path / "test-key.txt"
    keyfile.write_text("test-only-secret", encoding="utf-8")
    before = keyfile.read_bytes()
    assert read_api_key(keyfile) == "test-only-secret"
    assert keyfile.read_bytes() == before


def test_calibration_retries_invalid_answers_and_never_defaults(monkeypatch):
    import io
    from experiments.stage_zero import calibrate_teacher as calibration
    replies = iter(["not a score", "0.375"])
    calls = []
    def respond(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": next(replies)}}]}).encode())
    monkeypatch.setattr(calibration.urllib.request, "urlopen", respond)
    monkeypatch.setattr(calibration.time, "sleep", lambda _: None)
    score, info = calibration.request_score("test-only", [], max_attempts=2)
    assert score == 0.375 and info["attempts"] == 2 and calls[0] == calls[1]
    replies = iter(["NaN", "2"])
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        calibration.request_score("test-only", [], max_attempts=2)


def test_calibration_covers_cells_with_local_rng_and_actual_io(monkeypatch):
    from experiments.stage_zero import calibrate_teacher as calibration
    state = random.getstate()
    calls = []
    monkeypatch.setattr(calibration, "png_bytes", lambda frame: b"test-only-image")
    def score(key, messages, **kwargs):
        calls.append(messages)
        return 0.375, {"response_model": "test-only"}
    monkeypatch.setattr(calibration, "request_score", score)
    data = calibration.calibrate("test-only", samples_per_cell=2)
    assert random.getstate() == state and len(calls) == 108
    assert len(data["cells"]) == 54
    for name, cell in data["cells"].items():
        assert cell["mean"] == 0.375 and cell["sample_count"] == 2 and cell["std"] == 0
        for instance in cell["instances"]:
            assert action_quality_bucket(instance["target"], torch.tensor(instance["action_bits"])) == tuple(map(int, name.split(",")))
