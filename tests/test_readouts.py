from copy import deepcopy

import torch

from acnt.hand import (
    HAND_DISCRETE_NAMES, KEYBOARD_KEYS, MOUSE_BUTTONS, TRAINING_CONTROL_NAMES,
    read_training_controls,
)


def set_hand_controls(runtime, enabled):
    last_linear = runtime.adapters["hand"].network[-1]
    with torch.no_grad():
        for parameter in runtime.adapters["hand"].parameters():
            parameter.zero_()
        last_linear.bias[:85].fill_(-100.)
        for name in enabled:
            last_linear.bias[HAND_DISCRETE_NAMES.index(name)] = 100.
        last_linear.bias[85:] = torch.tensor([2.5, -3.5])


def test_full_keyboard_layout_and_trainer_subset_are_explicit():
    assert len(KEYBOARD_KEYS) == len(set(KEYBOARD_KEYS)) == 82
    assert len(HAND_DISCRETE_NAMES) == len(set(HAND_DISCRETE_NAMES)) == 85
    assert HAND_DISCRETE_NAMES == KEYBOARD_KEYS + MOUSE_BUTTONS
    assert MOUSE_BUTTONS == ("MOUSE_LEFT", "MOUSE_RIGHT", "MOUSE_MIDDLE")
    assert TRAINING_CONTROL_NAMES == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("MOUSE_LEFT",)
    assert set(TRAINING_CONTROL_NAMES).issubset(HAND_DISCRETE_NAMES)


def test_hand_samples_independent_full_controls_and_logs_key_combinations(make_runtime):
    runtime = make_runtime(ticktime=0.25)
    enabled = {"CTRL_LEFT", "C", "BRACKET_LEFT", "MOUSE_LEFT"}
    set_hand_controls(runtime, enabled)

    signal = runtime.generate_hand(now_ms=500, generator=torch.Generator().manual_seed(19))

    assert signal.discrete.q.shape == signal.discrete.p.shape == signal.discrete.a.shape == (85,)
    assert signal.discrete.noise.shape == (85,)
    assert signal.discrete.tau == runtime.noise_scale
    assert signal.discrete.a.dtype == torch.bool
    expected = torch.tensor([name in enabled for name in HAND_DISCRETE_NAMES])
    assert torch.equal(signal.discrete.a, expected)
    assert signal.discrete.p.sum() > 1  # A modifier and its key can both be active.
    assert len(runtime.mechanical_log.records) == 1
    record = runtime.mechanical_log.records[0]
    assert record["time_ms"] == 500 and record["organ"] == "hand"
    assert record["signal"]["discrete"] == dict(zip(HAND_DISCRETE_NAMES, expected.tolist()))
    assert record["signal"]["continuous"] == {"dx": 2.5, "dy": -3.5}


def test_hand_continuous_mouse_axes_bypass_noise_and_threshold(make_runtime):
    runtime = make_runtime()
    set_hand_controls(runtime, {"A"})

    first = runtime.generate_hand(now_ms=0, threshold=0., generator=torch.Generator().manual_seed(1))
    second = runtime.generate_hand(now_ms=250, threshold=200., generator=torch.Generator().manual_seed(2))

    assert first.discrete.a.any() and not second.discrete.a.any()
    assert not torch.equal(first.discrete.noise, second.discrete.noise)
    assert first.continuous.shape == second.continuous.shape == (2,)
    torch.testing.assert_close(first.continuous, torch.tensor([2.5, -3.5]))
    torch.testing.assert_close(second.continuous, first.continuous)


def test_trainer_reads_only_letters_and_left_click_without_modifying_full_log(make_runtime):
    runtime = make_runtime()
    set_hand_controls(runtime, {"A", "BRACKET_LEFT", "CTRL_LEFT", "MOUSE_LEFT", "MOUSE_RIGHT"})
    runtime.generate_hand(now_ms=0, generator=torch.Generator().manual_seed(3))
    record = runtime.mechanical_log.records[0]
    before = deepcopy(record)

    observed = read_training_controls(record)

    assert set(observed) == set(TRAINING_CONTROL_NAMES)
    assert observed["A"] and observed["MOUSE_LEFT"]
    assert not observed["B"]
    assert "BRACKET_LEFT" not in observed and "CTRL_LEFT" not in observed
    assert "MOUSE_RIGHT" not in observed and "dx" not in observed
    assert record == before
    assert record["signal"]["discrete"]["BRACKET_LEFT"]
    assert record["signal"]["discrete"]["MOUSE_RIGHT"]
    observed["A"] = False
    assert record["signal"]["discrete"]["A"]


def test_speak_returns_and_logs_all_30_continuous_parameters(make_runtime):
    runtime = make_runtime()
    expected = torch.linspace(-3., 3., 30)
    with torch.no_grad():
        runtime.adapters["speak"].linear.weight.zero_()
        runtime.adapters["speak"].linear.bias.copy_(expected)

    signal = runtime.generate_speak(now_ms=750)

    assert signal.shape == (30,)
    assert signal.dtype == torch.float32
    torch.testing.assert_close(signal, expected)
    assert runtime.mechanical_log.records == [{
        "time_ms": 750, "organ": "speak", "signal": {"controls": expected.tolist()},
        "execution_enabled": False,
    }]
    assert read_training_controls(runtime.mechanical_log.records[0]) == {}
