import math

import pytest
import torch

from acnt import Block


def make_block(**kwargs):
    torch.manual_seed(19)
    return Block(0, 3, [3], **kwargs)


def test_initialization_shapes_and_parameter_inventory():
    block = Block(1, 3, [2, 3, 5], hold_tick=4, readin=True)
    assert block.active
    assert block.neuron_size == 3
    assert block.ticktime == 1
    assert block.hold_tick == 4
    for name in ("z", "a", "h"):
        assert getattr(block, name).shape == (3,)
    assert block.r.shape == block.b.shape == block.o.shape == (6,)
    assert [tuple(w.shape) for w in block.W_ij] == [(6, 2), (6, 3), (6, 5)]
    assert all(w.shape == (3, 6) for w in block.W_c)
    assert all(b.shape == (3,) for b in block.b_c)
    assert len(block.W_c) == len(block.b_c) == 4
    assert len(list(block.parameters())) == 3 + 1 + 4 + 4
    assert all(p.dtype == torch.float32 for p in block.parameters())
    assert len(block.A) == len(block.At) == 0
    assert make_block().o is None


def test_sigma_and_layer_normalization():
    block = make_block()
    x = torch.tensor([-1000.0, 0.0, 1000.0])
    torch.testing.assert_close(block.sigma(x), torch.tensor([0.0, 0.5, 1.0]))
    x = torch.tensor([1.0, 2.0, 6.0])
    expected = (x - x.mean()) / math.sqrt(x.var(unbiased=False).item() + 1e-5)
    torch.testing.assert_close(block.LN(x), expected)
    torch.testing.assert_close(block.LN(torch.ones(3)), torch.zeros(3))


def test_first_update_matches_hand_calculation_and_readin_o():
    block = make_block(readin=True)
    with torch.no_grad():
        block.b.copy_(torch.tensor([1.0, -2.0, 3.0, 0.0, 0.0, 0.0]))
        block.o.copy_(torch.tensor([2.0, 1.0, -1.0, 0.0, 0.0, 0.0]))
        block.W_c[0].copy_(torch.cat((torch.eye(3), torch.zeros(3, 3)), dim=1))
    expected_r = block.o + block.b
    raw_a = expected_r[:3] * 0.5
    expected_a = (raw_a - raw_a.mean()) / torch.sqrt(raw_a.var(unbiased=False) + 1e-5)
    raw_h = expected_a
    expected_z = 0.5 * (raw_h - raw_h.mean()) / torch.sqrt(raw_h.var(unbiased=False) + 1e-5)
    z = block.update(now_ms=100)
    torch.testing.assert_close(block.r, expected_r)
    torch.testing.assert_close(block.a, expected_a)
    torch.testing.assert_close(z, expected_z)
    assert list(block.At) == [100]
    torch.testing.assert_close(block.A[0], expected_a)


def test_repeated_updates_push_pop_and_pairing():
    block = make_block(hold_tick=3)
    expected_history = []
    for step in range(30):
        with torch.no_grad():
            block.b[:3].copy_(torch.tensor([float(step), -1.0, 2.0]))
        block.update(now_ms=step * 7)
        expected_history.append(block.a.clone())
        assert len(block.A) == len(block.At) == min(step + 1, 3)
        assert list(block.At) == [i * 7 for i in range(max(0, step - 2), step + 1)]
        for actual, expected in zip(block.A, expected_history[-3:]):
            torch.testing.assert_close(actual, expected)
        assert all(torch.isfinite(getattr(block, name)).all() for name in ("z", "a", "r", "h"))


@pytest.mark.parametrize("hold_tick", [1, 2, 5])
def test_history_not_full_and_hold_tick_limit(hold_tick):
    block = make_block(hold_tick=hold_tick)
    for step in range(hold_tick + 3):
        block.update(now_ms=step)
        assert len(block.A) == len(block.At) == min(step + 1, hold_tick)


def test_backwards_time_rejected_without_state_change():
    block = make_block()
    block.update(now_ms=8)
    previous = block.z.clone()
    with pytest.raises(ValueError, match="backwards"):
        block.update(now_ms=7)
    assert list(block.At) == [8]
    torch.testing.assert_close(block.z, previous)


@pytest.mark.parametrize("kwargs", [{"hold_tick": 0}, {"ticktime": 0}, {"ticktime": -1}, {"ticktime": float("nan")}, {"ln_eps": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_block(**kwargs)
