import pytest
import torch

from acnt import Block, NeuronLevelModel


def make_block(**kwargs):
    return Block(0, 3, [3], **kwargs)


def test_initialization_shapes_and_parameter_inventory():
    block = Block(1, 3, [2, 3, 5], hold_tick=4, nlm_hidden_dim=4, readin=True)
    assert block.active and block.ticktime == 1 and block.hold_tick == 4
    assert block.z.shape == block.a.shape == (3,)
    assert block.r.shape == block.o.shape == (6,)
    assert [tuple(weight.shape) for weight in block.W_ij] == [(6, 2), (6, 3), (6, 5)]
    assert block.nlm.weight1.shape == (3, 8, 12)
    assert block.nlm.bias1.shape == (3, 8)
    assert block.nlm.weight2.shape == (3, 2, 4)
    assert block.nlm.bias2.shape == (3, 2)
    assert list(block.A) == [] and list(block.At) == []


def test_private_nlm_matches_explicit_per_neuron_calculation():
    torch.manual_seed(3)
    nlm = NeuronLevelModel(5, 9, hidden_dim=4)
    x = torch.randn(5, 9)
    expected = []
    for d in range(5):
        first = torch.nn.functional.glu(nlm.weight1[d] @ x[d] + nlm.bias1[d], dim=-1)
        expected.append(torch.nn.functional.glu(nlm.weight2[d] @ first + nlm.bias2[d], dim=-1))
    torch.testing.assert_close(nlm(x), torch.stack(expected).squeeze(-1))


def test_first_update_matches_synapse_glu_layernorm_and_nlm():
    block = make_block(readin=True, hold_tick=2)
    with torch.no_grad():
        block.b.copy_(torch.tensor([1.0, -2.0, 3.0, 0.0, 0.0, 0.0]))
        block.o.copy_(torch.tensor([2.0, 1.0, -1.0, 0.0, 0.0, 0.0]))
    expected_r = block.b + block.o
    raw = expected_r[:3] * torch.sigmoid(expected_r[3:])
    expected_a = torch.nn.functional.layer_norm(raw, (3,), eps=block.ln_eps)
    expected_x = block.nlm_input((expected_a,), (100,), now_ms=100)
    expected_z = block.nlm(expected_x)
    block.update(now_ms=100)
    torch.testing.assert_close(block.r, expected_r)
    torch.testing.assert_close(block.a, expected_a)
    torch.testing.assert_close(block.z, expected_z)
    torch.testing.assert_close(block.A[0], expected_a)
    assert list(block.At) == [100]


def test_repeated_updates_keep_finite_paired_fifo():
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
        assert all(torch.isfinite(getattr(block, name)).all() for name in ("z", "a", "r"))


@pytest.mark.parametrize("kwargs", [
    {"hold_tick": 0}, {"nlm_hidden_dim": 0}, {"ticktime": 0},
    {"ticktime": -1}, {"ticktime": float("nan")}, {"ln_eps": 0},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_block(**kwargs)


def test_inactive_update_changes_nothing():
    block = make_block(active=False)
    before = {name: getattr(block, name).clone() for name in ("z", "a", "r")}
    returned = block.update(now_ms=0, active_z={0: torch.ones(3)})
    assert returned.data_ptr() == block.z.data_ptr()
    assert list(block.A) == [] and list(block.At) == []
    for name, value in before.items():
        torch.testing.assert_close(getattr(block, name), value, rtol=0, atol=0)
