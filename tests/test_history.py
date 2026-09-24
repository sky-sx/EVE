import pytest
import torch

from acnt import Block


def make_block(*, hold_tick=3, ticktime=250, neurons=2):
    block = Block(0, neurons, [neurons], hold_tick=hold_tick, ticktime=ticktime, nlm_hidden_dim=1)
    with torch.no_grad():
        block.W_ij[0].zero_()
        block.b.zero_()
    return block


def configure_single_feature_reader(block, neuron, feature):
    """Make one private NLM monotonic in one selected input feature."""
    with torch.no_grad():
        for parameter in block.nlm.parameters():
            parameter.zero_()
        block.nlm.weight1[neuron, 0, feature] = 1
        block.nlm.bias1[neuron, 1] = 1
        block.nlm.weight2[neuron, 0, 0] = 1
        block.nlm.bias2[neuron, 1] = 1


def test_incomplete_history_is_right_aligned_with_age_and_validity():
    block = make_block(hold_tick=4, ticktime=250)
    history = (torch.tensor([1.0, 10.0]), torch.tensor([2.0, 20.0]))
    x = block.nlm_input(history, (500, 750), now_ms=1000)
    torch.testing.assert_close(x[0, :4], torch.tensor([0.0, 0.0, 1.0, 2.0]))
    torch.testing.assert_close(x[1, :4], torch.tensor([0.0, 0.0, 10.0, 20.0]))
    torch.testing.assert_close(x[:, 4:8], torch.tensor([[0.0, 0.0, 2.0, 1.0]]).expand(2, -1))
    torch.testing.assert_close(x[:, 8:12], torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(2, -1))


def test_nlm_locality_has_no_direct_cross_neuron_history_path():
    block = make_block(hold_tick=2)
    configure_single_feature_reader(block, neuron=0, feature=1)
    baseline = block.nlm_input((torch.tensor([0.0, 7.0]),), (0,), now_ms=0)
    changed = block.nlm_input((torch.tensor([9.0, 7.0]),), (0,), now_ms=0)
    z0 = block.nlm(baseline)
    z1 = block.nlm(changed)
    assert z0[0] != z1[0]
    torch.testing.assert_close(z0[1], z1[1], rtol=0, atol=0)


def test_real_time_age_changes_output_with_identical_activations():
    block = make_block(hold_tick=2, ticktime=250)
    age_feature_for_oldest_slot = block.hold_tick
    configure_single_feature_reader(block, neuron=0, feature=age_feature_for_oldest_slot)
    history = (torch.tensor([3.0, -2.0]), torch.tensor([4.0, 5.0]))
    young = block.nlm(block.nlm_input(history, (500, 750), now_ms=1000))
    old = block.nlm(block.nlm_input(history, (0, 250), now_ms=1000))
    assert young[0] != old[0]
    torch.testing.assert_close(young[1], old[1], rtol=0, atol=0)


def test_constant_time_and_state_are_exactly_deterministic():
    block = make_block()
    history = (torch.tensor([1.0, -1.0]), torch.tensor([2.0, -2.0]))
    x1 = block.nlm_input(history, (0, 250), now_ms=500)
    x2 = block.nlm_input(history, (0, 250), now_ms=500)
    torch.testing.assert_close(x1, x2, rtol=0, atol=0)
    torch.testing.assert_close(block.nlm(x1), block.nlm(x2), rtol=0, atol=0)


def test_history_fifo_evicts_oldest_pair_together():
    block = make_block(hold_tick=3, ticktime=1)
    recorded = []
    for now in (1, 2, 3, 4):
        with torch.no_grad():
            block.b[:2].copy_(torch.tensor([float(now), -float(now)]))
        block.update(now_ms=now)
        recorded.append(block.a.clone())
    assert list(block.At) == [2, 3, 4]
    for actual, expected in zip(block.A, recorded[-3:]):
        torch.testing.assert_close(actual, expected)


def test_time_rollback_rejected_without_partial_commit():
    block = make_block()
    block.update(now_ms=100)
    before = {name: getattr(block, name).clone() for name in ("r", "a", "z")}
    history = [value.clone() for value in block.A]
    times = list(block.At)
    with pytest.raises(ValueError, match="backwards"):
        block.update(now_ms=99)
    assert list(block.At) == times
    for name, value in before.items():
        torch.testing.assert_close(getattr(block, name), value, rtol=0, atol=0)
    for actual, expected in zip(block.A, history):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gap_ms", [1, 250, 1000, 10_000, 3_600_000])
def test_long_real_time_gaps_remain_finite_without_age_clipping(gap_ms):
    block = make_block(ticktime=250)
    configure_single_feature_reader(block, neuron=0, feature=block.hold_tick - 1)
    block.update(now_ms=0)
    block.update(now_ms=gap_ms)
    x = block.nlm_input(tuple(block.A), tuple(block.At), now_ms=gap_ms)
    expected_oldest_age = gap_ms / block.ticktime
    assert x[0, block.hold_tick + 1].item() == pytest.approx(expected_oldest_age)
    assert torch.isfinite(x).all()
    assert torch.isfinite(block.z).all()


@pytest.mark.parametrize("ticktime", [0, -1, float("inf"), float("nan")])
def test_ticktime_requires_finite_positive_milliseconds(ticktime):
    with pytest.raises(ValueError, match="ticktime"):
        make_block(ticktime=ticktime)
