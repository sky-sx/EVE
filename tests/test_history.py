"""Independent numerical checks of the canonical history recurrence."""

import math

import pytest
import torch

from acnt import Block


def ln_reference(values):
    """Population LN in Python arithmetic, independent of torch.layer_norm."""
    values = [float(value) for value in values]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + 1e-5)
    return torch.tensor([(value - mean) / scale for value in values], dtype=torch.float32)


def matvec_reference(matrix, vector):
    return [math.fsum(float(x) * float(y) for x, y in zip(row, vector)) for row in matrix]


def sigmoid_reference(value):
    return 1.0 / (1.0 + math.exp(-value))


def zero_block(*, hold_tick=3, ticktime=0.001):
    block = Block(0, 3, [3], hold_tick=hold_tick, ticktime=ticktime)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    return block


def bias_history_block(*, ticktime=0.001):
    block = zero_block(ticktime=ticktime)
    biases = ([1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 1.0, 0.0])
    with torch.no_grad():
        for parameter, bias in zip(block.b_c, biases):
            parameter.copy_(torch.tensor(bias))
    return block, [ln_reference(bias) for bias in biases]


@pytest.mark.parametrize("ticktime", [0.001, 0.002, 0.25])
def test_latest_history_has_zero_delta_and_half_weight(ticktime):
    block, candidates = bias_history_block(ticktime=ticktime)
    block.update(now_ms=100)
    # The just-appended entry has delta_t=0, hence gamma=1/2 for any ticktime.
    torch.testing.assert_close(block.z, 0.5 * candidates[0])
    assert list(block.At) == [100]


@pytest.mark.parametrize("ticktime", [0.001, 0.002])
def test_nonuniform_history_uses_adjacent_deltas_and_queue_indices(ticktime):
    block, (c0, c1, c2) = bias_history_block(ticktime=ticktime)
    block.update(now_ms=100)
    block.update(now_ms=101)
    gamma_one = sigmoid_reference(1 * ticktime)
    # With two entries, newest uses index 1 and oldest uses index 0.
    torch.testing.assert_close(block.z, (1 - gamma_one) * c0 + 0.5 * gamma_one * c1)

    block.update(now_ms=103)
    gamma_two = sigmoid_reference(2 * ticktime)
    # At=[100,101,103] traverses delta_t=[0,2,1]. This is the closed-form
    # expansion, deliberately without reproducing the implementation's loop.
    expected = (
        (1 - gamma_one) * c0
        + gamma_one * (1 - gamma_two) * c1
        + 0.5 * gamma_one * gamma_two * c2
    )
    torch.testing.assert_close(block.z, expected)
    assert list(block.At) == [100, 101, 103]
    assert len(block.A) == 3


def test_ticktime_seconds_multiply_adjacent_history_deltas():
    fast, (c0, c1, c2) = bias_history_block(ticktime=0.001)
    slow, _ = bias_history_block(ticktime=0.002)
    for now_ms in (100, 101, 103):
        fast.update(now_ms=now_ms)
        slow.update(now_ms=now_ms)
    # Canonical gamma uses the adjacent millisecond gap multiplied by
    # ticktime in seconds. A larger ticktime produces a larger gamma.
    gamma_one, gamma_two = sigmoid_reference(1 * slow.ticktime), sigmoid_reference(2 * slow.ticktime)
    expected_slow = (
        (1 - gamma_one) * c0
        + gamma_one * (1 - gamma_two) * c1
        + 0.5 * gamma_one * gamma_two * c2
    )
    torch.testing.assert_close(slow.z, expected_slow)
    assert torch.linalg.vector_norm(fast.z - slow.z).item() > 0


@pytest.mark.parametrize("ticktime", [0, -0.001, float("inf"), float("nan")])
def test_ticktime_requires_finite_positive_seconds(ticktime):
    with pytest.raises(ValueError):
        zero_block(ticktime=ticktime)


def test_overflow_reindexes_retained_history_against_current_wc_slots():
    block = zero_block()
    matrices = (
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0]],
        [[1.0, 1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]],
    )
    raw_a = ([1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 1.0, 0.0], [1.0, -2.0, 1.0])
    expected_a = [ln_reference(values) for values in raw_a]
    with torch.no_grad():
        for parameter, matrix in zip(block.W_c, matrices):
            parameter[:, :3].copy_(torch.tensor(matrix))
    for values, now_ms in zip(raw_a, (100, 101, 103, 106)):
        with torch.no_grad():
            # Gate logits remain zero, so r[:3] * sigmoid(r[3:]) = values.
            block.b[:3].copy_(2 * torch.tensor(values))
        block.update(now_ms=now_ms)

    assert list(block.At) == [101, 103, 106]
    for actual, expected in zip(block.A, expected_a[1:]):
        torch.testing.assert_close(actual, expected)
    # After eviction, a_101 shifts from W_c[1] to W_c[0], and a_103 from
    # W_c[2] to W_c[1]. Slots follow queue indices, not creation-time indices.
    c0 = ln_reference(matvec_reference(matrices[0], expected_a[1]))
    c1 = ln_reference(matvec_reference(matrices[1], expected_a[2]))
    c2 = ln_reference(matvec_reference(matrices[2], expected_a[3]))
    gamma_two, gamma_three = sigmoid_reference(2 * block.ticktime), sigmoid_reference(3 * block.ticktime)
    expected_z = (
        (1 - gamma_two) * c0
        + gamma_two * (1 - gamma_three) * c1
        + 0.5 * gamma_two * gamma_three * c2
    )
    torch.testing.assert_close(block.z, expected_z)


def test_wc_second_half_transforms_h_from_the_newer_history_entry():
    block = zero_block(hold_tick=2, ticktime=0.002)
    matrix = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    bias = [0.5, -1.0, 0.5]
    newest_raw = [2.0, -1.0, 0.0]
    with torch.no_grad():
        block.W_c[0][:, 3:].copy_(torch.tensor(matrix))
        block.b_c[0].copy_(torch.tensor(bias))
        block.W_c[1][:, :3].copy_(torch.eye(3))
    block.update(now_ms=100)
    with torch.no_grad():
        block.b[:3].copy_(2 * torch.tensor(newest_raw))
    block.update(now_ms=101)

    newest_c = ln_reference(ln_reference(newest_raw))
    newest_h = 0.5 * newest_c
    transformed_h = matvec_reference(matrix, newest_h)
    oldest_c = ln_reference([value + offset for value, offset in zip(transformed_h, bias)])
    gamma = sigmoid_reference(1 * block.ticktime)
    expected = (1 - gamma) * oldest_c + gamma * newest_h
    torch.testing.assert_close(block.z, expected)
    without_h_input = (1 - gamma) * ln_reference(bias) + gamma * newest_h
    assert torch.linalg.vector_norm(block.z - without_h_input).item() > 0.1


def test_each_update_initializes_h_to_zero_instead_of_reusing_prior_h_or_z():
    block = zero_block(hold_tick=1)
    bias = [1.0, -2.0, 1.0]
    with torch.no_grad():
        block.W_c[0][:, 3:].copy_(torch.eye(3))
        block.b_c[0].copy_(torch.tensor(bias))
    expected = 0.5 * ln_reference(bias)
    block.update(now_ms=100)
    torch.testing.assert_close(block.h, expected)
    # Prior h and z are both nonzero. With hold_tick=1, the next update must
    # nevertheless start its only history step at h=0 and produce the same z.
    block.update(now_ms=107)
    torch.testing.assert_close(block.h, expected)
    torch.testing.assert_close(block.z, expected)
    assert list(block.At) == [107]
