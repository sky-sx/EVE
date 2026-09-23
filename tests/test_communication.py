import pytest
import torch

from acnt import Block, Core


def make_core(sizes=(2, 3, 4)):
    torch.manual_seed(29)
    return Core([Block(i, n, sizes, ticktime=1) for i, n in enumerate(sizes)])


def test_two_blocks_heterogeneous_transfer_through_complete_update():
    core = make_core((2, 3))
    target, source = core.blocks
    with torch.no_grad():
        source.z.copy_(torch.tensor([1.0, 2.0, -1.0]))
        target.W_ij[0].zero_()
        target.W_ij[1].copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]))
        target.W_c[0].copy_(torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1))
    expected_r = torch.tensor([1., 2., -1., 3.])
    expected_raw = expected_r[:2] * torch.sigmoid(expected_r[2:])
    expected_a = (expected_raw - expected_raw.mean()) / torch.sqrt(expected_raw.var(unbiased=False) + 1e-5)
    expected_z = 0.5 * (expected_a - expected_a.mean()) / torch.sqrt(expected_a.var(unbiased=False) + 1e-5)
    core.update_block(0, now_ms=10)
    torch.testing.assert_close(target.r, expected_r)
    torch.testing.assert_close(target.a, expected_a)
    torch.testing.assert_close(target.A[0], expected_a)
    torch.testing.assert_close(target.z, expected_z)
    assert target.At[0] == 10


def test_three_active_sources_sum_including_self():
    core = make_core()
    target = core.blocks[1]
    with torch.no_grad():
        target.b.fill_(0.25)
        for index, block in enumerate(core.blocks):
            block.z.fill_(index + 1)
            target.W_ij[index].fill_(index + 0.5)
    expected_scalar = 0.25 + 2 * 0.5 * 1 + 3 * 1.5 * 2 + 4 * 2.5 * 3
    core.update_block(1, now_ms=0)
    torch.testing.assert_close(target.r, torch.full((6,), expected_scalar))


def test_self_edge_reads_previous_z():
    core = make_core((2,))
    block = core.blocks[0]
    with torch.no_grad():
        block.z.copy_(torch.tensor([2., -3.]))
        block.W_ij[0].copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]))
    core.step(now_ms=0)
    torch.testing.assert_close(block.r, torch.tensor([2., -3., 0., 0.]))


def test_inactive_nonzero_source_excluded_from_sum():
    core = make_core()
    target = core.blocks[0]
    with torch.no_grad():
        for block in core.blocks:
            block.z.fill_(1.0)
        for weight in target.W_ij:
            weight.fill_(1.0)
    core.blocks[1].active = False
    core.update_block(0, now_ms=0)
    torch.testing.assert_close(target.r, torch.full((4,), 2.0 + 4.0))
    assert len(core.blocks[1].At) == len(core.blocks[1].A) == 0


def test_same_round_reads_old_state_and_next_round_reads_committed_state():
    core = make_core((2, 2))
    first, second = core.blocks
    with torch.no_grad():
        first.b.copy_(torch.tensor([1., -1., 0., 0.]))
        second.W_ij[0].fill_(0.5)
        second.W_ij[0][0].copy_(torch.tensor([1., -1.]))
        second.W_ij[1].zero_()
    core.step(now_ms=1)
    torch.testing.assert_close(second.r, torch.zeros_like(second.r))
    previous = first.z.clone()
    core.step(now_ms=2)
    torch.testing.assert_close(second.r, second.W_ij[0] @ previous)
    assert abs(second.r[0].item()) > 0.1


@pytest.mark.parametrize("sizes", [(2, 3), (2, 3, 4, 5)])
def test_repeated_snapshot_and_asynchronous_updates(sizes):
    core = make_core(sizes)
    for tick in range(50):
        order = list(reversed(range(len(sizes)))) if tick % 2 else [tick % len(sizes)]
        outputs = core.step(now_ms=tick, order=order)
        assert set(outputs) == set(order)
        assert all(torch.isfinite(z).all() for z in outputs.values())


def test_incompatible_topology_rejected():
    with pytest.raises(ValueError, match="source neuron sizes"):
        Core([Block(0, 2, [2, 9]), Block(1, 3, [2, 3])])


def test_relabeling_block_ids_preserves_corresponding_states():
    from copy import deepcopy
    original = make_core((3, 3, 3))
    with torch.no_grad():
        for i, block in enumerate(original.blocks):
            block.b.copy_(torch.tensor([0.1, -0.3, 0.7, 0.2, -0.1, 0.4]) * (i + 1))
            block.z.copy_(torch.tensor([0.2, -0.4, 0.1]) * (i + 1))
    original.step(now_ms=0)
    permutation = [2, 0, 1]  # new ID -> old ID; remap all incoming edges too.
    blocks = []
    for new_id, old_id in enumerate(permutation):
        block = deepcopy(original.blocks[old_id])
        block.block_id = new_id
        block.W_ij = torch.nn.ParameterList([deepcopy(original.blocks[old_id].W_ij[j]) for j in permutation])
        blocks.append(block)
    relabeled = Core(blocks)
    original.step(now_ms=1)
    relabeled.step(now_ms=1)
    for new_id, old_id in enumerate(permutation):
        left, right = original.blocks[old_id], relabeled.blocks[new_id]
        for name in ("r", "a", "h", "z"):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=1e-5, atol=1e-6)
        assert list(left.At) == list(right.At)
