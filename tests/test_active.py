import pytest
import torch

from acnt import Block, Core


def make_core():
    torch.manual_seed(31)
    core = Core([Block(i, 3, [3, 3, 3], ticktime=0.001) for i in range(3)])
    with torch.no_grad():
        for block in core.blocks:
            block.b.copy_(torch.tensor([1., 2., -1., 0., 0., 0.]))
    return core


def snapshot(block):
    return {name: getattr(block, name).clone() for name in ("z", "a", "r", "h")}, list(block.At), [a.clone() for a in block.A]


def assert_unchanged(block, saved):
    values, times, history = saved
    for name, expected in values.items():
        torch.testing.assert_close(getattr(block, name), expected, rtol=0, atol=0)
    assert list(block.At) == times
    assert len(block.A) == len(history)
    for a, expected in zip(block.A, history):
        torch.testing.assert_close(a, expected, rtol=0, atol=0)


def test_all_active_then_partial_then_reactivate():
    core = make_core()
    assert set(core.step(now_ms=0)) == {0, 1, 2}
    saved = snapshot(core.blocks[1])
    parameters = list(core.blocks[1].parameters())
    core.set_active_mask([True, False, True])
    assert core.active_ids == (0, 2)
    assert set(core.step(now_ms=1)) == {0, 2}
    assert len(core.blocks) == 3
    assert_unchanged(core.blocks[1], saved)
    assert all(old is new for old, new in zip(parameters, core.blocks[1].parameters()))
    core.set_active(1, True)
    assert set(core.step(now_ms=2)) == {0, 1, 2}
    assert list(core.blocks[1].At) == [0, 2]


def test_inactive_direct_update_skips_every_state():
    core = make_core()
    block = core.blocks[1]
    block.update(now_ms=10)
    core.set_active(1, False)
    saved = snapshot(block)
    block.update(now_ms=11, active_z={0: torch.ones(3)})
    core.update_block(1, now_ms=12)
    assert_unchanged(block, saved)


def test_all_inactive_then_active_set_changes_repeatedly():
    core = make_core()
    core.set_active_mask([False, False, False])
    assert core.step(now_ms=0) == {}
    assert len(core.blocks) == 3
    for tick in range(1, 31):
        mask = [(tick + i) % 3 != 0 for i in range(3)]
        core.set_active_mask(mask)
        saved = [snapshot(block) for block in core.blocks]
        assert set(core.step(now_ms=tick)) == {i for i, value in enumerate(mask) if value}
        for i, value in enumerate(mask):
            if not value:
                assert_unchanged(core.blocks[i], saved[i])


def test_mask_has_effect_on_actual_source_sum_next_step():
    core = make_core()
    target = core.blocks[0]
    with torch.no_grad():
        target.b.zero_()
        for block in core.blocks:
            block.z.fill_(1.)
        for weight in target.W_ij:
            weight.fill_(1.)
    core.set_active_mask([True, False, True])
    core.step(now_ms=0, order=[0])
    torch.testing.assert_close(target.r, torch.full((6,), 6.))
    core.set_active_mask([True, False, False])
    with torch.no_grad():
        target.z.fill_(1.)
    core.step(now_ms=1, order=[0])
    torch.testing.assert_close(target.r, torch.full((6,), 3.))


def test_one_update_override_can_preserve_existing_block_and_history():
    # Only the active primitive is tested here. Actual ReadIn/route roles wait
    # for their later phases, after the mandatory Adapter selection gate.
    core = make_core()
    core.set_active(1, False)
    core.set_active(1, True)
    core.step(now_ms=0, order=[1])
    core.set_active(1, False)
    core.step(now_ms=1)
    assert list(core.blocks[1].At) == [0]
    assert not core.blocks[1].active


@pytest.mark.parametrize("mask", [[True], [True, False, 1]])
def test_invalid_mask_does_not_partially_apply(mask):
    core = make_core()
    with pytest.raises(ValueError, match="one bool"):
        core.set_active_mask(mask)
    assert core.active_ids == (0, 1, 2)


def test_core_snapshot_makes_visitation_order_irrelevant():
    left, right = make_core(), make_core()
    for now in (0, 1, 2):
        left.step(now_ms=now, order=[0, 1, 2])
        right.step(now_ms=now, order=[2, 0, 1])
        for a, b in zip(left.blocks, right.blocks):
            assert_unchanged(a, snapshot(b))
