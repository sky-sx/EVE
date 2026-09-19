"""Observable Core scheduling at per-Block tick intervals in seconds."""

import pytest
import torch

from acnt import Block, Core


def timed_core():
    return Core(
        [
            Block(0, 3, [3, 3], ticktime=0.25, hold_tick=8),
            Block(1, 3, [3, 3], ticktime=0.5, hold_tick=8),
        ]
    )


def test_step_obeys_each_block_interval_in_seconds():
    core = timed_core()
    core.step(now_ms=0)
    assert [list(block.At) for block in core.blocks] == [[0], [0]]

    before = [block.z.clone() for block in core.blocks]
    core.step(now_ms=249)
    assert [list(block.At) for block in core.blocks] == [[0], [0]]
    for block, previous in zip(core.blocks, before):
        torch.testing.assert_close(block.z, previous, rtol=0, atol=0)

    core.step(now_ms=250)
    assert [list(block.At) for block in core.blocks] == [[0, 250], [0]]
    core.step(now_ms=500)
    assert [list(block.At) for block in core.blocks] == [[0, 250, 500], [0, 500]]


def test_update_block_obeys_same_interval_as_step():
    core = timed_core()
    core.update_block(0, now_ms=0)
    original = core.blocks[0].z.clone()
    early = core.update_block(0, now_ms=249)
    torch.testing.assert_close(early, original, rtol=0, atol=0)
    assert list(core.blocks[0].At) == [0]
    core.update_block(0, now_ms=250)
    assert list(core.blocks[0].At) == [0, 250]
    assert list(core.blocks[1].At) == []


def test_inactive_block_is_not_updated_when_interval_elapsed():
    core = timed_core()
    core.step(now_ms=0)
    core.set_active(0, False)
    inactive_z = core.blocks[0].z.clone()
    core.step(now_ms=500)
    core.update_block(0, now_ms=750)
    assert list(core.blocks[0].At) == [0]
    torch.testing.assert_close(core.blocks[0].z, inactive_z, rtol=0, atol=0)
    assert list(core.blocks[1].At) == [0, 500]


@pytest.mark.parametrize("method", ["step", "update_block"])
def test_scheduler_rejects_time_moving_backwards(method):
    core = timed_core()
    core.step(now_ms=0)
    core.step(now_ms=500)
    with pytest.raises(ValueError, match="backwards"):
        if method == "step":
            core.step(now_ms=499)
        else:
            core.update_block(0, now_ms=499)
