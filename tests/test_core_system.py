import json
import subprocess
import sys

import pytest
import torch

from acnt import Block, Core


def test_500_steps_with_intermittent_asynchronous_updates():
    torch.manual_seed(37)
    sizes = (1, 2, 3, 5)
    core = Core([Block(i, n, sizes, hold_tick=3, ticktime=i + 1) for i, n in enumerate(sizes)])
    with torch.no_grad():
        for block in core.blocks:
            block.z.copy_(torch.linspace(-1, 1, block.neuron_size))
    for tick in range(500):
        core.set_active_mask([(tick + i) % 4 != 0 for i in range(4)])
        core.step(now_ms=tick * tick, order=[3, 1, 0] if tick % 2 else [0, 2, 1, 3])
        for block in core.blocks:
            assert len(block.A) == len(block.At) <= 3
            assert all(torch.isfinite(getattr(block, name)).all() for name in ("z", "a", "r", "h"))
            assert all(torch.isfinite(a).all() for a in block.A)


def test_nonfinite_calculation_rejected_without_partial_commit():
    block = Block(0, 2, [2])
    block.update(now_ms=0)
    old_z, old_a, old_r, old_h = [getattr(block, name).clone() for name in ("z", "a", "r", "h")]
    old_history = [a.clone() for a in block.A]
    with torch.no_grad():
        block.b.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        block.update(now_ms=1)
    assert list(block.At) == [0]
    for name, expected in zip(("z", "a", "r", "h"), (old_z, old_a, old_r, old_h)):
        torch.testing.assert_close(getattr(block, name), expected)
    torch.testing.assert_close(block.A[0], old_history[0])


@pytest.mark.parametrize("bad_source", [torch.ones(3), torch.ones(2, dtype=torch.float64), torch.tensor([float("nan"), 0.])])
def test_invalid_source_cannot_corrupt_history(bad_source):
    block = Block(0, 2, [2])
    with pytest.raises(ValueError):
        block.update(now_ms=0, active_z={0: bad_source})
    assert len(block.A) == len(block.At) == 0


def test_cli_starts_and_finishes_several_core_steps():
    result = subprocess.run([sys.executable, "-m", "acnt", "--core-only", "--steps", "12"], capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 12
    assert rows[-1]["tick"] == 11
    assert len({tuple(row["active"]) for row in rows}) > 1
    assert all(row["active"] == row["updated"] for row in rows)
    assert all(max(row["history_lengths"]) <= 4 for row in rows)
