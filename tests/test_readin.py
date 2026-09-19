import pytest
import torch


@pytest.mark.parametrize("organ,shape", [("eye", (3, 1080, 1920)), ("ear", (1, 32))])
def test_new_input_overrides_inactive_and_clock_once(make_runtime, organ, shape):
    runtime = make_runtime(ticktime=1.0)
    block_id = runtime.organ_blocks[organ]
    block = runtime.core.blocks[block_id]
    runtime.core.set_active(block_id, False)
    runtime.update_blocks(now_ms=0, readins={organ: torch.ones(shape)})
    assert list(block.At) == [0]
    assert not block.active
    assert torch.isfinite(block.o).all()
    # A new input forces an early update only 1ms into a one-second period.
    result = runtime.update_blocks(now_ms=1, readins={organ: torch.zeros(shape)})
    assert block_id in result
    assert list(block.At) == [0, 1]
    assert not block.active
    frozen_z = block.z.clone()
    result = runtime.update_blocks(now_ms=2)
    assert block_id not in result
    assert list(block.At) == [0, 1]
    torch.testing.assert_close(block.z, frozen_z)


def test_forced_eye_is_available_to_other_blocks_in_same_round(make_runtime):
    runtime = make_runtime()
    runtime.core.set_active_mask([False, False, True, False, False, False, False, False])
    runtime.set_goodness_active(False)
    result = runtime.update_blocks(now_ms=0, readins={"eye": torch.ones(3, 1080, 1920)})
    eye, target = runtime.core.blocks[0], runtime.core.blocks[2]
    assert set(result) == {0, 2, 5}
    torch.testing.assert_close(target.r, target.b + target.W_ij[0] @ eye.z)
    assert eye.z.norm() > 0
    assert not eye.active


def test_two_new_readins_each_update_once_and_restore_base_active(make_runtime):
    runtime = make_runtime()
    runtime.core.set_active(0, False)
    runtime.core.set_active(1, True)
    runtime.update_blocks(now_ms=0, readins={"eye": torch.zeros(3, 1080, 1920), "ear": torch.ones(1, 32)})
    assert list(runtime.core.blocks[0].At) == [0]
    assert list(runtime.core.blocks[1].At) == [0]
    assert not runtime.core.blocks[0].active
    assert runtime.core.blocks[1].active
