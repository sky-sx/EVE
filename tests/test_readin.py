import pytest
import torch


@pytest.mark.parametrize("organ,shape", [("eye", (3, 1080, 1920)), ("ear", (1, 32))])
def test_new_input_overrides_inactive_and_clock_once(make_runtime, organ, shape):
    runtime = make_runtime(ticktime=1000)
    block_id = runtime.organ_blocks[organ]
    block = runtime.core.blocks[block_id]
    runtime.core.set_active(block_id, False)
    runtime.update_blocks(now_ms=0, readins={organ: torch.ones(shape)})
    assert list(block.At) == [0]
    assert not block.active
    assert torch.count_nonzero(block.o) == 0
    assert not block.o.requires_grad
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


def test_forced_eye_new_state_is_visible_only_next_round(make_runtime):
    runtime = make_runtime()
    runtime.core.set_active_mask([False, False, True, False, False, False, False, False])
    runtime.set_goodness_active(False)
    result = runtime.update_blocks(now_ms=0, readins={"eye": torch.ones(3, 1080, 1920)})
    eye, target = runtime.core.blocks[0], runtime.core.blocks[2]
    assert set(result) == {0, 2, 5}
    torch.testing.assert_close(target.r, target.b)
    old_sources = {i: runtime.core.blocks[i].z.clone() for i in (0, 2, 5)}
    # Admit the eye as an active source on the next propagation round.
    runtime.core.set_active(0, True)
    runtime.update_blocks(now_ms=250)
    torch.testing.assert_close(target.r, target.b + sum(target.W_ij[i] @ z for i, z in old_sources.items()))
    assert eye.z.norm() > 0
    assert eye.active


def test_two_new_readins_each_update_once_and_restore_base_active(make_runtime):
    runtime = make_runtime()
    runtime.core.set_active(0, False)
    runtime.core.set_active(1, True)
    runtime.update_blocks(now_ms=0, readins={"eye": torch.zeros(3, 1080, 1920), "ear": torch.ones(1, 32)})
    assert list(runtime.core.blocks[0].At) == [0]
    assert list(runtime.core.blocks[1].At) == [0]
    assert not runtime.core.blocks[0].active
    assert runtime.core.blocks[1].active


def test_readin_o_is_one_pulse_and_zero_sample_is_a_real_event(make_runtime):
    runtime = make_runtime()
    block = runtime.core.blocks[1]
    with torch.no_grad():
        runtime.adapters["ear"].linear.weight.zero_()
        runtime.adapters["ear"].linear.bias.fill_(0.7)
    calls = []
    hook = runtime.adapters["ear"].register_forward_hook(lambda *args: calls.append(1))
    runtime.update_blocks(now_ms=0, readins={"ear": torch.zeros(1, 32)})
    torch.testing.assert_close(block.r, block.b + 0.7)
    assert torch.count_nonzero(block.o) == 0 and len(calls) == 1
    sources = runtime.core.source_snapshot()
    runtime.update_blocks(now_ms=250)
    torch.testing.assert_close(block.r, block.b + sum(block.W_ij[i] @ z for i, z in sources.items()))
    assert torch.count_nonzero(block.o) == 0 and len(calls) == 1
    runtime.update_blocks(now_ms=251, readins={"ear": torch.zeros(1, 32)})
    assert list(block.At) == [0, 250, 251]
    assert torch.count_nonzero(block.o) == 0 and len(calls) == 2
    hook.remove()


def test_runtime_forced_readin_and_local_tags_ignore_visitation_order(make_runtime):
    left, right = make_runtime(), make_runtime()
    for runtime in (left, right):
        runtime.enable_plasticity()
        runtime.core.set_active(1, False)
    for now in (0, 1, 251):
        for runtime, order in ((left, list(range(8))), (right, list(reversed(range(8))))):
            runtime.update_blocks(now_ms=now, readins={"ear": torch.ones(1, 32)}, order=order)
            runtime.generate_hand(now_ms=now, generator=torch.Generator().manual_seed(31))
        for a, b in zip(left.core.blocks, right.core.blocks):
            for key in ("z", "r", "a"):
                torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
            assert list(a.At) == list(b.At)
        for i in range(8):
            for name, parameter in left.plasticity.groups[i].items():
                other = right.plasticity.groups[i][name]
                torch.testing.assert_close(left.plasticity.traces[id(parameter)], right.plasticity.traces[id(other)], rtol=0, atol=0)
