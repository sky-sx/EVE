import pytest
import torch

from acnt import Runtime


@pytest.mark.parametrize("organ,shape", [("eye", (3, 1080, 1920)), ("ear", (1, 32))])
def test_external_input_reaches_own_block_update(make_runtime, organ, shape):
    runtime = make_runtime()
    external = torch.rand(shape)
    o = runtime.encode_readin(organ, external)
    block_id = runtime.organ_blocks[organ]
    block = runtime.core.blocks[block_id]
    assert block.o is o
    runtime.core.update_block(block_id, now_ms=0)
    torch.testing.assert_close(block.r, block.b + o)
    assert len(block.A) == len(block.At) == 1
    assert block.z.shape == (3,)
    assert torch.isfinite(block.z).all()


@pytest.mark.parametrize("organ,expected_size", [("hand", 87), ("speak", 30), ("goodness", 1), ("route", 8)])
def test_each_readout_reads_its_own_unique_block(make_runtime, organ, expected_size):
    runtime = make_runtime()
    own = runtime.core.blocks[runtime.organ_blocks[organ]]
    with torch.no_grad():
        own.z.copy_(torch.tensor([1., -1., 2.]))
    expected = runtime.adapters[organ](own.z)
    output = runtime.decode_readout(organ)
    assert output.shape == (expected_size,)
    torch.testing.assert_close(output, expected)
    assert len(set(runtime.organ_blocks.values())) == 6


def test_shared_organ_block_rejected(make_runtime):
    runtime = make_runtime()
    invalid = dict(runtime.organ_blocks)
    invalid["ear"] = invalid["eye"]
    with pytest.raises(ValueError, match="different Block"):
        Runtime(runtime.core, runtime.adapters, invalid)


def test_readout_cannot_reuse_readin_block(make_runtime):
    runtime = make_runtime()
    invalid = dict(runtime.organ_blocks)
    invalid["eye"], invalid["hand"] = invalid["hand"], invalid["eye"]
    with pytest.raises(ValueError, match="o vector"):
        Runtime(runtime.core, runtime.adapters, invalid)


def test_core_rejects_adapter_batch_without_overwriting_o(make_runtime):
    runtime = make_runtime()
    block = runtime.core.blocks[runtime.organ_blocks["ear"]]
    before = block.o.clone()
    with pytest.raises(ValueError, match="shape"):
        runtime.encode_readin("ear", torch.ones(2, 1, 32))
    torch.testing.assert_close(block.o, before)


def test_unbound_ordinary_block_cannot_have_readin_state(make_runtime):
    runtime = make_runtime()
    runtime.core.blocks[6].o = torch.zeros(6)
    with pytest.raises(ValueError, match="only bound eye/ear"):
        Runtime(runtime.core, runtime.adapters, runtime.organ_blocks)
