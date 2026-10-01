import pytest
import torch
from acnt.grouped_write_independent import IndependentGroupAdapter, IndependentGroupedWriteKernel, independent_group_count
from acnt.grouped_write import GroupedWriteLearner


def test_independent_readout_budget_includes_its_own_parameters():
    kernel = IndependentGroupedWriteKernel(group_size=32)
    assert kernel.parameter_size == 3756
    assert kernel.group_count == 143
    assert kernel.writer_indices.numel() == 715
    assert kernel.body.group_index.numel() == 3756
    assert (kernel.body.group_sizes <= 32).all()
    with pytest.raises(ValueError):
        IndependentGroupedWriteKernel(group_size=4)


def test_independent_self_writes_and_streaming_step():
    kernel = IndependentGroupedWriteKernel(width=2, hold=2, group_size=8)
    kernel.eligibility.fill_(.1)
    before = kernel.initial_state()
    new = kernel(before, tuple(() for _ in range(7)), 0, None, torch.tensor([.7, 1., 1., 1.]))
    assert torch.count_nonzero(new[kernel.neural_size:][kernel.writer_indices]) == kernel.writer_indices.numel()
    model = GroupedWriteLearner(kernel, lr=0)
    model.step(now_ms=0)
    assert model.meta.shape[1] == kernel.writer_indices.numel()


def test_vector_readout_has_separate_output_rows():
    adapter = IndependentGroupAdapter(4, 143)
    hidden = torch.tensor([.2, -.3, .4, .1])
    before = adapter(hidden).detach()
    with torch.no_grad():
        adapter.linear.bias[17] += .5
    after = adapter(hidden).detach()
    changed = torch.nonzero(before != after).flatten().tolist()
    assert changed == [17]


def test_vector_kernel_does_not_read_group_addresses_or_weight_means():
    kernel = IndependentGroupedWriteKernel(width=2, hold=2, group_size=8)
    state = kernel.initial_state()
    kernel.eligibility.fill_(.1)
    times = tuple(() for _ in range(7))
    obs = torch.tensor([.7, 1., 1., 1.])
    before = kernel(state, times, 0, None, obs)
    # Metadata used by the old scalar generator must not enter vector ReadOut.
    kernel.body.group_addresses.fill_(float('nan'))
    kernel.body.group_sizes.fill_(float('nan'))
    after = kernel(state, times, 0, None, obs)
    assert torch.equal(before, after)


def test_near_critical_group_budget_converges_beyond_old_iteration_limit():
    assert independent_group_count(120,4,32) == 143
    assert independent_group_count(2943,29,32) == 47094
    assert independent_group_count(3146,30,32) == 100672
    with pytest.raises(ValueError):
        independent_group_count(1,31,32)
