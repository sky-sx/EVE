"""Count vector Write self-coverage; no behavioral training or rollout.

The current seven-Block toy supplies real per-tensor base sizes. The writer is
counted algebraically, so even an infeasible architecture is never allocated.
"""
from __future__ import annotations

import json
from pathlib import Path

from acnt.grouped_write import GroupedWriteKernel


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def vector_budget(base_sizes, neurons, group_size):
    if neurons < 1 or group_size < 1 or not base_sizes or any(s < 1 for s in base_sizes):
        raise ValueError('positive sizes required')
    base = sum(base_sizes)
    row = dict(neurons=neurons, group_size=group_size, base_parameters=base,
               writer_parameters_per_output=neurons+1)
    surplus = group_size-neurons-1
    if surplus <= 0:
        return dict(row, finite_self_coverage=False,
                    reason='nonpositive per-output surplus with a nonempty base')
    base_groups = sum(ceil_div(s, group_size) for s in base_sizes)
    upper = ceil_div(group_size*(base_groups+2), surplus)
    groups, iterations = base_groups, 0
    while True:
        following = base_groups+ceil_div(neurons*groups, group_size)+ceil_div(groups, group_size)
        assert groups <= following <= upper
        if following == groups:
            break
        groups = following
        iterations += 1
    writer = (neurons+1)*groups
    assert group_size*groups >= base+writer
    return dict(row, finite_self_coverage=True,
                ideal_min_outputs=ceil_div(base, surplus), base_tensor_groups=base_groups,
                actual_outputs=groups, writer_parameters=writer,
                total_parameters=base+writer, allocated_group_slots=group_size*groups,
                unused_group_slots=group_size*groups-base-writer, iterations=iterations)


def main():
    rows = []
    for neurons in (2, 4, 8):
        kernel = GroupedWriteKernel(width=neurons, hold=3, group_size=32)
        sizes = [p.numel() for name, p in kernel.body.named_parameters()
                 if not name.startswith('write.')]
        # Seven fully connected Blocks, private NLM hidden=2/history=3,
        # Ear=4 samples, one-output Hand hidden=8, feedback=4 inputs.
        assert sum(sizes) == 98*neurons**2+364*neurons+17
        for group_size in (8, 9, 10, 16, 32):
            rows.append(vector_budget(sizes, neurons, group_size))
    result = dict(kind='structural parameter count, not a learning experiment',
                  base_formula='98*N^2 + 364*N + 17',
                  writer_formula='(N+1)*M',
                  ideal_condition='(g-N-1)*M >= P0(N)',
                  tensor_fixed_point='M = A + ceil(N*M/g) + ceil(M/g)',
                  rows=rows)
    path = Path('reports/write_vector_budget_2026-10-01.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
