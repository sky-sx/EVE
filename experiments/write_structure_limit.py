"""Exact structural counts, never allocate a large writer or train it."""
import json
from pathlib import Path

from acnt.grouped_write import GroupedWriteKernel
from acnt.grouped_write_independent import independent_group_count
from experiments.write_vector_budget import vector_budget


def main():
    rows=[]
    for n in (4,8,16,24,28,29,30,31):
        base=GroupedWriteKernel(width=n,hold=3,group_size=32)
        sizes=[p.numel() for name,p in base.body.named_parameters() if not name.startswith('write.')]
        row=vector_budget(sizes,n,32)
        row['core_neurons']=7*n
        if row['finite_self_coverage']:
            assert row['actual_outputs']==independent_group_count(row['base_tensor_groups'],n,32)
            p,h=row['total_parameters'],28*n
            row['reference_persistent_tensor_bytes']=4*(h+h*p+7*p+8*n)+p
        rows.append(row)
    result=dict(kind='structural count only, not a scaled learning run',
        scope='seven equal-width Blocks, direct Linear Write, g=32, minimal fixed-point F count',
        maximum_width=30,rows=rows,
        fixed143=dict(global_slots=4576,writer_parameters=715,global_other_capacity=3861,
            tensor_writer_groups=23,tensor_other_groups=120,tensor_other_capacity_bound=3840,
            tensor_total_capacity_bound=4555,current_other_parameters=3041,
            current_other_tail_slots=799,current_writer_tail_slots=21),
        scalable_scope='if F and Block count may grow with fixed Write input width, no finite coverage-only parameter ceiling')
    path=Path('reports/write_structure_limit_2026-10-01.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
