"""G3 current state: async runtime semantics PASS; developmental robustness pending.

This script validates eager-vs-lazy continuous-time materialization. It then
prints a birth-sensitivity diagnostic rather than claiming G3 training PASS.
"""
from pathlib import Path
import sys

import torch
from acnt.event_flow import EventFlowBlock


def main():
    torch.manual_seed(4); dtype=torch.float64
    blocks=[EventFlowBlock(2,M,1,2,dtype=dtype) for M in (2,4,8,16)]
    for b in blocks: b.neutral_initialize_()
    states=[b.initial_state() for b in blocks]
    # Give each Block one semantic event at t=0 to set its flow parameters.
    for i,b in enumerate(blocks):
        states[i]=b.semantic_event(states[i],0.0,torch.zeros(2,dtype=dtype),torch.tensor([0.2*(i+1)],dtype=dtype))

    final_t=0.8
    eager=[s.clone() for s in states]
    # Eager scheduler materializes everyone at every global scheduler tick.
    ticks=sorted(set([k*0.02 for k in range(1,41)] + [k*0.05 for k in range(1,17)] + [k*.1 for k in range(1,9)] + [k*.2 for k in range(1,5)]))
    for t in ticks:
        eager=[b.materialize(s,t) for b,s in zip(blocks,eager)]

    # Lazy scheduler touches each Block only at final read.
    lazy=[b.materialize(s,final_t) for b,s in zip(blocks,states)]
    errs=[float(torch.max(torch.abs(a.z-l.z)).detach()) for a,l in zip(eager,lazy)]
    print("eager vs lazy state max errors:", [f"{x:.3e}" for x in errs])
    assert max(errs)<1e-12

    # Birth transmissibility diagnostic: not a pass condition yet.
    z0=torch.cat([s.z for s in states]).detach().requires_grad_(True)
    # toy readout with a chain-like attenuation to show what G3 must monitor
    y=(z0[:2].sum()*1e-3 + z0[2:4].sum()*1e-2 + z0[4:6].sum()*1e-1 + z0[6:].sum())
    g=torch.autograd.grad(y,z0)[0]
    print("example distant/local sensitivity ratio:", float(torch.linalg.vector_norm(g[:2])/(torch.linalg.vector_norm(g[6:])+1e-30)))
    print("G3 runtime semantics: PASS; G3 developmental robustness: PENDING")

if __name__=='__main__': main()
