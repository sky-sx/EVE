"""G0: local mathematical correctness / partition-gradient regression.

This current-code regression targets the post-G2 event+exp Block. The original
chat also checked the superseded full-CfC interpolation version; see
results/reported_chat_results.md.
"""
import math, os, sys
from pathlib import Path
import torch
from acnt.event_flow import EventFlowBlock


def main():
    torch.manual_seed(0)
    dtype=torch.float64
    b=EventFlowBlock(d=2, history_len=3, input_dim=1, message_dim=2, dtype=dtype)
    b.neutral_initialize_()
    s=b.initial_state()
    msg=torch.tensor([0.2,-0.1],dtype=dtype)
    x=torch.tensor([0.7],dtype=dtype)

    # Same semantic events; only scheduler slicing between events differs.
    def run(slices):
        st=b.initial_state()
        st=b.semantic_event(st,0.0,msg,x)
        # pure time passage to 0.37, optionally materialized many times
        for k in range(1,slices+1):
            st=b.materialize(st,0.37*k/slices)
        st=b.semantic_event(st,0.37,msg*0.5,x*0.0)
        for k in range(1,slices+1):
            st=b.materialize(st,0.91-(0.91-0.37)*(1-k/slices))
        return st.z

    z1=run(1); z32=run(32)
    fwd_err=float(torch.max(torch.abs(z1-z32)).detach())

    # Gradient invariance wrt scheduler slicing.
    params=[p for p in b.parameters()]
    g=[]
    for slices in (1,32):
        b.zero_grad(set_to_none=True)
        loss=run(slices).square().sum()
        loss.backward()
        g.append(torch.cat([p.grad.reshape(-1) for p in params]))
    grad_err=float(torch.max(torch.abs(g[0]-g[1])).detach())
    rel=float((torch.linalg.vector_norm(g[0]-g[1])/(torch.linalg.vector_norm(g[0])+1e-30)).detach())

    print(f"G0 current event-flow partition forward max_abs={fwd_err:.3e}")
    print(f"G0 current event-flow partition gradient max_abs={grad_err:.3e}, rel={rel:.3e}")
    assert fwd_err < 1e-12
    assert grad_err < 1e-11
    print("G0 smoke: PASS")

if __name__=='__main__': main()
