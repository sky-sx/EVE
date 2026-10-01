"""G2: physical-time observability + scheduler partition invariance.

A tiny supervised oracle is used here to isolate architecture/time semantics.
The chat also ran Bernoulli-action + scalar-Goodness versions; those reported
results are in results/reported_chat_results.md.
"""
from pathlib import Path
import sys

import torch
from acnt.event_flow import EventFlowBlock


def partition_invariance():
    torch.manual_seed(1)
    b=EventFlowBlock(d=3,history_len=4,input_dim=1,message_dim=3,dtype=torch.float64)
    b.neutral_initialize_()
    msg=torch.tensor([.1,-.2,.05],dtype=torch.float64)
    x=torch.tensor([.4],dtype=torch.float64)
    def run(n):
        s=b.initial_state(); s=b.semantic_event(s,0.0,msg,x)
        for k in range(1,n+1): s=b.materialize(s,0.73*k/n)
        return s.z
    ref=run(1)
    ns=[2,4,8,16,32,128]
    errs=[float(torch.max(torch.abs(run(n)-ref)).detach()) for n in ns]
    print("partition errors:", {n:f"{e:.3e}" for n,e in zip(ns,errs)})
    assert max(errs)<1e-12


def time_observability():
    torch.manual_seed(2)
    dtype=torch.float64
    b=EventFlowBlock(d=4,history_len=3,input_dim=1,message_dim=4,dtype=dtype)
    b.neutral_initialize_(gain=0.25)
    out=torch.nn.Linear(4,1,dtype=dtype)
    opt=torch.optim.Adam(list(b.parameters())+list(out.parameters()),lr=1.5e-2)

    batch=256
    for step in range(350):
        T=0.2+0.6*torch.rand(batch,dtype=dtype)
        ys=(T<0.5).to(dtype)
        s=b.initial_state(batch_shape=(batch,))
        msg=torch.zeros(batch,4,dtype=dtype)
        pulse=torch.ones(batch,1,dtype=dtype)
        s=b.semantic_event(s,0.0,msg,pulse)
        zT=b.flow_tensor(s.z,s.target,s.lam,T)
        logits=out(zT).squeeze(-1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ys)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        T=0.2+0.6*torch.rand(3000,dtype=dtype); ys=(T<.5)
        s=b.initial_state(batch_shape=(T.numel(),))
        msg=torch.zeros(T.numel(),4,dtype=dtype)
        pulse=torch.ones(T.numel(),1,dtype=dtype)
        s=b.semantic_event(s,0.0,msg,pulse)
        pred=out(b.flow_tensor(s.z,s.target,s.lam,T)).squeeze(-1)>0
        fakeT=torch.full_like(T,0.4)
        fake=out(b.flow_tensor(s.z,s.target,s.lam,fakeT)).squeeze(-1)>0
        acc=float((pred==ys).double().mean())
        fake_acc=float((fake==ys).double().mean())
    print(f"physical-time acc={acc:.4f}; fake-dt acc={fake_acc:.4f}")
    assert acc>0.85 and abs(fake_acc-0.5)<0.08


def main():
    partition_invariance(); time_observability(); print("G2 smoke: PASS")
if __name__=='__main__': main()
