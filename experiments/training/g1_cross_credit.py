"""G1 mechanism/efficient-credit demonstration.

This smoke checks the *statistical* property expected from a UORO-style
streaming cross-credit estimator: individual estimates are noisy, while the
mean of independent estimators should align with an exact-RTRL oracle.
"""
from pathlib import Path
import sys

import torch
from acnt.rtrl import ExactRTRL, BlockLowRankRTRL, split_slices


def make_sequence(seed=3, steps=30, Bn=4, d=2, pb=3):
    torch.manual_seed(seed); dtype=torch.float64
    S=Bn*d; P=Bn*pb
    seq=[]
    for _ in range(steps):
        A=torch.eye(S,dtype=dtype)*0.78
        for i in range(Bn-1):
            A[slice((i+1)*d,(i+2)*d), slice(i*d,(i+1)*d)] += 0.18*torch.eye(d,dtype=dtype)
        A += 0.01*torch.randn(S,S,dtype=dtype)
        Bmat=torch.zeros(S,P,dtype=dtype)
        for i in range(Bn):
            Bmat[slice(i*d,(i+1)*d), slice(i*pb,(i+1)*pb)] = 0.1*torch.randn(d,pb,dtype=dtype)
        seq.append((A,Bmat))
    score=torch.randn(S,dtype=dtype)
    return seq,score


def main():
    dtype=torch.float64; Bn=4; d=2; pb=3
    S=Bn*d; P=Bn*pb
    ss=split_slices([d]*Bn); ps=split_slices([pb]*Bn)
    seq,score=make_sequence()

    exact=ExactRTRL.zeros(S,P,dtype=dtype)
    for A,Bmat in seq: exact.step(A,Bmat)
    gx=score@exact.J

    grads=[]; one_cos=[]
    for seed in range(128):
        low=BlockLowRankRTRL(ss,ps,rank=4,dtype=dtype,seed=seed)
        for A,Bmat in seq: low.step(A,Bmat)
        gl=score@low.reconstruct()
        grads.append(gl)
        c=float(torch.dot(gx,gl)/(torch.linalg.vector_norm(gx)*torch.linalg.vector_norm(gl)+1e-30))
        one_cos.append(c)
    gmean=torch.stack(grads).mean(0)
    cos=float(torch.dot(gx,gmean)/(torch.linalg.vector_norm(gx)*torch.linalg.vector_norm(gmean)+1e-30))
    rel=float(torch.linalg.vector_norm(gmean-gx)/(torch.linalg.vector_norm(gx)+1e-30))
    print(f"single-estimator cosine: mean={sum(one_cos)/len(one_cos):.4f}, min={min(one_cos):.4f}, max={max(one_cos):.4f}")
    print(f"mean of 128 rank-4 estimators vs exact: cosine={cos:.6f}, relative_error={rel:.6f}")
    assert cos > 0.95
    assert rel < 0.35
    print("G1 streaming low-rank statistical smoke: PASS")

if __name__=='__main__': main()
