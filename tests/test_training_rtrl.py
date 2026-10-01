from pathlib import Path
import sys

import torch
from acnt.rtrl import ExactRTRL, BlockLowRankRTRL, split_slices


def test_low_rank_shapes_and_finite():
    dtype=torch.float64
    ss=split_slices([2,2]); ps=split_slices([3,3])
    lr=BlockLowRankRTRL(ss,ps,rank=2,dtype=dtype,seed=0)
    A=torch.eye(4,dtype=dtype)*.8
    A[2:,0:2]=.1*torch.eye(2,dtype=dtype)
    B=torch.zeros(4,6,dtype=dtype); B[0:2,0:3]=.1; B[2:4,3:6]=.1
    for _ in range(5): lr.step(A,B)
    J=lr.reconstruct()
    assert J.shape==(4,6)
    assert torch.isfinite(J).all()
