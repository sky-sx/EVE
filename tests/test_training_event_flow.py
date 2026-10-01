from pathlib import Path
import sys

import torch
from acnt.event_flow import EventFlowBlock


def test_semigroup_fixed_target_lambda():
    torch.manual_seed(0)
    z=torch.randn(5,dtype=torch.float64)
    target=torch.randn(5,dtype=torch.float64)
    lam=torch.rand(5,dtype=torch.float64)+0.1
    a=EventFlowBlock.flow_tensor(z,target,lam,0.17)
    ab=EventFlowBlock.flow_tensor(a,target,lam,0.31)
    direct=EventFlowBlock.flow_tensor(z,target,lam,0.48)
    assert torch.allclose(ab,direct,atol=1e-12,rtol=1e-12)


def test_history_does_not_shift_on_materialize():
    b=EventFlowBlock(2,4,1,2,dtype=torch.float64)
    s=b.initial_state(); H=s.H.clone()
    s=b.materialize(s,0.5)
    assert torch.equal(s.H,H)


def test_history_shifts_once_on_semantic_event():
    b=EventFlowBlock(2,4,1,2,dtype=torch.float64)
    s=b.initial_state();
    s.H.copy_(torch.arange(8,dtype=torch.float64).reshape(4,2))
    s.z.fill_(0.4)
    before=b.materialize(s,0.2)
    expected=torch.cat([before.H[1:],before.z.unsqueeze(0)])
    s=b.semantic_event(s,0.2,torch.zeros(2,dtype=torch.float64),torch.ones(1,dtype=torch.float64))
    assert torch.equal(s.H,expected)
    assert s.t_last==0.2
