import pytest
import torch
from acnt import EventFlowBlock, EventFlowCore, BlockLowRankRTRL
from acnt.rtrl import split_slices

def test_time_propagation_partition_and_rng():
    ss=split_slices([2,2]); ps=split_slices([3,3])
    a=BlockLowRankRTRL(ss,ps,2); b=BlockLowRankRTRL(ss,ps,2)
    torch.manual_seed(7)
    a.E.normal_(); a.U.normal_(); a.V.normal_()
    b.E.copy_(a.E); b.U.copy_(a.U); b.V.copy_(a.V)
    before=a.generator.get_state().clone()
    rates=torch.tensor([0.2,0.7,0.4,0.8],dtype=torch.float64)
    a.propagate_time(torch.diag(torch.exp(-rates*0.9)))
    for _ in range(30):
        b.propagate_time(torch.diag(torch.exp(-rates*0.03)))
    torch.testing.assert_close(a.reconstruct(),b.reconstruct(),atol=1e-12,rtol=1e-12)
    assert torch.equal(before,a.generator.get_state())
    assert torch.equal(before,b.generator.get_state())

@pytest.mark.parametrize("time",[-0.1,float("nan"),float("inf")])
def test_invalid_time_rejected(time):
    b=EventFlowBlock(2,4)
    with pytest.raises(ValueError): b.materialize(b.initial_state(),time)

@pytest.mark.parametrize("rank",[0,-1,1.5,True])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError): BlockLowRankRTRL(split_slices([2]),split_slices([2]),rank)

def test_initial_state_tracks_model_dtype():
    b=EventFlowBlock(2,4).float()
    assert b.initial_state().z.dtype == torch.float32

def test_scheduler_ticks_only_flow_selected_events_shift_history():
    core=EventFlowCore([EventFlowBlock(2,4),EventFlowBlock(2,4)])
    core.states[0].z.fill_(0.4)
    before=[s.H.clone() for s in core.states]
    core.materialize(0.5)
    for s,h in zip(core.states,before): assert torch.equal(s.H,h)
    core.semantic_events(0.7,{0:(torch.zeros(2,dtype=torch.float64),torch.ones(1,dtype=torch.float64))})
    assert not torch.equal(core.states[0].H,before[0])
    assert torch.equal(core.states[1].H,before[1])
    committed=core.states
    with pytest.raises(ValueError): core.semantic_events(0.9,{5:(None,None)})
    assert core.states is committed


def test_toy_sequence_rejects_silent_truncation():
    from acnt.toy_chain import FourBlockEventFlow
    with pytest.raises(ValueError): FourBlockEventFlow(history_lengths=(2,4))
    model=FourBlockEventFlow()
    with pytest.raises(ValueError): model.forward_event_sequence([0.0],[])
