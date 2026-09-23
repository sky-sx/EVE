import torch
from torch import nn
from acnt.control import sample_discrete
from acnt.plasticity import CorrelationRule, LocalEvent, Plasticity


def test_terminal_control_uses_actual_action_noise_threshold():
    pre=torch.tensor([2.,-1.])
    q=torch.tensor([.2,-.1])
    signal=sample_discrete(q,tau=.25,threshold=.1,generator=torch.Generator().manual_seed(97))
    event=LocalEvent(pre,q,0,"control",action=signal.a,noise=signal.noise,
                     threshold=signal.threshold,control_count=2)
    value=CorrelationRule().F_e(event)
    drive=torch.where(signal.a,1.,-1.)*(q+signal.noise-signal.threshold).abs().clamp(max=1)
    torch.testing.assert_close(value,drive[:,None]*pre[None,:])


def test_terminal_observer_records_local_state_only():
    adapter=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2))
    group={f"adapter.{name}":p for name,p in adapter.named_parameters()}
    learner=Plasticity({0:group},None)
    learner.attach_adapters({"hand":adapter})
    with torch.no_grad():
        learner.set_event_time(0)
        q=adapter(torch.ones(3))
        signal=sample_discrete(q,tau=.25,generator=torch.Generator().manual_seed(4))
        learner.observe_control(adapter,signal,now_ms=0)
    terminal=adapter[-1]
    e=learner.states[id(terminal.weight)]
    assert e.abs().sum()>0
    assert torch.isfinite(e).all()
    assert all(p.grad is None for p in adapter.parameters())


def test_discrete_state_and_update_shrink_across_idle_delays():
    outcomes=[]
    for delay in (250,500,1000):
        p=nn.Parameter(torch.ones((2,2)))
        learner=Plasticity({0:{"w":p}},None)
        learner.observe(p,LocalEvent(torch.tensor([2.,3.]),torch.tensor([1.,-1.]),0))
        before=p.clone()
        e=learner.states[id(p)].clone()
        learner.apply_goodness(1.,now_ms=delay)
        decayed=e*torch.exp(torch.tensor(-delay/1000.))
        torch.testing.assert_close(p,before+.0005*decayed)
        torch.testing.assert_close(learner.states[id(p)],decayed)
        outcomes.append(p.detach().clone())
    deltas=[float((x-torch.ones_like(x)).norm()) for x in outcomes]
    assert deltas[0]>deltas[1]>deltas[2]
