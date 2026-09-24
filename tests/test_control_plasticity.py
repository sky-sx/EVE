import torch
from torch import nn
from acnt.control import sample_discrete
from acnt.plasticity import Plasticity


def test_terminal_control_uses_bernoulli_score():
    q = torch.tensor(
        [0.2, -0.1],
        dtype=torch.float32,
    )

    generator = torch.Generator().manual_seed(97)

    signal = sample_discrete(
        q,
        tau=0.25,
        threshold=0.1,
        generator=generator,
    )

    expected = (
        signal.a.to(signal.p.dtype)
        - signal.p
    ) / signal.tau

    from acnt.control import discrete_score

    torch.testing.assert_close(
        discrete_score(signal),
        expected,
    )


def test_terminal_observer_records_local_state_only():
    adapter=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2))
    group={f"adapter.{name}":p for name,p in adapter.named_parameters()}
    learner=Plasticity({0:group},None)
    learner.attach_adapters({"hand":adapter})
    with torch.no_grad():
        q=adapter(torch.ones(3))
        signal=sample_discrete(q,tau=.25,generator=torch.Generator().manual_seed(4))
        learner.observe_control("hand",adapter,signal,now_ms=0)
    terminal=adapter[-1]
    e=learner.traces[id(terminal.weight)]
    assert e.abs().sum()>0
    assert torch.isfinite(e).all()
    assert all(p.grad is None for p in adapter.parameters())


def test_discrete_state_and_update_shrink_across_idle_delays():
    outcomes=[]
    for delay in (250,500,1000):
        p=nn.Parameter(torch.ones((2,2)))
        learner=Plasticity({0:{"w":p}},None)
        pre=torch.tensor([2.,3.])
        post=torch.tensor([1.,-1.])
        learner.accumulate(p,post.unsqueeze(1)*pre.unsqueeze(0),now_ms=0)
        before=p.clone()
        e=learner.traces[id(p)].clone()
        learner.apply_goodness(1.,now_ms=delay)
        decayed=e*torch.exp(torch.tensor(-delay/1000.))
        torch.testing.assert_close(p,before+.0005*decayed)
        torch.testing.assert_close(learner.traces[id(p)],decayed)
        outcomes.append(p.detach().clone())
    deltas=[float((x-torch.ones_like(x)).norm()) for x in outcomes]
    assert deltas[0]>deltas[1]>deltas[2]