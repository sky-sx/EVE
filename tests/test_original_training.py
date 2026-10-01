import copy
import pytest
import torch
from acnt import OriginalTrainingRuntime, GoodnessTrainer

def test_original_forward_parity_with_async_forced_readins(make_runtime):
    runtime=make_runtime()
    reference=copy.deepcopy(runtime)
    model=OriginalTrainingRuntime(runtime)
    masks=[[True]*8,[False,True,True,False,True,True,False,True],[True]*8]
    for t,mask in zip([0,31,250],masks):
        reference.core.set_active_mask(mask)
        reference.core.set_active(reference.organ_blocks["route"],True)
        model.set_active_mask(mask)
        inputs={"ear":torch.linspace(-1,1,32).reshape(1,32)}
        expected=reference.update_blocks(now_ms=t,readins=inputs)
        actual=model.step(now_ms=t,readins=inputs)
        assert set(actual)==set(expected)
        for block,state in zip(reference.core.blocks,model.states):
            torch.testing.assert_close(block.z,state.z,rtol=0,atol=0)
            torch.testing.assert_close(block.r,state.r,rtol=0,atol=0)
            assert tuple(block.At)==state.times
            for a,b in zip(block.A,state.history): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for name in ["hand","speak","goodness","route"]:
            torch.testing.assert_close(reference.decode_readout(name),model.decode(name),rtol=0,atol=0)

def test_gradients_cross_input_core_history_output(make_runtime):
    runtime=make_runtime(neuron_size=4,ticktime=1)
    model=OriginalTrainingRuntime(runtime)
    for t in range(4):
        model.step(now_ms=t,readins={"ear":torch.randn(1,32)} if t==0 else None)
    loss=model.decode("hand")[0].square()
    loss.backward()
    parameters=[runtime.adapters["ear"].linear.weight,
                runtime.core.blocks[1].nlm.weight1,
                runtime.core.blocks[2].W_ij[1],
                runtime.core.blocks[2].nlm.weight1,
                runtime.adapters["hand"].network[2].weight]
    for p in parameters:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.norm()>0
    model.truncate()
    for s in model.states:
        assert s.z.grad_fn is None
        assert all(a.grad_fn is None for a in s.history)

def test_eye_adapter_has_real_differentiable_path(make_runtime):
    runtime=make_runtime(neuron_size=3,ticktime=1)
    model=OriginalTrainingRuntime(runtime)
    model.step(now_ms=0,readins={"eye":torch.rand(3,1080,1920)})
    model.step(now_ms=1)
    model.decode("hand")[0].backward()
    for p in [runtime.adapters["eye"].features[0].weight,runtime.adapters["eye"].projection.weight]:
        assert p.grad is not None and p.grad.norm()>0

def test_finite_difference_of_input_adapter_through_history(make_runtime):
    runtime=make_runtime(neuron_size=4,ticktime=1)
    model=OriginalTrainingRuntime(runtime)
    audio=torch.linspace(-.5,.5,32).reshape(1,32)
    def objective():
        model.reset()
        for t in range(3): model.step(now_ms=t,readins={"ear":audio} if t==0 else None)
        return model.decode("hand")[0]
    p=runtime.adapters["ear"].linear.weight
    objective().backward()
    index=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape))
    analytic=p.grad[index].item()
    old=p[index].item(); eps=1e-3
    with torch.no_grad():
        p[index]=old+eps; plus=objective().item()
        p[index]=old-eps; minus=objective().item()
        p[index]=old
    numeric=(plus-minus)/(2*eps)
    assert analytic==pytest.approx(numeric,rel=.05,abs=1e-5)

def test_route_credit_excludes_role_overrides(make_runtime):
    runtime=make_runtime()
    model=OriginalTrainingRuntime(runtime)
    sample=model.sample_route(generator=torch.Generator().manual_seed(9))
    raw=model.decode("route")
    grad=torch.autograd.grad(sample.log_prob,runtime.adapters["route"].linear.bias)[0]
    rid=runtime.organ_blocks["route"]; gid=runtime.organ_blocks["goodness"]
    assert grad[rid]==0 and grad[gid]==0
    assert sample.action[rid]==1
    assert sample.action[gid]==float(runtime.goodness_active)

def test_optimizer_updates_complete_path_and_excludes_goodness(make_runtime):
    runtime=make_runtime(neuron_size=4,ticktime=1)
    model=OriginalTrainingRuntime(runtime)
    trainer=GoodnessTrainer(model)
    before={n:p.detach().clone() for n,p in runtime.named_parameters()}
    for t in range(3): model.step(now_ms=t,readins={"ear":torch.randn(1,32)} if t==0 else None)
    trainer.update_loss(model.decode("hand")[0].square())
    changed=[n for n,p in runtime.named_parameters() if not torch.equal(p,before[n])]
    assert any(n.startswith("adapters.ear.") for n in changed)
    assert any(n.startswith("core.blocks.1.") for n in changed)
    assert any(n.startswith("core.blocks.2.") for n in changed)
    assert any(n.startswith("adapters.hand.") for n in changed)
    assert not any(n.startswith(("adapters.goodness.","core.blocks.4.")) for n in changed)
    model.commit()
    for b,s in zip(runtime.core.blocks,model.states): torch.testing.assert_close(b.z,s.z)

def test_window_guard_and_timestamp_validation(make_runtime):
    model=OriginalTrainingRuntime(make_runtime(),max_window=1)
    model.step(now_ms=0)
    with pytest.raises(RuntimeError): model.step(now_ms=250)
    model.truncate(); model.step(now_ms=250)
    with pytest.raises(ValueError): model.step(now_ms=249)
    assert GoodnessTrainer.returns([0,1],[0,100],discount_tau_ms=100)[0]==pytest.approx(torch.exp(torch.tensor(-1.)).item())
    with pytest.raises(ValueError): GoodnessTrainer.returns([2],[0])


def test_goodness_calibration_only_updates_its_own_parameters(make_runtime):
    runtime=make_runtime(neuron_size=4,ticktime=1)
    model=OriginalTrainingRuntime(runtime); trainer=GoodnessTrainer(model)
    for t in range(3): model.step(now_ms=t,readins={"ear":torch.randn(1,32)})
    before={n:p.detach().clone() for n,p in runtime.named_parameters()}
    trainer.calibrate_goodness(.8)
    changed=[n for n,p in runtime.named_parameters() if not torch.equal(p,before[n])]
    assert changed
    assert all(n.startswith(("adapters.goodness.","core.blocks.4.")) for n in changed)

def test_checkpoint_restores_history_optimizer_and_action_rng(make_runtime,tmp_path):
    model=OriginalTrainingRuntime(make_runtime(neuron_size=4,ticktime=1))
    trainer=GoodnessTrainer(model)
    for t in range(3): model.step(now_ms=t,readins={"ear":torch.randn(1,32)})
    generator=torch.Generator().manual_seed(42)
    sample=model.bernoulli(model.decode("hand")[:1],generator=generator)
    trainer.update([([sample],[1.],[2])])
    checkpoint=tmp_path/"training.pt"
    trainer.save_checkpoint(checkpoint,generator=generator)
    restored=OriginalTrainingRuntime(make_runtime(neuron_size=4,ticktime=1))
    other=GoodnessTrainer(restored); rng=torch.Generator()
    other.load_checkpoint(checkpoint,generator=rng)
    assert other.baseline==trainer.baseline
    assert torch.equal(rng.get_state(),generator.get_state())
    for a,b in zip(model.states,restored.states):
        assert a.times==b.times
        torch.testing.assert_close(a.z,b.z,atol=0,rtol=0)
        for x,y in zip(a.history,b.history): torch.testing.assert_close(x,y,atol=0,rtol=0)
    # Same next transition and sampled action, then same optimizer update.
    for m in (model,restored): m.step(now_ms=3)
    s=model.bernoulli(model.decode("hand")[:1],generator=generator)
    r=restored.bernoulli(restored.decode("hand")[:1],generator=rng)
    assert torch.equal(s.action,r.action)
    trainer.update([([s],[1.],[3])]); other.update([([r],[1.],[3])])
    for a,b in zip(model.parameters(),restored.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_continuous_hand_and_speak_have_policy_credit(make_runtime):
    runtime=make_runtime(neuron_size=4,ticktime=1)
    model=OriginalTrainingRuntime(runtime)
    for t in range(2): model.step(now_ms=t,readins={"ear":torch.randn(1,32)})
    hand=model.sample_hand(continuous_std=.2)
    speak=model.sample_speak(std=.2)
    (hand.log_prob+speak.log_prob).backward()
    assert runtime.adapters["speak"].linear.weight.grad.norm()>0
    assert runtime.adapters["hand"].network[2].weight.grad[-2:].norm()>0
