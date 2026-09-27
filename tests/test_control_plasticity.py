import torch


def test_learning_rng_does_not_consume_action_rng(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    action = torch.Generator().manual_seed(97)
    before = action.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    learner.apply_goodness(.5, now_ms=0)
    learner.begin_trial()
    learner.apply_goodness(.4, now_ms=10)
    assert torch.equal(action.get_state(), before)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_hand_route_forward_does_not_change_learning_state(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity()
    d = {k:v.clone() for k,v in learner.delta_w.items()}
    rng = learner.rng.getstate()
    runtime.generate_hand(generator=torch.Generator().manual_seed(9))
    runtime.generate_route(generator=torch.Generator().manual_seed(10))
    assert learner.rng.getstate() == rng and learner.pending_indices is None
    assert all(torch.equal(v,d[k]) for k,v in learner.delta_w.items())
    assert all(not module._forward_hooks for module in runtime.adapters.modules())
