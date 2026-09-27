import math

import pytest
import torch
from torch import nn

from acnt.plasticity import Plasticity


def make_learner(seed=17, fraction=.1, **kwargs):
    parameters = {"weight": nn.Parameter(torch.zeros(7, 9)),
                  "bias": nn.Parameter(torch.zeros(37))}
    return Plasticity({0: parameters}, None, plasticity_seed=seed,
                      subset_fraction=fraction, **kwargs)


def flat(learner, directions=False):
    values = learner.delta_w.values() if directions else learner.parameters.values()
    return torch.cat([p.detach().flatten() for p in values]).clone()


def test_initialization_and_independent_reproducible_signs():
    left, right = make_learner(), make_learner()
    for key, p in left.parameters.items():
        d = left.delta_w[key]
        assert d.shape == p.shape and d.device == p.device
        assert d.dtype == torch.float32 and not d.requires_grad
        assert not isinstance(d, nn.Parameter)
        assert torch.equal(d.abs(), torch.full_like(d, .001))
    assert set(flat(left, True).sign().tolist()) == {-1., 1.}
    assert torch.equal(flat(left, True), flat(right, True))
    assert not torch.equal(flat(left, True), flat(make_learner(seed=18), True))


def test_first_goodness_only_establishes_baseline():
    learner = make_learner()
    w, d, rng = flat(learner), flat(learner, True), learner.rng.getstate()
    assert learner.begin_trial() == 0
    assert learner.apply_goodness(.6, now_ms=0) is None
    assert learner.previous_goodness == .6 and learner.pending_indices is None
    assert torch.equal(flat(learner), w) and torch.equal(flat(learner, True), d)
    assert learner.rng.getstate() == rng


@pytest.mark.parametrize('fraction', [.001, .1, 1.])
def test_global_element_subset_size_reproducibility_and_sparse_movement(fraction):
    left, right = make_learner(fraction=fraction), make_learner(fraction=fraction)
    before, delta = flat(left), flat(left, True)
    for learner in (left, right):
        learner.apply_goodness(.5, now_ms=0)
        assert learner.begin_trial() == max(1, math.floor(100 * fraction))
    indices = left.pending_indices
    assert indices == right.pending_indices and len(set(indices)) == len(indices)
    expected = before.clone()
    expected[list(indices)] += delta[list(indices)]
    assert torch.equal(flat(left), expected)
    if fraction == .1:
        assert any(i < 63 for i in indices) and any(i >= 63 for i in indices)


@pytest.mark.parametrize('goodness', [.4, .6, .8])
def test_goodness_changes_only_selected_directions_never_rolls_back(goodness):
    learner = make_learner()
    learner.apply_goodness(.6, now_ms=0)
    old_d = flat(learner, True)
    learner.begin_trial()
    indices, moved = learner.pending_indices, flat(learner)
    assert learner.apply_goodness(goodness, now_ms=1000) == pytest.approx(goodness-.6)
    expected = old_d.clone()
    if goodness < .6:
        expected[list(indices)] *= -1
    assert torch.equal(flat(learner, True), expected)
    assert torch.equal(flat(learner), moved)
    assert learner.pending_indices is None
    assert learner.delta_flip_count == (len(indices) if goodness < .6 else 0)
    assert all(p.grad is None for p in learner.parameters.values())


def test_same_element_moves_back_only_when_selected_again():
    p = nn.Parameter(torch.tensor([.5]))
    learner = Plasticity({0:{'w':p}}, None)
    learner.delta_w[id(p)].fill_(.001)
    learner.apply_goodness(.6, now_ms=0)
    learner.begin_trial()
    learner.apply_goodness(.4, now_ms=1)
    assert p.item() == pytest.approx(.501)
    learner.begin_trial()
    assert p.item() == pytest.approx(.5)


def test_unresolved_trial_cannot_be_overwritten():
    learner = make_learner()
    learner.apply_goodness(.5, now_ms=0)
    learner.begin_trial()
    selected, w, rng = learner.pending_indices, flat(learner), learner.rng.getstate()
    with pytest.raises(RuntimeError, match='unresolved'):
        learner.begin_trial()
    assert selected == learner.pending_indices and rng == learner.rng.getstate()
    assert torch.equal(w, flat(learner))
    learner.apply_goodness(.4, now_ms=100000)
    assert learner.begin_trial() > 0


def test_freeze_preserves_directions_rng_and_resumes_with_new_baseline():
    learner = make_learner()
    learner.apply_goodness(.5, now_ms=0)
    learner.begin_trial()
    w, d, rng = flat(learner), flat(learner, True), learner.rng.getstate()
    learner.set_learning(False)
    for tick in range(3):
        assert learner.begin_trial() == 0
        assert learner.apply_goodness(0., now_ms=tick) is None
    assert learner.pending_indices is None and learner.previous_goodness is None
    assert torch.equal(w, flat(learner)) and torch.equal(d, flat(learner, True))
    assert rng == learner.rng.getstate()
    learner.set_learning(True)
    assert learner.begin_trial() == 0
    assert learner.apply_goodness(.1, now_ms=10) is None
    assert torch.equal(w, flat(learner)) and torch.equal(d, flat(learner, True))
    assert learner.begin_trial() > 0


def test_runtime_groups_include_biases_and_exclude_goodness(make_runtime):
    runtime = make_runtime()
    learner = runtime.enable_plasticity(subset_fraction=1.)
    excluded = list(learner.groups[learner.goodness_id].values())
    before = [p.clone() for p in excluded]
    assert not {id(p) for p in excluded} & learner.parameters.keys()
    assert any(name.endswith('bias') for group in learner.groups.values() for name in group)
    learner.apply_goodness(.5, now_ms=0)
    learner.begin_trial()
    learner.apply_goodness(.1, now_ms=1)
    for p, initial in zip(excluded, before):
        assert torch.equal(p, initial)
    assert all(p.grad is None for p in runtime.parameters())


def test_clip_only_selected_values_and_keeps_direction():
    learner = make_learner(parameter_clip=(-.0001,.0001))
    learner.apply_goodness(.5, now_ms=0)
    d = flat(learner, True)
    learner.begin_trial()
    assert flat(learner).abs().max() <= .0001
    assert torch.equal(d, flat(learner, True))


@pytest.mark.parametrize('state', ['parameter', 'delta_w'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_state_raises_before_movement(state, bad):
    learner = make_learner()
    learner.apply_goodness(.5, now_ms=0)
    values = learner.parameters if state == 'parameter' else learner.delta_w
    with torch.no_grad():
        next(iter(values.values())).view(-1)[0] = bad
    with pytest.raises(FloatingPointError):
        learner.begin_trial()


@pytest.mark.parametrize('kwargs', [dict(delta_magnitude=0), dict(delta_magnitude=float('nan')),
    dict(subset_fraction=0), dict(subset_fraction=1.1), dict(plasticity_seed=True),
    dict(parameter_clip=(2,1))])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Plasticity({0:{'w':nn.Parameter(torch.zeros(2))}}, None, **kwargs)


def test_overflow_rejected_before_clip_can_hide_it():
    p = nn.Parameter(torch.tensor([3e38]))
    learner = Plasticity({0:{'w':p}}, None, delta_magnitude=3e38, parameter_clip=(-1.,1.))
    learner.delta_w[id(p)].fill_(3e38)
    learner.apply_goodness(.5, now_ms=0)
    with pytest.raises(FloatingPointError):
        learner.begin_trial()
    assert p.item() == pytest.approx(3e38)
    assert learner.pending_indices is None


def test_invalid_goodness_preserves_pending_trial():
    learner = make_learner()
    learner.apply_goodness(.5, now_ms=100)
    learner.begin_trial()
    selected, w, d = learner.pending_indices, flat(learner), flat(learner, True)
    for goodness, time in [(float('nan'),101), (2.,101), (.1,99)]:
        with pytest.raises(ValueError):
            learner.apply_goodness(goodness, now_ms=time)
        assert learner.pending_indices == selected
        assert torch.equal(flat(learner),w) and torch.equal(flat(learner,True),d)
