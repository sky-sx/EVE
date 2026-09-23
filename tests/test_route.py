import pytest
import torch

from acnt.control import sample_discrete


def test_actual_inverse_logistic_noise_threshold_and_probability():
    q = torch.tensor([-0.3, 0.2, 1.4, 0.0])
    theta = torch.tensor([0., 0.1, 0.5, -0.4])
    generator = torch.Generator().manual_seed(83)
    oracle_generator = torch.Generator().manual_seed(83)
    u = torch.rand(q.shape, generator=oracle_generator).clamp(torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps)
    expected_noise = 0.25 * (torch.log(u) - torch.log1p(-u))
    result = sample_discrete(q, tau=0.25, threshold=theta, generator=generator)
    torch.testing.assert_close(result.noise, expected_noise)
    torch.testing.assert_close(result.p, torch.sigmoid((q - theta) / 0.25))
    assert torch.equal(result.a, q + expected_noise > theta)
    assert result.noise.unique().numel() == q.numel()


def test_multiple_actions_can_be_on_or_off_without_softmax():
    high = sample_discrete(torch.full((8,), 100.), tau=0.25)
    low = sample_discrete(torch.full((8,), -100.), tau=0.25)
    assert high.a.all() and not low.a.any()
    assert high.p.sum() > 1


def set_route_bias(runtime, values):
    with torch.no_grad():
        runtime.adapters["route"].linear.weight.zero_()
        runtime.adapters["route"].linear.bias.copy_(torch.tensor(values, dtype=torch.float32))


def test_route_changes_next_computation_and_cannot_disable_itself(make_runtime):
    runtime = make_runtime(ticktime=250)
    runtime.set_goodness_active(False)
    runtime.update_blocks(now_ms=0)
    set_route_bias(runtime, [-100.] * 8)
    result = runtime.generate_route(generator=torch.Generator().manual_seed(2))
    assert result.q.shape == result.noise.shape == result.a.shape == (8,)
    assert not result.a.any()  # Raw stochastic proposal is all off.
    assert runtime.core.active_ids == (5,)  # Route is permanently active.
    assert set(runtime.update_blocks(now_ms=250)) == {5}
    set_route_bias(runtime, [100., -100., 100., -100., 100., -100., 100., -100.])
    runtime.generate_route(generator=torch.Generator().manual_seed(2))
    assert runtime.core.active_ids == (0, 2, 5, 6)
    assert set(runtime.update_blocks(now_ms=500)) == {0, 2, 5, 6}


def test_goodness_active_is_human_controlled_even_when_route_disagrees(make_runtime):
    runtime = make_runtime()
    set_route_bias(runtime, [-100.] * 8)
    runtime.generate_route()
    assert runtime.core.blocks[4].active
    runtime.set_goodness_active(False)
    set_route_bias(runtime, [100.] * 8)
    runtime.generate_route()
    assert not runtime.core.blocks[4].active
    assert runtime.core.blocks[5].active


def test_route_uses_runtime_noise_scale(make_runtime):
    runtime = make_runtime(ticktime=125)
    assert runtime.generate_route().tau == runtime.noise_scale == 1.0


@pytest.mark.parametrize("tau", [0., -1., float("nan")])
def test_invalid_noise_tau_rejected(tau):
    with pytest.raises((ValueError, TypeError)):
        sample_discrete(torch.zeros(2), tau=tau)
