import pytest
import torch
from torch import nn

from acnt.adapters import (
    EarAdapter, EyeAdapter, GoodnessAdapter, HandAdapter, RouteAdapter, SpeakAdapter,
)


CASES = [
    (lambda: EyeAdapter(3), (3, 1080, 1920), 6),
    (lambda: EarAdapter(3, window_samples=32), (1, 32), 6),
    (lambda: HandAdapter(3), (3,), 87),
    (lambda: SpeakAdapter(3), (3,), 30),
    (lambda: GoodnessAdapter(3), (3,), 1),
    (lambda: RouteAdapter(3, max_blocks=9), (3,), 9),
]


@pytest.mark.parametrize("factory,input_shape,out_size", CASES)
@pytest.mark.parametrize("batch", [False, True])
def test_forward_shapes_parameters_and_local_derivatives(factory, input_shape, out_size, batch):
    torch.manual_seed(71)
    adapter = factory()
    prefix = (2,) if batch else ()
    x = torch.rand(*prefix, *input_shape, dtype=torch.float32)
    output = adapter(x)
    assert output.shape == (*prefix, out_size)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    parameters = list(adapter.named_parameters())
    assert parameters and len({id(p) for _, p in parameters}) == len(parameters)
    # Production adapter forward intentionally has no autograd graph.
    assert not output.requires_grad
    for _, parameter in parameters:
        assert parameter.dtype == torch.float32
        assert torch.isfinite(parameter).all()
        assert parameter.grad is None
    if batch:
        with torch.no_grad():
            torch.testing.assert_close(output[0], adapter(x[0]))


def test_eye_uses_learned_spatial_reduction_without_average_pooling():
    adapter = EyeAdapter(3)
    convolutions = [layer for layer in adapter.modules() if isinstance(layer, nn.Conv2d)]
    assert len(convolutions) == 2
    assert not any(isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) for layer in adapter.modules())
    shapes = []
    handles = [layer.register_forward_hook(lambda _module, _inputs, output: shapes.append(tuple(output.shape[-3:]))) for layer in convolutions]
    try:
        with torch.no_grad():
            adapter(torch.zeros(3, 1080, 1920))
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == [(8, 108, 192), (16, 18, 32)]


def test_hand_supports_full_keyboard_and_mouse_with_optional_continuous_axes():
    assert HandAdapter(3, continuous_controls=0)(torch.ones(3)).shape == (85,)
    assert HandAdapter(3, continuous_controls=2)(torch.ones(3)).shape == (87,)


@pytest.mark.parametrize("adapter", [RouteAdapter(3, max_blocks=6), GoodnessAdapter(3)])
def test_readout_adapters_return_raw_values_before_runtime_control(adapter):
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        last_linear = [layer for layer in adapter.modules() if isinstance(layer, nn.Linear)][-1]
        last_linear.bias.fill_(2.)
        output = adapter(torch.ones(3))
    torch.testing.assert_close(output, torch.full_like(output, 2.))


@pytest.mark.parametrize("factory,input_shape,_out_size", CASES)
def test_wrong_shape_dtype_and_nonfinite_inputs_are_rejected(factory, input_shape, _out_size):
    adapter = factory()
    with pytest.raises((ValueError, TypeError)):
        adapter(torch.zeros((*input_shape[:-1], input_shape[-1] + 1)))
    with pytest.raises((ValueError, TypeError)):
        adapter(torch.zeros(input_shape, dtype=torch.float64))
    bad = torch.zeros(input_shape)
    bad.flatten()[0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        adapter(bad)


def test_ear_fixed_window_rejects_missing_channel_and_extra_batch_axes():
    adapter = EarAdapter(3, window_samples=32, channels=2)
    assert adapter(torch.ones(2, 32)).shape == (6,)
    assert adapter(torch.ones(4, 2, 32)).shape == (4, 6)
    with pytest.raises((ValueError, TypeError)):
        adapter(torch.ones(32))
    with pytest.raises((ValueError, TypeError)):
        adapter(torch.ones(2, 4, 2, 32))
