"""Standalone FP32 adapters selected for Phase 6.

Adapters map tensors; their parametrized layers expose local pre/post activity
to the Runtime observer. Production forward passes do not build gradient graphs.
"""

import torch
from torch import Tensor, nn


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _Adapter(nn.Module):
    """Validate unbatched or singly batched FP32 tensors."""

    def _check_input(self, value: Tensor, sample_shape: tuple[int, ...]) -> None:
        if not isinstance(value, Tensor):
            raise ValueError("input must be a tensor")
        valid_shape = value.shape == sample_shape or (
            value.ndim == len(sample_shape) + 1
            and value.shape[0] > 0
            and value.shape[1:] == sample_shape
        )
        if not valid_shape:
            raise ValueError(
                f"input must have shape {sample_shape} or (batch, {', '.join(map(str, sample_shape))})"
            )
        for parameter in self.parameters():
            if parameter.dtype != torch.float32:
                raise ValueError("Adapter parameters must use FP32")
            if value.device != parameter.device:
                raise ValueError("input must use the Adapter's device")
        if value.dtype != torch.float32:
            raise ValueError("input must use FP32")
        if not torch.isfinite(value).all():
            raise ValueError("input must contain finite values")

    @staticmethod
    def _check_output(value: Tensor) -> Tensor:
        if value.dtype != torch.float32:
            raise ValueError("Adapter output must use FP32; disable mixed precision")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Adapter produced non-finite output")
        return value


class EyeAdapter(_Adapter):
    """Learn image structure using strided convolutions, without averaging.

    Input: [3, 1080, 1920] or [batch, 3, 1080, 1920].
    Features: [8, 108, 192], then [16, 18, 32].
    Output: [2 * neuron_size] or [batch, 2 * neuron_size].
    """

    def __init__(self, neuron_size: int) -> None:
        super().__init__()
        self.neuron_size = _positive_integer(neuron_size, "neuron_size")
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=11, stride=10, padding=5, dtype=torch.float32),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, stride=6, padding=3, dtype=torch.float32),
            nn.ReLU(),
        )
        self.projection = nn.Linear(16 * 18 * 32, 2 * neuron_size, dtype=torch.float32)
        # Mark the layers whose ReLU output is what keeps propagating forward.
        self.features[0]._acnt_post_relu = True
        self.features[2]._acnt_post_relu = True

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        self._check_input(image, (3, 1080, 1920))
        features = self.features(image)
        # Preserve the optional batch axis; flatten only C, H, W.
        return self._check_output(self.projection(features.flatten(start_dim=-3)))


class EarAdapter(_Adapter):
    """Map one fixed audio window to the ReadIn vector; no audio capture."""

    def __init__(self, neuron_size: int, window_samples: int = 1600, channels: int = 1) -> None:
        super().__init__()
        self.neuron_size = _positive_integer(neuron_size, "neuron_size")
        self.window_samples = _positive_integer(window_samples, "window_samples")
        self.channels = _positive_integer(channels, "channels")
        self.linear = nn.Linear(channels * window_samples, 2 * neuron_size, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, audio: Tensor) -> Tensor:
        self._check_input(audio, (self.channels, self.window_samples))
        return self._check_output(self.linear(audio.flatten(start_dim=-2)))


class HandAdapter(_Adapter):
    """Return discrete tendencies followed by continuous mouse controls.

    The first ``discrete_controls`` coordinates are raw q values. Remaining
    coordinates are raw continuous controls. Sampling and mouse execution are
    deliberately outside this tensor mapping.
    """

    def __init__(
        self,
        neuron_size: int,
        discrete_controls: int = 85,
        continuous_controls: int = 2,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.neuron_size = _positive_integer(neuron_size, "neuron_size")
        self.discrete_controls = _positive_integer(discrete_controls, "discrete_controls")
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        if type(continuous_controls) is not int or continuous_controls < 0:
            raise ValueError("continuous_controls must be a nonnegative integer")
        self.continuous_controls = continuous_controls
        self.output_size = discrete_controls + continuous_controls
        self.network = nn.Sequential(
            nn.Linear(neuron_size, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size, dtype=torch.float32),
        )
        # Mark the layer whose ReLU output is what keeps propagating forward.
        self.network[0]._acnt_post_relu = True

    @torch.no_grad()
    def forward(self, z: Tensor) -> Tensor:
        self._check_input(z, (self.neuron_size,))
        return self._check_output(self.network(z))


class _LinearReadout(_Adapter):
    def __init__(self, neuron_size: int, output_size: int) -> None:
        super().__init__()
        self.neuron_size = _positive_integer(neuron_size, "neuron_size")
        self.output_size = _positive_integer(output_size, "output_size")
        self.linear = nn.Linear(neuron_size, output_size, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, z: Tensor) -> Tensor:
        self._check_input(z, (self.neuron_size,))
        return self._check_output(self.linear(z))


class SpeakAdapter(_LinearReadout):
    """Map z to 30 continuous speech parameters by default."""

    def __init__(self, neuron_size: int, output_size: int = 30) -> None:
        super().__init__(neuron_size, output_size)


class GoodnessAdapter(_LinearReadout):
    """Return raw goodness with a final dimension of one, without clamping."""

    def __init__(self, neuron_size: int) -> None:
        super().__init__(neuron_size, 1)


class RouteAdapter(_LinearReadout):
    """Return independent raw q coordinates; no Softmax, noise, or mask."""

    def __init__(self, neuron_size: int, max_blocks: int) -> None:
        self.max_blocks = _positive_integer(max_blocks, "max_blocks")
        super().__init__(neuron_size, max_blocks)
