"""Independent discrete controls driven by explicit Logistic noise."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiscreteSignal:
    """One control sample, retaining the local graph of q and p."""

    q: Tensor
    noise: Tensor
    threshold: Tensor
    p: Tensor
    a: Tensor
    tau: float


def sample_discrete(
    q: Tensor,
    *,
    tau: float,
    threshold: float | Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> DiscreteSignal:
    """Sample each coordinate with ``q + Logistic(0, tau) > threshold``.

    The matching marginal probability is ``sigmoid((q - threshold) / tau)``.
    No gradients flow through the noise or the Boolean outcome; q and p keep
    their local autograd graph for subsequent eligibility calculations.
    """
    if not isinstance(q, Tensor) or q.ndim != 1 or q.numel() == 0:
        raise ValueError("q must be a nonempty one-dimensional tensor")
    if q.dtype != torch.float32:
        raise ValueError("q must use FP32")
    if not torch.isfinite(q).all():
        raise ValueError("q must contain finite values")
    if isinstance(tau, bool) or not isinstance(tau, Real) or not isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a finite positive number")
    tau = float(tau)
    tau_fp32 = q.new_tensor(tau)
    if not torch.isfinite(tau_fp32) or tau_fp32 <= 0:
        raise ValueError("tau must be representable as a finite positive FP32 value")
    if isinstance(threshold, Tensor):
        if threshold.ndim != 0 and threshold.shape != q.shape:
            raise ValueError("threshold must be scalar or have the same shape as q")
        if threshold.dtype != torch.float32 or threshold.device != q.device:
            raise ValueError("threshold must use FP32 on q's device")
        threshold_tensor = threshold.expand_as(q)
    else:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a number or tensor")
        if not isfinite(threshold):
            raise ValueError("threshold must contain finite values")
        threshold_tensor = q.new_tensor(float(threshold)).expand_as(q)
    if not torch.isfinite(threshold_tensor).all():
        raise ValueError("threshold must contain finite FP32 values")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    eps = torch.finfo(q.dtype).eps
    uniform = torch.rand(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    uniform = uniform.clamp(min=eps, max=1 - eps)
    noise = tau_fp32 * (torch.log(uniform) - torch.log1p(-uniform))
    p = torch.sigmoid((q - threshold_tensor) / tau_fp32)
    if not torch.isfinite(noise).all() or not torch.isfinite(p).all():
        raise FloatingPointError("discrete sampling produced non-finite noise or probabilities")
    a = q + noise > threshold_tensor
    return DiscreteSignal(q=q, noise=noise, threshold=threshold_tensor, p=p, a=a, tau=tau)
