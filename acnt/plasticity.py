"""Local e-prop traces and the one-scalar, delayed goodness update."""

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn

from .control import DiscreteSignal


class EligibilityBank:
    """One detached eligibility tensor per concrete parameter tensor."""

    def __init__(self, parameters: Mapping[str, nn.Parameter], tau: float) -> None:
        if not parameters or not math.isfinite(tau) or tau <= 0:
            raise ValueError("eligibility requires parameters and positive finite tau")
        self.parameters = dict(parameters)
        self.tau = float(tau)
        self.values = {name: torch.zeros_like(p) for name, p in self.parameters.items()}
        self.last_ms: int | None = None

    def advance(self, now_ms: int) -> None:
        if type(now_ms) is not int:
            raise ValueError("trace time must be integer milliseconds")
        if self.last_ms is not None:
            if now_ms < self.last_ms:
                raise ValueError("trace time must not move backwards")
            # lambda = exp(-delta_seconds / tau), with tau == Block.ticktime.
            decay = math.exp(-((now_ms - self.last_ms) / 1000.0) / self.tau)
            for trace in self.values.values():
                trace.mul_(decay)
        self.last_ms = now_ms

    def observe_scalar(self, scalar: Tensor, *, now_ms: int) -> None:
        if scalar.shape != () or not torch.isfinite(scalar):
            raise ValueError("local contraction must be one finite scalar")
        parameters = tuple(self.parameters.values())
        derivatives = torch.autograd.grad(scalar, parameters, allow_unused=True, retain_graph=True) if scalar.requires_grad else (None,) * len(parameters)
        local = {
            name: torch.zeros_like(parameter) if derivative is None else derivative.detach()
            for (name, parameter), derivative in zip(self.parameters.items(), derivatives)
        }
        if any(not torch.isfinite(value).all() for value in local.values()):
            raise FloatingPointError("non-finite local eligibility")
        self.advance(now_ms)
        for name, derivative in local.items():
            self.values[name].add_(derivative)

    def observe_mean(self, output: Tensor, *, now_ms: int) -> None:
        if output.numel() == 0:
            raise ValueError("cannot average an empty output")
        # User-confirmed output-axis contraction: (1/n) sum_k dz_k/dtheta.
        self.observe_scalar(output.mean(), now_ms=now_ms)

    def observe_control(self, signal: DiscreteSignal, *, now_ms: int) -> Tensor:
        # Score-function estimator of the ACTUAL sampled event. Do not
        # differentiate p, hard threshold or the external world.
        d_logpi = ((signal.a.to(signal.q.dtype) - signal.p) / signal.tau).detach()
        # Independent discrete controls contribute a SUM, not an average.
        self.observe_scalar((d_logpi * signal.q).sum(), now_ms=now_ms)
        return d_logpi

    def clear(self) -> None:
        for trace in self.values.values():
            trace.zero_()
        self.last_ms = None


class Plasticity:
    """Block-local channels, consumed together by delta = g_eff - g_bar."""

    def __init__(
        self, groups: Mapping[int, Mapping[str, nn.Parameter]], taus: Mapping[int, float],
        *, learning_rate: float = 0.001, rho: float = 0.9, ema_alpha: float = 0.1,
        initial_g_bar: float = 0.5, parameter_clip: tuple[float, float] | None = None,
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (rho, ema_alpha, initial_g_bar)):
            raise ValueError("rho, EMA alpha and initial g_bar must be in [0,1]")
        if parameter_clip is not None and (len(parameter_clip) != 2 or not all(math.isfinite(v) for v in parameter_clip) or parameter_clip[0] > parameter_clip[1]):
            raise ValueError("parameter_clip must be ordered finite bounds")
        self.groups = {i: dict(parameters) for i, parameters in groups.items()}
        all_ids = [id(p) for parameters in self.groups.values() for p in parameters.values()]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("each parameter must belong to exactly one Block group")
        self.internal = {i: EligibilityBank(parameters, taus[i]) for i, parameters in self.groups.items()}
        self.continuous = {i: EligibilityBank(parameters, taus[i]) for i, parameters in self.groups.items()}
        self.control = {i: EligibilityBank(parameters, taus[i]) for i, parameters in self.groups.items()}
        self.learning_rate = learning_rate
        self.rho = rho
        self.ema_alpha = ema_alpha
        self.g_bar = initial_g_bar
        self.parameter_clip = parameter_clip

    def banks(self):
        return (self.internal, self.continuous, self.control)

    @torch.no_grad()
    def apply_goodness(self, g_eff: float, *, now_ms: int) -> float:
        if not math.isfinite(g_eff) or not 0 <= g_eff <= 1:
            raise ValueError("effective goodness must be a finite scalar in [0,1]")
        banks = [bank for channel in self.banks() for bank in channel.values()]
        if type(now_ms) is not int or any(bank.last_ms is not None and now_ms < bank.last_ms for bank in banks):
            raise ValueError("goodness delivery cannot precede current trace time")
        for bank in banks:
            bank.advance(now_ms)
        delta = g_eff - self.g_bar  # Use the baseline BEFORE incorporating g_eff.
        proposals = []
        for block_id, parameters in self.groups.items():
            for name, parameter in parameters.items():
                trace = sum(channel[block_id].values[name] for channel in self.banks())
                value = parameter + self.learning_rate * delta * trace
                if self.parameter_clip is not None:
                    value = value.clamp(*self.parameter_clip)
                if not torch.isfinite(value).all():
                    raise FloatingPointError("goodness would produce a non-finite parameter")
                proposals.append((parameter, value))
        for parameter, value in proposals:
            parameter.copy_(value)
        for bank in banks:
            for trace in bank.values.values():
                trace.mul_(self.rho)
        self.g_bar += self.ema_alpha * delta
        return delta

    def clear(self) -> None:
        """Explicit episode boundary; never resets the world or its state."""
        for channel in self.banks():
            for bank in channel.values():
                bank.clear()
