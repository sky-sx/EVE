"""Connection-local plasticity for the production ACNT runtime.

F_e and F_w are deliberately replaceable. CorrelationRule is a small
experimental candidate, not a fixed equation of the ACNT architecture.
No autograd, Jacobian, eligibility trace, or global backward signal is used.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalEvent:
    """Activities at the two real ends of one parametrized local operation."""

    pre: Tensor
    post: Tensor
    kind: str = "dense"
    module: nn.Module | None = None
    action: Tensor | None = None
    noise: Tensor | None = None
    threshold: Tensor | None = None
    control_count: int = 0
    excluded_control: int | None = None


class CorrelationRule:
    """Replaceable experimental F_e/F_w candidate; no learning claim implied."""

    def __init__(self, *, learning_rate: float = 0.001, retention: float = 0.95) -> None:
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if not math.isfinite(retention) or not 0 <= retention <= 1:
            raise ValueError("retention must be in [0, 1]")
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)

    def F_e(self, state: Tensor, event: LocalEvent) -> Tensor:
        pre, post = event.pre, event.post
        if event.kind == "conv":
            module = event.module
            if not isinstance(module, nn.Conv2d):
                raise ValueError("conv event requires its local Conv2d")
            if pre.ndim == 3:
                pre = pre.unsqueeze(0)
                post = post.unsqueeze(0)
            patches = F.unfold(pre, module.kernel_size, dilation=module.dilation,
                               padding=module.padding, stride=module.stride)
            if module.groups != 1:
                raise ValueError("grouped convolution needs its own local rule")
            outputs = post.flatten(2)
            product = torch.einsum("bol,bil->oi", outputs, patches)
            product = product.reshape_as(state) / (outputs.shape[0] * outputs.shape[-1])
        elif event.kind == "bias":
            product = post.reshape(-1, post.shape[-1]).mean(0)
        elif event.kind in ("dense", "control"):
            x = pre.reshape(-1, pre.shape[-1])
            y = post.reshape(-1, post.shape[-1])
            if event.kind == "control":
                # Terminal-local facts: q, this noise realization, and the
                # event actually formed. No probability/score derivative.
                if event.action is None or event.noise is None or event.threshold is None:
                    raise ValueError("control event requires actual action and noise")
                actual = event.action.to(y.dtype).reshape_as(y)
                noise = event.noise.reshape_as(y)
                threshold = event.threshold.reshape_as(y)
                y = y.clone()
                count = event.control_count
                y[:, :count] = torch.where(actual[:, :count].bool(), 1.0, -1.0) * (
                    y[:, :count] + noise[:, :count] - threshold[:, :count]
                ).abs().clamp(max=1)
                if event.excluded_control is not None:
                    y[:, event.excluded_control] = 0
            product = (y.T @ x) / x.shape[0]
        else:
            raise ValueError(f"unknown local event kind: {event.kind}")
        if product.shape != state.shape:
            raise ValueError("local activity dimensions do not match the connection")
        return self.retention * state + product

    def F_w(self, weight: Tensor, state: Tensor, g_eff: float) -> Tensor:
        # Centering is a choice of this candidate, not an extra reward stream.
        return weight + self.learning_rate * (g_eff - 0.5) * state

    def F_w_g(self, weight: Tensor, state: Tensor, c_g: float) -> Tensor:
        return weight + self.learning_rate * c_g * state


class Plasticity:
    """One e_c per parameter element, owned by its Block or Adapter.

    The goodness Block and Adapter only receive c_g. Every other connection
    receives the one global g_eff when that scalar arrives.
    """

    def __init__(
        self, groups: Mapping[int, Mapping[str, nn.Parameter]], goodness_id: int,
        *, rule=None, learning_rate: float = 0.001, retention: float = 0.95,
        parameter_clip: tuple[float, float] | None = None,
    ) -> None:
        self.groups = {i: dict(parameters) for i, parameters in groups.items()}
        if goodness_id not in self.groups:
            raise ValueError("goodness Block must have a parameter group")
        self.goodness_id = goodness_id
        self.parameters = {id(p): p for group in self.groups.values() for p in group.values()}
        if len(self.parameters) != sum(len(group) for group in self.groups.values()):
            raise ValueError("each parameter must belong to exactly one Block group")
        if parameter_clip is not None and (
            len(parameter_clip) != 2 or
            not all(math.isfinite(v) for v in parameter_clip) or
            parameter_clip[0] > parameter_clip[1]
        ):
            raise ValueError("parameter_clip must be ordered finite bounds")
        self.parameter_clip = parameter_clip
        self.rule = CorrelationRule(learning_rate=learning_rate, retention=retention) if rule is None else rule
        if any(not callable(getattr(self.rule, name, None)) for name in ("F_e", "F_w", "F_w_g")):
            raise TypeError("local rule must supply F_e, F_w, and F_w_g")
        self.states = {key: torch.zeros_like(p) for key, p in self.parameters.items()}
        self._terminal_inputs: dict[int, tuple[Tensor, Tensor] | None] = {}
        self._excluded_controls: dict[int, int] = {}
        self._hooks = []

    def attach_adapters(self, adapters: Mapping[str, nn.Module], *, route_id: int) -> None:
        """Observe each parametrized layer's real input and output locally."""
        self.detach_adapters()
        for name, adapter in adapters.items():
            for module in adapter.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if id(module.weight) not in self.states:
                        raise ValueError(f"{name} Adapter parameters are not in a Block group")
                    self._hooks.append(module.register_forward_hook(self._layer_hook))
            terminal = next((layer for layer in reversed(list(adapter.modules()))
                             if isinstance(layer, nn.Linear)), None)
            if name in ("hand", "route") and terminal is not None:
                self._terminal_inputs[id(terminal)] = None
                if name == "route":
                    self._excluded_controls[id(terminal)] = route_id

    def detach_adapters(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._terminal_inputs.clear()
        self._excluded_controls.clear()

    @torch.no_grad()
    def _layer_hook(self, module: nn.Module, inputs: tuple, output: Tensor) -> None:
        pre, post = inputs[0].detach(), output.detach()
        if id(module) in self._terminal_inputs:
            self._terminal_inputs[id(module)] = (pre.clone(), post.clone())
            return
        kind = "conv" if isinstance(module, nn.Conv2d) else "dense"
        self.observe(module.weight, LocalEvent(pre, post, kind, module))
        if module.bias is not None:
            bias_post = post.movedim(-3, -1) if kind == "conv" else post
            self.observe(module.bias, LocalEvent(torch.ones_like(bias_post), bias_post, "bias"))

    @torch.no_grad()
    def observe(self, parameter: nn.Parameter, event: LocalEvent) -> None:
        key = id(parameter)
        if key not in self.states:
            raise ValueError("observed connection is not registered")
        tensors = (event.pre, event.post, event.action, event.noise, event.threshold)
        if any(value is not None and not bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("non-finite local activity")
        state = self.rule.F_e(self.states[key], event)
        if state.shape != parameter.shape or not bool(torch.isfinite(state).all()):
            raise FloatingPointError("invalid local plastic state")
        self.states[key] = state.detach().clone()

    @torch.no_grad()
    def observe_control(self, adapter: nn.Module, signal) -> None:
        terminal = next((layer for layer in reversed(list(adapter.modules()))
                         if isinstance(layer, nn.Linear)), None)
        if terminal is None or id(terminal) not in self._terminal_inputs:
            raise ValueError("discrete readout has no observed terminal layer")
        recorded = self._terminal_inputs[id(terminal)]
        if recorded is None:
            raise ValueError("readout must run before its terminal event")
        pre, post = recorded
        count = signal.q.numel()
        if not torch.equal(post[:count], signal.q):
            raise ValueError("terminal event does not match the sampled tendencies")
        action = torch.zeros_like(post, dtype=torch.bool)
        action[:count] = signal.a
        noise = torch.zeros_like(post)
        noise[:count] = signal.noise
        threshold = torch.zeros_like(post)
        threshold[:count] = signal.threshold
        excluded = self._excluded_controls.get(id(terminal))
        event = LocalEvent(pre, post, "control", terminal, action, noise, threshold, count, excluded)
        self.observe(terminal.weight, event)
        if terminal.bias is not None:
            drive = post.clone()
            drive[:count] = torch.where(signal.a, 1.0, -1.0) * (
                signal.q + signal.noise - signal.threshold
            ).abs().clamp(max=1)
            if excluded is not None:
                drive[excluded] = 0
            self.observe(terminal.bias, LocalEvent(torch.ones_like(drive), drive, "bias"))
        self._terminal_inputs[id(terminal)] = None

    @torch.no_grad()
    def _apply(self, block_ids, scalar: float, *, teacher: bool = False) -> float:
        if not math.isfinite(scalar):
            raise ValueError("plasticity modulator must be finite")
        proposals = []
        for block_id in block_ids:
            for parameter in self.groups[block_id].values():
                update = self.rule.F_w_g if teacher else self.rule.F_w
                value = update(parameter.detach(), self.states[id(parameter)], scalar)
                if self.parameter_clip is not None:
                    value = value.clamp(*self.parameter_clip)
                if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("local rule proposed an invalid parameter")
                proposals.append((parameter, value))
        for parameter, value in proposals:
            parameter.copy_(value)
        return scalar

    def apply_goodness(self, g_eff: float, *, now_ms: int | None = None) -> float:
        if not math.isfinite(g_eff) or not 0 <= g_eff <= 1:
            raise ValueError("effective goodness must be in [0,1]")
        if now_ms is not None and type(now_ms) is not int:
            raise ValueError("goodness delivery time must be integer milliseconds")
        return self._apply((i for i in self.groups if i != self.goodness_id), float(g_eff))

    def calibrate_goodness(self, c_g: float) -> float:
        if not math.isfinite(c_g) or not -1 <= c_g <= 1:
            raise ValueError("calibration scalar must be in [-1,1]")
        return self._apply((self.goodness_id,), float(c_g), teacher=True)

    def clear(self) -> None:
        for state in self.states.values():
            state.zero_()
