"""One-event forward sensitivity and delayed policy credit for original ACNT.

Research reference, not a production-scale trainer. The original transition
is used verbatim. Local autograd constructs derivatives of ONE event only;
no recurrent graph or action graph survives that event. Dense sensitivity is
exact for fixed weights and is a stop-update tangent for changing weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor
from .original_training import TrainingState
from .runtime import Runtime, READINS


@dataclass(frozen=True)
class OnlineAction:
    action: Tensor
    probability: Tensor
    score: Tensor


class OriginalOnlineRuntime:
    """Original state/snapshot scheduling with persistent forward sensitivity.

    Timestamps, masks and sampled routing are conditioned-on event metadata.
    The differentiable state is z AND every retained pre-activation, not z
    alone. Diagnostic r/a and consumed ReadIn buffers need no extra tangent.
    A sensitivity limit deliberately makes the oracle approximate when hit.
    """
    def __init__(self, runtime: Runtime, *, train_adapters=("ear", "hand"),
                 train_core=True, sensitivity_limit=None, max_elements=5_000_000,
                 cut_state_credit=False):
        self.runtime = runtime
        if runtime.plasticity is not None and runtime.plasticity.learning:
            raise ValueError("disable legacy plasticity before online training")
        gid = runtime.organ_blocks["goodness"]
        if any(name not in READINS + ("hand", "speak", "route") for name in train_adapters):
            raise ValueError("Goodness is excluded; select valid actor adapters")
        if sensitivity_limit is not None and (not math.isfinite(sensitivity_limit) or sensitivity_limit <= 0):
            raise ValueError("sensitivity_limit must be positive or None")
        self.named_parameters = [(n, p) for n, p in runtime.named_parameters()
            if p.requires_grad and ((train_core and n.startswith("core.blocks.")
                and not n.startswith(f"core.blocks.{gid}."))
                or any(n.startswith(f"adapters.{a}.") for a in train_adapters))]
        if not self.named_parameters:
            raise ValueError("at least one actor parameter is required")
        self.parameters = [p for _, p in self.named_parameters]
        self.param_dim = sum(p.numel() for p in self.parameters)
        self.slices = []
        offset = 0
        for b in runtime.core.blocks:
            size = b.neuron_size * (1 + b.hold_tick)
            self.slices.append(slice(offset, offset + size))
            offset += size
        self.state_dim = offset
        if self.state_dim * self.param_dim > max_elements:
            raise ValueError("dense reference exceeds budget; select parameters or use a compressed implementation")
        self.states = tuple(TrainingState(b.z.detach().clone(), b.r.detach().clone(),
            b.a.detach().clone(), tuple(x.detach().clone() for x in b.A), tuple(b.At),
            None if b.o is None else b.o.detach().clone()) for b in runtime.core.blocks)
        self.state = self.pack(self.states).detach()
        self.sensitivity = self.state.new_zeros(self.state_dim, self.param_dim)
        self.last_time = max((t for s in self.states for t in s.times), default=None)
        self.sensitivity_limit = sensitivity_limit
        self.cut_state_credit = cut_state_credit
        self.events = 0
        self.sensitivity_clips = 0
        self.max_sensitivity_norm = 0.0
        runtime.learning = False

    def pack(self, states):
        pieces = []
        for b, s in zip(self.runtime.core.blocks, states):
            pad = b.hold_tick - len(s.history)
            hist = [s.z.new_zeros(b.neuron_size) for _ in range(pad)] + list(s.history)
            pieces.extend((s.z, torch.cat(hist)))
        return torch.cat(pieces)

    def unpack(self, vector):
        states = []
        for b, old, part in zip(self.runtime.core.blocks, self.states, self.slices):
            values = vector[part]
            n = b.neuron_size
            hist = values[n:].reshape(b.hold_tick, n)
            k = len(old.times)
            states.append(TrainingState(values[:n], old.r, old.a,
                tuple(hist[-k:].unbind()) if k else (), old.times, old.o))
        return tuple(states)

    def _flat_parameter_grad(self, gradients, *, batch=None):
        return torch.cat([(self.state.new_zeros((batch, p.numel())) if batch is not None
                          else self.state.new_zeros(p.numel())) if g is None
                         else g.reshape((batch, -1) if batch is not None else (-1,))
                         for p, g in zip(self.parameters, gradients)], dim=-1)

    def step(self, *, now_ms: int, readins: Mapping[str, Tensor] | None = None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms < self.last_time):
            raise ValueError("time must be monotonic integer milliseconds")
        inputs = {} if readins is None else dict(readins)
        if not set(inputs).issubset(READINS):
            raise ValueError("ReadIn must be eye or ear")
        leaf = self.state.detach().requires_grad_(True)
        old = self.unpack(leaf)
        active = [b.active for b in self.runtime.core.blocks]
        active[self.runtime.organ_blocks["route"]] = True
        active[self.runtime.organ_blocks["goodness"]] = self.runtime.goodness_active
        forced = {self.runtime.organ_blocks[n] for n in inputs}
        sources = {i: s.z for i, s in enumerate(old) if active[i] or i in forced}
        encoded = {self.runtime.organ_blocks[n]: self.runtime.adapters[n].forward_train(x)
                   for n, x in inputs.items()}
        new = list(old)
        updated = []
        for i, (b, s) in enumerate(zip(self.runtime.core.blocks, old)):
            due = not s.times or now_ms - s.times[-1] >= b.ticktime
            if i not in forced and (not active[i] or not due):
                continue
            r, a, z, hist, times = b.transition(history=s.history, times=s.times,
                now_ms=now_ms, active_z=sources, o=encoded.get(i, s.o))
            new[i] = TrainingState(z, r, a, hist, times,
                                  None if s.o is None else torch.zeros_like(s.o))
            updated.append(i)
        vector = self.pack(new)
        grads = torch.autograd.grad(vector, (leaf, *self.parameters),
            grad_outputs=torch.eye(self.state_dim, device=leaf.device, dtype=leaf.dtype),
            is_grads_batched=True, allow_unused=True)
        a_jac = grads[0]
        b_jac = self._flat_parameter_grad(grads[1:], batch=self.state_dim)
        with torch.no_grad():
            tangent = b_jac if self.cut_state_credit else a_jac @ self.sensitivity + b_jac
            norm = float(tangent.norm())
            self.max_sensitivity_norm = max(self.max_sensitivity_norm, norm)
            if not math.isfinite(norm):
                raise FloatingPointError("nonfinite forward sensitivity")
            if self.sensitivity_limit is not None and norm > self.sensitivity_limit:
                tangent = tangent * (self.sensitivity_limit / norm)
                self.sensitivity_clips += 1
            self.sensitivity = tangent.detach()
            self.state = vector.detach()
            self.states = tuple(s.detached() for s in new)
            for b, s, enabled in zip(self.runtime.core.blocks, self.states, active):
                b.z = s.z.clone(); b.r = s.r.clone(); b.a = s.a.clone()
                b.A.clear(); b.A.extend(x.clone() for x in s.history)
                b.At.clear(); b.At.extend(s.times)
                b.o = None if s.o is None else s.o.clone()
                b.active = enabled
        self.last_time = now_ms
        self.events += 1
        return tuple(updated)

    def sample_discrete(self, name="hand", *, coordinates=(0,), generator=None,
                        threshold=0.0):
        if name not in ("hand", "route") or not math.isfinite(threshold):
            raise ValueError("select a discrete Hand or Route policy")
        leaf = self.state.detach().requires_grad_(True)
        states = self.unpack(leaf)
        raw = self.runtime.adapters[name].forward_train(states[self.runtime.organ_blocks[name]].z)
        count = self.runtime.adapters["hand"].discrete_controls if name == "hand" else raw.numel()
        indices = tuple(coordinates)
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < count for i in indices):
            raise ValueError("coordinates must be unique valid discrete coordinates")
        logits = (raw[list(indices)] - threshold) / self.runtime.noise_scale
        probability = torch.sigmoid(logits)
        action = torch.bernoulli(probability.detach(), generator=generator)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(logits, action, reduction="none")
        if name == "route":
            if not self.runtime.execution_enabled["route"]:
                mask = torch.zeros_like(logp)
            else:
                mask = torch.tensor([i not in (self.runtime.organ_blocks["route"],
                    self.runtime.organ_blocks["goodness"]) for i in indices], device=leaf.device)
            logp = logp * mask
        grads = torch.autograd.grad(logp.sum(), (leaf, *self.parameters), allow_unused=True)
        score = self._flat_parameter_grad(grads[1:]) + grads[0] @ self.sensitivity
        if not torch.isfinite(score).all():
            raise FloatingPointError("nonfinite action credit")
        if name == "route":
            for k, i in enumerate(indices):
                if i == self.runtime.organ_blocks["route"]:
                    action[k] = 1
                elif i == self.runtime.organ_blocks["goodness"]:
                    action[k] = float(self.runtime.goodness_active)
                elif self.runtime.execution_enabled["route"]:
                    self.runtime.core.blocks[i].active = bool(action[k])
        return OnlineAction(action.detach(), probability.detach(), score.detach())

    @property
    def learning_state_bytes(self):
        return self.sensitivity.numel() * self.sensitivity.element_size()


class DelayedOnlineTrainer:
    """Persistent multi-time-scale action traces modulated by one Goodness.

    No action IDs, targets or retrospective graphs are accepted. Reward arrival
    modulates the surviving trace of all past decisions. Trace mixing, reward
    centering, clipping and changing weights make this an approximate update.
    """
    def __init__(self, model: OriginalOnlineRuntime, *, taus_ms=(24.0, 96.0),
                 lr=0.001, baseline_decay=0.95, max_update_norm=0.02,
                 score_limit=10.0, trace_limit=100.0, cut_delay_credit=False):
        scalars = (lr, max_update_norm, score_limit, trace_limit, *taus_ms)
        if not taus_ms or any(not math.isfinite(v) or v <= 0 for v in scalars):
            raise ValueError("learning rates, limits and trace times must be positive")
        if not 0 <= baseline_decay < 1:
            raise ValueError("invalid baseline decay")
        self.model = model
        self.taus = tuple(taus_ms)
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.baseline = 0.5
        self.max_update_norm = max_update_norm
        self.score_limit = score_limit
        self.trace_limit = trace_limit
        self.cut_delay_credit = cut_delay_credit
        self.traces = model.state.new_zeros(len(self.taus), model.param_dim)
        # Capture the baseline BEFORE each action, not at delayed delivery.
        self.baseline_traces = torch.zeros_like(self.traces)
        self.m = model.state.new_zeros(model.param_dim)
        self.v = torch.zeros_like(self.m)
        self.last_time = None
        self.updates = 0
        self.score_clips = 0
        self.trace_clips = 0
        self.total_parameter_path = 0.0

    @torch.no_grad()
    def advance(self, now_ms):
        if type(now_ms) is not int or (self.last_time is not None and now_ms < self.last_time):
            raise ValueError("trace time must be monotonic integer milliseconds")
        if self.last_time is not None:
            dt = now_ms - self.last_time
            if dt and self.cut_delay_credit:
                self.traces.zero_()
                self.baseline_traces.zero_()
            else:
                decay = self.traces.new_tensor([math.exp(-dt / tau) for tau in self.taus])
                self.traces.mul_(decay[:, None])
                self.baseline_traces.mul_(decay[:, None])
        self.last_time = now_ms

    @torch.no_grad()
    def record_action(self, action: OnlineAction, *, now_ms):
        self.advance(now_ms)
        score = action.score
        norm = float(score.norm())
        if norm > self.score_limit:
            score = score * (self.score_limit / norm)
            self.score_clips += 1
        self.traces.add_(score)
        self.baseline_traces.add_(score, alpha=self.baseline)
        norms = self.traces.norm(dim=1)
        factors = (self.trace_limit / norms.clamp_min(1e-30)).clamp_max(1)
        self.trace_clips += int((factors < 1).sum())
        self.traces.mul_(factors[:, None])
        self.baseline_traces.mul_(factors[:, None])

    @torch.no_grad()
    def observe_goodness(self, goodness, *, now_ms):
        if not math.isfinite(goodness) or not 0 <= goodness <= 1:
            raise ValueError("one finite Goodness in [0,1] is required")
        self.advance(now_ms)
        advantage = goodness - self.baseline
        direction = goodness * self.traces.mean(dim=0) - self.baseline_traces.mean(dim=0)
        norm = float(direction.norm())
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite learning direction")
        direction = direction / max(1.0, norm)
        self.updates += 1
        self.m.mul_(0.9).add_(direction, alpha=0.1)
        self.v.mul_(0.999).addcmul_(direction, direction, value=0.001)
        proposal = self.lr * (self.m / (1 - 0.9 ** self.updates)) / (
            (self.v / (1 - 0.999 ** self.updates)).sqrt() + 1e-8)
        delta_norm = float(proposal.norm())
        proposal.mul_(min(1.0, self.max_update_norm / max(delta_norm, 1e-30)))
        offset = 0
        for parameter in self.model.parameters:
            size = parameter.numel()
            candidate = parameter + proposal[offset:offset + size].reshape_as(parameter)
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("nonfinite proposed parameter")
            offset += size
        offset = 0
        for parameter in self.model.parameters:
            size = parameter.numel()
            parameter.add_(proposal[offset:offset + size].reshape_as(parameter))
            offset += size
        actual_norm = float(proposal.norm())
        self.total_parameter_path += actual_norm
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * goodness
        return {"delivery_centered_goodness": advantage, "direction_norm": norm, "update_norm": actual_norm,
                "baseline": self.baseline, "updates": self.updates}

    @property
    def learning_state_bytes(self):
        return self.model.learning_state_bytes + sum(x.numel() * x.element_size()
                                                   for x in (self.traces, self.baseline_traces, self.m, self.v))
