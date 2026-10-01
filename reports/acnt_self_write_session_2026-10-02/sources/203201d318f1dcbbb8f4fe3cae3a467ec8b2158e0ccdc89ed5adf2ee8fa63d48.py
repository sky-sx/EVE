"""Original seven-Block forward with address-generated writes to ALL weights.

All effective parameters, including the writer, are base + packed deviation.
The base is a coordinate origin, not a second protected effective network.
An external bootstrap updates only writer origins. Its streaming derivative is
exact at fixed origins, approximate across bootstrap steps and tangent clipping.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.func import functional_call

from .self_write import SelfWriteKernel
from .full_write import SharedWriteAdapter


class _AddressBody(nn.Module):
    def __init__(self, original, write_scale):
        super().__init__()
        self.transitions = original.transitions
        self.ear, self.hand = original.ear, original.hand
        self.learning_readin = original.learning_readin
        self.width, self.hold = original.width, original.hold
        self.write = SharedWriteAdapter(self.width, hidden_size=8, step_scale=write_scale)
        # Identity initial write. Bootstrap must learn useful nonzero commands.
        nn.init.zeros_(self.write.network[-1].weight)
        nn.init.zeros_(self.write.network[-1].bias)

    def forward(self, neural, times, now_ms, cue, observation, weights, addresses,
                constant_controller=False):
        n, stride = self.width, self.width * (1 + self.hold)
        sources = {i: neural[i*stride:i*stride+n] for i in range(7)}
        readin = None if cue is None else self.ear.forward_train(cue)
        feedback = self.learning_readin(observation)
        pieces = []
        for i, transition in enumerate(self.transitions):
            part = neural[i*stride:(i+1)*stride]
            history = part[n:].reshape(self.hold, n)
            valid = len(times[i])
            old = tuple(history[-valid:].unbind()) if valid else ()
            incoming = readin if i == 1 else feedback if i == 6 else None
            _, _, z, history, _ = transition(old, times[i], now_ms, sources, incoming)
            pieces.extend((z, torch.cat([z.new_zeros(n)] * (self.hold-len(history)) + list(history))))
        neural = torch.cat(pieces)
        h = neural[6*stride:6*stride+n]
        if constant_controller:
            h = torch.zeros_like(h)
        command = self.write(h, weights, addresses)
        return neural, command


class AddressWriteKernel(nn.Module):
    def __init__(self, *, seed=11, width=4, hold=3, write_scale=.002):
        super().__init__()
        if not math.isfinite(write_scale) or not 0 < write_scale <= .1:
            raise ValueError("write_scale must be in (0,.1]")
        original = SelfWriteKernel(seed=seed, width=width, hold=hold)
        self.body = _AddressBody(original, write_scale)
        self.width, self.hold = width, hold
        self.neural_size = 7 * width * (1 + hold)
        self.schema = []
        named = tuple(self.body.named_parameters())
        start = 0
        addresses = []
        total = sum(p.numel() for _, p in named)
        for i, (name, p) in enumerate(named):
            # All are writable; requires_grad only selects external bootstrap.
            p.requires_grad_(name.startswith("write."))
            end = start + p.numel()
            self.schema.append((name, start, end, p.shape))
            coordinate = torch.arange(p.numel(), dtype=torch.float32)
            addresses.append(torch.stack((
                torch.full_like(coordinate, i/max(1, len(named)-1)),
                coordinate/max(1, p.numel()-1),
                torch.full_like(coordinate, p.numel()/total)), dim=1))
            start = end
        self.parameter_size = total
        self.total_state_size = self.neural_size + total
        self.register_buffer("addresses", torch.cat(addresses))
        self.writer_indices = torch.cat([torch.arange(a, b) for name, a, b, _ in self.schema
                                         if name.startswith("write.")])

    def base(self):
        return torch.cat([p.reshape(-1) for p in self.body.parameters()])

    def effective(self, state):
        return self.base() + state[self.neural_size:]

    def mapping(self, flat):
        return {name: flat[a:b].reshape(shape) for name, a, b, shape in self.schema}

    def initial_state(self):
        return self.addresses.new_zeros(self.total_state_size)

    def forward(self, state, times, now_ms, cue, observation, *, write_enabled=True,
                constant_controller=False):
        base = self.base()
        weights = base + state[self.neural_size:]
        neural, command = functional_call(self.body, (self.mapping(weights), dict(self.body.named_buffers())),
            (state[:self.neural_size], times, now_ms, cue, observation, weights,
             self.addresses, constant_controller), strict=True)
        if write_enabled:
            delta = (weights + command).clamp(-4., 4.) - base
        else:
            delta = state[self.neural_size:]
        return torch.cat((neural, delta))

    def logits(self, state):
        weights = self.mapping(self.effective(state))
        hand = {name.removeprefix("hand."): value for name, value in weights.items()
                if name.startswith("hand.")}
        stride = self.width * (1 + self.hold)
        z = state[2*stride:2*stride+self.width]
        # HandAdapter.forward_train, with every Hand weight replaced by actual state.
        raw = functional_call(self.body.hand.network,
            {name.removeprefix("network."): value for name, value in hand.items()}, (z,), strict=True)
        return raw[0]


class AddressWriteLearner:
    """Bounded streaming bootstrap; no old events or trajectory graph retained.

T = d(state)/d(writer origin); update via double-backward JVP, never form a
dense state-by-state Jacobian. Observed actions/G are conditional constants.
"""
    def __init__(self, kernel, *, lr=.001, meta_limit=1000., write_enabled=True,
                 cut_write_credit=False, constant_controller=False):
        self.kernel, self.lr, self.meta_limit = kernel, lr, meta_limit
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("finite nonnegative lr required")
        self.parameters = [p for p in kernel.parameters() if p.requires_grad]
        self.param_size = sum(p.numel() for p in self.parameters)
        self.state = kernel.initial_state().detach()
        self.meta = self.state.new_zeros(self.state.numel(), self.param_size)
        self.moment = self.state.new_zeros(self.param_size)
        self.square = torch.zeros_like(self.moment)
        self.times = tuple(() for _ in range(7))
        self.last_time = None
        self.write_enabled, self.cut_write_credit = write_enabled, cut_write_credit
        self.constant_controller = constant_controller
        self.steps = self.updates = self.meta_clips = 0
        self.write_path = self.self_write_path = 0.

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms <= self.last_time):
            raise ValueError("strictly increasing integer timestamps required")
        if goodness is not None and (not math.isfinite(goodness) or not 0 <= goodness <= 1):
            raise ValueError("goodness in [0,1] required")
        if action is not None and (type(action) is not int or action not in (0, 1)):
            raise ValueError("binary action required")
        obs = self.state.new_tensor([goodness or 0., float(goodness is not None),
                                    action or 0., float(action is not None)])
        leaf = self.state.detach().requires_grad_(True)
        new = self.kernel(leaf, self.times, now_ms, cue, obs,
                          write_enabled=self.write_enabled,
                          constant_controller=self.constant_controller)
        dual = torch.zeros_like(new, requires_grad=True)
        reverse = torch.autograd.grad(new, (leaf, *self.parameters), dual,
                                      create_graph=True, allow_unused=True)
        targets, seeds = [reverse[0]], [self.meta.T]
        eye = torch.eye(self.param_size, dtype=new.dtype)
        start = 0
        for p, vjp in zip(self.parameters, reverse[1:]):
            if vjp is not None:
                targets.append(vjp)
                seeds.append(eye[:, start:start+p.numel()].reshape(self.param_size, *p.shape))
            start += p.numel()
        tangent = torch.autograd.grad(targets, dual, grad_outputs=seeds,
                                      is_grads_batched=True)[0].T
        with torch.no_grad():
            if self.cut_write_credit:
                tangent[self.kernel.neural_size:] = 0.
            norm = float(tangent.norm())
            if not torch.isfinite(new).all() or not math.isfinite(norm):
                raise FloatingPointError("nonfinite joint state or tangent")
            if self.meta_limit is not None and norm > self.meta_limit:
                tangent *= self.meta_limit/norm
                self.meta_clips += 1
            change = new[self.kernel.neural_size:] - leaf[self.kernel.neural_size:]
            self.write_path += float(change.norm())
            self.self_write_path += float(change[self.kernel.writer_indices].norm())
            self.state, self.meta = new.detach(), tangent.detach()
        self.times = tuple(tuple((*old, now_ms)[-self.kernel.hold:]) for old in self.times)
        self.last_time, self.steps = now_ms, self.steps+1

    def policy_score(self, action):
        leaf = self.state.detach().requires_grad_(True)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(
            self.kernel.logits(leaf), leaf.new_tensor(float(action)))
        gradient = torch.autograd.grad(logp, leaf)[0]
        return (gradient @ self.meta).detach()

    def sample(self, generator):
        with torch.no_grad():
            probability = float(torch.sigmoid(self.kernel.logits(self.state)))
            action = int(torch.bernoulli(self.state.new_tensor(probability), generator=generator))
        return action, probability, self.policy_score(action)

    def learn(self, score, goodness):
        if score.shape != (self.param_size,) or not torch.isfinite(score).all():
            raise ValueError("finite bootstrap score required")
        if not math.isfinite(goodness) or not 0 <= goodness <= 1:
            raise ValueError("goodness in [0,1] required")
        if self.lr == 0:
            return 0.
        direction = (goodness-.5) * score
        direction /= max(1., float(direction.norm()))
        with torch.no_grad():
            m = .9*self.moment + .1*direction
            v = .999*self.square + .001*direction.square()
            count = self.updates+1
            update = self.lr*m/(1-.9**count)/(torch.sqrt(v/(1-.999**count))+1e-8)
            update *= min(1., .02/max(float(update.norm()), 1e-12))
            # Origin update writes the same effective writer coordinates;
            # cap actual weights, including their accumulated self-writes.
            effective = self.kernel.effective(self.state).detach()
            proposed = (effective[self.kernel.writer_indices]+update).clamp(-4., 4.)
            update = proposed-effective[self.kernel.writer_indices]
            if not torch.isfinite(proposed).all():
                raise FloatingPointError("invalid bootstrap update")
            start = 0
            for p in self.parameters:
                p.add_(update[start:start+p.numel()].reshape_as(p))
                start += p.numel()
            self.moment, self.square, self.updates = m, v, count
        return float(update.norm())

    @property
    def persistent_tensor_bytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.state, self.meta, self.moment, self.square))
