"""Grouped scalar F times signed exponential eligibility, with all-weight writing.

Research bootstrap. Actor eligibility uses streaming original-forward score;
writer eligibility uses the address prototype's conditional self-write score.
E's exponential decay/clipping is preserved. Its value is stopped in outer
derivatives; neither this hybrid eligibility nor bootstrap is a lifetime oracle.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.func import functional_call

from .address_write import AddressWriteKernel, AddressWriteLearner, _AddressBody
from .adapters import HandAdapter


class _GroupedBody(_AddressBody):
    def __init__(self, original):
        super().__init__(original, 1.)
        # Preserve exactly the old task's used Hand computation, drop unused outputs.
        self.hand = HandAdapter(self.width, discrete_controls=1, continuous_controls=0, hidden_size=8)
        with torch.no_grad():
            self.hand.network[0].load_state_dict(original.hand.network[0].state_dict())
            self.hand.network[2].weight.copy_(original.hand.network[2].weight[:1])
            self.hand.network[2].bias.copy_(original.hand.network[2].bias[:1])
            self.write.network[-1].bias.fill_(.05)

    def forward(self, neural, times, now_ms, cue, observation, weights, addresses,
                constant_controller=False):
        n, stride = self.width, self.width * (1+self.hold)
        sources = {i: neural[i*stride:i*stride+n] for i in range(7)}
        readin = None if cue is None else self.ear.forward_train(cue)
        feedback = self.learning_readin(observation)
        pieces = []
        for i, transition in enumerate(self.transitions):
            part = neural[i*stride:(i+1)*stride]
            history = part[n:].reshape(self.hold, n)
            k = len(times[i])
            old = tuple(history[-k:].unbind()) if k else ()
            incoming = readin if i == 1 else feedback if i == 6 else None
            _, _, z, history, _ = transition(old, times[i], now_ms, sources, incoming)
            pieces.extend((z, torch.cat([z.new_zeros(n)]*(self.hold-len(history))+list(history))))
        neural = torch.cat(pieces)
        h = neural[6*stride:6*stride+n]
        if constant_controller:
            h = torch.zeros_like(h)
        values = weights.new_zeros(self.group_count).scatter_add(0, self.group_index, weights)
        values = values/self.group_sizes
        # One scalar per group; all members receive the same signed modulation.
        group_f = self.write(h, values, self.group_addresses)
        return neural, group_f[self.group_index]


class GroupedWriteKernel(AddressWriteKernel):
    def __init__(self, *, seed=11, width=4, hold=3, group_size=32,
                 write_scale=.01, modulation='core'):
        if type(group_size) is not int or group_size < 1:
            raise ValueError('positive group_size required')
        if modulation not in ('core', 'anchored', 'broadcast'):
            raise ValueError('unknown modulation')
        super().__init__(seed=seed, width=width, hold=hold)
        self.body = _GroupedBody(self.body)
        self.schema = []
        start = 0
        named = tuple(self.body.named_parameters())
        self.parameter_size = sum(p.numel() for _, p in named)
        indices, features, counts = [], [], []
        for tensor_id, (name, p) in enumerate(named):
            p.requires_grad_(name.startswith('write.'))
            end = start+p.numel()
            self.schema.append((name, start, end, p.shape))
            for local in range(0, p.numel(), group_size):
                size = min(group_size, p.numel()-local)
                indices.extend([len(counts)]*size)
                counts.append(size)
                features.append((tensor_id/max(1, len(named)-1),
                    (local+(size-1)/2)/max(1, p.numel()-1), p.numel()/self.parameter_size))
            start = end
        self.total_state_size = self.neural_size+self.parameter_size
        self.group_size, self.group_count = group_size, len(counts)
        self.modulation, self.write_scale = modulation, write_scale
        self.body.group_count = self.group_count
        self.body.register_buffer('group_index', torch.tensor(indices, dtype=torch.long))
        self.body.register_buffer('group_sizes', torch.tensor(counts, dtype=torch.float32))
        self.body.register_buffer('group_addresses', torch.tensor(features, dtype=torch.float32))
        self.addresses = self.body.group_addresses
        self.register_buffer('eligibility', self.addresses.new_zeros(self.parameter_size))
        self.writer_indices = torch.cat([torch.arange(a, b) for name, a, b, _ in self.schema
                                         if name.startswith('write.')])

    def forward(self, state, times, now_ms, cue, observation, *, write_enabled=True,
                constant_controller=False):
        base = self.base()
        weights = base+state[self.neural_size:]
        neural, f = functional_call(self.body, (self.mapping(weights), dict(self.body.named_buffers())),
            (state[:self.neural_size], times, now_ms, cue, observation, weights,
             self.addresses, constant_controller), strict=True)
        if self.modulation == 'broadcast':
            f = torch.ones_like(f)*(observation[0]-.5)
        elif self.modulation == 'anchored':
            # Explicit conventional G anchor control, not autonomous Core analysis.
            f = torch.tanh(observation[0]-.5+.25*f)
        command = self.write_scale*self.eligibility*f*observation[1]
        command = command*torch.clamp(.02/command.norm().clamp_min(1e-12), max=1.)
        delta = (weights+command).clamp(-4., 4.)-base if write_enabled else state[self.neural_size:]
        return torch.cat((neural, delta))


class GroupedWriteLearner(AddressWriteLearner):
    def __init__(self, kernel, *, taus_ms=(8., 32., 128.), **settings):
        super().__init__(kernel, **settings)
        self.taus = tuple(taus_ms)
        if any(not math.isfinite(tau) or tau <= 0 for tau in self.taus):
            raise ValueError('positive trace times required')
        self.traces = self.state.new_zeros(len(self.taus), kernel.parameter_size)
        self.actor_tangent = self.state.new_zeros(kernel.neural_size, kernel.parameter_size)
        self.trace_time = None
        self.actor_clips = self.trace_clips = self.score_clips = 0

    @torch.no_grad()
    def advance_eligibility(self, now_ms):
        if self.trace_time is not None:
            if now_ms < self.trace_time:
                raise ValueError('eligibility time must be monotonic')
            decay = self.traces.new_tensor([math.exp(-(now_ms-self.trace_time)/tau) for tau in self.taus])
            self.traces *= decay[:, None]
        self.trace_time = now_ms
        self.kernel.eligibility.copy_(self.traces.mean(dim=0))

    @torch.no_grad()
    def record_action(self, score, *, now_ms):
        self.advance_eligibility(now_ms)
        norm = float(score.norm())
        if norm > 10.:
            score = score*(10./norm)
            self.score_clips += 1
        self.traces += score
        factors = (100./self.traces.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)
        self.trace_clips += int((factors < 1.).sum())
        self.traces *= factors[:, None]
        self.kernel.eligibility.copy_(self.traces.mean(dim=0))

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms <= self.last_time):
            raise ValueError('strictly increasing timestamps required')
        if goodness is not None and (not math.isfinite(goodness) or not 0 <= goodness <= 1):
            raise ValueError('goodness in [0,1] required')
        self.advance_eligibility(now_ms)
        obs = self.state.new_tensor([goodness or 0., float(goodness is not None),
                                    action or 0., float(action is not None)])
        leaf = self.state.detach().requires_grad_(True)
        new = self.kernel(leaf, self.times, now_ms, cue, obs,
            write_enabled=self.write_enabled, constant_controller=self.constant_controller)
        h = self.kernel.neural_size
        local = torch.autograd.grad(new[:h], leaf, grad_outputs=torch.eye(h),
                                    is_grads_batched=True, retain_graph=True)[0]
        actor = local[:, :h]@self.actor_tangent+local[:, h:]
        dual = torch.zeros_like(new, requires_grad=True)
        reverse = torch.autograd.grad(new, (leaf, *self.parameters), dual,
                                      create_graph=True, allow_unused=True)
        targets, seeds = [reverse[0]], [self.meta.T]
        eye = torch.eye(self.param_size)
        start = 0
        for p, vjp in zip(self.parameters, reverse[1:]):
            if vjp is not None:
                targets.append(vjp)
                seeds.append(eye[:, start:start+p.numel()].reshape(self.param_size, *p.shape))
            start += p.numel()
        tangent = torch.autograd.grad(targets, dual, grad_outputs=seeds, is_grads_batched=True)[0].T
        with torch.no_grad():
            if self.cut_write_credit:
                tangent[h:] = 0.
            for value, name in ((tangent, 'meta'), (actor, 'actor')):
                norm = float(value.norm())
                if not math.isfinite(norm) or not torch.isfinite(new).all():
                    raise FloatingPointError('nonfinite streaming state')
                if self.meta_limit is not None and norm > self.meta_limit:
                    value *= self.meta_limit/norm
                    if name == 'meta':
                        self.meta_clips += 1
                    else:
                        self.actor_clips += 1
            change = new[h:]-leaf[h:]
            self.write_path += float(change.norm())
            self.self_write_path += float(change[self.kernel.writer_indices].norm())
            self.state, self.meta, self.actor_tangent = new.detach(), tangent.detach(), actor.detach()
        self.times = tuple(tuple((*old, now_ms)[-self.kernel.hold:]) for old in self.times)
        self.last_time, self.steps = now_ms, self.steps+1

    def policy_score(self, action):
        leaf = self.state.detach().requires_grad_(True)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(
            self.kernel.logits(leaf), leaf.new_tensor(float(action)))
        gradient = torch.autograd.grad(logp, leaf)[0]
        h = self.kernel.neural_size
        score = gradient[:h]@self.actor_tangent+gradient[h:]
        # Writer has behavior credit through previous actual weight writes.
        score[self.kernel.writer_indices] = gradient@self.meta
        return score.detach()

    @property
    def persistent_tensor_bytes(self):
        return super().persistent_tensor_bytes+sum(x.numel()*x.element_size() for x in
            (self.traces, self.actor_tangent, self.kernel.eligibility))
