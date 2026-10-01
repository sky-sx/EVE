"""Self-calibrating vector Write using finite streaming numerical credit.

Seven ordinary original Blocks are unchanged. New adapter normalization is
causal, finite and detached. This is an immediate-reward reference, not an
exact lifetime gradient: parameter writes, moments and statistics are stopped.
No outer optimizer changes any parameter. A task-independent calibration loss
provides controller eligibility; Write gates also control those actual writes.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.func import functional_call

from .grouped_write_independent import IndependentGroupedWriteKernel


class CalibratedVectorAdapter(nn.Module):
    def __init__(self, previous):
        super().__init__()
        self.linear = previous.linear
        self.register_buffer('mean', torch.zeros(self.linear.in_features))
        self.register_buffer('scale', torch.ones(self.linear.in_features))

    def forward(self, hidden):
        return torch.tanh(self.linear((hidden-self.mean)/self.scale))


class CalibratedWriteKernel(IndependentGroupedWriteKernel):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.body.write = CalibratedVectorAdapter(self.body.write)
        self.register_buffer('hand_mean', torch.zeros(self.width))
        self.register_buffer('hand_scale', torch.ones(self.width))
        control = torch.zeros(self.parameter_size, dtype=torch.bool)
        for name, start, end, _ in self.schema:
            if name.startswith(('transitions.6.', 'learning_readin.', 'write.')):
                control[start:end] = True
        self.register_buffer('control_coordinates', control)
        sizes = self.body.group_sizes.to(dtype=torch.long)
        first = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
        # Groups are contiguous within tensors. Build/check membership in O(P),
        # rather than scanning all P coordinates separately for every group.
        assert torch.equal(control[first][self.body.group_index], control)
        self.register_buffer('group_first', first)
        self.register_buffer('control_groups', control[self.group_first])

    def logits(self, state):
        mapping = self.mapping(self.effective(state))
        params = {name.removeprefix('hand.network.'): value for name, value in mapping.items()
                  if name.startswith('hand.network.')}
        stride = self.width*(1+self.hold)
        z = state[2*stride:2*stride+self.width]
        return functional_call(self.body.hand.network, params,
            ((z-self.hand_mean)/self.hand_scale,), strict=True)[0]

    def readout(self, neural, weights):
        stride = self.width*(1+self.hold)
        params = {name.removeprefix('write.'): value for name, value in self.mapping(weights).items()
                  if name.startswith('write.')}
        raw = functional_call(self.body.write,
            (params, dict(self.body.write.named_buffers())),
            (neural[6*stride:6*stride+self.width],), strict=True)
        # Actor groups receive a signed reward prediction. Controller groups
        # receive positive calibration gain, including groups of Write itself.
        return torch.where(self.control_groups, (raw+1)*.5, raw)


class CalibratedWriteLearner:
    def __init__(self, kernel, *, mode='core', taus_ms=(8., 32., 128.),
                 actor_lr=.001, head_lr=.01, control_lr=.003,
                 max_update=.02, tangent_limit=1000., normalize=True):
        if mode not in ('core', 'broadcast', 'cut_calibration', 'no_write'):
            raise ValueError('unknown mode')
        if not taus_ms or any(not math.isfinite(t) or t <= 0 for t in taus_ms):
            raise ValueError('positive time constants required')
        self.kernel, self.mode, self.taus = kernel, mode, tuple(taus_ms)
        self.max_update, self.tangent_limit, self.normalize = max_update, tangent_limit, normalize
        self.state = kernel.initial_state().detach()
        p, h = kernel.parameter_size, kernel.neural_size
        self.tangent = torch.zeros(h, p)
        self.traces = torch.zeros(len(self.taus), p)
        self.moment, self.square = torch.zeros(p), torch.zeros(p)
        self.rates = torch.full((p,), actor_lr)
        self.rates[kernel.control_coordinates] = control_lr
        for name, start, end, _ in kernel.schema:
            if name.startswith('hand.'):
                self.rates[start:end] = head_lr
        self.stats = torch.zeros(2, 2, kernel.width)  # role, first/second moments, neuron
        self.stats_count = [0, 0]
        self.times = tuple(() for _ in range(7))
        self.last_time = self.trace_time = None
        self.steps = self.updates = self.tangent_clips = self.update_clips = 0
        self.ever_written = torch.zeros(p, dtype=torch.bool)

    @torch.no_grad()
    def normalize_role(self, role, hidden):
        if not self.normalize:
            return
        self.stats_count[role] += 1
        self.stats[role, 0].mul_(.99).add_(hidden, alpha=.01)
        self.stats[role, 1].mul_(.99).add_(hidden.square(), alpha=.01)
        correction = 1-.99**self.stats_count[role]
        mean = self.stats[role, 0]/correction
        variance = self.stats[role, 1]/correction-mean.square()
        scale = variance.clamp_min(1e-6).sqrt()
        if role == 0:
            self.kernel.hand_mean.copy_(mean)
            self.kernel.hand_scale.copy_(scale)
        else:
            self.kernel.body.write.mean.copy_(mean)
            self.kernel.body.write.scale.copy_(scale)

    @torch.no_grad()
    def advance_trace(self, now_ms):
        if self.trace_time is not None:
            if now_ms < self.trace_time:
                raise ValueError('monotonic eligibility time required')
            decay = self.traces.new_tensor([math.exp(-(now_ms-self.trace_time)/tau) for tau in self.taus])
            self.traces.mul_(decay[:, None])
        self.trace_time = now_ms

    def policy_score(self, action):
        leaf = self.state.detach().requires_grad_(True)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(
            self.kernel.logits(leaf), leaf.new_tensor(float(action)))
        gradient = torch.autograd.grad(logp, leaf)[0]
        h = self.kernel.neural_size
        return (gradient[:h]@self.tangent+gradient[h:]).detach()

    def sample(self, generator):
        stride = self.kernel.width*(1+self.kernel.hold)
        self.normalize_role(0, self.state[2*stride:2*stride+self.kernel.width])
        with torch.no_grad():
            probability = float(torch.sigmoid(self.kernel.logits(self.state)))
            action = int(torch.bernoulli(self.state.new_tensor(probability), generator=generator))
        return action, probability, self.policy_score(action)

    @torch.no_grad()
    def record_action(self, score, *, now_ms):
        self.advance_trace(now_ms)
        score = score/max(1., float(score.norm())/10.)
        self.traces.add_(score)
        self.traces.mul_((100./self.traces.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None])

    @torch.no_grad()
    def commit(self, eligibility, group_signals):
        signals = group_signals[self.kernel.body.group_index]
        direction = eligibility*signals
        direction /= max(1., float(direction.norm()))
        count = self.updates+1
        self.moment.mul_(.9).add_(direction, alpha=.1)
        self.square.mul_(.999).addcmul_(direction, direction, value=.001)
        delta = self.rates*(self.moment/(1-.9**count))/(
            (self.square/(1-.999**count)).sqrt()+1e-8)
        norm = float(delta.norm())
        self.update_clips += int(norm > self.max_update)
        delta *= min(1., self.max_update/max(norm, 1e-30))
        before = self.kernel.effective(self.state).detach()
        after = (before+delta).clamp(-4., 4.)
        if not torch.isfinite(after).all():
            raise FloatingPointError('nonfinite actual parameter write')
        self.state[self.kernel.neural_size:] = after-self.kernel.base().detach()
        self.ever_written |= after != before
        self.updates = count
        return float((after-before).norm())

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms <= self.last_time):
            raise ValueError('strictly increasing integer timestamps required')
        if goodness is not None and (not math.isfinite(goodness) or not 0 <= goodness <= 1):
            raise ValueError('goodness in [0,1] required')
        if action is not None and (type(action) is not int or action not in (0, 1)):
            raise ValueError('binary action required')
        self.advance_trace(now_ms)
        kernel, h = self.kernel, self.kernel.neural_size
        weights = kernel.effective(self.state).detach().requires_grad_(True)
        old = self.state[:h].detach().requires_grad_(True)
        obs = old.new_tensor([goodness or 0., float(goodness is not None),
                              action or 0., float(action is not None)])
        new, _ = functional_call(kernel.body,
            (kernel.mapping(weights), dict(kernel.body.named_buffers())),
            (old, self.times, now_ms, cue, obs, weights, kernel.addresses, False), strict=True)
        a, b = torch.autograd.grad(new, (old, weights), grad_outputs=torch.eye(h),
                                  is_grads_batched=True)
        tangent = a@self.tangent+b
        norm = float(tangent.norm())
        if not torch.isfinite(new).all() or not math.isfinite(norm):
            raise FloatingPointError('nonfinite streaming neural credit')
        if self.tangent_limit is not None and norm > self.tangent_limit:
            tangent *= self.tangent_limit/norm
            self.tangent_clips += 1
        self.state = torch.cat((new.detach(), self.state[h:])).detach()
        self.tangent = tangent.detach()
        diagnostics = {}
        if goodness is not None:
            stride = kernel.width*(1+kernel.hold)
            self.normalize_role(1, new.detach()[6*stride:6*stride+kernel.width])
            neural = new.detach().requires_grad_(True)
            actual = kernel.effective(self.state).detach().requires_grad_(True)
            f = kernel.readout(neural, actual)
            target = torch.where(kernel.control_groups, torch.full_like(f, .5),
                                 torch.full_like(f, 2*goodness-1))
            loss = (f-target).square().mean()
            state_grad, direct = torch.autograd.grad(loss, (neural, actual))
            calibration = -(state_grad@self.tangent+direct)
            e = self.traces.mean(dim=0).clone()
            e[kernel.control_coordinates] = calibration[kernel.control_coordinates]
            if self.mode == 'cut_calibration':
                e[kernel.control_coordinates] = 0.
            signals = f.detach().clone()
            if self.mode == 'broadcast':
                signals[~kernel.control_groups] = 2*goodness-1
            update = self.commit(e.detach(), signals) if self.mode != 'no_write' else 0.
            diagnostics = dict(calibration_loss=float(loss.detach()),
                actor_f_mean=float(f[~kernel.control_groups].detach().mean()),
                actor_f_sign_accuracy=float(((f[~kernel.control_groups].detach() >= 0) == bool(goodness >= .5)).float().mean()),
                update_norm=update)
        self.times = tuple(tuple((*old_times, now_ms)[-kernel.hold:]) for old_times in self.times)
        self.last_time, self.steps = now_ms, self.steps+1
        return diagnostics

    @property
    def persistent_tensor_bytes(self):
        k = self.kernel
        values = (self.state, self.tangent, self.traces, self.moment, self.square,
                  self.rates, self.stats, self.ever_written, k.hand_mean, k.hand_scale,
                  k.body.write.mean, k.body.write.scale)
        return sum(v.numel()*v.element_size() for v in values)
