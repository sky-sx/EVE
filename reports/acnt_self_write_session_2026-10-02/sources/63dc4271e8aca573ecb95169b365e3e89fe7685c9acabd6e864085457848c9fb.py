"""Tiny forward-mode meta-training oracle for original ACNT self-writing.

Actor weights are runtime state. A seventh original Block plus a Write Adapter
produce bounded writes. Only that controller's parameters are outer-trained.
No trajectory tape, reward/action ledger or externally supplied write direction.
All-active synchronous scheduling is deliberate; this is not Runtime replacement.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.func import functional_call
from .block import Block
from .adapters import EarAdapter, HandAdapter


class _Transition(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, history, times, now_ms, sources, readin):
        return self.block.transition(history=history, times=times, now_ms=now_ms,
                                     active_z=sources, o=readin)


class SelfWriteKernel(nn.Module):
    """Six original organ Blocks plus one ordinary Block feeding Write.

No ordinary outgoing connections from the Write-connected Block enter the actor.
Its only behavioral influence is through the actual parameter-write interface.
LearningReadIn consumes [received G, G valid, own last action, action valid].
Actor targets: Hand-block <- Ear-block connection and Hand binary logit bias.
"""
    def __init__(self, *, seed=11, width=4, hold=3, write_rate=.02, write_mode="target"):
        super().__init__()
        if not math.isfinite(write_rate) or not 0 < write_rate <= .1:
            raise ValueError("write_rate must be in (0, .1]")
        if write_mode not in ("target", "delta"):
            raise ValueError("write_mode must be target or delta")
        torch.manual_seed(seed)
        self.width, self.hold, self.write_rate = width, hold, write_rate
        self.write_mode = write_mode
        sizes = [width] * 7
        self.transitions = nn.ModuleList([_Transition(Block(i, width, sizes,
            hold_tick=hold, nlm_hidden_dim=2, readin=i in (1, 6))) for i in range(7)])
        self.ear = EarAdapter(width, window_samples=4)
        self.hand = HandAdapter(width, hidden_size=8)
        self.learning_readin = nn.Linear(4, 2 * width, dtype=torch.float32)
        # u[2n], v[n], logit-bias target, write gate; rank-one matrix write.
        self.write = nn.Linear(width, 3 * width + 2, dtype=torch.float32)
        nn.init.normal_(self.write.weight, std=.1)
        nn.init.zeros_(self.write.bias)
        with torch.no_grad():
            for destination in self.transitions[:6]:
                destination.block.W_ij[6].zero_()
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(("transitions.6.", "learning_readin.", "write.")))
        self.state_size = 7 * width * (1 + hold)
        self.fast_size = 2 * width * width + 1
        self.total_state_size = self.state_size + self.fast_size
        self.register_buffer("initial_fast", torch.cat((
            self.transitions[2].block.W_ij[1].detach().flatten(),
            self.hand.network[2].bias[:1].detach())).clone())

    def initial_state(self):
        return torch.cat((self.initial_fast.new_zeros(self.state_size), self.initial_fast))

    def unpack(self, state, times):
        n, stride = self.width, self.width * (1 + self.hold)
        states = []
        for i in range(7):
            part = state[i * stride:(i + 1) * stride]
            history = part[n:].reshape(self.hold, n)
            k = len(times[i])
            states.append((part[:n], tuple(history[-k:].unbind()) if k else ()))
        return states, state[self.state_size:]

    def logits(self, state):
        n, stride = self.width, self.width * (1 + self.hold)
        hand_z = state[2 * stride:2 * stride + n]
        # Raw output coordinate 0: fixed adapter mapping, mutable actual bias.
        raw = self.hand.forward_train(hand_z)[0]
        return raw - self.hand.network[2].bias[0] + state[-1]

    def forward(self, state, times, now_ms, cue, learning_input, *, write_enabled=True,
                constant_controller=False):
        states, fast = self.unpack(state, times)
        sources = {i: old[0] for i, old in enumerate(states)}
        actor_readin = None if cue is None else self.ear.forward_train(cue)
        learner_readin = self.learning_readin(learning_input)
        pieces = []
        learning_z = None
        for i, (transition, old) in enumerate(zip(self.transitions, states)):
            readin = actor_readin if i == 1 else learner_readin if i == 6 else None
            args = (old[1], times[i], now_ms, sources, readin)
            if i == 2:
                params = {"block.W_ij.1": fast[:-1].reshape(2 * self.width, self.width)}
                _, _, z, history, _ = functional_call(transition, params, args)
            else:
                _, _, z, history, _ = transition(*args)
            padding = [z.new_zeros(self.width)] * (self.hold - len(history))
            pieces.extend((z, torch.cat(padding + list(history))))
            if i == 6:
                learning_z = z
        neural = torch.cat(pieces)
        raw = self.write(torch.zeros_like(learning_z) if constant_controller else learning_z)
        n = self.width
        u, v = torch.tanh(raw[:2*n]), torch.tanh(raw[2*n:3*n])
        target = torch.cat(((2 * u[:, None] * v[None, :]).flatten(),
                            (4 * torch.tanh(raw[3*n])).reshape(1)))
        rate = self.write_rate * torch.sigmoid(raw[-1])
        if not write_enabled:
            new_fast = fast
        elif self.write_mode == "target":
            # Legacy target interpolation: zero target is NOT a no-op.
            new_fast = fast + rate * (target - fast)
        else:
            # Residual write: zero proposed change keeps actual actor weights.
            proposal = fast + rate * target
            new_fast = torch.cat((proposal[:-1].clamp(-2, 2), proposal[-1:].clamp(-4, 4)))
        return torch.cat((neural, new_fast))


class SelfWriteLearner:
    """One-event derivatives carry credit through *actual writes* into the future.

Fixed phi: exact conditional common-trajectory tangent including writes.
Changing phi: stop-outer-update approximation, not an exact lifetime gradient.
Observed rewards and sampled actions are external constants in local derivatives.
"""
    def __init__(self, kernel: SelfWriteKernel, *, lr=.001, meta_limit=1000.,
                 write_enabled=True, cut_write_credit=False, constant_controller=False):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if meta_limit is not None and (not math.isfinite(meta_limit) or meta_limit <= 0):
            raise ValueError("meta_limit must be positive or None")
        self.kernel = kernel
        self.named_parameters = [(name, p) for name, p in kernel.named_parameters() if p.requires_grad]
        self.parameters = [p for _, p in self.named_parameters]
        self.param_size = sum(p.numel() for p in self.parameters)
        self.state = kernel.initial_state().detach()
        self.meta = self.state.new_zeros(kernel.total_state_size, self.param_size)
        self.moment = self.state.new_zeros(self.param_size)
        self.square = self.state.new_zeros(self.param_size)
        self.times = tuple(() for _ in range(7))
        self.last_time = None
        self.lr, self.meta_limit = lr, meta_limit
        self.write_enabled, self.cut_write_credit = write_enabled, cut_write_credit
        self.constant_controller = constant_controller
        self.steps, self.updates, self.meta_clips = 0, 0, 0
        self.write_path = 0.

    def _flatten(self, gradients, batch=None):
        return torch.cat([(self.state.new_zeros(p.numel()) if batch is None else
                           self.state.new_zeros(batch, p.numel())) if g is None else
                          g.reshape(-1) if batch is None else g.reshape(batch, -1)
                          for p, g in zip(self.parameters, gradients)], dim=-1)

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms <= self.last_time):
            raise ValueError("strictly increasing integer timestamps required")
        if goodness is not None and (not math.isfinite(goodness) or not 0 <= goodness <= 1):
            raise ValueError("goodness must be absent or in [0,1]")
        if action is not None and (type(action) is not int or action not in (0, 1)):
            raise ValueError("action must be absent or binary")
        obs = self.state.new_tensor([0. if goodness is None else goodness, float(goodness is not None),
                                     0. if action is None else action, float(action is not None)])
        leaf = self.state.detach().requires_grad_(True)
        new = self.kernel(leaf, self.times, now_ms, cue, obs,
                          write_enabled=self.write_enabled, constant_controller=self.constant_controller)
        gradients = torch.autograd.grad(new, (leaf, *self.parameters),
            grad_outputs=torch.eye(new.numel(), dtype=new.dtype, device=new.device),
            is_grads_batched=True, allow_unused=True)
        with torch.no_grad():
            tangent = gradients[0] @ self.meta + self._flatten(gradients[1:], new.numel())
            if self.cut_write_credit:
                tangent[self.kernel.state_size:] = 0
            norm = float(tangent.norm())
            if not math.isfinite(norm) or not torch.isfinite(new).all():
                raise FloatingPointError("nonfinite state/meta tangent")
            if self.meta_limit is not None and norm > self.meta_limit:
                tangent *= self.meta_limit / norm
                self.meta_clips += 1
            self.write_path += float((new[self.kernel.state_size:] - leaf[self.kernel.state_size:]).norm())
            self.state, self.meta = new.detach(), tangent.detach()
        self.times = tuple(tuple((*old, now_ms)[-self.kernel.hold:]) for old in self.times)
        self.last_time, self.steps = now_ms, self.steps + 1

    def policy_score(self, action):
        if type(action) is not int or action not in (0, 1):
            raise ValueError("binary action required")
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
            raise ValueError("finite controller score required")
        if not math.isfinite(goodness) or not 0 <= goodness <= 1:
            raise ValueError("goodness must be in [0,1]")
        if self.lr == 0:
            return 0.
        # Constant pre-action baseline: no feedback analysis substituted by an EMA.
        direction = (goodness - .5) * score
        direction = direction / max(1., float(direction.norm()))
        with torch.no_grad():
            next_m = .9 * self.moment + .1 * direction
            next_v = .999 * self.square + .001 * direction.square()
            count = self.updates + 1
            delta = self.lr * next_m / (1 - .9**count) / (torch.sqrt(next_v / (1 - .999**count)) + 1e-8)
            delta *= min(1., .02 / max(float(delta.norm()), 1e-12))
            proposed = []
            offset = 0
            for p in self.parameters:
                value = p + delta[offset:offset + p.numel()].reshape_as(p)
                if not torch.isfinite(value).all():
                    raise FloatingPointError("invalid proposed controller update")
                proposed.append(value)
                offset += p.numel()
            for p, value in zip(self.parameters, proposed):
                p.copy_(value)
            self.moment, self.square, self.updates = next_m, next_v, count
        return float(delta.norm())

    @property
    def persistent_tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.state, self.meta, self.moment, self.square))
