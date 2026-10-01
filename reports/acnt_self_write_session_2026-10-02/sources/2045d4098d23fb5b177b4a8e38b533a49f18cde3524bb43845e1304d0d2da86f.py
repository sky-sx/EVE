"""Joint action learning and pre-teacher action-conditional goodness prediction.

An ordinary existing Block (5) feeds a two-output Goodness Adapter. Selected
output supervision uses only teacher G after the actual action. All parameters,
including this adapter and Write, remain covered by the self-write interface.
Predicted G never substitutes for teacher G in calibration or actor modulation.
"""
import math
import torch
from torch import nn
from torch.func import functional_call

from .calibrated_write import CalibratedVectorAdapter
from .grouped_write_independent import IndependentGroupAdapter, independent_group_count
from experiments.calibrated_write_exploration import ExploringKernel, ExploringLearner


class GoodnessAdapter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 8), nn.LeakyReLU(.1), nn.Linear(8, 2))
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))

    def forward(self, hidden):
        return self.network((hidden-self.mean)/self.scale)


class GoodnessPredictionKernel(ExploringKernel):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.body.hand.network[1] = nn.LeakyReLU(.1)
        self.body.goodness = GoodnessAdapter(self.width)
        actor = [(name, p) for name, p in self.body.named_parameters() if not name.startswith('write.')]
        groups = independent_group_count(sum(math.ceil(p.numel()/self.group_size) for _, p in actor),
                                         self.width, self.group_size)
        previous = self.body.write.linear
        replacement = IndependentGroupAdapter(self.width, groups)
        with torch.no_grad():
            count = min(previous.out_features, groups)
            replacement.linear.weight[:count].copy_(previous.weight[:count])
            replacement.linear.bias[:count].copy_(previous.bias[:count])
        self.body.write = CalibratedVectorAdapter(replacement)
        named = tuple(self.body.named_parameters())
        self.parameter_size = sum(p.numel() for _, p in named)
        self.total_state_size = self.neural_size+self.parameter_size
        self.schema, indices, counts, features = [], [], [], []
        start = 0
        for tensor_id, (name, p) in enumerate(named):
            p.requires_grad_(name.startswith(('write.', 'goodness.')))
            self.schema.append((name, start, start+p.numel(), p.shape))
            for local in range(0, p.numel(), self.group_size):
                size = min(self.group_size, p.numel()-local)
                indices.extend([len(counts)]*size)
                counts.append(size)
                features.append((tensor_id/max(1,len(named)-1),
                    (local+(size-1)/2)/max(1,p.numel()-1), p.numel()/self.parameter_size))
            start += p.numel()
        assert len(counts) == groups
        self.group_count = self.body.group_count = groups
        self.body.group_index = torch.tensor(indices, dtype=torch.long)
        self.body.group_sizes = torch.tensor(counts, dtype=torch.float32)
        self.body.group_addresses = torch.tensor(features, dtype=torch.float32)
        self.addresses = self.body.group_addresses
        self.eligibility = self.addresses.new_zeros(self.parameter_size)
        self.writer_indices = torch.cat([torch.arange(a,b) for name,a,b,_ in self.schema if name.startswith('write.')])
        control = torch.zeros(self.parameter_size, dtype=torch.bool)
        evaluation = torch.zeros_like(control)
        for name, a, b, _ in self.schema:
            if name.startswith(('transitions.5.', 'goodness.')):
                evaluation[a:b] = True
            if name.startswith(('transitions.5.', 'transitions.6.', 'learning_readin.', 'write.', 'goodness.')):
                control[a:b] = True
        self.control_coordinates = control
        self.register_buffer('evaluation_coordinates', evaluation)
        sizes = self.body.group_sizes.to(torch.long)
        self.group_first = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
        assert torch.equal(control[self.group_first][self.body.group_index], control)
        self.control_groups = control[self.group_first]

    def goodness_logits(self, state):
        params = {name.removeprefix('goodness.'): value for name,value in
            self.mapping(self.effective(state)).items() if name.startswith('goodness.')}
        stride = self.width*(1+self.hold)
        hidden = state[5*stride:5*stride+self.width]
        return functional_call(self.body.goodness, (params, dict(self.body.goodness.named_buffers())),
                               (hidden,), strict=True)


class GoodnessPredictionLearner(ExploringLearner):
    def __init__(self, kernel, *, goodness_lr=.01, evaluation_core_lr=.001,
                 evaluation_weight=1., **settings):
        super().__init__(kernel, **settings)
        if not math.isfinite(evaluation_weight) or evaluation_weight <= 0:
            raise ValueError('positive finite evaluation loss weight required')
        self.evaluation_weight = evaluation_weight
        self.evaluation_traces = torch.zeros_like(self.traces)
        self.evaluation_stats = torch.zeros(2, kernel.width)
        self.evaluation_stats_count = 0
        self.pending_prediction = None
        self.teacher_for_commit = None
        self.auxiliary_direction_norm = 0.
        for name,a,b,_ in kernel.schema:
            if name.startswith('goodness.'):
                self.rates[a:b] = goodness_lr
            elif name.startswith('transitions.5.'):
                self.rates[a:b] = evaluation_core_lr

    @torch.no_grad()
    def advance_trace(self, now_ms):
        previous = self.trace_time
        super().advance_trace(now_ms)
        if previous is not None:
            decay = self.traces.new_tensor([math.exp(-(now_ms-previous)/tau) for tau in self.taus])
            self.evaluation_traces.mul_(decay[:, None])

    def judge(self, action, *, now_ms):
        if type(action) is not int or action not in (0,1):
            raise ValueError('binary actual action required')
        if self.pending_prediction is not None:
            raise RuntimeError('previous prediction still awaiting teacher feedback')
        self.advance_trace(now_ms)
        kernel, stride = self.kernel, self.kernel.width*(1+self.kernel.hold)
        with torch.no_grad():
            hidden = self.state[5*stride:5*stride+kernel.width]
            self.evaluation_stats_count += 1
            self.evaluation_stats[0].mul_(.99).add_(hidden, alpha=.01)
            self.evaluation_stats[1].mul_(.99).add_(hidden.square(), alpha=.01)
            correction = 1-.99**self.evaluation_stats_count
            mean = self.evaluation_stats[0]/correction
            variance = self.evaluation_stats[1]/correction-mean.square()
            kernel.body.goodness.mean.copy_(mean)
            kernel.body.goodness.scale.copy_(variance.clamp_min(1e-6).sqrt())
        leaf = self.state.detach().requires_grad_(True)
        logits = kernel.goodness_logits(leaf)
        probabilities = torch.sigmoid(logits).detach()
        gradient = torch.autograd.grad(logits[action], leaf)[0]
        h = kernel.neural_size
        tangent = (gradient[:h]@self.tangent+gradient[h:]).detach()
        tangent = tangent*kernel.evaluation_coordinates
        if not torch.isfinite(tangent).all():
            raise FloatingPointError('nonfinite pre-feedback evaluation credit')
        tangent = tangent/max(1., float(tangent.norm())/10.)
        with torch.no_grad():
            self.evaluation_traces.add_(tangent)
            self.evaluation_traces.mul_((100./self.evaluation_traces.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None])
        self.pending_prediction = float(probabilities[action])
        return dict(predicted_goodness=self.pending_prediction,
                    goodness_if_0=float(probabilities[0]), goodness_if_1=float(probabilities[1]))

    @torch.no_grad()
    def commit(self, eligibility, group_signals):
        if self.teacher_for_commit is None or self.pending_prediction is None:
            raise RuntimeError('teacher feedback must follow a pre-feedback prediction')
        # Negative BCE gradient: (teacher-p) * derivative of the selected logit.
        # Its forward features and tangent were captured BEFORE current teacher G.
        auxiliary = self.evaluation_weight*(self.teacher_for_commit-self.pending_prediction)*self.evaluation_traces.mean(0)
        self.auxiliary_direction_norm = float(auxiliary.norm())
        return super().commit(eligibility+auxiliary, group_signals)

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        if goodness is not None and self.pending_prediction is None:
            raise RuntimeError('cannot train evaluation without a prior prediction')
        self.teacher_for_commit = goodness
        try:
            diagnostics = super().step(now_ms=now_ms, cue=cue, goodness=goodness, action=action)
            if goodness is not None:
                diagnostics['evaluation_direction_norm'] = self.auxiliary_direction_norm
                self.pending_prediction = None
            return diagnostics
        finally:
            self.teacher_for_commit = None

    @property
    def persistent_tensor_bytes(self):
        values = (self.evaluation_traces, self.evaluation_stats,
                  self.kernel.body.goodness.mean, self.kernel.body.goodness.scale)
        return super().persistent_tensor_bytes+sum(v.numel()*v.element_size() for v in values)
