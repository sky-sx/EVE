"""Separate amplitude diagnostic: leave raw eligibility trace unchanged.

Uses detached per-coordinate RMS at the actuator, not a changed trace or new
learned parameters. The outer bootstrap also stops derivatives through RMS.
"""
import torch
from .grouped_write import GroupedWriteKernel, GroupedWriteLearner


class RMSGroupedWriteKernel(GroupedWriteKernel):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.register_buffer('normalizer', torch.ones(self.parameter_size))

    def forward(self, *args, **kwargs):
        raw = self.eligibility
        self.eligibility = raw/self.normalizer
        try:
            return super().forward(*args, **kwargs)
        finally:
            self.eligibility = raw


class RMSGroupedWriteLearner(GroupedWriteLearner):
    def __init__(self, kernel, **settings):
        super().__init__(kernel, **settings)
        self.rms = self.state.new_zeros(kernel.parameter_size)
        self.rms_updates = 0

    def step(self, *, now_ms, cue=None, goodness=None, action=None):
        self.advance_eligibility(now_ms)
        if goodness is not None:
            with torch.no_grad():
                self.rms.mul_(.999).addcmul_(self.kernel.eligibility, self.kernel.eligibility, value=.001)
                self.rms_updates += 1
                self.kernel.normalizer.copy_(torch.sqrt(self.rms/(1-.999**self.rms_updates))+1e-8)
        super().step(now_ms=now_ms, cue=cue, goodness=goodness, action=action)

    @property
    def persistent_tensor_bytes(self):
        return super().persistent_tensor_bytes+(self.rms.numel()+self.kernel.normalizer.numel())*4
