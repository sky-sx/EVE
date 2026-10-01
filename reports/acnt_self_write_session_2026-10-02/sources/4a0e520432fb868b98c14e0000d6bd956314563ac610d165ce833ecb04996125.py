"""Independent group readout, including its own parameters in the group budget."""
import math
import torch
from torch import nn

from .grouped_write import GroupedWriteKernel


class IndependentGroupAdapter(nn.Module):
    def __init__(self, width, groups):
        super().__init__()
        self.linear = nn.Linear(width, groups, dtype=torch.float32)
        nn.init.normal_(self.linear.weight, std=.01)
        nn.init.constant_(self.linear.bias, .05)

    def forward(self, hidden, values, addresses):
        return torch.tanh(self.linear(hidden))


class IndependentGroupedWriteKernel(GroupedWriteKernel):
    def __init__(self, *, group_size=32, **settings):
        super().__init__(group_size=group_size, **settings)
        if group_size <= self.width+1:
            raise ValueError('independent readout needs group_size > width+1 to close its self-coverage budget')
        actor = [(name, p) for name, p in self.body.named_parameters() if not name.startswith('write.')]
        actor_groups = sum(math.ceil(p.numel()/group_size) for _, p in actor)
        groups = actor_groups
        for _ in range(100):
            total = actor_groups+math.ceil(self.width*groups/group_size)+math.ceil(groups/group_size)
            if total == groups:
                break
            groups = total
        else:
            raise ValueError('group budget did not converge')
        self.body.write = IndependentGroupAdapter(self.width, groups)
        named = tuple(self.body.named_parameters())
        self.parameter_size = sum(p.numel() for _, p in named)
        self.total_state_size = self.neural_size+self.parameter_size
        self.schema = []
        start, indices, counts, features = 0, [], [], []
        for tensor_id, (name, p) in enumerate(named):
            p.requires_grad_(name.startswith('write.'))
            self.schema.append((name, start, start+p.numel(), p.shape))
            for local in range(0, p.numel(), group_size):
                size = min(group_size, p.numel()-local)
                indices.extend([len(counts)]*size)
                counts.append(size)
                features.append((tensor_id/max(1, len(named)-1),
                    (local+(size-1)/2)/max(1, p.numel()-1), p.numel()/self.parameter_size))
            start += p.numel()
        assert len(counts) == groups
        self.group_count = self.body.group_count = groups
        self.body.group_index = torch.tensor(indices, dtype=torch.long)
        self.body.group_sizes = torch.tensor(counts, dtype=torch.float32)
        self.body.group_addresses = torch.tensor(features, dtype=torch.float32)
        self.addresses = self.body.group_addresses
        self.eligibility = self.addresses.new_zeros(self.parameter_size)
        self.writer_indices = torch.cat([torch.arange(a, b) for name, a, b, _ in self.schema
                                         if name.startswith('write.')])
