"""Full-parameter write interface, including its own neural generator.

This is an interface reference, NOT a trained lifelong learner. Apply to models
whose authoritative weights are their registered Parameters, not the older
self_write joint-state prototype that overrides weights from another vector.
"""
from dataclasses import dataclass
import torch
from torch import nn


class SharedWriteAdapter(nn.Module):
    """One shared generator called for every scalar parameter address.

The complete command has P independent coordinate slots; the generator's
parameter count depends on local input width, not P. Shared generation does
not guarantee learning an arbitrary command from a particular hidden state.
"""
    def __init__(self, hidden_state_size, *, hidden_size=8, step_scale=.01):
        super().__init__()
        if type(hidden_state_size) is not int or hidden_state_size < 1:
            raise ValueError("positive hidden_state_size required")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("positive hidden_size required")
        if not 0 < step_scale < float('inf'):
            raise ValueError("positive finite step_scale required")
        self.hidden_state_size, self.step_scale = hidden_state_size, float(step_scale)
        # H, current scalar weight, tensor id, coordinate, relative tensor size.
        self.network = nn.Sequential(nn.Linear(hidden_state_size + 4, hidden_size),
                                     nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, hidden, values, addresses):
        if hidden.shape != (self.hidden_state_size,) or values.ndim != 1 or addresses.shape != (values.numel(), 3):
            raise ValueError("invalid H / values / addresses shape")
        features = torch.cat((hidden.expand(values.numel(), -1), values[:, None], addresses), dim=1)
        return self.step_scale * torch.tanh(self.network(features).flatten())


@dataclass(frozen=True)
class ParameterSlot:
    name: str
    parameter: nn.Parameter
    start: int
    end: int


class FullParameterWriteSurface:
    """Parameter-free actuator: theta_new[i] = theta_old[i] + command[i].

Enumerates ALL unique registered Parameters, including requires_grad=False
and all generator parameters. No learned output decoder or low-rank command
constraint. Generates the whole command before committing any scalar.
The registry is immutable schema metadata, not an experience memory.
"""
    def __init__(self, model, adapter, *, max_parameters=2_000_000):
        self.model, self.adapter = model, adapter
        named = tuple(model.named_parameters())
        if not named:
            raise ValueError("model has no parameters")
        available = {id(p) for _, p in named}
        if not all(id(p) in available for p in adapter.parameters()):
            raise ValueError("Write Adapter must be registered inside the writable model")
        device = named[0][1].device
        if any(p.dtype != torch.float32 or p.device != device for _, p in named):
            raise ValueError("reference requires one FP32 device")
        self.slots, offset = [], 0
        for name, p in named:
            self.slots.append(ParameterSlot(name, p, offset, offset + p.numel()))
            offset += p.numel()
        self.parameter_count = offset
        self.control_count = offset
        if offset > max_parameters:
            raise ValueError("reference command exceeds parameter budget")
        self.device = device
        self.signature = tuple((name, id(p), tuple(p.shape)) for name, p in named)

    def _check_registry(self):
        current = tuple((name, id(p), tuple(p.shape)) for name, p in self.model.named_parameters())
        if current != self.signature:
            raise ValueError("parameter registry changed; rebuild full write surface")

    def pack(self):
        self._check_registry()
        return torch.cat([slot.parameter.reshape(-1) for slot in self.slots])

    def generate(self, hidden, *, chunk_size=256):
        self._check_registry()
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("positive chunk_size required")
        if hidden.dtype != torch.float32 or hidden.device != self.device or not torch.isfinite(hidden).all():
            raise ValueError("finite FP32 H on model device required")
        commands = []
        tensor_count = len(self.slots)
        for tensor_id, slot in enumerate(self.slots):
            values = slot.parameter.reshape(-1)
            for start in range(0, values.numel(), chunk_size):
                local = values[start:start + chunk_size]
                coordinates = torch.arange(start, start + local.numel(), device=self.device, dtype=torch.float32)
                addresses = torch.stack((
                    torch.full_like(coordinates, tensor_id / max(1, tensor_count-1)),
                    coordinates / max(1, values.numel()-1),
                    torch.full_like(coordinates, values.numel() / self.parameter_count)), dim=1)
                commands.append(self.adapter(hidden, local, addresses))
        return torch.cat(commands)

    def proposed(self, command):
        self._check_registry()
        if command.shape != (self.control_count,) or command.dtype != torch.float32 or command.device != self.device:
            raise ValueError("one FP32 command per scalar parameter required")
        if not torch.isfinite(command).all():
            raise ValueError("commands must be finite")
        result = self.pack().detach() + command
        if not torch.isfinite(result).all():
            raise ValueError("proposed parameters must be finite")
        return result

    @torch.no_grad()
    def apply(self, command):
        proposed = self.proposed(command.detach())
        for slot in self.slots:
            slot.parameter.copy_(proposed[slot.start:slot.end].reshape_as(slot.parameter))
        # No state/History reset. Caller makes these weights visible next event.
