"""Explicit semantic-event scheduler for the post-G2 reference Blocks.

All timestamps and rates use one caller-selected physical time unit.
No periodic wakeup is interpreted as a semantic event.
"""
import math
from collections.abc import Mapping, Sequence
import torch
from torch import nn
from .event_flow import BlockState, EventFlowBlock

class EventFlowCore(nn.Module):
    def __init__(self, blocks: Sequence[EventFlowBlock], *, t0: float = 0.0):
        super().__init__()
        if not blocks:
            raise ValueError("at least one Block is required")
        self.blocks = nn.ModuleList(blocks)
        self.states = [block.initial_state(t0=t0) for block in blocks]
        self.time = float(t0)

    def _check_time(self, t_now):
        if not math.isfinite(t_now) or t_now < self.time:
            raise ValueError("time must be finite and monotonic")

    def reset(self, *, t0: float = 0.0):
        states = [block.initial_state(t0=t0) for block in self.blocks]
        self.states = states
        self.time = float(t0)

    def materialize(self, t_now: float):
        self._check_time(t_now)
        states = [b.materialize(s, t_now) for b, s in zip(self.blocks, self.states)]
        self.states = states
        self.time = float(t_now)
        return tuple(states)

    def semantic_events(self, t_now: float, events: Mapping[int, tuple[torch.Tensor, torch.Tensor]]):
        """Commit explicit events together, with caller-supplied snapshot messages.

        Only selected histories shift. Inputs are (message, external_input).
        Changes are committed only after every selected event succeeds.
        """
        self._check_time(t_now)
        if any(type(i) is not int or not 0 <= i < len(self.blocks) for i in events):
            raise ValueError("event id must index a Block")
        states = [b.materialize(s, t_now) for b, s in zip(self.blocks, self.states)]
        for i, (message, external_input) in events.items():
            states[i] = self.blocks[i].semantic_event(states[i], t_now, message, external_input)
        self.states = states
        self.time = float(t_now)
        return tuple(states)
