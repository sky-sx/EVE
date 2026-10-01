from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import torch

from .event_flow import EventFlowBlock, BlockState


class FourBlockEventFlow(torch.nn.Module):
    """Small reference network used by G0-G3 regression scripts.

    This is deliberately not the production ACNT runtime. It is a transparent,
    tiny differentiable model for validating equations, partition semantics and
    online-credit approximations.
    """
    def __init__(
        self,
        d: int = 2,
        history_lengths: Sequence[int] = (2, 4, 8, 16),
        input_dim: int = 1,
        dtype=torch.float64,
        edge_scale: float = 0.15,
    ):
        super().__init__()
        if len(history_lengths) != 4:
            raise ValueError("FourBlockEventFlow requires four history lengths")
        self.d = d
        self.history_lengths = tuple(int(x) for x in history_lengths)
        self.blocks = torch.nn.ModuleList([
            EventFlowBlock(d=d, history_len=M, input_dim=input_dim, message_dim=d, dtype=dtype)
            for M in self.history_lengths
        ])
        # Dense logical edge set for tiny experiments. W[target,source].
        self.edges = torch.nn.Parameter(torch.empty(4, 4, d, d, dtype=dtype))
        with torch.no_grad():
            self.edges.normal_(0.0, edge_scale)
            for b in self.blocks:
                b.neutral_initialize_(gain=0.2)

    def initial_states(self, t0: float = 0.0):
        return [b.initial_state(t0=t0) for b in self.blocks]

    def materialize_all(self, states: List[BlockState], t: float) -> List[BlockState]:
        return [b.materialize(s, t) for b, s in zip(self.blocks, states)]

    def semantic_event_all(self, states, t: float, external_inputs: Sequence[torch.Tensor]):
        if len(states) != 4 or len(external_inputs) != 4:
            raise ValueError("four states and external inputs are required")
        states_now = self.materialize_all(states, t)
        z_stack = torch.stack([s.z for s in states_now], dim=0)  # [B,d]
        messages = []
        for i in range(4):
            # active subgraph is all-active in G0-G2 reference.
            m = torch.zeros(self.d, dtype=z_stack.dtype, device=z_stack.device)
            for j in range(4):
                if i != j:
                    m = m + self.edges[i, j] @ z_stack[j]
            messages.append(m)
        return [
            self.blocks[i].semantic_event(states_now[i], t, messages[i], external_inputs[i])
            for i in range(4)
        ]

    def forward_event_sequence(self, event_times, external_event_inputs, final_time=None):
        if len(event_times) != len(external_event_inputs):
            raise ValueError("event times and inputs must correspond")
        states = self.initial_states(t0=0.0)
        for t, inputs in zip(event_times, external_event_inputs):
            states = self.semantic_event_all(states, float(t), inputs)
        if final_time is not None:
            states = self.materialize_all(states, float(final_time))
        return states


def zero_inputs(dtype=torch.float64):
    return [torch.zeros(1, dtype=dtype) for _ in range(4)]
