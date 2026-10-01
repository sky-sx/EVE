from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch
from torch import nn


@dataclass
class BlockState:
    """State of one ACNT Block.

    H: semantic-event history. It changes only on semantic_event().
    z: current continuous state.
    target: asymptotic continuous-time target between semantic events.
    lam: positive decay/rate vector between semantic events.
    t_last: physical time at which z was last materialized.
    """
    H: torch.Tensor
    z: torch.Tensor
    target: torch.Tensor
    lam: torch.Tensor
    t_last: float = 0.0

    def clone(self) -> "BlockState":
        return BlockState(
            self.H.clone(), self.z.clone(), self.target.clone(), self.lam.clone(), self.t_last
        )


class EventFlowBlock(nn.Module):
    """Current post-G2 ACNT Block time semantics.

    Physical time passage:
        z(t+dt) = target + exp(-lam*dt) * (z(t)-target)

    Semantic event:
        1. materialize z at event time using the equation above
        2. shift semantic history H once
        3. compute event jump z+ with tanh/sigmoid
        4. set target and positive lambda for the next inter-event interval

    exp is used only for physical-time flow. History never shifts because of a scheduler tick.
    """
    def __init__(
        self,
        d: int,
        history_len: int,
        input_dim: int = 1,
        message_dim: Optional[int] = None,
        history_hidden: Optional[int] = None,
        lambda_min: float = 0.2,
        lambda_max: float = 5.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        dimensions = (d, history_len, input_dim)
        if any(type(v) is not int or v < 1 for v in dimensions):
            raise ValueError("dimensions must be positive integers")
        if any(v is not None and (type(v) is not int or v < 1) for v in (message_dim, history_hidden)):
            raise ValueError("optional dimensions must be positive integers")
        if not (math.isfinite(lambda_min) and math.isfinite(lambda_max) and 0 < lambda_min < lambda_max):
            raise ValueError("rates must satisfy 0 < lambda_min < lambda_max < infinity")
        self.d = int(d)
        self.M = int(history_len)
        self.input_dim = int(input_dim)
        self.message_dim = int(message_dim if message_dim is not None else d)
        hh = int(history_hidden if history_hidden is not None else max(d, 2))
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.dtype = dtype

        self.history_1 = nn.Linear(self.M * self.d, hh, bias=True, dtype=dtype)
        self.history_2 = nn.Linear(hh, self.d, bias=True, dtype=dtype)

        qdim = self.d + self.message_dim + self.input_dim
        self.jump_u = nn.Linear(qdim, self.d, bias=True, dtype=dtype)
        self.jump_g = nn.Linear(qdim, self.d, bias=True, dtype=dtype)
        self.target_head = nn.Linear(qdim, self.d, bias=True, dtype=dtype)
        self.lambda_head = nn.Linear(qdim, self.d, bias=True, dtype=dtype)

    def initial_state(self, batch_shape=(), t0: float = 0.0, device=None) -> BlockState:
        if not math.isfinite(t0):
            raise ValueError("initial time must be finite")
        parameter = self.jump_u.weight
        device = parameter.device if device is None else device
        shape_h = tuple(batch_shape) + (self.M, self.d)
        shape_z = tuple(batch_shape) + (self.d,)
        H = torch.zeros(shape_h, dtype=parameter.dtype, device=device)
        z = torch.zeros(shape_z, dtype=parameter.dtype, device=device)
        target = torch.zeros_like(z)
        lam_mid = 0.5 * (self.lambda_min + self.lambda_max)
        lam = torch.full_like(z, lam_mid)
        return BlockState(H=H, z=z, target=target, lam=lam, t_last=float(t0))

    @staticmethod
    def flow_tensor(z: torch.Tensor, target: torch.Tensor, lam: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
        dt_t = torch.as_tensor(dt, dtype=z.dtype, device=z.device)
        if not torch.isfinite(dt_t).all() or (dt_t < 0).any():
            raise ValueError("elapsed time must be finite and nonnegative")
        if not torch.isfinite(lam).all() or (lam <= 0).any():
            raise ValueError("flow rates must be finite and positive")
        while dt_t.ndim < z.ndim:
            dt_t = dt_t.unsqueeze(-1)
        a = torch.exp(-lam * dt_t)
        return target + a * (z - target)

    def materialize(self, state: BlockState, t_now: float) -> BlockState:
        dt = float(t_now) - float(state.t_last)
        if not math.isfinite(float(t_now)) or not math.isfinite(float(state.t_last)):
            raise ValueError("time must be finite")
        if dt < 0:
            raise ValueError("time must be monotonic")
        if dt == 0:
            return state
        z = self.flow_tensor(state.z, state.target, state.lam, dt)
        return BlockState(state.H, z, state.target, state.lam, float(t_now))

    def history_repr(self, H: torch.Tensor) -> torch.Tensor:
        x = H.reshape(*H.shape[:-2], self.M * self.d)
        return torch.tanh(self.history_2(torch.tanh(self.history_1(x))))

    def semantic_event(
        self,
        state: BlockState,
        t_now: float,
        message: torch.Tensor,
        external_input: torch.Tensor,
    ) -> BlockState:
        s = self.materialize(state, t_now)
        # History records the pre-event state exactly once per semantic event.
        H_new = torch.cat([s.H[..., 1:, :], s.z.unsqueeze(-2)], dim=-2)
        h = self.history_repr(H_new)
        q = torch.cat([h, message, external_input], dim=-1)

        u = torch.tanh(self.jump_u(q))
        g = torch.sigmoid(self.jump_g(q))
        z_plus = (1.0 - g) * s.z + g * u
        target = torch.tanh(self.target_head(q))
        sig = torch.sigmoid(self.lambda_head(q))
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * sig
        return BlockState(H_new, z_plus, target, lam, float(t_now))

    def neutral_initialize_(self, gain: float = 0.15, jump_bias: float = -1.0) -> None:
        """Task-agnostic, plasticity-friendly birth initialization.

        This is a reference starting point, not a locked architectural constant.
        It avoids making all message/signal paths numerically dead at birth.
        """
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p, gain=gain)
                else:
                    p.zero_()
            # Modest immediate event response, not a saturated gate.
            self.jump_g.bias.fill_(jump_bias)
            # Keep lambda in a moderate range at birth.
            self.lambda_head.weight.mul_(0.2)
            self.lambda_head.bias.zero_()
