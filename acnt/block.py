"""Canonical Block equations with an optional one-update local graph."""

from collections import deque
from collections.abc import Mapping, Sequence
import math
from time import monotonic_ns

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Block(nn.Module):
    """One FP32, non-batched Block. Times are integer milliseconds.

    A/At store oldest first; W_c[idx] follows the current queue index.
    LN has no learned affine parameters: these are absent from canonical theta.
    Persistent states are detached; an optional local graph serves e-prop.
    """

    def __init__(
        self,
        block_id: int,
        neuron_size: int,
        source_sizes: Sequence[int],
        *,
        ticktime: float = 1.0,
        hold_tick: int = 4,
        active: bool = True,
        readin: bool = False,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if type(neuron_size) is not int or neuron_size < 1:
            raise ValueError("neuron_size must be a positive integer")
        if type(hold_tick) is not int or hold_tick < 1:
            raise ValueError("hold_tick must be a positive integer")
        if isinstance(ticktime, bool) or not isinstance(ticktime, (int, float)) or not math.isfinite(ticktime) or ticktime <= 0:
            raise ValueError("ticktime must be a finite positive number of seconds")
        if not source_sizes or any(type(n) is not int or n < 1 for n in source_sizes):
            raise ValueError("source_sizes must contain positive integers")
        if type(block_id) is not int or not 0 <= block_id < len(source_sizes):
            raise ValueError("block_id must index source_sizes")
        if source_sizes[block_id] != neuron_size:
            raise ValueError("source_sizes[block_id] must equal neuron_size")
        if type(active) is not bool or type(readin) is not bool:
            raise TypeError("active and readin must be bool")
        if not 0 < ln_eps < float("inf"):
            raise ValueError("ln_eps must be finite and positive")

        self.block_id = block_id
        self.neuron_size = neuron_size
        self.source_sizes = tuple(source_sizes)
        self.ticktime = float(ticktime)
        self.hold_tick = hold_tick
        self.active = active
        self.ln_eps = ln_eps
        n = neuron_size
        self.W_ij = nn.ParameterList(
            [nn.Parameter(torch.empty(2 * n, size, dtype=torch.float32)) for size in source_sizes]
        )
        self.b = nn.Parameter(torch.zeros(2 * n, dtype=torch.float32))
        self.W_c = nn.ParameterList(
            [nn.Parameter(torch.empty(n, 2 * n, dtype=torch.float32)) for _ in range(hold_tick)]
        )
        self.b_c = nn.ParameterList(
            [nn.Parameter(torch.zeros(n, dtype=torch.float32)) for _ in range(hold_tick)]
        )
        for weight in [*self.W_ij, *self.W_c]:
            nn.init.xavier_uniform_(weight)

        for name, size in (("z", n), ("a", n), ("r", 2 * n), ("h", n)):
            self.register_buffer(name, torch.zeros(size, dtype=torch.float32))
        # Only a ReadIn Block has an o vector. Otherwise the formula uses zero.
        self.register_buffer("o", torch.zeros(2 * n, dtype=torch.float32) if readin else None)
        self.A: deque[Tensor] = deque(maxlen=hold_tick)
        self.At: deque[int] = deque(maxlen=hold_tick)

    @staticmethod
    def sigma(x: Tensor) -> Tensor:
        """Elementwise 1 / (1 + exp(-x)), evaluated stably."""
        return torch.sigmoid(x)

    def LN(self, x: Tensor) -> Tensor:
        """Normalize all neurons; population variance, epsilon, no affine gain."""
        return F.layer_norm(x, (self.neuron_size,), eps=self.ln_eps)

    def _check_vector(self, value: Tensor, size: int, name: str) -> None:
        if not isinstance(value, Tensor) or value.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if value.dtype != torch.float32 or value.device != self.b.device:
            raise ValueError(f"{name} must use FP32 on the Block's device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    def update(self, *, now_ms: int | None = None, active_z: Mapping[int, Tensor] | None = None, track_grad: bool = False) -> Tensor:
        with torch.set_grad_enabled(track_grad):
            return self._update(now_ms=now_ms, active_z=active_z, track_grad=track_grad)

    def _update(self, *, now_ms: int | None, active_z: Mapping[int, Tensor] | None, track_grad: bool) -> Tensor:
        """Update once using exactly the supplied active source states.

        An omitted/empty mapping means no source contributes, including self.
        Core constructs this mapping from the active set.
        An inactive destination leaves all of its states and history untouched.
        Explicit now_ms makes history timing deterministic in tests.
        """
        if not self.active:
            return self.z
        if len(self.A) != len(self.At):
            raise ValueError("A and At must correspond one-to-one")

        # r_i = o_i + b_i + sum_{j active} W_ij z_j.
        r = self.b.clone()
        if self.o is not None:
            self._check_vector(self.o, 2 * self.neuron_size, "o")
            r = r + self.o
        for j, z_j in ({} if active_z is None else active_z).items():
            if type(j) is not int or not 0 <= j < len(self.W_ij):
                raise ValueError("source id must index W_ij")
            self._check_vector(z_j, self.source_sizes[j], f"z_{j}")
            r = r + self.W_ij[j] @ z_j.detach()
        n = self.neuron_size
        # a_i = LN(r_i[:n] * sigma(r_i[n:])).
        a = self.LN(r[:n] * self.sigma(r[n:]))

        # Canonical t_last is sampled after a is formed. Tests may supply t.
        now_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")
        if self.At and now_ms < self.At[-1]:
            raise ValueError("time must not move backwards")

        # Pop the oldest pair when full, then append the new pair.
        history = [*(entry.detach() for entry in self.A), a][-self.hold_tick :]
        times = [*self.At, now_ms][-self.hold_tick :]
        h = torch.zeros_like(self.z)
        t_last = now_ms
        for idx in range(len(history) - 1, -1, -1):
            a_k, t_k = history[idx], times[idx]
            delta_t = t_last - t_k
            t_last = t_k
            m = torch.cat((a_k, h))
            h_c = self.LN(self.W_c[idx] @ m + self.b_c[idx])
            # User clarification: ticktime is seconds per tick; divide the
            # elapsed seconds by ticktime. This supersedes the original product.
            gamma = self.sigma(self.b.new_tensor((delta_t / 1000.0) / self.ticktime))
            h = (1 - gamma) * h_c + gamma * h

        for name, value in (("r", r), ("a", a), ("h", h), ("z", h)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Block {self.block_id}: non-finite {name}")
        # Commit together so a failed calculation cannot leave half a history.
        self.r, self.a, self.h, self.z = [value.detach().clone() for value in (r, a, h, h)]
        self.A.clear()
        self.A.extend(value.detach().clone() for value in history)
        self.At.clear()
        self.At.extend(times)
        if self.o is not None:
            self.o = self.o.detach().clone()
        return h if track_grad else self.z
