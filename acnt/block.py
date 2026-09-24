"""Canonical ACNT Block with CTM-style private neuron-level models."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
import math
from time import monotonic_ns

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LocalObserver = Callable[[nn.Parameter, Tensor, Tensor, str], None]


class NeuronLevelModel(nn.Module):
    """Two private grouped Linear/GLU layers, evaluated for all neurons at once."""

    def __init__(self, neuron_size: int, input_dim: int, hidden_dim: int = 4) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (neuron_size, input_dim, hidden_dim)):
            raise ValueError("NLM dimensions must be positive integers")
        self.neuron_size = neuron_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight1 = nn.Parameter(torch.empty(neuron_size, 2 * hidden_dim, input_dim))
        self.bias1 = nn.Parameter(torch.zeros(neuron_size, 2 * hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(neuron_size, 2, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(neuron_size, 2))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound1 = math.sqrt(6.0 / (self.input_dim + 2 * self.hidden_dim))
        bound2 = math.sqrt(6.0 / (self.hidden_dim + 2))
        nn.init.uniform_(self.weight1, -bound1, bound1)
        nn.init.uniform_(self.weight2, -bound2, bound2)

    def forward(self, x: Tensor, observer: LocalObserver | None = None) -> Tensor:
        if x.shape != (self.neuron_size, self.input_dim):
            raise ValueError(f"NLM input must have shape ({self.neuron_size}, {self.input_dim})")
        linear1 = torch.einsum("noi,ni->no", self.weight1, x) + self.bias1
        if observer is not None:
            observer(self.weight1, x.detach(), linear1.detach(), "grouped_dense")
            observer(self.bias1, torch.ones_like(linear1), linear1.detach(), "grouped_bias")
        hidden = F.glu(linear1, dim=-1)
        linear2 = torch.einsum("noi,ni->no", self.weight2, hidden) + self.bias2
        if observer is not None:
            observer(self.weight2, hidden.detach(), linear2.detach(), "grouped_dense")
            observer(self.bias2, torch.ones_like(linear2), linear2.detach(), "grouped_bias")
        return F.glu(linear2, dim=-1).squeeze(-1)


class Block(nn.Module):
    """One FP32, non-batched Block. Times and ticktime are milliseconds.

    A/At store pre-activations and their logical timestamps, oldest first.
    Persistent states are detached; observers see only local connection ends.
    """

    def __init__(
        self,
        block_id: int,
        neuron_size: int,
        source_sizes: Sequence[int],
        *,
        ticktime: float = 1.0,
        hold_tick: int = 4,
        nlm_hidden_dim: int = 4,
        active: bool = True,
        readin: bool = False,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if type(neuron_size) is not int or neuron_size < 1:
            raise ValueError("neuron_size must be a positive integer")
        if type(hold_tick) is not int or hold_tick < 1:
            raise ValueError("hold_tick must be a positive integer")
        if type(nlm_hidden_dim) is not int or nlm_hidden_dim < 1:
            raise ValueError("nlm_hidden_dim must be a positive integer")
        if isinstance(ticktime, bool) or not isinstance(ticktime, (int, float)) or not math.isfinite(ticktime) or ticktime <= 0:
            raise ValueError("ticktime must be a finite positive number of milliseconds")
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
        self.nlm_hidden_dim = nlm_hidden_dim
        self.active = active
        self.ln_eps = ln_eps
        n = neuron_size
        self.W_ij = nn.ParameterList(
            [nn.Parameter(torch.empty(2 * n, size, dtype=torch.float32)) for size in source_sizes]
        )
        self.b = nn.Parameter(torch.zeros(2 * n, dtype=torch.float32))
        for weight in self.W_ij:
            nn.init.xavier_uniform_(weight)
        self.nlm = NeuronLevelModel(n, 3 * hold_tick, nlm_hidden_dim)

        for name, size in (("z", n), ("a", n), ("r", 2 * n)):
            self.register_buffer(name, torch.zeros(size, dtype=torch.float32))
        self.register_buffer("o", torch.zeros(2 * n, dtype=torch.float32) if readin else None)
        self.A: deque[Tensor] = deque(maxlen=hold_tick)
        self.At: deque[int] = deque(maxlen=hold_tick)
        self.local_observer: LocalObserver | None = None

    @staticmethod
    def sigma(x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def LN(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (self.neuron_size,), eps=self.ln_eps)

    def _check_vector(self, value: Tensor, size: int, name: str) -> None:
        if not isinstance(value, Tensor) or value.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if value.dtype != torch.float32 or value.device != self.b.device:
            raise ValueError(f"{name} must use FP32 on the Block's device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    def nlm_input(self, history: Sequence[Tensor], times: Sequence[int], *, now_ms: int) -> Tensor:
        """Build right-aligned [activation | real-time age | validity] rows."""
        if len(history) != len(times) or len(history) > self.hold_tick:
            raise ValueError("history and timestamps must correspond within hold_tick")
        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")
        if times and (any(type(t) is not int for t in times) or any(t > now_ms for t in times)):
            raise ValueError("history timestamps must be integer milliseconds not later than now")
        m, n, k = self.hold_tick, self.neuron_size, len(history)
        activation = self.b.new_zeros((n, m))
        age = self.b.new_zeros(m)
        validity = self.b.new_zeros(m)
        if k:
            values = torch.stack(tuple(history))
            if values.shape != (k, n) or values.dtype != torch.float32 or values.device != self.b.device:
                raise ValueError("A entries must be FP32 neuron vectors on the Block device")
            activation[:, -k:] = values.transpose(0, 1)
            age[-k:] = self.b.new_tensor([(now_ms - t) / self.ticktime for t in times])
            validity[-k:] = 1
        result = torch.cat((activation, age.expand(n, -1), validity.expand(n, -1)), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError(f"Block {self.block_id}: non-finite NLM input")
        return result

    def update(self, *, now_ms: int | None = None, active_z: Mapping[int, Tensor] | None = None) -> Tensor:
        with torch.no_grad():
            return self._update(now_ms=now_ms, active_z=active_z)

    def _update(self, *, now_ms: int | None, active_z: Mapping[int, Tensor] | None) -> Tensor:
        if not self.active:
            return self.z
        if len(self.A) != len(self.At):
            raise ValueError("A and At must correspond one-to-one")
        now_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")
        if self.At and now_ms < self.At[-1]:
            raise ValueError("time must not move backwards")

        r = self.b.clone()
        if self.o is not None:
            self._check_vector(self.o, 2 * self.neuron_size, "o")
            r = r + self.o
        sources = {} if active_z is None else active_z
        for j, z_j in sources.items():
            if type(j) is not int or not 0 <= j < len(self.W_ij):
                raise ValueError("source id must index W_ij")
            self._check_vector(z_j, self.source_sizes[j], f"z_{j}")
            r = r + self.W_ij[j] @ z_j.detach()
        n = self.neuron_size
        a = self.LN(r[:n] * self.sigma(r[n:]))
        if self.local_observer is not None:
            for j, z_j in sources.items():
                self.local_observer(self.W_ij[j], z_j.detach(), r.detach(), "dense")
            self.local_observer(self.b, torch.ones_like(r), r.detach(), "bias")

        history = [*(entry.detach() for entry in self.A), a][-self.hold_tick :]
        times = [*self.At, now_ms][-self.hold_tick :]
        z = self.nlm(self.nlm_input(history, times, now_ms=now_ms), self.local_observer)
        for name, value in (("r", r), ("a", a), ("z", z)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Block {self.block_id}: non-finite {name}")

        self.r, self.a, self.z = [value.detach().clone() for value in (r, a, z)]
        self.A.clear()
        self.A.extend(value.detach().clone() for value in history)
        self.At.clear()
        self.At.extend(times)
        if self.o is not None:
            self.o = self.o.detach().clone()
        return self.z
