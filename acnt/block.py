"""Canonical ACNT Block with CTM-style private neuron-level models."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from time import monotonic_ns

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LocalObserver = Callable[[nn.Parameter, Tensor, int], None]


@dataclass(frozen=True)
class SynapseFrame:
    """Detached local facts of one committed Block update."""

    time_ms: int
    l: Tensor
    rho: Tensor
    a: Tensor
    ln_scale: Tensor
    sources: dict[int, Tensor]


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

    def forward(self, x: Tensor, *, return_cache: bool = False):
        if x.shape != (self.neuron_size, self.input_dim):
            raise ValueError(
                f"NLM input must have shape ({self.neuron_size}, {self.input_dim})"
            )

        linear1 = torch.einsum("noi,ni->no", self.weight1, x) + self.bias1
        h = self.hidden_dim

        left1 = linear1[:, :h]
        gate1 = linear1[:, h:]
        rho1 = torch.sigmoid(gate1)
        hidden = left1 * rho1

        linear2 = torch.einsum("noi,ni->no", self.weight2, hidden) + self.bias2

        left2 = linear2[:, :1]
        gate2 = linear2[:, 1:]
        rho2 = torch.sigmoid(gate2)
        z = (left2 * rho2).squeeze(-1)

        if not return_cache:
            return z

        cache = {
            "x": x.detach().clone(),
            "left1": left1.detach().clone(),
            "rho1": rho1.detach().clone(),
            "hidden": hidden.detach().clone(),
            "left2": left2.detach().clone(),
            "rho2": rho2.detach().clone(),
        }
        return z, cache

    def local_vjp(self, cache: dict[str, Tensor], cotangent: Tensor) -> dict[str, Tensor]:
        if cotangent.shape != (self.neuron_size,):
            raise ValueError(
                f"NLM cotangent must have shape ({self.neuron_size},)"
            )
        if cotangent.dtype != torch.float32:
            raise ValueError("NLM cotangent must use FP32")

        x = cache["x"]
        left1 = cache["left1"]
        rho1 = cache["rho1"]
        hidden = cache["hidden"]
        left2 = cache["left2"]
        rho2 = cache["rho2"]

        c = cotangent.unsqueeze(-1)

        d_left2 = c * rho2
        d_gate2 = c * left2 * rho2 * (1.0 - rho2)
        d_linear2 = torch.cat((d_left2, d_gate2), dim=-1)

        grad_weight2 = d_linear2.unsqueeze(-1) * hidden.unsqueeze(1)
        grad_bias2 = d_linear2

        d_hidden = torch.einsum(
            "no,noh->nh",
            d_linear2,
            self.weight2.detach(),
        )

        d_left1 = d_hidden * rho1
        d_gate1 = d_hidden * left1 * rho1 * (1.0 - rho1)
        d_linear1 = torch.cat((d_left1, d_gate1), dim=-1)

        grad_weight1 = d_linear1.unsqueeze(-1) * x.unsqueeze(1)
        grad_bias1 = d_linear1

        grad_x = torch.einsum(
            "no,noi->ni",
            d_linear1,
            self.weight1.detach(),
        )

        return {
            "weight1": grad_weight1.detach(),
            "bias1": grad_bias1.detach(),
            "weight2": grad_weight2.detach(),
            "bias2": grad_bias2.detach(),
            "x": grad_x.detach(),
        }


class Block(nn.Module):
    """One FP32, non-batched Block. Times and ticktime are milliseconds.

    A/At store pre-activations and their logical timestamps, oldest first.
    Persistent states are detached; local learning uses only these frames.
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

        for name, size in (
            ("z", n),
            ("z_bar", n),
            ("a", n),
            ("r", 2 * n),
        ):
            self.register_buffer(name, torch.zeros(size, dtype=torch.float32))
        self.register_buffer("o", torch.zeros(2 * n, dtype=torch.float32) if readin else None)
        self.A: deque[Tensor] = deque(maxlen=hold_tick)
        self.At: deque[int] = deque(maxlen=hold_tick)

        self.learning_frames: deque[SynapseFrame] = deque(maxlen=hold_tick)

        self.local_observer: LocalObserver | None = None
        self.learning_enabled = False
        self.perturbation_scale = 0.0
        self.perturbation_generator: torch.Generator | None = None

    @staticmethod
    def sigma(x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def LN(self, x: Tensor, *, return_scale: bool = False):
        mean = x.mean()
        centered = x - mean
        scale = torch.sqrt(centered.square().mean() + self.ln_eps)
        a = centered / scale
        if return_scale:
            return a, scale.detach().clone()
        return a

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

    def LN_vjp(
        self,
        a: Tensor,
        scale: Tensor,
        cotangent: Tensor,
    ) -> Tensor:
        if a.shape != (self.neuron_size,):
            raise ValueError("LN activation has wrong shape")
        if cotangent.shape != (self.neuron_size,):
            raise ValueError("LN cotangent has wrong shape")

        return (
            cotangent
            - cotangent.mean()
            - a * (a * cotangent).mean()
        ) / scale

    def set_learning(
        self,
        enabled: bool,
        *,
        perturbation_scale: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if perturbation_scale < 0 or not math.isfinite(perturbation_scale):
            raise ValueError("perturbation_scale must be finite and nonnegative")

        self.learning_enabled = enabled
        self.perturbation_scale = float(perturbation_scale)
        self.perturbation_generator = generator

    def _emit_eligibility(
        self,
        parameter: nn.Parameter,
        contribution: Tensor,
        *,
        now_ms: int,
    ) -> None:
        if self.local_observer is None:
            return
        if contribution.shape != parameter.shape:
            raise ValueError("eligibility contribution shape mismatch")
        if not torch.isfinite(contribution).all():
            raise FloatingPointError("non-finite eligibility contribution")
        self.local_observer(
            parameter,
            contribution.detach(),
            now_ms,
        )

    def _accumulate_local_eligibility(
        self,
        *,
        now_ms: int,
        xi: Tensor,
        nlm_cache: dict[str, Tensor],
        frames: Sequence[SynapseFrame],
    ) -> None:
        if self.local_observer is None:
            return

        c = self.perturbation_scale
        if c <= 0:
            return

        cotangent = xi / c

        nlm_grads = self.nlm.local_vjp(
            nlm_cache,
            cotangent,
        )

        self._emit_eligibility(
            self.nlm.weight1,
            nlm_grads["weight1"],
            now_ms=now_ms,
        )
        self._emit_eligibility(
            self.nlm.bias1,
            nlm_grads["bias1"],
            now_ms=now_ms,
        )
        self._emit_eligibility(
            self.nlm.weight2,
            nlm_grads["weight2"],
            now_ms=now_ms,
        )
        self._emit_eligibility(
            self.nlm.bias2,
            nlm_grads["bias2"],
            now_ms=now_ms,
        )

        history_grad = nlm_grads["x"][:, :self.hold_tick]

        k = len(frames)
        if k == 0:
            return

        history_grad = history_grad[:, -k:]

        weight_contributions = [
            torch.zeros_like(weight)
            for weight in self.W_ij
        ]
        bias_contribution = torch.zeros_like(self.b)

        for slot, frame in enumerate(frames):
            v = history_grad[:, slot]

            c_a = self.LN_vjp(
                frame.a,
                frame.ln_scale,
                v,
            )

            delta_l = c_a * frame.rho
            delta_g = (
                c_a
                * frame.l
                * frame.rho
                * (1.0 - frame.rho)
            )
            delta_r = torch.cat(
                (delta_l, delta_g),
                dim=0,
            )

            bias_contribution.add_(delta_r)

            for source_id, source_z in frame.sources.items():
                weight_contributions[source_id].add_(
                    delta_r.unsqueeze(1)
                    * source_z.unsqueeze(0)
                )

        for source_id, contribution in enumerate(weight_contributions):
            if contribution.count_nonzero():
                self._emit_eligibility(
                    self.W_ij[source_id],
                    contribution,
                    now_ms=now_ms,
                )

        self._emit_eligibility(
            self.b,
            bias_contribution,
            now_ms=now_ms,
        )

    def update(self, *, now_ms: int | None = None, active_z: Mapping[int, Tensor] | None = None) -> Tensor:
        with torch.no_grad():
            return self._update(now_ms=now_ms, active_z=active_z)

    def _update(
        self,
        *,
        now_ms: int | None,
        active_z: Mapping[int, Tensor] | None,
    ) -> Tensor:
        if not self.active:
            return self.z

        if len(self.A) != len(self.At):
            raise ValueError("A and At must correspond one-to-one")

        if len(self.learning_frames) not in (0, len(self.A)):
            raise ValueError(
                "learning frame history must align with activation history"
            )

        now_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms

        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")

        if self.At and now_ms < self.At[-1]:
            raise ValueError("time must not move backwards")

        r = self.b.clone()

        if self.o is not None:
            self._check_vector(
                self.o,
                2 * self.neuron_size,
                "o",
            )
            r = r + self.o

        sources = {} if active_z is None else active_z

        detached_sources: dict[int, Tensor] = {}

        for j, z_j in sources.items():
            if type(j) is not int or not 0 <= j < len(self.W_ij):
                raise ValueError("source id must index W_ij")

            self._check_vector(
                z_j,
                self.source_sizes[j],
                f"z_{j}",
            )

            source = z_j.detach()
            detached_sources[j] = source.clone()
            r = r + self.W_ij[j] @ source

        n = self.neuron_size

        l = r[:n]
        g = r[n:]
        rho = torch.sigmoid(g)
        u = l * rho

        a, ln_scale = self.LN(
            u,
            return_scale=True,
        )

        current_frame = SynapseFrame(
            time_ms=now_ms,
            l=l.detach().clone(),
            rho=rho.detach().clone(),
            a=a.detach().clone(),
            ln_scale=ln_scale.detach().clone(),
            sources=detached_sources,
        )

        history = [
            *(entry.detach() for entry in self.A),
            a,
        ][-self.hold_tick:]

        times = [
            *self.At,
            now_ms,
        ][-self.hold_tick:]

        frames = [
            *self.learning_frames,
            current_frame,
        ][-self.hold_tick:]

        nlm_x = self.nlm_input(
            history,
            times,
            now_ms=now_ms,
        )

        z_bar, nlm_cache = self.nlm(
            nlm_x,
            return_cache=True,
        )

        if (
            self.learning_enabled
            and self.local_observer is not None
            and self.perturbation_scale > 0
        ):
            xi = torch.randn(
                z_bar.shape,
                dtype=z_bar.dtype,
                device=z_bar.device,
                generator=self.perturbation_generator,
            )

            z = z_bar + self.perturbation_scale * xi

            self._accumulate_local_eligibility(
                now_ms=now_ms,
                xi=xi.detach(),
                nlm_cache=nlm_cache,
                frames=frames,
            )
        else:
            z = z_bar

        for name, value in (
            ("r", r),
            ("a", a),
            ("z_bar", z_bar),
            ("z", z),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Block {self.block_id}: non-finite {name}"
                )

        self.r = r.detach().clone()
        self.a = a.detach().clone()
        self.z_bar = z_bar.detach().clone()
        self.z = z.detach().clone()

        self.A.clear()
        self.A.extend(
            value.detach().clone()
            for value in history
        )

        self.At.clear()
        self.At.extend(times)

        self.learning_frames.clear()
        self.learning_frames.extend(frames)

        if self.o is not None:
            self.o = self.o.detach().clone()

        return self.z