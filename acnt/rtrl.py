from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math
import torch


@dataclass
class ExactRTRL:
    """Dense exact-RTRL oracle for tiny validation models only."""
    J: torch.Tensor  # [state_dim, param_dim]

    @classmethod
    def zeros(cls, state_dim: int, param_dim: int, *, dtype=torch.float64, device=None):
        return cls(torch.zeros((state_dim, param_dim), dtype=dtype, device=device))

    def step(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.J = A @ self.J + B


class BlockLowRankRTRL:
    """Reference Block-local exact + streaming low-rank cross influence.

    Approximation:
        J ~= E_local + U V^T

    E_local is stored densely in this *toy/reference* implementation, but only
    block-local entries are used. A production implementation should store it
    block-sparsely.

    Cross residual update is streamed without materializing a full cross J.
    Old low-rank channels are propagated as A@U. New local->cross injections
    are decomposed into rank-1 components and stochastically merged into r
    buckets. Random signs make the merge unbiased in expectation.
    """
    def __init__(
        self,
        state_slices: Sequence[slice],
        param_slices: Sequence[slice],
        rank: int,
        *,
        dtype=torch.float64,
        device=None,
        seed: int = 0,
    ) -> None:
        if type(rank) is not int or rank < 1:
            raise ValueError("rank must be a positive integer")
        for slices in (state_slices, param_slices):
            end = 0
            if not slices:
                raise ValueError("Block partitions must not be empty")
            for part in slices:
                if part.start != end or part.stop is None or part.stop <= end or part.step not in (None, 1):
                    raise ValueError("Block slices must be contiguous positive partitions")
                end = part.stop
        if len(state_slices) != len(param_slices):
            raise ValueError("state_slices and param_slices must align by Block")
        self.state_slices = list(state_slices)
        self.param_slices = list(param_slices)
        self.rank = int(rank)
        self.S = max(s.stop for s in self.state_slices)
        self.P = max(s.stop for s in self.param_slices)
        self.E = torch.zeros((self.S, self.P), dtype=dtype, device=device)
        self.U = torch.zeros((self.S, self.rank), dtype=dtype, device=device)
        self.V = torch.zeros((self.P, self.rank), dtype=dtype, device=device)
        self.generator = torch.Generator(device=device if device is not None else "cpu")
        self.generator.manual_seed(seed)

    @property
    def dtype(self):
        return self.E.dtype

    @property
    def device(self):
        return self.E.device

    def propagate_time(self, A: torch.Tensor, B: torch.Tensor | None = None) -> None:
        """Deterministic time-only recurrence, without consuming merge randomness.

        A must be Block diagonal. B carries local direct time derivatives,
        including target/rate derivatives when those are in the state model.
        Callers must include target and lambda in the recurrent state whenever
        learned event heads determine them; a z-only decay Jacobian is insufficient.
        """
        if A.shape != (self.S, self.S):
            raise ValueError("time Jacobian has incorrect shape")
        direct = torch.zeros_like(self.E) if B is None else B
        if direct.shape != self.E.shape:
            raise ValueError("direct influence has incorrect shape")
        mask_a = torch.zeros_like(A, dtype=torch.bool)
        mask_b = torch.zeros_like(self.E, dtype=torch.bool)
        for ss, ps in zip(self.state_slices, self.param_slices):
            mask_a[ss, ss] = True
            mask_b[ss, ps] = True
        if torch.count_nonzero(A[~mask_a]) or torch.count_nonzero(direct[~mask_b]):
            raise ValueError("pure time propagation must be Block local")
        self.E = A @ self.E + direct
        self.U = A @ self.U

    def reconstruct(self) -> torch.Tensor:
        return self.E + self.U @ self.V.T

    def _bucket_add(self, U_new, V_new, u, v):
        # Skip numerically dead components.
        nu = torch.linalg.vector_norm(u)
        nv = torch.linalg.vector_norm(v)
        if float(nu * nv) < 1e-30:
            return
        b = int(torch.randint(self.rank, (), generator=self.generator, device=self.device).item())
        sign = -1.0 if bool(torch.randint(2, (), generator=self.generator, device=self.device).item()) else 1.0
        # Variance-reducing scaling for rank-one trick.
        rho = torch.sqrt((nv + 1e-30) / (nu + 1e-30))
        U_new[:, b] += sign * rho * u
        V_new[:, b] += sign * (1.0 / rho) * v

    def step(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """Advance one semantic-event Jacobian recurrence.

        Preconditions used by this reference implementation:
        - each parameter belongs to exactly one Block (param_slices)
        - B is directly local to the owning Block; any direct cross B should be
          emitted as extra rank-1 injections by a production implementation.
        """
        E_old = self.E
        U_old = self.U
        V_old = self.V

        E_new = torch.zeros_like(E_old)
        U_new = torch.zeros_like(U_old)
        V_new = torch.zeros_like(V_old)

        # Propagate existing cross channels; each channel is itself one rank-1 term.
        AU = A @ U_old
        for q in range(self.rank):
            self._bucket_add(U_new, V_new, AU[:, q], V_old[:, q])

        # Propagate each Block-local exact trace.
        for ss, ps in zip(self.state_slices, self.param_slices):
            E_block = E_old[ss, ps]                      # [d_s, p_b]
            A_from = A[:, ss]                            # [S, d_s]
            propagated = A_from @ E_block               # [S, p_b], small toy matrix
            direct = B[:, ps]
            total = propagated + direct

            # Keep same-Block rows exact in E.
            E_new[ss, ps] = total[ss, :]

            # Cross rows from local trace can be represented as <= d_s rank-1 terms:
            # A_cross[:,k] outer E_block[k,:], plus direct cross B if present.
            # We stream those terms immediately rather than storing a full cross J.
            for k in range(ss.stop - ss.start):
                u = A_from[:, k].clone()
                u[ss] = 0.0
                v = torch.zeros(self.P, dtype=self.dtype, device=self.device)
                v[ps] = E_block[k, :]
                self._bucket_add(U_new, V_new, u, v)

            # Handle any direct cross B robustly. For tiny reference dimensions,
            # decompose each nonlocal row as one rank-1 injection.
            cross_B = direct.clone()
            cross_B[ss, :] = 0.0
            nz_rows = torch.where(torch.linalg.vector_norm(cross_B, dim=1) > 1e-30)[0]
            for rr in nz_rows.tolist():
                u = torch.zeros(self.S, dtype=self.dtype, device=self.device)
                u[rr] = 1.0
                v = torch.zeros(self.P, dtype=self.dtype, device=self.device)
                v[ps] = cross_B[rr, :]
                self._bucket_add(U_new, V_new, u, v)

        self.E = E_new
        self.U = U_new
        self.V = V_new


def split_slices(lengths: Sequence[int]) -> List[slice]:
    out = []
    start = 0
    for n in lengths:
        out.append(slice(start, start + int(n)))
        start += int(n)
    return out
