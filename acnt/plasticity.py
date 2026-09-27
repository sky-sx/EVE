"""Persistent parameter directions on a single irreversible trajectory."""

from bisect import bisect_right
from collections.abc import Mapping
import math
import random

import torch
from torch import nn


class Plasticity:
    """Move a global sparse subset before forward; evaluate its direction once.

    A bad result reverses selected directions, never the parameter movement.
    Parameter registration order defines the reproducible flat sampling space.
    Move modules to their final device before registering their parameters.
    """

    @torch.no_grad()
    def __init__(self, groups: Mapping[int, Mapping[str, nn.Parameter]],
                 goodness_id: int | None, *, delta_magnitude: float = 0.001,
                 subset_fraction: float = 0.001, plasticity_seed: int = 0,
                 parameter_clip: tuple[float, float] | None = None) -> None:
        self.groups = {i: dict(group) for i, group in groups.items()}
        if goodness_id is not None and goodness_id not in self.groups:
            raise ValueError("goodness Block must have a parameter group")
        self.goodness_id = goodness_id
        all_parameters = [p for group in self.groups.values() for p in group.values()]
        if len({id(p) for p in all_parameters}) != len(all_parameters):
            raise ValueError("each parameter must belong to exactly one Block group")
        if not math.isfinite(delta_magnitude) or delta_magnitude <= 0:
            raise ValueError("delta_magnitude must be finite and positive")
        magnitude = torch.tensor(delta_magnitude, dtype=torch.float32)
        if not torch.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("delta_magnitude must be representable in FP32")
        if not math.isfinite(subset_fraction) or not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0,1]")
        if type(plasticity_seed) is not int:
            raise ValueError("plasticity_seed must be an integer")
        if parameter_clip is not None and (
            len(parameter_clip) != 2 or
            not all(math.isfinite(v) for v in parameter_clip) or
            parameter_clip[0] > parameter_clip[1]
        ):
            raise ValueError("parameter_clip must be ordered finite bounds")
        for p in all_parameters:
            if p.dtype != torch.float32 or not p.is_contiguous():
                raise ValueError("parameters must be contiguous FP32 tensors")
            if not torch.isfinite(p).all():
                raise FloatingPointError("non-finite parameter")
        self.parameters = {id(p): p for i, group in self.groups.items()
                           if i != goodness_id for p in group.values()}
        self.delta_magnitude = float(delta_magnitude)
        self.subset_fraction = float(subset_fraction)
        self.plasticity_seed = plasticity_seed
        self.parameter_clip = parameter_clip
        # Both streams belong exclusively to learning, independent of action RNG.
        self.rng = random.Random(plasticity_seed)
        signs = torch.Generator(device="cpu").manual_seed(plasticity_seed)
        self.delta_w = {
            key: (torch.randint(0, 2, p.shape, generator=signs, dtype=torch.int64)
                  .to(device=p.device, dtype=torch.float32).mul_(2).sub_(1).mul_(magnitude.item()))
            for key, p in self.parameters.items()
        }
        self._keys = list(self.parameters)
        self._ends = []
        total = 0
        for p in self.parameters.values():
            total += p.numel()
            self._ends.append(total)
        self.total_elements = total
        self.learning = True
        self.reset_credit()

    def reset_credit(self) -> None:
        """Discard runtime comparison state, preserving W, directions and RNG."""
        self.previous_goodness: float | None = None
        self.goodness_time_ms: int | None = None
        self.pending_indices: tuple[int, ...] | None = None
        self._pending: dict[int, torch.Tensor] = {}
        self.selected_parameter_count = 0
        self.delta_flip_count = 0
        self.goodness_delta: float | None = None

    def set_learning(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if not enabled or enabled != self.learning:
            self.reset_credit()
        self.learning = enabled

    def _validate_finite(self) -> None:
        for key, p in self.parameters.items():
            d = self.delta_w[key]
            if p.device != d.device or p.dtype != torch.float32:
                raise ValueError("registered parameter device/dtype changed")
            if not torch.isfinite(p).all() or not torch.isfinite(d).all():
                raise FloatingPointError("non-finite parameter or delta_w")

    @torch.no_grad()
    def begin_trial(self) -> int:
        if not self.learning:
            return 0
        if self.pending_indices is not None:
            raise RuntimeError("unresolved learning trial requires Goodness before another movement")
        self.selected_parameter_count = 0
        self.delta_flip_count = 0
        self.goodness_delta = None
        if self.previous_goodness is None or not self.total_elements:
            return 0
        self._validate_finite()
        count = max(1, math.floor(self.total_elements * self.subset_fraction))
        # Uniform sampling without replacement, O(k) storage at sparse fractions.
        selected = tuple(sorted(self.rng.sample(range(self.total_elements), count)))
        local = {}
        for index in selected:
            slot = bisect_right(self._ends, index)
            offset = self._ends[slot - 1] if slot else 0
            local.setdefault(self._keys[slot], []).append(index - offset)
        updates = {}
        pending = {}
        for key, indices in local.items():
            p = self.parameters[key]
            idx = torch.tensor(indices, dtype=torch.long, device=p.device)
            proposed = p.view(-1)[idx] + self.delta_w[key].view(-1)[idx]
            # Detect overflow before clipping could hide it.
            if not torch.isfinite(proposed).all():
                raise FloatingPointError("non-finite proposed parameter")
            if self.parameter_clip is not None:
                proposed.clamp_(*self.parameter_clip)
            if not torch.isfinite(proposed).all():
                raise FloatingPointError("non-finite clipped parameter")
            pending[key] = idx
            updates[key] = proposed
        for key, values in updates.items():
            self.parameters[key].view(-1)[pending[key]] = values
        self.pending_indices = selected
        self._pending = pending
        self.selected_parameter_count = count
        self._validate_finite()
        return count

    @torch.no_grad()
    def apply_goodness(self, g_eff: float, *, now_ms: int) -> float | None:
        if not self.learning:
            return None
        if not math.isfinite(g_eff) or not 0 <= g_eff <= 1:
            raise ValueError("effective goodness must be in [0,1]")
        if type(now_ms) is not int:
            raise ValueError("goodness time must be integer milliseconds")
        if self.goodness_time_ms is not None and now_ms < self.goodness_time_ms:
            raise ValueError("goodness time must not move backwards")
        self._validate_finite()
        change = None if self.previous_goodness is None else float(g_eff) - self.previous_goodness
        self.delta_flip_count = 0
        if change is not None and change < 0 and self.pending_indices is not None:
            for key, indices in self._pending.items():
                d = self.delta_w[key].view(-1)
                d[indices] = -d[indices]
            self.delta_flip_count = len(self.pending_indices)
        self.previous_goodness = float(g_eff)
        self.goodness_time_ms = now_ms
        self.goodness_delta = change
        self.pending_indices = None
        self._pending = {}
        self._validate_finite()
        return change

    def calibrate_goodness(self, c_g: float, *, now_ms: int) -> float:
        """Preserve the existing scalar-only calibration interface; no update."""
        if self.goodness_id is None:
            raise RuntimeError("no goodness Block is registered")
        if not math.isfinite(c_g) or not -1 <= c_g <= 1:
            raise ValueError("calibration scalar must be in [-1,1]")
        if type(now_ms) is not int:
            raise ValueError("goodness calibration time must be integer milliseconds")
        return float(c_g)

    def diagnostics(self) -> dict:
        return {"parameter_elements": self.total_elements,
                "selected_parameter_count": self.selected_parameter_count,
                "delta_flip_count": self.delta_flip_count,
                "goodness_delta": self.goodness_delta,
                "previous_goodness": self.previous_goodness,
                "pending": self.pending_indices is not None,
                "delta_magnitude": self.delta_magnitude,
                "subset_fraction": self.subset_fraction,
                "plasticity_seed": self.plasticity_seed}
