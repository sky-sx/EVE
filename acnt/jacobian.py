from __future__ import annotations
from typing import Callable, Iterable, List, Sequence, Tuple
import torch


def flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params]) if params else torch.empty(0)


def param_slices(params: Sequence[torch.Tensor]) -> List[slice]:
    out=[]; start=0
    for p in params:
        n=p.numel(); out.append(slice(start,start+n)); start+=n
    return out


def finite_difference_grad(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps=1e-6):
    y0 = fn(x)
    if y0.ndim != 0:
        raise ValueError("finite_difference_grad expects scalar fn")
    g = torch.zeros_like(x)
    for i in range(x.numel()):
        xp=x.clone(); xm=x.clone(); xp[i]+=eps; xm[i]-=eps
        g[i]=(fn(xp)-fn(xm))/(2*eps)
    return g
