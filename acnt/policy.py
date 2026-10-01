from __future__ import annotations
import torch
from torch import nn


class BernoulliReadout(nn.Module):
    def __init__(self, d: int, dtype=torch.float64, init_scale: float = 0.12):
        super().__init__()
        self.linear = nn.Linear(d, 1, bias=True, dtype=dtype)
        with torch.no_grad():
            self.linear.weight.normal_(0.0, init_scale)
            self.linear.bias.zero_()

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z).squeeze(-1)

    def prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(z))

    def sample(self, z: torch.Tensor, generator=None):
        p = self.prob(z)
        if generator is None:
            a = torch.bernoulli(p)
        else:
            a = torch.bernoulli(p, generator=generator)
        return a, p

    @staticmethod
    def score_from_action_prob(action: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        # d log Bernoulli / d logit
        return action - prob
