"""ARCHIVED REFERENCE — activity conditioning experiment.

The session found that scalar-Goodness ReadOut learning became dramatically
more stable after mean/variance normalization + online decorrelation. This is
still a useful conditioning idea, but it is not the core G1 cross-credit rule.
"""
import torch


class OnlineConditioner:
    def __init__(self, d, beta=1e-3, decor_lr=1e-4, dtype=torch.float64):
        self.mean=torch.zeros(d,dtype=dtype)
        self.var=torch.ones(d,dtype=dtype)
        self.D=torch.eye(d,dtype=dtype)
        self.beta=beta; self.decor_lr=decor_lr

    @torch.no_grad()
    def __call__(self, h):
        delta=h-self.mean
        self.mean += self.beta*delta
        self.var = (1-self.beta)*self.var + self.beta*delta.square()
        x=(h-self.mean)/torch.sqrt(self.var+1e-6)
        y=self.D@x
        off=torch.outer(y,y)-torch.diag(y.square())
        self.D -= self.decor_lr*(off@self.D)
        return y
