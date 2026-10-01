"""DEPRECATED time rule retained only for lineage/regression comparison.

This was the earlier ACNT time core before G2 exposed scheduler/event-count
entanglement. Do not restore it as the sole physical-time mechanism.
"""
import torch


def legacy_cfc_update(x, dt, W_u, b_u, W_v, b_v, W_ta, b_ta, W_tb, b_tb):
    u=torch.tanh(W_u@x+b_u)
    v=torch.tanh(W_v@x+b_v)
    a=W_ta@x+b_ta
    b=W_tb@x+b_tb
    gamma=torch.sigmoid(a*dt+b)
    return (1-gamma)*u+gamma*v
