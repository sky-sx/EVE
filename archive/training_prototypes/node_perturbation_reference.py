"""ARCHIVED REFERENCE — not the current canonical ACNT credit algorithm.

Early in the session, a real-trajectory rule of the form

    z_i = F_i + sigma * eps_i
    s_p = (eps_i / sigma) * chi_p
    e_p <- exp(-dt/tau) e_p + s_p
    dtheta_p ~ eta (G-Gbar) e_p

was used to establish that a single scalar Goodness can train delayed recurrent
behavior. It was later superseded as the main Core rule because cross-Block
credit became the limiting issue in G1.
"""
import torch


def node_perturbation_step(pre, weight, sigma=0.05, generator=None):
    u = weight @ pre
    mu = torch.tanh(u)
    eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
    z = mu + sigma * eps
    chi = (1.0 - mu.square()).unsqueeze(-1) * pre.unsqueeze(0)
    local_score = (eps / sigma).unsqueeze(-1) * chi
    return z, local_score


def update_trace(trace, local_score, dt, tau):
    return torch.exp(torch.as_tensor(-dt / tau, dtype=trace.dtype, device=trace.device)) * trace + local_score
