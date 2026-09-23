"""Only orchestration: all F_e and F_w calls use acnt.plasticity."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from acnt import Block, Core
from acnt.adapters import EyeAdapter, HandAdapter
from acnt.control import sample_discrete
from acnt.plasticity import LocalEvent, Plasticity
from .environment import ACTIONS, exact_success, potential_goodness, render, stimulus_hash


class StageZero:
    def __init__(self, seed: int, device: str = "cpu", *,
                 tau_e_s: float = 1.0, tau_g_s: float = 5.0):
        self.seed, self.device = seed, torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        sizes = [100] * 10
        self.core = Core([Block(i, 100, sizes, readin=(i == 0),
                                hold_tick=4, ticktime=250) for i in range(10)]).to(self.device)
        self.eye = EyeAdapter(100).to(self.device)
        self.hand = HandAdapter(100, discrete_controls=27, continuous_controls=0).to(self.device)
        self.groups = {}
        for i, block in enumerate(self.core.blocks):
            group = {"block."+name: p for name, p in block.named_parameters()}
            adapter = self.eye if i == 0 else self.hand if i == 1 else None
            if adapter is not None:
                group.update({"adapter."+name: p for name, p in adapter.named_parameters()})
            self.groups[i] = group
        self.plasticity = Plasticity(
            self.groups, goodness_id=None, tau_e_s=tau_e_s, tau_g_s=tau_g_s)
        self.cache = {}
        self._learning = False
        self._previous = {id(p): p.detach().clone() for p in self.plasticity.parameters.values()}
        self.set_learning(False)

    def set_learning(self, learning: bool):
        self.plasticity.detach_adapters()
        self._learning = learning
        if learning:
            self.plasticity.attach_adapters({"eye": self.eye, "hand": self.hand})
        for block in self.core.blocks:
            block.local_observer = self._observe_block if learning else None

    def _observe_block(self, parameter, pre, post):
        self.plasticity.observe(parameter, LocalEvent(
            pre, post, self.plasticity._event_time_ms,
            "dense" if parameter.ndim == 2 else "bias"))

    def reset_phase(self, learning: bool):
        self.set_learning(learning)
        self.plasticity.clear()
        for block in self.core.blocks:
            for name in ("z", "a", "r", "h"):
                getattr(block, name).zero_()
            if block.o is not None:
                block.o.zero_()
            block.A.clear()
            block.At.clear()
            block.active = True
        self._previous = {id(p): p.detach().clone() for p in self.plasticity.parameters.values()}

    def frame(self, target, color):
        key = (target, color)
        if key not in self.cache:
            # The hash is computed from the exact RGB FP32 tensor delivered to Eye.
            image = render(target, color, device=self.device)
            self.cache[key] = image, stimulus_hash(image)
        return self.cache[key]

    def _parameter_stats(self):
        params = list(self.plasticity.parameters.values())
        norm = float(torch.stack([p.detach().square().sum() for p in params]).sum().sqrt())
        delta = float(torch.stack([(p.detach()-self._previous[id(p)]).square().sum()
                                   for p in params]).sum().sqrt())
        for p in params:
            self._previous[id(p)].copy_(p.detach())
        return norm, delta

    @staticmethod
    def _summarize(tensors):
        if not tensors:
            return {"l2": 0., "mean": 0., "std": 0., "max_abs": 0., "nonzero_fraction": 0.}
        total = sum(t.numel() for t in tensors)
        squares = torch.stack([t.square().sum() for t in tensors]).sum()
        mean = torch.stack([t.sum() for t in tensors]).sum() / total
        max_abs = torch.stack([t.abs().max() for t in tensors]).max()
        nonzero = torch.stack([(t != 0).sum() for t in tensors]).sum()
        std = (squares / total - mean.square()).clamp_min(0).sqrt()
        return {"l2": float(squares.sqrt()), "mean": float(mean), "std": float(std),
                "max_abs": float(max_abs), "nonzero_fraction": float(nonzero / total)}

    def state_stats(self):
        group = {}
        for i, params in self.groups.items():
            group[str(i)] = self._summarize([self.plasticity.states[id(p)] for p in params.values()])
        eye = [self.plasticity.states[id(p)] for p in self.eye.parameters()]
        hand = [self.plasticity.states[id(p)] for p in self.hand.parameters()]
        core = [self.plasticity.states[id(p)] for b in self.core.blocks for p in b.parameters()]
        terminal = list(self.hand.network[-1].parameters())
        all_states = list(self.plasticity.states.values())
        return {
            "groups": group, "eye_adapter": self._summarize(eye),
            "hand_adapter": self._summarize(hand),
            "ordinary_blocks": self._summarize([self.plasticity.states[id(p)]
                                                for b in list(self.core.blocks)[2:]
                                                for p in b.parameters()]),
            "core_all": self._summarize(core),
            "terminal_hand": self._summarize([self.plasticity.states[id(p)] for p in terminal]),
            "total": self._summarize(all_states),
        }

    @torch.no_grad()
    def episode(self, *, phase, episode, target, color, delay_ms, start_ms, generator):
        frame, sha = self.frame(target, color)
        frame_times = [start_ms + 250 * i for i in range(3)]
        for t in frame_times:
            self.plasticity.set_event_time(t)
            self.core.blocks[0].o = self.eye(frame)
            updated = self.core.step(now_ms=t)
            if tuple(updated) != tuple(range(10)):
                raise AssertionError("all ten Blocks must update on every frame")
        action_time = frame_times[-1]
        self.plasticity.set_event_time(action_time)
        q = self.hand(self.core.blocks[1].z)
        signal = sample_discrete(q, tau=0.25, threshold=0., generator=generator)
        if self._learning:
            self.plasticity.observe_control(self.hand, signal, now_ms=action_time)
        g = potential_goodness(target, signal.a)
        exact = exact_success(target, signal.a)
        delivery = action_time + delay_ms
        # No Core, Adapter, or F_e call takes place between action and delivery.
        before_state = self.state_stats()
        g_bar_before = self.plasticity.g_bar
        modulation = None
        if self._learning:
            modulation = self.plasticity.apply_goodness(g, now_ms=delivery)
        after_state = self.state_stats()
        g_bar_after = self.plasticity.g_bar
        norm, delta = self._parameter_stats()
        bits = signal.a.cpu().tolist()
        probs = signal.p.cpu()
        ptarget = float(probs[target])
        exact_event_probability = float(probs[target] * torch.prod(torch.cat(
            (1-probs[:target], 1-probs[target+1:]))))
        finite_tensors = [*self.plasticity.parameters.values(), *self.plasticity.states.values()]
        nan_count = sum(int(torch.isnan(v).sum()) for v in finite_tensors)
        inf_count = sum(int(torch.isinf(v).sum()) for v in finite_tensors)
        return {
            "seed": self.seed, "phase": phase, "episode": episode,
            "target_class": ACTIONS[target], "target_index": target,
            "target_action": ACTIONS[target], "color": color, "stimulus_hash": sha,
            "frame_times_ms": frame_times, "action_time_ms": action_time,
            "goodness_delay_ms": delay_ms, "goodness_time_ms": delivery,
            "q": signal.q.cpu().tolist(), "p": probs.tolist(),
            "noise": signal.noise.cpu().tolist(), "threshold": signal.threshold.cpu().tolist(),
            "action_bits": bits, "sampled_actions": [ACTIONS[i] for i, bit in enumerate(bits) if bit],
            "g_star": g, "exact_success": exact,
            "target_q": float(signal.q[target]), "target_p": ptarget,
            "non_target_mean_p": float((probs.sum()-probs[target])/26),
            "target_bit_actual": bool(bits[target]),
            "non_target_false_rate": (sum(bits)-int(bits[target]))/26,
            "active_bit_count": sum(bits), "exact_event_probability": exact_event_probability,
            "parameter_norm": norm, "parameter_delta_norm": delta,
            "plastic_state": after_state,
            "pre_goodness_plastic_state": before_state,
            "g_bar_before": g_bar_before, "g_bar_after": g_bar_after,
            "goodness_modulation": modulation,
            "nan_count": nan_count, "inf_count": inf_count,
        }

    def checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"core": self.core.state_dict(), "eye": self.eye.state_dict(),
                    "hand": self.hand.state_dict(),
                    "plastic_states": {f"{i}:{name}": self.plasticity.states[id(p)].cpu()
                                       for i, group in self.groups.items()
                                       for name, p in group.items()},
                    "seed": self.seed}, path)

    def changed_groups(self, initial):
        return {str(i): {name: float((p.detach()-initial[id(p)]).norm())
                         for name, p in group.items() if float((p.detach()-initial[id(p)]).norm()) > 0}
                for i, group in self.groups.items()}

