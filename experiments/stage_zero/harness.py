"""Thin composition of canonical ACNT; no experimental learning equations."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from acnt.adapters import EyeAdapter, HandAdapter
from acnt.block import Block
from acnt.control import DiscreteSignal, sample_discrete
from acnt.core import Core
from acnt.runtime import Runtime, HandSignal


@dataclass(frozen=True)
class Protocol:
    blocks: int = 10
    neurons: int = 100
    ticktime: float = 0.25
    hold_tick: int = 4
    frames_per_episode: int = 3
    frame_interval_ms: int = 250
    goodness_delay_ms: int = 250
    learning_rate: float = 0.001
    rho: float = 0.9
    ema_alpha: float = 0.1
    initial_g_bar: float = 0.5

    def __post_init__(self):
        # This entry point intentionally exposes no architecture/tuning search.
        if (self.blocks, self.neurons, self.ticktime, self.hold_tick,
                self.frames_per_episode, self.frame_interval_ms,
                self.goodness_delay_ms) != (10, 100, 0.25, 4, 3, 250, 250):
            raise ValueError("Stage Zero fixes 10x100, three 250 ms frames and 250 ms delay")

    @property
    def episode_interval_ms(self):
        return self.frames_per_episode * self.frame_interval_ms + self.goodness_delay_ms


class StageZero(nn.Module):
    """Only Eye and Hand are bound; the six-organ Runtime is never constructed.

    Reuse the actual Runtime learning wiring by method reference. Its feedback
    initializer also creates dormant route buffers; no route adapter or graph
    exists here. In particular this class has no step(), executors or _emit().
    Move modules before making eligibility banks, and before history exists.
    """

    encode_readin = Runtime.encode_readin
    decode_readout = Runtime.decode_readout
    _observe_discrete = Runtime._observe_discrete

    def __init__(self, seed: int, *, device="cpu", protocol: Protocol | None = None):
        super().__init__()
        self.protocol = protocol or Protocol()
        cfg = self.protocol
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.core = Core([
                Block(i, cfg.neurons, [cfg.neurons] * cfg.blocks,
                      readin=i == 0, active=True, ticktime=cfg.ticktime,
                      hold_tick=cfg.hold_tick)
                for i in range(cfg.blocks)
            ])
            self.adapters = nn.ModuleDict({
                "eye": EyeAdapter(cfg.neurons),
                "hand": HandAdapter(cfg.neurons, discrete_controls=27, continuous_controls=0),
            })
        self.organ_blocks = {"eye": 0, "hand": 1}
        self._local_z: dict[int, Tensor] = {}
        self.to(device=device, dtype=torch.float32)
        self.plasticity = Runtime.enable_plasticity(
            self, feedback_seed=seed, learning_rate=cfg.learning_rate,
            rho=cfg.rho, ema_alpha=cfg.ema_alpha, initial_g_bar=cfg.initial_g_bar,
        )
        self.pending: tuple[int, bool] | None = None
        self.last_delivery_ms: int | None = None

    @property
    def device(self):
        return self.core.blocks[0].b.device

    def update_frame(self, *, now_ms: int, image: Tensor | None = None,
                     learn: bool = True, order=None) -> dict[int, Tensor]:
        """One committed-snapshot round; even a zero frame is a new event."""
        if self.pending is not None:
            raise RuntimeError("deliver the pending goodness before another frame")
        if self.last_delivery_ms is not None and now_ms < self.last_delivery_ms:
            raise ValueError("frame cannot precede previous goodness delivery")
        ids = list(range(self.protocol.blocks)) if order is None else list(order)
        if sorted(ids) != list(range(self.protocol.blocks)):
            raise ValueError("order must visit every Block exactly once")
        if self.core.active_ids != tuple(range(self.protocol.blocks)):
            raise RuntimeError("Stage Zero requires all Blocks active")
        for i in ids:
            self.core.is_due(i, now_ms)
        self._local_z.clear()
        with torch.set_grad_enabled(learn):
            if image is not None:
                self.encode_readin("eye", image)
            sources = self.core.source_snapshot()
            outputs = {}
            for i in ids:
                forced = i == self.organ_blocks["eye"] and image is not None
                if forced or self.core.is_due(i, now_ms):
                    outputs[i] = self.core.update_block(
                        i, now_ms=now_ms, force=forced, track_grad=learn,
                        source_snapshot=sources,
                    )
                    if learn:
                        self._local_z[i] = outputs[i]
                        self.plasticity.internal[i].advance(now_ms)
        return outputs

    def act(self, image: Tensor, *, start_ms: int, generator: torch.Generator,
            learn: bool) -> tuple[int, HandSignal]:
        """Three fresh Eye frames, then one action. No target argument exists."""
        cfg = self.protocol
        for frame in range(cfg.frames_per_episode):
            now_ms = start_ms + frame * cfg.frame_interval_ms
            # No action exists on the first two rounds; no local graph survives.
            self.update_frame(now_ms=now_ms, image=image,
                              learn=learn and frame == cfg.frames_per_episode - 1)
        with torch.set_grad_enabled(learn):
            raw = self.decode_readout("hand")
            hand = sample_discrete(raw, tau=self.core.blocks[1].ticktime,
                                   generator=generator)
            if learn:
                # Canonical adapter score eligibility and fixed-random Block VJP.
                self._observe_discrete("hand", hand, now_ms=now_ms)
        self._local_z.clear()
        self.pending = (now_ms, learn)
        detached = DiscreteSignal(**{
            name: getattr(hand, name).detach() if isinstance(getattr(hand, name), Tensor)
            else getattr(hand, name)
            for name in ("q", "noise", "threshold", "p", "a", "tau")
        })
        return now_ms, HandSignal(detached, raw[27:].detach())

    def deliver_goodness(self, goodness: float, *, now_ms: int) -> float | None:
        """Only a scalar enters Plasticity, exactly 250 logical ms after action."""
        if self.pending is None:
            raise RuntimeError("no pending action")
        action_ms, learn = self.pending
        if now_ms != action_ms + self.protocol.goodness_delay_ms:
            raise ValueError("goodness must arrive exactly 250 ms after action")
        if goodness not in (0.0, 1.0):
            raise ValueError("Stage Zero goodness must be binary")
        delta = self.plasticity.apply_goodness(float(goodness), now_ms=now_ms) if learn else None
        self.pending = None
        self.last_delivery_ms = now_ms
        return delta

    def reset_dynamics(self):
        """Identical phase-start protocol; weights and goodness baseline persist."""
        if self.pending is not None:
            raise RuntimeError("cannot reset with undelivered goodness")
        self._local_z.clear()
        for block in self.core.blocks:
            for name in ("z", "a", "r", "h", "o"):
                value = getattr(block, name)
                if value is not None:
                    setattr(block, name, torch.zeros_like(value))
            block.A.clear()
            block.At.clear()
        self.plasticity.clear()
        self.last_delivery_ms = None

    def parameter_vector(self):
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def eligibility_vector(self):
        return torch.cat([t.flatten() for channel in self.plasticity.banks()
                          for bank in channel.values() for t in bank.values.values()])

    def check_devices(self):
        tensors = [*self.parameters(), *self.buffers()]
        tensors += [t for block in self.core.blocks for t in block.A]
        tensors += [t for channel in self.plasticity.banks()
                    for bank in channel.values() for t in bank.values.values()]
        return all(t.dtype == torch.float32 and t.device == self.device for t in tensors)
