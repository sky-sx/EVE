"""Minimal 2-Block ABC harness for perturbation-eprop validation."""

from __future__ import annotations

from pathlib import Path

import torch

from acnt import Block, Core
from acnt.adapters import HandAdapter
from acnt.control import sample_discrete
from acnt.plasticity import Plasticity

from .environment import (
    ACTIONS,
    CLASS_COUNT,
    encode_target,
    exact_success,
    potential_goodness,
)


class StageZeroABC(torch.nn.Module):
    BLOCK_COUNT = 2
    NEURON_SIZE = 10
    HOLD_TICK = 4
    TICKTIME_MS = 250
    FRAMES_PER_EPISODE = 3

    def __init__(
        self,
        seed: int,
        device: str = "cpu",
        *,
        learning_rate: float = 0.001,
        tau_q_s: float = 1.0,
        tau_g_s: float = 5.0,
        perturbation_scale: float = 0.1,
        action_tau: float = 0.25,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.device = torch.device(device)

        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        sizes = [
            self.NEURON_SIZE,
            self.NEURON_SIZE,
        ]

        self.core = Core([
            Block(
                0,
                self.NEURON_SIZE,
                sizes,
                readin=True,
                hold_tick=self.HOLD_TICK,
                ticktime=self.TICKTIME_MS,
            ),
            Block(
                1,
                self.NEURON_SIZE,
                sizes,
                readin=False,
                hold_tick=self.HOLD_TICK,
                ticktime=self.TICKTIME_MS,
            ),
        ]).to(self.device)

        self.hand = HandAdapter(
            self.NEURON_SIZE,
            discrete_controls=CLASS_COUNT,
            continuous_controls=0,
            hidden_size=self.NEURON_SIZE,
        ).to(self.device)

        self.groups = {}

        for i, block in enumerate(self.core.blocks):
            group = {
                "block." + name: parameter
                for name, parameter
                in block.named_parameters()
            }

            if i == 1:
                group.update({
                    "adapter." + name: parameter
                    for name, parameter
                    in self.hand.named_parameters()
                })

            self.groups[i] = group

        self.plasticity = Plasticity(
            self.groups,
            goodness_id=None,
            learning_rate=learning_rate,
            tau_q_s=tau_q_s,
            tau_g_s=tau_g_s,
            perturbation_scale=perturbation_scale,
        )

        self.action_tau = float(action_tau)

        self.perturbation_generator = torch.Generator(
            device=self.device,
        )

        self.perturbation_generator.manual_seed(
            seed * 1000003 + 900001
        )

        self._learning = False

        self._previous = {
            id(parameter):
            parameter.detach().clone()
            for parameter
            in self.plasticity.parameters.values()
        }

        self.set_learning(False)

    def _observe_block(
        self,
        parameter,
        contribution,
        time_ms,
    ) -> None:
        self.plasticity.accumulate(
            parameter,
            contribution,
            now_ms=time_ms,
        )

    def set_learning(
        self,
        learning: bool,
    ) -> None:
        if type(learning) is not bool:
            raise TypeError(
                "learning must be bool"
            )

        self.plasticity.detach_adapters()

        self._learning = learning

        if learning:
            self.plasticity.attach_adapters({
                "hand": self.hand,
            })

        for block in self.core.blocks:
            block.local_observer = (
                self._observe_block
                if learning
                else None
            )

            block.set_learning(
                learning,
                perturbation_scale=(
                    self.plasticity.perturbation_scale
                    if learning
                    else 0.0
                ),
                generator=(
                    self.perturbation_generator
                    if learning
                    else None
                ),
            )

    def reset_phase(
        self,
        learning: bool,
    ) -> None:
        self.set_learning(learning)

        self.plasticity.clear()

        for block in self.core.blocks:
            for name in (
                "z",
                "z_bar",
                "a",
                "r",
            ):
                getattr(
                    block,
                    name,
                ).zero_()

            if block.o is not None:
                block.o.zero_()

            block.A.clear()
            block.At.clear()
            block.learning_frames.clear()

            block.active = True

        self._previous = {
            id(parameter):
            parameter.detach().clone()
            for parameter
            in self.plasticity.parameters.values()
        }

    def _parameter_stats(self):
        parameters = list(
            self.plasticity.parameters.values()
        )

        norm = float(
            torch.stack([
                parameter.detach()
                .square()
                .sum()
                for parameter
                in parameters
            ])
            .sum()
            .sqrt()
        )

        delta = float(
            torch.stack([
                (
                    parameter.detach()
                    - self._previous[id(parameter)]
                )
                .square()
                .sum()
                for parameter
                in parameters
            ])
            .sum()
            .sqrt()
        )

        for parameter in parameters:
            self._previous[
                id(parameter)
            ].copy_(
                parameter.detach()
            )

        return norm, delta

    @staticmethod
    def _summarize(
        tensors,
    ):
        tensors = list(tensors)

        if not tensors:
            return {
                "l2": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "max_abs": 0.0,
                "nonzero_fraction": 0.0,
            }

        total = sum(
            tensor.numel()
            for tensor in tensors
        )

        squares = torch.stack([
            tensor.square().sum()
            for tensor in tensors
        ]).sum()

        mean = torch.stack([
            tensor.sum()
            for tensor in tensors
        ]).sum() / total

        max_abs = torch.stack([
            tensor.abs().max()
            for tensor in tensors
        ]).max()

        nonzero = torch.stack([
            (tensor != 0).sum()
            for tensor in tensors
        ]).sum()

        std = (
            squares / total
            - mean.square()
        ).clamp_min(0).sqrt()

        return {
            "l2": float(
                squares.sqrt()
            ),
            "mean": float(mean),
            "std": float(std),
            "max_abs": float(max_abs),
            "nonzero_fraction": float(
                nonzero / total
            ),
        }

    def trace_stats(self):
        block0 = list(
            self.core.blocks[0]
            .parameters()
        )

        block1 = list(
            self.core.blocks[1]
            .parameters()
        )

        hand = list(
            self.hand.parameters()
        )

        terminal = list(
            self.hand.network[-1]
            .parameters()
        )

        return {
            "block0": self._summarize(
                self.plasticity.traces[
                    id(parameter)
                ]
                for parameter
                in block0
            ),
            "block1": self._summarize(
                self.plasticity.traces[
                    id(parameter)
                ]
                for parameter
                in block1
            ),
            "hand": self._summarize(
                self.plasticity.traces[
                    id(parameter)
                ]
                for parameter
                in hand
            ),
            "terminal_hand": self._summarize(
                self.plasticity.traces[
                    id(parameter)
                ]
                for parameter
                in terminal
            ),
            "total": self._summarize(
                self.plasticity.traces.values()
            ),
        }

    @torch.no_grad()
    def episode(
        self,
        *,
        phase: str,
        episode: int,
        target: int,
        delay_ms: int,
        start_ms: int,
        action_generator: torch.Generator,
    ):
        if delay_ms != 0:
            raise ValueError(
                "ABC baseline requires delay_ms == 0"
            )

        frame_times = [
            start_ms
            + self.TICKTIME_MS * i
            for i in range(
                self.FRAMES_PER_EPISODE
            )
        ]

        input_o = encode_target(
            target,
            neuron_size=self.NEURON_SIZE,
            device=self.device,
        )

        for now_ms in frame_times:
            self.core.blocks[0].o = (
                input_o.clone()
            )

            updated = self.core.step(
                now_ms=now_ms,
            )

            if tuple(updated) != (0, 1):
                raise AssertionError(
                    "both ABC Blocks must update every tick"
                )

        action_time = frame_times[-1]

        q = self.hand(
            self.core.blocks[1].z
        )

        signal = sample_discrete(
            q,
            tau=self.action_tau,
            threshold=0.0,
            generator=action_generator,
        )

        if self._learning:
            self.plasticity.observe_control(
                "hand",
                self.hand,
                signal,
                now_ms=action_time,
            )

        goodness = potential_goodness(
            target,
            signal.a,
        )

        exact = exact_success(
            target,
            signal.a,
        )

        goodness_time = (
            action_time + delay_ms
        )

        trace_before_goodness = (
            self.trace_stats()
        )

        g_bar_before = (
            self.plasticity.g_bar
        )

        modulation = None

        if self._learning:
            modulation = (
                self.plasticity
                .apply_goodness(
                    goodness,
                    now_ms=goodness_time,
                )
            )

        trace_after_goodness = (
            self.trace_stats()
        )

        g_bar_after = (
            self.plasticity.g_bar
        )

        parameter_norm, parameter_delta = (
            self._parameter_stats()
        )

        probabilities = (
            signal.p.detach().cpu()
        )

        actions = (
            signal.a.detach().cpu()
        )

        target_probability = float(
            probabilities[target]
        )

        non_target_probability = float(
            (
                probabilities.sum()
                - probabilities[target]
            )
            / (CLASS_COUNT - 1)
        )

        target_bit = bool(
            actions[target]
        )

        wrong_active = (
            int(actions.sum())
            - int(target_bit)
        )

        exact_event_probability = float(
            probabilities[target]
            * torch.prod(
                torch.cat((
                    1.0
                    - probabilities[:target],
                    1.0
                    - probabilities[
                        target + 1:
                    ],
                ))
            )
        )

        finite_tensors = [
            *self.plasticity
            .parameters
            .values(),
            *self.plasticity
            .traces
            .values(),
        ]

        nan_count = sum(
            int(
                torch.isnan(tensor)
                .sum()
            )
            for tensor
            in finite_tensors
        )

        inf_count = sum(
            int(
                torch.isinf(tensor)
                .sum()
            )
            for tensor
            in finite_tensors
        )

        return {
            "seed": self.seed,
            "phase": phase,
            "episode": episode,
            "target_index": target,
            "target_action": ACTIONS[target],
            "frame_times_ms": frame_times,
            "action_time_ms": action_time,
            "goodness_delay_ms": delay_ms,
            "goodness_time_ms": goodness_time,
            "q": (
                signal.q
                .detach()
                .cpu()
                .tolist()
            ),
            "p": probabilities.tolist(),
            "action_bits": actions.tolist(),
            "g_star": goodness,
            "exact_success": exact,
            "target_q": float(
                signal.q[target]
            ),
            "target_p": target_probability,
            "non_target_mean_p":
                non_target_probability,
            "target_bit_actual":
                target_bit,
            "non_target_false_rate":
                wrong_active
                / (CLASS_COUNT - 1),
            "active_bit_count":
                int(actions.sum()),
            "exact_event_probability":
                exact_event_probability,
            "parameter_norm":
                parameter_norm,
            "parameter_delta_norm":
                parameter_delta,
            "eligibility_trace":
                trace_after_goodness,
            "pre_goodness_eligibility_trace":
                trace_before_goodness,
            "g_bar_before":
                g_bar_before,
            "g_bar_after":
                g_bar_after,
            "goodness_modulation":
                modulation,
            "nan_count":
                nan_count,
            "inf_count":
                inf_count,
        }

    def changed_groups(
        self,
        initial,
    ):
        return {
            str(i): {
                name: float(
                    (
                        parameter.detach()
                        - initial[
                            id(parameter)
                        ]
                    )
                    .norm()
                )
                for name, parameter
                in group.items()
                if float(
                    (
                        parameter.detach()
                        - initial[
                            id(parameter)
                        ]
                    )
                    .norm()
                ) > 0
            }
            for i, group
            in self.groups.items()
        }

    def checkpoint(
        self,
        path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "core":
                    self.core.state_dict(),
                "hand":
                    self.hand.state_dict(),
                "eligibility_traces": {
                    f"{i}:{name}":
                        self.plasticity
                        .traces[
                            id(parameter)
                        ]
                        .detach()
                        .cpu()
                    for i, group
                    in self.groups.items()
                    for name, parameter
                    in group.items()
                },
                "seed":
                    self.seed,
            },
            path,
        )