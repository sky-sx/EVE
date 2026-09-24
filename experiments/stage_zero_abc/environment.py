"""Minimal deterministic ABC environment for ACNT Stage Zero."""

from __future__ import annotations

import random

import torch


ACTIONS = ("A", "B", "C")
CLASS_COUNT = len(ACTIONS)

DELAYS_MS = (0,)


def encode_target(
    target: int,
    *,
    neuron_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if type(target) is not int or target not in range(CLASS_COUNT):
        raise ValueError("target must be an ABC class index")

    if type(neuron_size) is not int or neuron_size < CLASS_COUNT:
        raise ValueError(
            "neuron_size must be at least the number of ABC classes"
        )

    o = torch.zeros(
        2 * neuron_size,
        dtype=torch.float32,
        device=device,
    )

    o[target] = 1.0

    return o


def potential_goodness(
    target: int,
    action: torch.Tensor,
) -> float:
    if type(target) is not int or target not in range(CLASS_COUNT):
        raise ValueError("target must be an ABC class index")

    if (
        not isinstance(action, torch.Tensor)
        or action.shape != (CLASS_COUNT,)
        or action.dtype != torch.bool
    ):
        raise ValueError(
            "action must be a bool tensor of shape (3,)"
        )

    if not bool(action[target]):
        return 0.0

    active = int(action.sum())

    if active <= 0:
        raise AssertionError(
            "target-active action must contain at least one active bit"
        )

    return 1.0 / active


def exact_success(
    target: int,
    action: torch.Tensor,
) -> bool:
    if type(target) is not int or target not in range(CLASS_COUNT):
        raise ValueError("target must be an ABC class index")

    if (
        not isinstance(action, torch.Tensor)
        or action.shape != (CLASS_COUNT,)
        or action.dtype != torch.bool
    ):
        raise ValueError(
            "action must be a bool tensor of shape (3,)"
        )

    return (
        bool(action[target])
        and int(action.sum()) == 1
    )


def balanced_targets(
    count: int,
    rng: random.Random,
) -> list[int]:
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")

    order: list[int] = []

    while len(order) < count:
        cycle = list(range(CLASS_COUNT))
        rng.shuffle(cycle)

        remaining = count - len(order)
        order.extend(cycle[:remaining])

    return order


def phase_schedule(
    count: int,
    seed: int,
    phase_index: int,
):
    if type(seed) is not int:
        raise ValueError("seed must be int")

    if type(phase_index) is not int or phase_index < 0:
        raise ValueError(
            "phase_index must be a nonnegative int"
        )

    target_seed = (
        seed * 1000003
        + phase_index * 10007
        + 1
    )

    hand_seed = (
        seed * 1000003
        + phase_index * 10007
        + 2
    )

    targets = balanced_targets(
        count,
        random.Random(target_seed),
    )

    schedule = [
        (target, 0)
        for target in targets
    ]

    streams = {
        "target": target_seed,
        "hand": hand_seed,
    }

    return schedule, streams