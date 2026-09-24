"""Runner for the minimal ABC perturbation-eprop baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch

from .environment import (
    ACTIONS,
    DELAYS_MS,
    phase_schedule,
)
from .harness import StageZeroABC


DEFAULT_COUNTS = {
    "initial": 300,
    "training": 3000,
    "frozen": 600,
}

DEFAULT_SEEDS = (
    11,
    22,
    33,
    44,
    55,
)

DEFAULT_DEVICE = "cuda"

LEARNING_RATE = 0.001
TAU_Q_S = 1.0
TAU_G_S = 5.0
PERTURBATION_SCALE = 0.1
ACTION_TAU = 0.25

SCHEMA = 1


def source_hashes():
    paths = (
        "acnt/block.py",
        "acnt/plasticity.py",
        "acnt/core.py",
        "acnt/adapters.py",
        "acnt/control.py",
        "experiments/stage_zero_abc/environment.py",
        "experiments/stage_zero_abc/harness.py",
        "experiments/stage_zero_abc/runner.py",
    )

    return {
        path:
        hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest()
        for path in paths
    }


def config(
    *,
    device,
    counts,
    seeds,
):
    return {
        "schema": SCHEMA,
        "device": device,
        "counts": counts,
        "seeds": list(seeds),

        "task":
            "ABC one-hot -> ABC discrete action",

        "block_count": 2,
        "neuron_size": 10,
        "hold_tick": 4,
        "ticktime_ms": 250,

        "readin_block": 0,
        "hand_block": 1,

        "frames_per_episode": 3,
        "frame_offsets_ms": [
            0,
            250,
            500,
        ],

        "inter_episode_gap_ms": 250,

        "goodness_delays_ms":
            list(DELAYS_MS),

        "goodness":
            "target active: 1 / active bits; otherwise 0",

        "exact_success":
            "target is the only active bit; evaluation only",

        "learning_rate":
            LEARNING_RATE,

        "tau_q_s":
            TAU_Q_S,

        "tau_g_s":
            TAU_G_S,

        "perturbation_scale":
            PERTURBATION_SCALE,

        "action_tau":
            ACTION_TAU,

        "threshold": 0.0,

        "g_bar_initial": 0.5,

        "parameter_clip": None,

        "input":
            "deterministic direct Block0 o vector; o[target]=1; all remaining entries zero",

        "input_adapter":
            None,

        "teacher":
            None,

        "autograd_training":
            False,

        "bptt":
            False,

        "curriculum":
            False,

        "perturbation_seed_rule":
            "seed*1000003+900001",

        "target_seed_rule":
            "seed*1000003+phase_index*10007+1",

        "hand_seed_rule":
            "seed*1000003+phase_index*10007+2",

        "source_sha256":
            source_hashes(),

        "python":
            platform.python_version(),

        "torch":
            torch.__version__,

        "cuda":
            torch.version.cuda,

        "gpu":
            (
                torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None
            ),
    }


def aggregate(
    rows,
):
    if not rows:
        return {}

    n = len(rows)

    exact_count = sum(
        int(row["exact_success"])
        for row in rows
    )

    target_hits = sum(
        int(row["target_bit_actual"])
        for row in rows
    )

    result = {
        "episodes": n,
        "exact_success":
            exact_count,
        "exact_success_rate":
            exact_count / n,
        "positive_g_star":
            sum(
                row["g_star"] > 0
                for row in rows
            ),
        "target_bit_hit_rate":
            target_hits / n,
        "mean_g_star":
            sum(
                row["g_star"]
                for row in rows
            ) / n,
        "mean_active_bits":
            sum(
                row["active_bit_count"]
                for row in rows
            ) / n,
        "target_p":
            sum(
                row["target_p"]
                for row in rows
            ) / n,
        "non_target_mean_p":
            sum(
                row["non_target_mean_p"]
                for row in rows
            ) / n,
        "specificity":
            sum(
                (
                    row["target_p"]
                    - row[
                        "non_target_mean_p"
                    ]
                )
                for row in rows
            ) / n,
        "non_target_false_rate":
            sum(
                row[
                    "non_target_false_rate"
                ]
                for row in rows
            ) / n,
        "exact_event_probability":
            sum(
                row[
                    "exact_event_probability"
                ]
                for row in rows
            ) / n,
        "parameter_delta_norm":
            sum(
                row[
                    "parameter_delta_norm"
                ]
                for row in rows
            ) / n,
        "trace_total_l2_mean":
            sum(
                row[
                    "eligibility_trace"
                ]["total"]["l2"]
                for row in rows
            ) / n,
        "trace_total_l2_max":
            max(
                row[
                    "eligibility_trace"
                ]["total"]["l2"]
                for row in rows
            ),
        "nan_count":
            sum(
                row["nan_count"]
                for row in rows
            ),
        "inf_count":
            sum(
                row["inf_count"]
                for row in rows
            ),
    }

    return result


def summarize(
    rows,
):
    result = {
        "overall":
            aggregate(rows),
        "phase": {},
        "per_seed": {},
        "per_class": {},
    }

    seeds = sorted({
        row["seed"]
        for row in rows
    })

    for phase in (
        "initial",
        "training",
        "frozen",
    ):
        subset = [
            row
            for row in rows
            if row["phase"] == phase
        ]

        result["phase"][phase] = (
            aggregate(subset)
        )

        result["per_seed"][phase] = {
            str(seed):
            aggregate([
                row
                for row in subset
                if row["seed"] == seed
            ])
            for seed in seeds
        }

        result["per_class"][phase] = {
            action:
            aggregate([
                row
                for row in subset
                if row["target_action"]
                == action
            ])
            for action in ACTIONS
        }

    return result


def run_seed(
    seed,
    *,
    device,
    counts,
    output,
):
    output = Path(output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = StageZeroABC(
        seed,
        device,
        learning_rate=LEARNING_RATE,
        tau_q_s=TAU_Q_S,
        tau_g_s=TAU_G_S,
        perturbation_scale=
            PERTURBATION_SCALE,
        action_tau=ACTION_TAU,
    )

    initial_parameters = {
        id(parameter):
        parameter.detach().clone()
        for parameter
        in model.plasticity
        .parameters
        .values()
    }

    rows = []

    clock_ms = 0

    start = time.perf_counter()

    raw_path = (
        output
        / f"seed_{seed}.jsonl"
    )

    initial_g_bar = None

    with raw_path.open(
        "w",
        encoding="utf-8",
    ) as log:
        for phase_index, (
            phase,
            count,
        ) in enumerate(
            counts.items()
        ):
            model.reset_phase(
                phase == "training"
            )

            schedule, streams = (
                phase_schedule(
                    count,
                    seed,
                    phase_index,
                )
            )

            if phase == "training":
                if initial_g_bar is None:
                    raise AssertionError(
                        "initial phase must run before training"
                    )

                model.plasticity.g_bar = (
                    initial_g_bar
                )

            action_generator = (
                torch.Generator(
                    device=model.device
                )
            )

            action_generator.manual_seed(
                streams["hand"]
            )

            phase_rows = []

            for episode, (
                target,
                delay_ms,
            ) in enumerate(schedule):
                row = model.episode(
                    phase=phase,
                    episode=episode,
                    target=target,
                    delay_ms=delay_ms,
                    start_ms=clock_ms,
                    action_generator=
                        action_generator,
                )

                row["rng_seeds"] = (
                    streams
                )

                log.write(
                    json.dumps(
                        row,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                rows.append(row)
                phase_rows.append(row)

                clock_ms = (
                    row["goodness_time_ms"]
                    + 250
                )

            if phase == "initial":
                initial_g_bar = (
                    sum(
                        row["g_star"]
                        for row
                        in phase_rows
                    )
                    / len(phase_rows)
                )

            log.flush()

    changed = model.changed_groups(
        initial_parameters
    )

    result = {
        "seed": seed,
        "elapsed_s":
            time.perf_counter()
            - start,
        "raw_path":
            str(raw_path),
        "raw_sha256":
            hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
        "metrics":
            summarize(rows),
        "changed_parameter_groups":
            changed,
        "positive_count":
            sum(
                row["g_star"] > 0
                for row in rows
            ),
    }

    (
        output
        / f"seed_{seed}_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return result


def main(
    argv=None,
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
        ),
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--output",
        default=
            "runs/stage_zero_abc",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--initial",
        type=int,
        default=
            DEFAULT_COUNTS[
                "initial"
            ],
    )

    parser.add_argument(
        "--training",
        type=int,
        default=
            DEFAULT_COUNTS[
                "training"
            ],
    )

    parser.add_argument(
        "--frozen",
        type=int,
        default=
            DEFAULT_COUNTS[
                "frozen"
            ],
    )

    args = parser.parse_args(
        argv
    )

    counts = {
        "initial":
            args.initial,
        "training":
            args.training,
        "frozen":
            args.frozen,
    }

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = config(
        device=args.device,
        counts=counts,
        seeds=args.seeds,
    )

    (
        output
        / "config.json"
    ).write_text(
        json.dumps(
            cfg,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    results = [
        run_seed(
            seed,
            device=args.device,
            counts=counts,
            output=output,
        )
        for seed in args.seeds
    ]

    summary = {
        "config":
            cfg,
        "seeds":
            results,
    }

    (
        output
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()