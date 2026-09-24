"""ACNT Runtime v0 mock smoke, with an optional isolated Core smoke."""

import argparse
import json
from pathlib import Path

import torch

from .block import Block
from .core import Core
from .runtime import ORGANS, Runtime
from .adapters import EarAdapter, EyeAdapter, GoodnessAdapter, HandAdapter, RouteAdapter, SpeakAdapter
from .mechanical import MechanicalLog


def run_core(steps: int) -> None:
    torch.manual_seed(41)
    sizes = (3, 4, 5)
    core = Core([Block(i, n, sizes, hold_tick=4, ticktime=2) for i, n in enumerate(sizes)])
    with torch.no_grad():
        for block in core.blocks:
            block.z.copy_(torch.linspace(-1, 1, block.neuron_size))
    for tick in range(steps):
        # Manually specified masks exercise the primitive, not route behavior.
        core.set_active_mask([True, tick % 3 != 1, tick % 4 != 2])
        outputs = core.step(now_ms=tick * 2)
        print(json.dumps({
            "tick": tick,
            "time_ms": tick * 2,
            "active": core.active_ids,
            "updated": list(outputs),
            "history_lengths": [len(block.A) for block in core.blocks],
            "z_norms": [float(block.z.norm()) for block in core.blocks],
        }))


def build_mock_runtime(*, log_file: str | None = None, seed: int = 41) -> Runtime:
    """Six distinct organ Blocks plus two ordinary Blocks; no Stage 0 task."""
    torch.manual_seed(seed)
    n, count = 4, 8
    core = Core([Block(i, n, [n] * count, ticktime=250, hold_tick=4, readin=i < 2) for i in range(count)])
    adapters = {
        "eye": EyeAdapter(n), "ear": EarAdapter(n), "hand": HandAdapter(n),
        "speak": SpeakAdapter(n), "goodness": GoodnessAdapter(n), "route": RouteAdapter(n, count),
    }
    runtime = Runtime(core, adapters, dict(zip(ORGANS, range(6))))
    runtime.mechanical_log = MechanicalLog(log_file)
    runtime.enable_plasticity(learning_rate=0.0001, tau_q_s=1.0, tau_g_s=5.0, parameter_clip=(-10., 10.))
    with torch.no_grad():
        for block in core.blocks:
            block.z.copy_(torch.linspace(-0.5, 0.5, n))
        # Center the mock goodness predictor away from clamp saturation.
        adapters["goodness"].linear.weight.zero_()
        adapters["goodness"].linear.bias.fill_(0.5)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="ACNT Runtime v0 mock smoke; never runs Stage 0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--log-file", default="runs/runtime-v0/mechanical.jsonl")
    parser.add_argument("--summary-file")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.core_only:
        run_core(args.steps)
        return
    runtime = build_mock_runtime(log_file=args.log_file, seed=args.seed)
    before = {name: p.detach().clone() for name, p in runtime.named_parameters()}
    generator = torch.Generator().manual_seed(args.seed + 1)
    rows = []
    for tick in range(args.steps):
        image = torch.full((3, 1080, 1920), (tick % 5) / 4.)
        audio = torch.linspace(-1., 1., 1600).reshape(1, 1600) * ((tick % 3) / 2.)
        # Sparse, coincident mock labels; no letter task or training corpus.
        teacher = (0.8 if tick % 8 == 0 else 0.2) if tick % 4 == 0 else None
        row = runtime.step(now_ms=tick * 250, readins={"eye": image, "ear": audio}, teacher=teacher, generator=generator)
        row["tick"] = tick
        rows.append(row)
        print(json.dumps(row, allow_nan=False))
    if args.summary_file:
        summary = {
            "steps": args.steps, "blocks": len(runtime.core.blocks),
            "mechanical_records": len(runtime.mechanical_log.records),
            "finite_parameters": all(bool(torch.isfinite(p).all()) for p in runtime.parameters()),
            "changed_parameter_tensors": [name for name, p in runtime.named_parameters() if not torch.equal(p, before[name])],
            "teacher_steps": [row["tick"] for row in rows if row["teacher"] is not None],
            "distinct_active_sets": len({tuple(row["next_active"]) for row in rows}),
            "local_plasticity": True, "stage_0_run": False,
        }
        target = Path(args.summary_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
