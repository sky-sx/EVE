"""OR/XOR mechanism sanity, using the production Persistent Delta-W rule.

A static four-row truth table defines this toy objective, G = 1 - MSE.
Each trial advances W once and evaluates it once; no +/- comparison or rollback.
This toy metric is not the ACNT Goodness definition or behavioral validation.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from acnt.plasticity import Plasticity


@torch.no_grad()
def run(task, seed, steps, subset_fraction, delta_magnitude, device="cpu"):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1), nn.Sigmoid()).to(device)
    learner = Plasticity({0: dict(model.named_parameters())}, None,
                         plasticity_seed=seed + 300000,
                         subset_fraction=subset_fraction, delta_magnitude=delta_magnitude)
    x = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
    y = torch.tensor([0.,1.,1.,1. if task == "OR" else 0.], device=device).unsqueeze(1)

    def evaluate():
        prediction = model(x)
        return float((prediction-y).square().mean()), float(((prediction>=.5)==(y>=.5)).float().mean())

    initial_loss, initial_accuracy = evaluate()
    learner.apply_goodness(1-initial_loss, now_ms=0)
    for step in range(1, steps+1):
        learner.begin_trial()
        loss, accuracy = evaluate()
        learner.apply_goodness(1-loss, now_ms=step)
    final_loss, final_accuracy = evaluate()
    assert all(p.grad is None for p in model.parameters())
    return {"task": task, "seed": seed, "plasticity_seed": seed+300000,
            "steps": steps, "subset_fraction": subset_fraction, "delta_magnitude": delta_magnitude,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "initial_accuracy": initial_accuracy, "final_accuracy": final_accuracy,
            "success": final_accuracy == 1., "success_definition": "all four thresholded predictions correct",
            "selected_per_trial": learner.selected_parameter_count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,22,33])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--subset-fraction", type=float, default=.001)
    parser.add_argument("--delta-magnitude", type=float, default=.001)
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    torch.set_num_threads(1)
    rows = [run(task, seed, args.steps, args.subset_fraction, args.delta_magnitude, args.device)
            for task in ("OR","XOR") for seed in args.seeds]
    result = {"mechanism": "Persistent Delta-W Plasticity", "goodness": "1 - MSE on the toy truth table",
              "device": args.device, "results": rows}
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
