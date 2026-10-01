"""Immediate scalar feedback bootstrap; no resets/replay/subject event ledger."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from acnt.self_write import SelfWriteKernel, SelfWriteLearner


def run(seed=11, *, mode="full", decisions=1000, gap=3, task="constant", lr=.001,
        write_rate=.02, output="runs/self_write_online/pilot"):
    if type(decisions) is not int or decisions < 1 or type(gap) is not int or gap < 1:
        raise ValueError("positive integer decisions and gap required")
    if task not in ("constant", "cue", "switch"):
        raise ValueError("unknown task")
    kernel = SelfWriteKernel(seed=seed, write_rate=write_rate)
    model = SelfWriteLearner(kernel, lr=0 if mode == "frozen" else lr,
        write_enabled=mode != "no_write", cut_write_credit=mode == "cut_write",
        constant_controller=mode == "constant_controller")
    rng = random.Random(seed + 70000)
    action_rng = torch.Generator().manual_seed(seed + 10000)
    source_names = ("acnt/self_write.py", "acnt/block.py", "acnt/adapters.py", __file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_names}
    initial_phi = torch.cat([p.detach().flatten().clone() for p in model.parameters])
    rows, bit = [], 0
    resources = model.persistent_tensor_bytes
    started = time.perf_counter()
    period = gap + 1
    for event in range(decisions * period):
        phase, cue, goodness, action = event % period, None, None, None
        if phase == 0:
            bit = rng.randrange(2)
            amplitude = rng.uniform(.5, 1.)
            sign = 2 * bit - 1
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude, .5*sign*amplitude, -.5*sign*amplitude]])
        if phase == gap:
            decision = len(rows)
            target = 1 if task == "constant" else bit if task == "cue" else int(decision // 250 % 2 == 0)
            action, probability, score = model.sample(action_rng)
            goodness = float(action == target)
            row = {"decision": decision + 1, "event": event, "target": target,
                "action": action, "goodness": goodness,
                "expected_goodness": probability if target else 1 - probability,
                "controller_score_norm": float(score.norm()), "actor_bias": float(model.state[-1])}
        # Input feedback only after the action, never target or private expected G.
        model.step(now_ms=event * 2, cue=cue, goodness=goodness, action=action)
        if goodness is not None:
            row["outer_update_norm"] = model.learn(score, goodness)
            rows.append(row)
            if len(rows) % 100 == 0:
                print(json.dumps({"seed": seed, "mode": mode, "task": task, "decisions": len(rows),
                    "expected_goodness": sum(r['expected_goodness'] for r in rows[-100:]) / 100,
                    "seconds": time.perf_counter() - started}), flush=True)
        assert resources == model.persistent_tensor_bytes
        assert model.state.grad_fn is None and model.meta.grad_fn is None
    def stats(part):
        return {"count": len(part), "expected_goodness": sum(r['expected_goodness'] for r in part)/len(part),
                "sample_accuracy": sum(r['goodness'] for r in part)/len(part)}
    window = min(100, max(1, decisions // 4))
    final_phi = torch.cat([p.detach().flatten() for p in model.parameters])
    result = {"seed": seed, "mode": mode, "task": task, "decisions": decisions,
        "events": model.steps, "gap": gap, "lr": model.lr, "write_rate": write_rate, "neural_resets": 0,
        "first": stats(rows[:window]), "last": stats(rows[-window:]), "all": stats(rows),
        "outer_updates": model.updates, "controller_parameter_dim": model.param_size,
        "joint_state_dim": model.state.numel(), "actor_fast_parameter_dim": kernel.fast_size,
        "persistent_state_and_training_tensor_bytes": resources, "meta_clips": model.meta_clips,
        "actor_write_path_length": model.write_path, "controller_change_norm": float((final_phi-initial_phi).norm()),
        "source_hashes": hashes, "seconds": time.perf_counter()-started, "rows": rows,
        "limitations": ["immediate feedback only", "all-active synchronous seven-Block oracle",
            "write one original Core connection and one Hand bias only",
            "dense forward meta tangent, stop outer updates and clipped", "synthetic world, no mechanical effects",
            "constant/cue/switch tasks are not lifelong no-forgetting tests"]}
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f"{task}_seed_{seed}_{mode}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("seed", "mode", "task", "first", "last", "seconds", "meta_clips")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--modes", nargs="+", choices=["full", "frozen", "no_write", "cut_write", "constant_controller"], default=["full"])
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--task", choices=["constant", "cue", "switch"], default="constant")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--write-rate", type=float, default=.02)
    parser.add_argument("--output", default="runs/self_write_online/pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for mode in args.modes:
        for seed in args.seeds:
            run(seed, mode=mode, decisions=args.decisions, task=args.task, lr=args.lr,
                write_rate=args.write_rate, output=args.output)


if __name__ == "__main__":
    main()
