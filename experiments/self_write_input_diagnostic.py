"""Observer-only signal probes. Private targets never enter the ACNT learner.

Probes fit on the first 70% of one continuous trajectory and evaluate later
states. Offline probe data is an experiment artifact, not subject memory.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import torch
from acnt.self_write import SelfWriteKernel, SelfWriteLearner


def probe(x, y):
    x = torch.stack(x).double()
    y = torch.tensor(y, dtype=torch.float64)
    split = int(.7 * len(y))
    mean, std = x[:split].mean(0), x[:split].std(0).clamp_min(1e-10)
    xn = (x - mean) / std
    head = torch.nn.Linear(x.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = head(xn[:split]).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[:split])
        loss = loss + .001 * head.weight.square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logits = head(xn).flatten()
        correct = (logits >= 0) == (y >= .5)
        difference = x[y == 1].mean(0) - x[y == 0].mean(0)
        return {"train_count": split, "test_count": len(y) - split,
                "train_accuracy": float(correct[:split].double().mean()),
                "future_accuracy": float(correct[split:].double().mean()),
                "class_mean_distance": float(difference.norm()),
                "feature_std_rms": float(x.std(0).square().mean().sqrt()),
                "normalized_class_distance": float((difference / std).norm())}


def hand_probe(features, targets, adapter):
    """Original Hand architecture, observer-supervised, no input standardization."""
    x, y = torch.stack(features), torch.tensor(targets, dtype=torch.float32)
    split = int(.7 * len(y))
    head = deepcopy(adapter)
    head.requires_grad_(True)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    for _ in range(2000):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head.forward_train(x[:split])[:, 0], y[:split])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        correct = (head.forward_train(x)[:, 0] >= 0) == (y >= .5)
    return {"train_accuracy": float(correct[:split].float().mean()),
            "future_accuracy": float(correct[split:].float().mean()),
            "steps": 2000, "training_signal": "observer labels, diagnostic only"}


def run(seed=11, *, mode="no_write", decisions=1000, write_mode="target", output="runs/self_write_input_diagnostic"):
    kernel = SelfWriteKernel(seed=seed, write_mode=write_mode)
    model = SelfWriteLearner(kernel) if mode == "trained_write" else None
    state = kernel.initial_state().detach()
    times = tuple(() for _ in range(7))
    rng = random.Random(seed + 70000)
    generator = torch.Generator().manual_seed(seed + 10000)
    stages = {name: [] for name in ("raw_audio", "ear_adapter", "ear_z_at_cue", "ear_z_at_action",
                                   "hand_z_at_action", "hand_history_at_action", "hand_hidden", "write_block_z_at_action")}
    targets, policy, credit = [], [], []
    n, stride = kernel.width, kernel.width * (kernel.hold + 1)
    bit = 0
    for event in range(decisions * 4):
        phase, cue, goodness, action = event % 4, None, None, None
        if phase == 0:
            bit = rng.randrange(2)
            amplitude = rng.uniform(.5, 1.)
            sign = 2 * bit - 1
            cue = state.new_tensor([[sign*amplitude, -sign*amplitude, .5*sign*amplitude, -.5*sign*amplitude]])
            audio_at_cue = cue.flatten().clone()
            encoded_at_cue = kernel.ear.forward_train(cue).detach()
        if phase == 3:
            if model is not None:
                action, probability, score = model.sample(generator)
                leaf = state.detach().requires_grad_(True)
                dq = torch.autograd.grad(kernel.logits(leaf), leaf)[0]
                a = dq[:kernel.state_size] @ model.meta[:kernel.state_size]
                b = dq[-1] * model.meta[-1]
                credit.append({"neural_path_norm": float(a.norm()), "bias_path_norm": float(b.norm()),
                               "total_norm": float((a+b).norm())})
            else:
                with torch.no_grad():
                    probability = float(torch.sigmoid(kernel.logits(state)))
                    action = int(torch.bernoulli(state.new_tensor(probability), generator=generator))
            goodness = float(action == bit)
            hand_z = state[2*stride:2*stride+n].detach()
            stage_values = {"raw_audio": audio_at_cue, "ear_adapter": encoded_at_cue,
                "ear_z_at_cue": ear_at_cue, "ear_z_at_action": state[stride:stride+n],
                "hand_z_at_action": hand_z, "hand_history_at_action": state[2*stride+n:3*stride],
                "hand_hidden": torch.relu(kernel.hand.network[0](hand_z)),
                "write_block_z_at_action": state[6*stride:6*stride+n]}
            for name, values in stage_values.items():
                stages[name].append(values.detach().clone())
            targets.append(bit)
            policy.append({"probability_one": probability, "expected_goodness": probability if bit else 1-probability,
                           "correct": goodness})
        if model is not None:
            model.step(now_ms=2*event, cue=cue, goodness=goodness, action=action)
            if goodness is not None:
                model.learn(score, goodness)
            state, times = model.state, model.times
        else:
            obs = state.new_tensor([goodness or 0., float(goodness is not None), action or 0., float(action is not None)])
            with torch.no_grad():
                state = kernel(state, times, 2*event, cue, obs, write_enabled=mode != "no_write").detach()
            times = tuple(tuple((*old, 2*event)[-kernel.hold:]) for old in times)
        if phase == 0:
            ear_at_cue = state[stride:stride+n].detach().clone()
        if phase == 3 and len(targets) % 200 == 0:
            print(json.dumps({"mode": mode, "decisions": len(targets),
                "expected_goodness_last_100": sum(r['expected_goodness'] for r in policy[-100:])/100}), flush=True)
    split = int(.7 * len(targets))
    def stats(rows):
        return {"expected_goodness": sum(r['expected_goodness'] for r in rows)/len(rows),
                "sample_accuracy": sum(r['correct'] for r in rows)/len(rows)}
    result = {"seed": seed, "mode": mode, "write_mode": write_mode, "decisions": decisions, "neural_resets": 0,
        "policy_future": stats(policy[split:]), "policy_last_100": stats(policy[-100:]),
        "future_majority_baseline": max(sum(targets[split:]), len(targets[split:])-sum(targets[split:]))/len(targets[split:]),
        "probes": {name: probe(values, targets) for name, values in stages.items()},
        "original_hand_supervised_probe": hand_probe(stages['hand_z_at_action'], targets, kernel.hand),
        "core_edge_initial_norm": float(kernel.initial_fast[:-1].norm()),
        "core_edge_final_norm": float(state[kernel.state_size:-1].norm()),
        "credit_last_100": credit[-100:],
        "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('acnt/self_write.py', 'acnt/block.py', 'acnt/adapters.py', 'experiments/self_write_input_diagnostic.py')},
        "limitations": ["observer probes use labels; not a training mechanism offered to the subject",
            "linear separability is not proof that the self-write rule can learn the readout",
            "one seed; chronological held-out prefix/suffix; only 1000 simulated decisions",
            "probe features collected before the current action reward enters Core"]}
    folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
    (folder/f"seed_{seed}_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"mode": mode, "policy": result['policy_last_100'],
        "future_probes": {k: v['future_accuracy'] for k,v in result['probes'].items()},
        "hand_probe": result['original_hand_supervised_probe']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=['no_write', 'frozen_write', 'trained_write'], default=['no_write'])
    parser.add_argument('--decisions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--write-mode', choices=['target', 'delta'], default='target')
    parser.add_argument('--output', default='runs/self_write_input_diagnostic')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for mode in args.modes:
        run(args.seed, mode=mode, decisions=args.decisions, write_mode=args.write_mode, output=args.output)


if __name__ == '__main__':
    main()
