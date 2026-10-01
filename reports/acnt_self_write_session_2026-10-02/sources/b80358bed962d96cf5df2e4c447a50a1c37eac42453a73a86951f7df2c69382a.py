"""A single uninterrupted synthetic lifetime for original ACNT online credit.

The world generates cues and judges actions. The learner sees only Ear input,
its sampled action and later one scalar Goodness, never labels/action IDs.
No neural reset, history recomputation, replay or episode batches occur.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from acnt import Block, Core, Runtime
from acnt.adapters import EyeAdapter, EarAdapter, HandAdapter, SpeakAdapter, GoodnessAdapter, RouteAdapter
from acnt.runtime import ORGANS
from acnt.original_online import OriginalOnlineRuntime, DelayedOnlineTrainer, FixedDelayLedgerTrainer


def build(seed, *, mode="full", width=4, hold=4, lr=.001, gap=6, credit_mode="trace", delay_periods=3):
    torch.manual_seed(seed)
    sizes = [width] * 6
    core = Core([Block(i, width, sizes, readin=i < 2, ticktime=1,
                       hold_tick=hold, nlm_hidden_dim=2) for i in range(6)])
    adapters = {"eye": EyeAdapter(width), "ear": EarAdapter(width, window_samples=4),
        "hand": HandAdapter(width, hidden_size=8), "speak": SpeakAdapter(width),
        "goodness": GoodnessAdapter(width), "route": RouteAdapter(width, 6)}
    runtime = Runtime(core, adapters, dict(zip(ORGANS, range(6))))
    runtime.execution_enabled["route"] = False
    model = OriginalOnlineRuntime(runtime, train_core=mode != "freeze_core",
                                  cut_state_credit=mode == "cut_state", sensitivity_limit=1000.)
    if credit_mode == "fixed_delay":
        trainer = FixedDelayLedgerTrainer(model, feedback_delay_ms=((gap + 2) * delay_periods + 2) * 2,
                                         lr=lr)
    else:
        trainer = DelayedOnlineTrainer(model, lr=lr, cut_delay_credit=mode == "cut_delay")
    return model, trainer


def run(seed=11, *, mode="full", decisions=400, lr=.001, width=4, hold=4,
        gap=6, output="runs/original_acnt_online/pilot", log_every=100,
        credit_mode="trace", delay_periods=3, feedback_protocol=None):
    if decisions < 1 or gap < 2:
        raise ValueError("positive decision budget and gap >=2 required")
    if credit_mode == "fixed_delay" and mode == "cut_delay":
        raise ValueError("cut_delay is a trace control, not a fixed-delay ledger mode")
    feedback_protocol = feedback_protocol or ("fixed" if credit_mode == "fixed_delay" else "variable")
    if credit_mode == "fixed_delay" and feedback_protocol != "fixed":
        raise ValueError("known-delay ledger requires the matching fixed-delay world")
    model, trainer = build(seed, mode=mode, width=width, hold=hold, lr=lr, gap=gap,
                          credit_mode=credit_mode, delay_periods=delay_periods)
    env_rng = random.Random(seed + 70000)
    action_rng = torch.Generator().manual_seed(seed + 10000)
    sources = ["acnt/original_online.py", "acnt/block.py", "acnt/adapters.py", __file__]
    source_hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
                     hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    initial = {n: p.detach().clone() for n, p in model.named_parameters}
    period = gap + 2
    pending = []
    queries = []
    feedback = []
    started = time.perf_counter()
    bit = 0
    memory_bytes = trainer.learning_state_bytes
    for event in range(decisions * period):
        now = event * 2
        phase = event % period
        readins = None
        if phase == 0:
            bit = env_rng.randrange(2)
            amplitude = env_rng.uniform(.5, 1.)
            sign = 2 * bit - 1
            readins = {"ear": model.state.new_tensor([[sign * amplitude,
                -sign * amplitude, .5 * sign * amplitude, -.5 * sign * amplitude]])}
        trainer.advance(now)
        model.step(now_ms=now, readins=readins)
        if phase == gap:
            sample = model.sample_discrete(generator=action_rng)
            if mode != "frozen":
                trainer.record_action(sample, now_ms=now)
            p = float(sample.probability[0])
            action = int(sample.action[0])
            # Consume the same RNG draw under either delivery protocol.
            drawn_periods = env_rng.choice((1, 2, 3))
            delay = period * (delay_periods if feedback_protocol == "fixed" else drawn_periods) + 2
            pending.append({"due": event + delay, "goodness": float(action == bit),
                            "action_event": event, "updates_at_action": trainer.updates})
            queries.append({"event": event, "time_ms": now, "bit": bit,
                "probability_one": p, "action": action, "correct": action == bit,
                "expected_goodness": p if bit else 1 - p,
                "diagnostic_correct": (p >= .5) == bool(bit), "delay_events": delay,
                "updates_at_action": trainer.updates})
            if len(queries) % log_every == 0:
                block = queries[-log_every:]
                print(json.dumps({"seed": seed, "mode": mode, "decisions": len(queries),
                    "expected_goodness": sum(q["expected_goodness"] for q in block) / len(block),
                    "sample_accuracy": sum(q["correct"] for q in block) / len(block),
                    "sensitivity_norm": float(model.sensitivity.norm()),
                    "seconds": time.perf_counter() - started}), flush=True)
        due = [r for r in pending if r["due"] == event]
        pending = [r for r in pending if r["due"] != event]
        for reward in due:
            crossed = trainer.updates - reward["updates_at_action"]
            stat = trainer.observe_goodness(reward["goodness"], now_ms=now) if mode != "frozen" else {}
            feedback.append({"event": event, "goodness": reward["goodness"],
                "delay_events": event - reward["action_event"], "intervening_updates": crossed, **stat})
        assert trainer.learning_state_bytes == memory_bytes
        assert model.state.grad_fn is None and model.sensitivity.grad_fn is None
        assert all(s.z.grad_fn is None and all(h.grad_fn is None for h in s.history) for s in model.states)
    changes = {n: float((p.detach() - initial[n]).norm()) for n, p in model.named_parameters}
    def stats(part):
        return {"count": len(part), "expected_goodness": sum(q["expected_goodness"] for q in part) / len(part),
            "sample_accuracy": sum(q["correct"] for q in part) / len(part),
            "diagnostic_accuracy": sum(q["diagnostic_correct"] for q in part) / len(part)}
    chunk = max(1, min(100, decisions // 4))
    result = {"seed": seed, "mode": mode, "credit_mode": credit_mode,
        "feedback_protocol": feedback_protocol,
        "delay_periods": delay_periods if feedback_protocol == "fixed" else None,
        "decisions": decisions, "events": model.events,
        "gap_events": gap, "hold_tick": hold, "width": width, "lr": lr,
        "seconds": time.perf_counter() - started, "initial_segment": stats(queries[:chunk]),
        "final_segment": stats(queries[-chunk:]), "all": stats(queries),
        "updates": trainer.updates, "feedback_crossing_updates": sum(r["intervening_updates"] > 0 for r in feedback),
        "neural_resets": 0, "learning_state_bytes": memory_bytes,
        "state_dim": model.state_dim, "param_dim": model.param_dim,
        "sensitivity_clips": model.sensitivity_clips, "max_sensitivity_norm": model.max_sensitivity_norm,
        "score_clips": trainer.score_clips, "trace_clips": trainer.trace_clips,
        "parameter_path_length": trainer.total_parameter_path,
        "parameter_changes": changes, "queries": queries, "feedback": feedback,
        "pending_at_stop": len(pending),
        "source_hashes": source_hashes,
        "limitations": ["synthetic exogenous cue stream; no real mechanical action", "one Hand binary coordinate only",
            "dense sensitivity reference, not scalable production kernels", "stop-update tangent, not full meta-gradient",
            "no lifelong no-forgetting or irreversible-world safety guarantee", "test lifetime stopped without draining pending feedback"]}
    target = Path(output) / f"seed_{seed}_{mode}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"seed": seed, "mode": mode, "final": result["final_segment"],
        "updates": trainer.updates, "feedback_crossing_updates": result["feedback_crossing_updates"],
        "seconds": result["seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--modes", nargs="+", choices=["full", "cut_state", "cut_delay", "freeze_core", "frozen"], default=["full"])
    parser.add_argument("--decisions", type=int, default=400)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--hold", type=int, default=4)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--credit-mode", choices=["trace", "fixed_delay"], default="trace")
    parser.add_argument("--delay-periods", type=int, default=3)
    parser.add_argument("--feedback-protocol", choices=["variable", "fixed"])
    parser.add_argument("--output", default="runs/original_acnt_online/pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [run(seed, mode=mode, decisions=args.decisions, lr=args.lr, width=args.width,
        hold=args.hold, gap=args.gap, output=args.output, credit_mode=args.credit_mode,
        delay_periods=args.delay_periods, feedback_protocol=args.feedback_protocol)
        for mode in args.modes for seed in args.seeds]
    compact = [{k: r[k] for k in ("seed", "mode", "initial_segment", "final_segment", "all", "updates",
        "feedback_crossing_updates", "learning_state_bytes", "seconds", "sensitivity_clips")} for r in results]
    (Path(args.output) / "summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
