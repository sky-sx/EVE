"""Predeclared Stage 0 budgets and deterministic five-stream schedules."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .environment import ACTIONS, DELAYS_MS, environment_config, phase_schedule
from .harness import StageZero

FORMAL_COUNTS = {"initial": 270, "training": 1080, "frozen": 270}
FORMAL_SEEDS = (11, 22, 33, 44, 55)
SCHEMA = 1


def source_hashes():
    paths = ("acnt/plasticity.py", "acnt/block.py", "acnt/core.py",
             "acnt/adapters.py", "acnt/control.py",
             "experiments/stage_zero/environment.py",
             "experiments/stage_zero/harness.py",
             "experiments/stage_zero/runner.py")
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def config(device, counts):
    return {
        "schema": SCHEMA, "device": device, "counts": counts, "seeds": FORMAL_SEEDS,
        "block_count": 10, "neuron_size": 100, "hold_tick": 4, "ticktime_s": .25,
        "active_blocks": list(range(10)), "eye_block": 0, "hand_block": 1,
        "ordinary_blocks": list(range(2,10)), "frames_per_episode": 3,
        "frame_offsets_ms": [0,250,500], "inter_episode_gap_ms": 250,
        "goodness_delays_ms": DELAYS_MS, "goodness": "target active: 1 / active Hand bits; otherwise 0; exact success evaluation only",
        "learning_rate": .001, "retention": .95, "parameter_clip": None,
        "tau": .25, "threshold": 0.,
        "rng_rule": "seed*1000003+phase_index*10007+stream_offset(1..4); phase order initial/training/frozen",
        "source_sha256": source_hashes(), "environment": environment_config(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def aggregate(rows):
    if not rows:
        return {}
    n=len(rows)
    fields=("target_p","non_target_mean_p","target_bit_actual","non_target_false_rate",
            "active_bit_count","exact_event_probability","parameter_norm",
            "parameter_delta_norm")
    target_hits=sum(int(r["target_bit_actual"]) for r in rows)
    result={"episodes":n, "exact_success":sum(int(r["exact_success"]) for r in rows),
            "positive_g_star":sum(r["g_star"]>0 for r in rows)}
    result["exact_success_rate"]=result["exact_success"]/n
    result["target_bit_hit_rate"]=target_hits/n
    result["mean_active_bits"]=sum(r["active_bit_count"] for r in rows)/n
    result["mean_active_bits_given_target_hit"]=(
        sum(r["active_bit_count"] for r in rows if r["target_bit_actual"])/target_hits
        if target_hits else None
    )
    result["mean_g_star"]=sum(r["g_star"] for r in rows)/n
    for key in fields:
        result[key]=sum(r[key] for r in rows)/n
    result["state_total_l2_mean"]=sum(r["plastic_state"]["total"]["l2"] for r in rows)/n
    result["state_total_l2_max"]=max(r["plastic_state"]["total"]["l2"] for r in rows)
    result["nan_count"]=sum(r["nan_count"] for r in rows)
    result["inf_count"]=sum(r["inf_count"] for r in rows)
    return result


def summarize(rows):
    result={"overall":aggregate(rows), "phase":{}, "per_seed":{},
            "per_class":{}, "per_color":{}, "per_delay_ms":{}, "training_windows":{}}
    for phase in ("initial","training","frozen"):
        subset=[r for r in rows if r["phase"]==phase]
        result["phase"][phase]=aggregate(subset)
        result["per_seed"][phase]={str(seed):aggregate([r for r in subset if r["seed"]==seed])
                                   for seed in sorted({r["seed"] for r in subset})}
        result["per_class"][phase]={name:aggregate([r for r in subset if r["target_class"]==name])
                                    for name in ACTIONS}
        result["per_color"][phase]={name:aggregate([r for r in subset if r["color"]==name])
                                    for name in ("red","blue","green")}
        result["per_delay_ms"][phase]={str(d):aggregate([r for r in subset if r["goodness_delay_ms"]==d])
                                       for d in DELAYS_MS}
        if phase=="training":
            result["training_windows"]={str(i):aggregate([r for r in subset
                if (r["episode"]*4)//max(1,len(subset)//max(1,len({r["seed"] for r in subset})))==i])
                for i in range(4)}
    return result


def run_seed(seed, *, device, counts, output, checkpoint_interval=270):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model=StageZero(seed,device)
    initial_parameters={id(p):p.detach().clone() for p in model.plasticity.parameters.values()}
    rows=[]
    start=time.perf_counter()
    clock_ms=0
    raw_path=output/f"seed_{seed}.jsonl"
    with raw_path.open("w",encoding="utf-8") as log:
        for phase_index,(phase,count) in enumerate(counts.items()):
            model.reset_phase(phase=="training")
            schedule,streams=phase_schedule(count,seed,phase_index)
            generator=torch.Generator(device=model.device).manual_seed(streams["hand"])
            for episode,(target,color,delay) in enumerate(schedule):
                row=model.episode(phase=phase,episode=episode,target=target,color=color,
                                  delay_ms=delay,start_ms=clock_ms,generator=generator)
                row["rng_seeds"]=streams
                log.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                rows.append(row)
                clock_ms=row["goodness_time_ms"]+250
                if phase=="training" and checkpoint_interval and (episode+1)%checkpoint_interval==0:
                    model.checkpoint(output/f"seed_{seed}_training_{episode+1}.pt")
            log.flush()
            print(json.dumps({"seed":seed,"phase":phase,"episodes":count,
                              "elapsed_s":time.perf_counter()-start,"metrics":aggregate(rows[-count:])}),
                  flush=True)
    changed=model.changed_groups(initial_parameters)
    result={"seed":seed,"elapsed_s":time.perf_counter()-start,
            "raw_path":str(raw_path),"raw_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "metrics":summarize(rows),"changed_parameter_groups":changed,
            "positive_count":sum(r["g_star"]>0 for r in rows)}
    (output/f"seed_{seed}_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",default="runs/stage_zero_local/potential_goodness")
    parser.add_argument("--seeds",nargs="+",type=int,default=list(FORMAL_SEEDS))
    parser.add_argument("--initial",type=int,default=270)
    parser.add_argument("--training",type=int,default=1080)
    parser.add_argument("--frozen",type=int,default=270)
    parser.add_argument("--checkpoint-interval",type=int,default=270)
    args=parser.parse_args(argv)
    counts={"initial":args.initial,"training":args.training,"frozen":args.frozen}
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cfg=config(args.device,counts)
    (output/"config.json").write_text(json.dumps(cfg,indent=2,allow_nan=False),encoding="utf-8")
    results=[run_seed(seed,device=args.device,counts=counts,output=output,
                      checkpoint_interval=args.checkpoint_interval) for seed in args.seeds]
    (output/"summary.json").write_text(json.dumps({"config":cfg,"seeds":results},indent=2,allow_nan=False),
                                        encoding="utf-8")


if __name__=="__main__":
    main()

