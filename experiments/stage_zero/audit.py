"""Independent raw-log audit; reconstruct behavior without trusting summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .environment import ACTIONS, DELAYS_MS, phase_schedule, render, stimulus_hash
from .runner import FORMAL_COUNTS, FORMAL_SEEDS, aggregate, source_hashes


def audit_goodness(target, bits):
    """Recompute both quantities from raw bits, independently of environment."""
    active=sum(bits)
    g=1.0/active if bits[target] else 0.0
    exact=bool(bits[target]) and active==1
    return g, exact


def audit(directory, config_path=None):
    directory=Path(directory)
    config_path=directory/"config.json" if config_path is None else Path(config_path)
    lock=json.loads(config_path.read_text(encoding="utf-8"))
    if lock["source_sha256"]!=source_hashes():
        raise AssertionError("production or experiment sources differ from locked SHA256")
    report={"schema":1,"config_path":str(config_path),"source_sha256_verified":True,
            "seed_results":{},"failures":[]}
    expected_hash={}
    for seed in FORMAL_SEEDS:
        raw_path=directory/f"seed_{seed}.jsonl"
        rows=[json.loads(line) for line in raw_path.open(encoding="utf-8")]
        expected_n=sum(FORMAL_COUNTS.values())
        if len(rows)!=expected_n:
            raise AssertionError(f"seed {seed}: {len(rows)} rows != {expected_n}")
        summary=json.loads((directory/f"seed_{seed}_summary.json").read_text(encoding="utf-8"))
        if summary["raw_sha256"]!=hashlib.sha256(raw_path.read_bytes()).hexdigest():
            raise AssertionError(f"seed {seed}: raw SHA mismatch")
        start=0
        phase_result={}
        for phase_index,(phase,count) in enumerate(FORMAL_COUNTS.items()):
            segment=rows[start:start+count]
            schedule,streams=phase_schedule(count,seed,phase_index)
            if len(Counter(r["target_index"] for r in segment))!=27:
                raise AssertionError("missing target class")
            counts=Counter(r["target_index"] for r in segment)
            if max(counts.values())-min(counts.values())>1:
                raise AssertionError("unbalanced targets")
            dc=Counter(r["goodness_delay_ms"] for r in segment)
            if max(dc.values())-min(dc.values())>1:
                raise AssertionError("unbalanced delays")
            for i,(row,(target,color,delay)) in enumerate(zip(segment,schedule)):
                if (row["seed"],row["phase"],row["episode"])!=(seed,phase,i):
                    raise AssertionError("seed/phase/index mismatch")
                if (row["target_index"],row["color"],row["goodness_delay_ms"])!=(target,color,delay):
                    raise AssertionError("RNG schedule mismatch")
                if row["rng_seeds"]!=streams:
                    raise AssertionError("RNG streams mismatch")
                key=(target,color)
                if key not in expected_hash:
                    expected_hash[key]=stimulus_hash(render(target,color))
                if row["stimulus_hash"]!=expected_hash[key]:
                    raise AssertionError("stimulus SHA mismatch")
                if row["target_class"]!=ACTIONS[target] or row["target_action"]!=ACTIONS[target]:
                    raise AssertionError("target/action map mismatch")
                q,p,noise,threshold,bits=[row[k] for k in ("q","p","noise","threshold","action_bits")]
                if not all(len(x)==27 for x in (q,p,noise,threshold,bits)):
                    raise AssertionError("control length mismatch")
                if row["sampled_actions"]!=[ACTIONS[j] for j,b in enumerate(bits) if b]:
                    raise AssertionError("sampled action names mismatch")
                for j in range(27):
                    if bits[j]!=(q[j]+noise[j]>threshold[j]):
                        raise AssertionError("terminal action mismatch")
                    pred=1/(1+math.exp(-(q[j]-threshold[j])/.25))
                    if not math.isclose(p[j],pred,rel_tol=2e-6,abs_tol=2e-6):
                        raise AssertionError("diagnostic p mismatch")
                active_bits=sum(bits)
                g,exact=audit_goodness(target,bits)
                if not math.isclose(row["g_star"],g,rel_tol=1e-12,abs_tol=1e-12):
                    raise AssertionError("potential Goodness mismatch")
                if row["exact_success"]!=exact:
                    raise AssertionError("exact success mismatch")
                if row["target_bit_actual"]!=bool(bits[target]) or row["active_bit_count"]!=active_bits:
                    raise AssertionError("action diagnostics mismatch")
                if row["frame_times_ms"]!=[row["frame_times_ms"][0]+j*250 for j in range(3)]:
                    raise AssertionError("frame cadence mismatch")
                if row["action_time_ms"]!=row["frame_times_ms"][-1]:
                    raise AssertionError("action time mismatch")
                if row["goodness_time_ms"]!=row["action_time_ms"]+delay:
                    raise AssertionError("Goodness delivery mismatch")
                if i>0 and row["frame_times_ms"][0]!=segment[i-1]["goodness_time_ms"]+250:
                    raise AssertionError("inter-episode gap mismatch")
                if phase in ("initial","frozen") and (row["parameter_delta_norm"]!=0 or
                                                     row["plastic_state"]["total"]["l2"]!=0):
                    raise AssertionError("evaluation updated parameters or e")
                if row["nan_count"] or row["inf_count"]:
                    raise AssertionError("nonfinite model state")
                if not math.isfinite(row["plastic_state"]["total"]["l2"]):
                    raise AssertionError("nonfinite local plastic norm")
                exact_p=p[target]
                for j in range(27):
                    if j!=target:
                        exact_p*=1-p[j]
                if not math.isclose(exact_p,row["exact_event_probability"],rel_tol=1e-5,abs_tol=1e-15):
                    raise AssertionError("exact event probability mismatch")
            phase_result[phase]=aggregate(segment)
            saved=summary["metrics"]["phase"][phase]
            for metric in ("exact_success","target_bit_hit_rate","mean_g_star",
                           "mean_active_bits","mean_active_bits_given_target_hit"):
                observed=phase_result[phase][metric]
                recorded=saved[metric]
                if observed is None or recorded is None:
                    if observed is not recorded:
                        raise AssertionError(f"summary {metric} mismatch")
                elif not math.isclose(observed,recorded,rel_tol=1e-12,abs_tol=1e-12):
                    raise AssertionError(f"summary {metric} mismatch")
            start+=count
        report["seed_results"][str(seed)]={"rows":len(rows),"sha256":summary["raw_sha256"],
            "phase":phase_result,"changed_parameter_groups":summary["changed_parameter_groups"]}
    report["stimulus_count"]=len(expected_hash)
    report["stimulus_sha256"]={f"{ACTIONS[t]}_{c}":h for (t,c),h in expected_hash.items()}
    report["all_rows_audited"]=sum(v["rows"] for v in report["seed_results"].values())
    report["passed"]=not report["failures"]
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default="runs/stage_zero_local/potential_goodness")
    parser.add_argument("--lock",default=None)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    report=audit(args.directory,args.lock)
    output=Path(args.output) if args.output is not None else Path(args.directory)/"audit.json"
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"all_rows_audited":report["all_rows_audited"],
                      "stimulus_count":report["stimulus_count"]}))


if __name__=="__main__":
    main()

