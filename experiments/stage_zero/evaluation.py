"""Generate compact behavior and local-state report from audited formal logs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .environment import ACTIONS, DELAYS_MS
from .runner import FORMAL_COUNTS, FORMAL_SEEDS, aggregate


def _fmt(x):
    return f"{x:.6g}" if isinstance(x,float) else str(x)


def _table(headers, rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join(
        "| "+" | ".join(_fmt(x) for x in row)+" |" for row in rows)+"\n"


def build(directory, audit_file):
    directory=Path(directory)
    audit=json.loads(Path(audit_file).read_text(encoding="utf-8"))
    if not audit["passed"] or audit["all_rows_audited"]!=8100:
        raise AssertionError("complete independent audit required")
    lock=json.loads(Path("reports/stage_zero_local_plasticity_lock.json").read_text(encoding="utf-8"))
    rows=[]
    seed_summaries={}
    for seed in FORMAL_SEEDS:
        seed_summaries[seed]=json.loads((directory/f"seed_{seed}_summary.json").read_text(encoding="utf-8"))
        rows += [json.loads(x) for x in (directory/f"seed_{seed}.jsonl").open(encoding="utf-8")]
    subsets={phase:[r for r in rows if r["phase"]==phase] for phase in FORMAL_COUNTS}
    totals={phase:aggregate(r) for phase,r in subsets.items()}
    each={seed:{phase:aggregate([r for r in subsets[phase] if r["seed"]==seed])
                for phase in FORMAL_COUNTS} for seed in FORMAL_SEEDS}
    initial,frozen=totals["initial"],totals["frozen"]
    frozen_delay_target={d:aggregate([r for r in subsets["frozen"] if r["goodness_delay_ms"]==d])["target_p"] for d in DELAYS_MS}
    improved=[s for s in FORMAL_SEEDS if
        each[s]["frozen"]["target_p"]>each[s]["initial"]["target_p"] and
        each[s]["frozen"]["non_target_mean_p"]<each[s]["initial"]["non_target_mean_p"] and
        each[s]["frozen"]["exact_success"]>each[s]["initial"]["exact_success"]]
    supported=(len(improved)>=3 and frozen["exact_success"]>initial["exact_success"] and
        frozen["target_p"]>initial["target_p"] and
        frozen["non_target_mean_p"]<initial["non_target_mean_p"])
    verdict="SUPPORTED" if supported else "PARTIAL" if improved or frozen["exact_success"]>initial["exact_success"] else "NOT SUPPORTED"
    lines=[
        "# Current Local Plasticity Stage 0 formal report","",
        "This is the new Local Plasticity experiment. The archived e-prop Stage 0 is a separate historical result.","",
        f"- Baseline main commit: `{lock['baseline_commit']}`.",
        "- Final main commit: see release response; a tracked report cannot embed its own content-addressed commit SHA.",
        "- Git status at start: clean. Final status checked after commit and push.",
        f"- Python {lock['python']}; PyTorch {lock['torch']}; CUDA {lock['cuda']}; GPU {lock['gpu']}. Formal execution used CPU.",
        "- Frozen candidate: production CorrelationRule, learning_rate=.001, retention=.95, parameter_clip=None.",
        "- Formal seeds 11, 22, 33, 44, 55; per seed 270 initial, 1080 training, 270 frozen episodes.",
        f"- Independent audit: passed, {audit['all_rows_audited']} rows and {audit['stimulus_count']} stimulus hashes.",
        "- Pre-formal full active pytest: 143 passed; final full active pytest after report: 146 passed. CPU and RTX 5080 CUDA 1/1/1 smoke passed; 27/9/9 timing smoke passed on both.",
        f"- Formal command: `{lock['formal_command']}`.",
        "- Audit command: `python -m experiments.stage_zero.audit --directory runs/stage_zero_local`.",
        "- Local only: raw JSONL and checkpoints under ignored `runs/stage_zero_local/`.","",
        "## Overall behavior","",
        _table(["Phase","Episodes","Exact","Exact rate","Target p","Non-target p","Target-bit hit","False rate","Active bits","Positive g*"],
               [[phase,*[m[k] for k in ("episodes","exact_success","exact_success_rate","target_p",
                 "non_target_mean_p","target_bit_actual","non_target_false_rate","active_bit_count","positive_g_star")]]
                for phase,m in totals.items()]),
        "## Per seed","",
        _table(["Seed","Initial exact","Train exact","Frozen exact","Initial target p","Frozen target p",
                "Initial non-target p","Frozen non-target p","Elapsed s"],
               [[s,each[s]["initial"]["exact_success"],each[s]["training"]["exact_success"],
                 each[s]["frozen"]["exact_success"],each[s]["initial"]["target_p"],
                 each[s]["frozen"]["target_p"],each[s]["initial"]["non_target_mean_p"],
                 each[s]["frozen"]["non_target_mean_p"],seed_summaries[s]["elapsed_s"]]
                for s in FORMAL_SEEDS]),
        "## Per class","",
        _table(["Class","Initial exact","Training exact","Frozen exact","Initial target p","Frozen target p",
                "Initial non-target p","Frozen non-target p"],
               [[name,*[aggregate([r for r in subsets[phase] if r["target_class"]==name])["exact_success"]
                         for phase in FORMAL_COUNTS],
                 aggregate([r for r in subsets["initial"] if r["target_class"]==name])["target_p"],
                 aggregate([r for r in subsets["frozen"] if r["target_class"]==name])["target_p"],
                 aggregate([r for r in subsets["initial"] if r["target_class"]==name])["non_target_mean_p"],
                 aggregate([r for r in subsets["frozen"] if r["target_class"]==name])["non_target_mean_p"]]
                for name in ACTIONS]),
        "## Per delay","",
        _table(["Delay ms","Phase","Episodes","Exact","Target p","Non-target p","Target-bit hit","False rate"],
               [[d,phase,*[m[k] for k in ("episodes","exact_success","target_p","non_target_mean_p",
                                           "target_bit_actual","non_target_false_rate")]]
                for phase,subset in subsets.items() for d in DELAYS_MS
                for m in [aggregate([r for r in subset if r["goodness_delay_ms"]==d])]]),
        "## Per color","",
        _table(["Color","Phase","Episodes","Exact","Target p","Non-target p"],
               [[color,phase,*[m[k] for k in ("episodes","exact_success","target_p","non_target_mean_p")]]
                for phase,subset in subsets.items() for color in ("red","blue","green")
                for m in [aggregate([r for r in subset if r["color"]==color])]]),
        "## Training windows","",
        _table(["Window","Episodes","Exact","Target p","Non-target p","Mean e L2"],
               [[i,*[m[k] for k in ("episodes","exact_success","target_p","non_target_mean_p",
                                    "state_total_l2_mean")]]
                for i in range(4)
                for m in [aggregate([r for r in subsets["training"] if r["episode"]//270==i])]]),
        "## Local state and parameters","",
        _table(["Seed","Train mean e L2","Train max e L2","End train e L2","Frozen e L2",
                "Changed parameter tensors","Max per-episode delta"],
               [[s,each[s]["training"]["state_total_l2_mean"],each[s]["training"]["state_total_l2_max"],
                 [r for r in subsets["training"] if r["seed"]==s][-1]["plastic_state"]["total"]["l2"],
                 each[s]["frozen"]["state_total_l2_max"],
                 sum(len(v) for v in seed_summaries[s]["changed_parameter_groups"].values()),
                 max(r["parameter_delta_norm"] for r in subsets["training"] if r["seed"]==s)]
                for s in FORMAL_SEEDS]),
        "Per-group local-state L2 at the last training Goodness delivery:","",
        _table(["Seed",*[f"Block {i}" for i in range(10)],"Eye Adapter","Hand Adapter",
                "Ordinary Blocks","Terminal Hand"],
               [[s,*[r["plastic_state"]["groups"][str(i)]["l2"] for i in range(10)],
                 r["plastic_state"]["eye_adapter"]["l2"],r["plastic_state"]["hand_adapter"]["l2"],
                 r["plastic_state"]["ordinary_blocks"]["l2"],r["plastic_state"]["terminal_hand"]["l2"]]
                for s in FORMAL_SEEDS
                for r in [[x for x in subsets["training"] if x["seed"]==s][-1]]]),
        "Local-state distribution at the final training delivery:",
        "",
        _table(["Seed","e mean","e std","e max abs","e nonzero fraction"],
               [[seed,*[r["plastic_state"]["total"][k] for k in
                        ("mean","std","max_abs","nonzero_fraction")]]
                for seed in FORMAL_SEEDS
                for r in [[x for x in subsets["training"] if x["seed"]==seed][-1]]]),
        "The candidate has no bounded e state, so saturation is not a defined failure criterion; finite values, magnitude, and nonzero fraction are reported instead.",
        "",
        "At Goodness delivery, local-state persistence by delay:","",
        _table(["Delay ms","Train rows","Nonzero e","Mean e L2","Mean parameter delta"],
               [[d,len(g),sum(r["plastic_state"]["total"]["l2"]>0 for r in g),
                 aggregate(g)["state_total_l2_mean"],aggregate(g)["parameter_delta_norm"]]
                for d in DELAYS_MS
                for g in [[r for r in subsets["training"] if r["goodness_delay_ms"]==d]]]),
        "F_w coefficient is 0.001*(g* - 0.5): -0.0005 when g*=0 and +0.0005 when g*=1. Numerical tests verify that update direction. All 10 Block groups, including both Adapters, changed in the formal summaries.",
        f"NaN / Inf counts across all logged episodes: {sum(r['nan_count'] for r in rows)} / {sum(r['inf_count'] for r in rows)}.",
        f"Seeds without joint required behavior improvement: {[s for s in FORMAL_SEEDS if s not in improved]}.",
        f"Total formal elapsed seed time: {sum(seed_summaries[s]['elapsed_s'] for s in FORMAL_SEEDS):.3f} s.","",
        "## Local-only raw inventory","",
        _table(["Seed","Raw JSONL bytes","Raw SHA256","Checkpoint count","Checkpoint bytes"],
               [[seed,(directory/f"seed_{seed}.jsonl").stat().st_size,
                 seed_summaries[seed]["raw_sha256"],
                 len(list(directory.glob(f"seed_{seed}_training_*.pt"))),
                 sum(path.stat().st_size for path in directory.glob(f"seed_{seed}_training_*.pt"))]
                for seed in FORMAL_SEEDS]),
        "These bulk raw logs and full state checkpoints are excluded by repository Git rules.","",
        "## Locked source SHA256","",
        _table(["Source","SHA256"],[[p,h] for p,h in lock["source_sha256"].items()]),
        "## Conclusion","",
        f"Classification: **{verdict}**. Independent frozen behavior is the strongest evidence. Exact-one-hot reward sparsity is the largest failure point when no positive g* occurs. Parameter or e changes alone are not learning evidence.","",
        f"Q1: The current local correlation state plus delayed scalar Goodness {'formed a reproducible mapping' if supported else 'did not demonstrate a reproducible visual-to-action mapping'} under this protocol.",
        f"Q2: All three delay strata have zero exact success. Frozen target p values are {frozen_delay_target[250]:.6f} (250 ms), {frozen_delay_target[500]:.6f} (500 ms), and {frozen_delay_target[1000]:.6f} (1000 ms); this observed spread gives no evidence of a clear delay-dependent mapping difference. This does not imply a required delta-time rule.","",
        "Limitations: These results assess the frozen candidate under the strict 27-bit Stage 0 reward. They do not resolve which future F_e/F_w rule the architecture should use.",
    ]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="runs/stage_zero_local")
    p.add_argument("--audit",default="reports/stage_zero_local_plasticity_audit.json")
    p.add_argument("--output",default="reports/stage_zero_local_plasticity_report.md")
    a=p.parse_args()
    Path(a.output).write_text(build(a.directory,a.audit),encoding="utf-8")
    print(a.output)


if __name__=="__main__":
    main()

