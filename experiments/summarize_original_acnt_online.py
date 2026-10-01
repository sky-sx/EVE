"""Render the completed 2026-10-01 experiments without rerunning training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/original_acnt_online"
OUT = ROOT / "reports"
STEM = "original_acnt_online_2026-10-01"


def load(relative):
    path = RUNS / relative
    result = json.loads(path.read_text(encoding="utf-8"))
    assert len(result["queries"]) == result["decisions"]
    assert result["neural_resets"] == 0
    return result


def compact(result, relative):
    record = {k: v for k, v in result.items()
              if k not in ("queries", "feedback", "parameter_changes")}
    record["raw_path"] = "runs/original_acnt_online/" + relative
    record["bins_100"] = []
    for start in range(0, result["decisions"], 100):
        rows = result["queries"][start:start + 100]
        record["bins_100"].append({
            "end_decision": start + len(rows),
            "expected_goodness": sum(r["expected_goodness"] for r in rows) / len(rows),
            "sample_accuracy": sum(r["correct"] for r in rows) / len(rows),
        })
    record["changed_parameter_groups"] = {
        group: sum(v * v for name, v in result["parameter_changes"].items()
                   if name.startswith(prefix)) ** .5
        for group, prefix in (("core", "core."), ("ear", "adapters.ear."),
                              ("hand", "adapters.hand."))
    }
    record["intervening_update_range"] = [
        min(r["intervening_updates"] for r in result["feedback"]),
        max(r["intervening_updates"] for r in result["feedback"]),
    ]
    return record


def main():
    OUT.mkdir(exist_ok=True)
    records = {}
    for family, dirname in (("fixed", "fixed_final"), ("anonymous", "full_2000")):
        records[family] = []
        for seed in (11, 22, 33):
            relative = f"{dirname}_seed_{seed}/seed_{seed}_full.json"
            records[family].append(compact(load(relative), relative))
    control_paths = {
        "Matched ledger": "fixed_delay_seed_11/seed_11_full.json",
        "Cut state credit": "fixed_cut_state/seed_11_cut_state.json",
        "Freeze Core": "fixed_freeze_core/seed_11_freeze_core.json",
        "Anonymous traces": "fixed_world_trace_controls/seed_11_full.json",
        "Cut delay credit": "fixed_world_trace_controls/seed_11_cut_delay.json",
    }
    records["controls_seed_11_at_1000"] = {
        label: compact(load(relative), relative) for label, relative in control_paths.items()
    }
    current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in records["fixed"][0]["source_hashes"]}
    assert all(r["source_hashes"] == current_hashes for r in records["fixed"])
    records["verified_current_source_hashes"] = current_hashes
    records["interpretation"] = (
        "Online conditional expected Goodness, not a separate evaluation trajectory. "
        "Controls have one seed and 1000 decisions; fixed and anonymous main runs "
        "have three seeds and 2000 decisions. Known fixed delay is stronger information. "
        "Earlier fixed runs are prefixes, not independent replications."
    )
    (OUT / f"{STEM}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, family, title in zip(axes[:2], ("fixed", "anonymous"),
                                 ("Known fixed-delay ledger", "Anonymous variable-delay traces")):
        for record in records[family]:
            bins = record["bins_100"]
            ax.plot([b["end_decision"] for b in bins],
                    [b["expected_goodness"] for b in bins],
                    marker=".", label=f"seed {record['seed']}")
        ax.set(title=title, xlabel="Continuous online decisions", ylim=(.35, 1.02))
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Online expected Goodness (100-decision bins)")
    controls = records["controls_seed_11_at_1000"]
    values = [r["final_segment"]["expected_goodness"] for r in controls.values()]
    axes[2].barh(list(controls), values, color=["#1b9e77"] + ["#929da7"] * 4)
    axes[2].invert_yaxis()
    axes[2].set(title="Same fixed world: seed 11, 1000 decisions", xlim=(0, 1.12),
                xlabel="Expected Goodness, final 100 decisions")
    axes[2].axvline(.5, color="gray", linestyle="--", linewidth=1)
    for i, value in enumerate(values):
        axes[2].text(value + .015, i, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("Original ACNT forward: persistent online credit, no neural resets", fontsize=13)
    fig.savefig(OUT / f"{STEM}.png", dpi=180)
    fig.savefig(OUT / f"{STEM}.svg")
    plt.close(fig)
    print(json.dumps({family: [r["final_segment"] for r in records[family]]
                      for family in ("fixed", "anonymous")}, indent=2))


if __name__ == "__main__":
    main()
