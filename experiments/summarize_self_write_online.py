"""Summarize the self-write bootstrap and its unsuccessful cue probes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/self_write_online"
OUT = ROOT / "reports"
STEM = "self_write_bootstrap_2026-10-01"


def load(relative):
    result = json.loads((RUNS / relative).read_text(encoding="utf-8"))
    assert result["neural_resets"] == 0 and len(result["rows"]) == result["decisions"]
    for name, expected in result["source_hashes"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            candidates = [RUNS / "source_snapshot" / Path(name).name,
                ROOT / "runs/self_write_input_diagnostic/source_snapshot" / Path(name).name]
            assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected
                       for p in candidates)
    rows = result.pop("rows")
    result["raw_path"] = "runs/self_write_online/" + relative
    result["write_rate"] = result.get("write_rate", .02)
    result["bins_100"] = [{"end_decision": start + len(rows[start:start+100]),
        "expected_goodness": sum(r["expected_goodness"] for r in rows[start:start+100]) / len(rows[start:start+100])}
        for start in range(0, len(rows), 100)]
    if len(rows) >= 500:
        result["decisions_401_500"] = {
            "expected_goodness": sum(r["expected_goodness"] for r in rows[400:500]) / 100,
            "sample_accuracy": sum(r["goodness"] for r in rows[400:500]) / 100}
    result["score_norm_max"] = max(r["controller_score_norm"] for r in rows)
    return result


def main():
    full = [load(f"constant_full{s}/constant_seed_{s}_full.json") for s in (11, 22, 33)]
    controls = [load(f"constant_controls/constant_seed_11_{mode}.json")
                for mode in ("frozen", "cut_write", "constant_controller")]
    cues = [load(f"{folder}/cue_seed_11_full.json") for folder in ("cue_probe", "cue_fast_write")]
    result = {"constant_full": full, "constant_controls_seed_11": controls, "cue_probes_seed_11": cues,
              "interpretation": "A trainable write mechanism, not a validated general learner. "
              "Constant-task success cannot demonstrate contextual learning or lifelong retention. "
              "Controls compared at 500 decisions; cue probes are one seed."}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{STEM}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    def plot(ax, record, label, stop=None):
        bins = record["bins_100"]
        if stop is not None:
            bins = [r for r in bins if r["end_decision"] <= stop]
        ax.plot([r["end_decision"] for r in bins], [r["expected_goodness"] for r in bins], marker=".", label=label)
    for record in full:
        plot(axes[0], record, f"seed {record['seed']}")
    plot(axes[1], full[0], "Full", 500)
    for record, label in zip(controls, ("Frozen controller", "Cut write credit", "Constant writer")):
        plot(axes[1], record, label)
    for record in cues:
        plot(axes[2], record, f"write rate {record['write_rate']}")
    for ax, title in zip(axes, ("Constant target: 3 seeds", "Constant target: seed 11 controls", "Changing cues: seed 11 probes")):
        ax.set(title=title, xlabel="Continuous online decisions", ylim=(.3, 1.02))
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Expected Goodness (100-decision bins)")
    fig.suptitle("Trainable self-write bootstrap: immediate scalar G, no subject diary", fontsize=13)
    fig.savefig(OUT / f"{STEM}.png", dpi=180)
    fig.savefig(OUT / f"{STEM}.svg")
    plt.close(fig)
    print(json.dumps({"constant_final": [r["last"] for r in full],
                      "controls_500": [(r["mode"], r["last"]) for r in controls],
                      "cue_final": [(r["write_rate"], r["last"]) for r in cues]}, indent=2))


if __name__ == "__main__":
    main()
