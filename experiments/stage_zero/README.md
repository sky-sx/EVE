# Current Local Plasticity Stage 0

Run from the repository root:

```powershell
python -m pytest -q
python -m experiments.stage_zero --device cpu --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local/smoke_cpu
python -m experiments.stage_zero --device cuda --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local/smoke_cuda
```

## Pilot run (CUDA, reduced scale)

The pilot keeps the Stage 0 network, Goodness, Local Plasticity, stimulus and timing unchanged and only reduces the budget: seeds 11 and 22, 135 initial + 540 training + 135 frozen episodes per seed, `tau_e_s` in `{0.5, 1, 2}`, `tau_G_s = 5`, CUDA device, and `--checkpoint-interval 0` so no full state checkpoints are written.

```powershell
python -m experiments.stage_zero --device cuda --seeds 11 22 --tau-e-s 0.5 1 2 --tau-g-s 5 --initial 135 --training 540 --frozen 135 --checkpoint-interval 0 --output runs/stage_zero_local/tau_scan
```

These are the command defaults, so `python -m experiments.stage_zero` alone runs the same pilot. The run writes `scan_config.json` plus one group directory per (`tau_e_s`, `tau_G_s`) pair; this pilot has three:

| Group directory |
| --- |
| `runs/stage_zero_local/tau_scan/tau_e_0p5_tau_g_5p0` |
| `runs/stage_zero_local/tau_scan/tau_e_1p0_tau_g_5p0` |
| `runs/stage_zero_local/tau_scan/tau_e_2p0_tau_g_5p0` |

## Audit and evaluation per group

Every group directory carries its own `config.json`. `audit.py` and `evaluation.py` read `seeds` and `counts` from that file and derive the expected row count and audit range from the actual configuration, so they are not tied to any fixed scale.

```powershell
python -m experiments.stage_zero.audit --directory runs/stage_zero_local/tau_scan/tau_e_0p5_tau_g_5p0
python -m experiments.stage_zero.evaluation --directory runs/stage_zero_local/tau_scan/tau_e_0p5_tau_g_5p0

python -m experiments.stage_zero.audit --directory runs/stage_zero_local/tau_scan/tau_e_1p0_tau_g_5p0
python -m experiments.stage_zero.evaluation --directory runs/stage_zero_local/tau_scan/tau_e_1p0_tau_g_5p0

python -m experiments.stage_zero.audit --directory runs/stage_zero_local/tau_scan/tau_e_2p0_tau_g_5p0
python -m experiments.stage_zero.evaluation --directory runs/stage_zero_local/tau_scan/tau_e_2p0_tau_g_5p0
```

Both commands default to `audit.json` and `report.md` inside the group directory; pass `--output` to place them elsewhere. The pilot is below the formal scale, so `report.md` lists trend indicators only and applies no SUPPORTED/PARTIAL/NOT SUPPORTED classification: it reports the frozen-minus-initial deltas in exact success, target p and non-target p, and which seeds move in each direction.

Review `DESIGN.md` for exact stimulus, timing and learning boundaries. Raw logs and full checkpoints are local only. The prior `reports/stage_zero_local_plasticity_*.md` files and the earlier formal 5-seed run describe the previous budget and are kept intact. `archive/stage_zero_legacy/` is historical material and is never imported by this package.