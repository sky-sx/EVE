# Current Local Plasticity Stage 0

Run from the repository root:

```powershell
python -m pytest -q
python -m experiments.stage_zero --device cpu --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local/smoke_cpu
python -m experiments.stage_zero --device cuda --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local/smoke_cuda
python -m experiments.stage_zero --device cpu --seeds 11 22 33 44 55 --initial 270 --training 1080 --frozen 270 --checkpoint-interval 270 --output runs/stage_zero_local/potential_goodness
python -m experiments.stage_zero.audit --directory runs/stage_zero_local/potential_goodness
```

The formal run writes its config and source hashes to `runs/stage_zero_local/potential_goodness/config.json`. Review `DESIGN.md` for exact stimulus, timing and learning boundaries. Raw logs and full checkpoints are local only. The prior `reports/stage_zero_local_plasticity_*.md` files and `runs/stage_zero_local/seed_*.jsonl` describe the earlier strict-reward run; this command keeps them intact. `archive/stage_zero_legacy/` is historical material and is never imported by this package.

