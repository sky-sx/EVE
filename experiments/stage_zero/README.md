# Current Local Plasticity Stage 0

Run from the repository root:

```powershell
python -m pytest -q
python -m experiments.stage_zero --device cpu --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local_smoke_cpu
python -m experiments.stage_zero --device cuda --seeds 11 --initial 1 --training 1 --frozen 1 --checkpoint-interval 0 --output runs/stage_zero_local_smoke_cuda
python -m experiments.stage_zero --device cpu --seeds 11 22 33 44 55 --initial 270 --training 1080 --frozen 270 --checkpoint-interval 270 --output runs/stage_zero_local
python -m experiments.stage_zero.audit --directory runs/stage_zero_local
```

The formal run uses the predeclared `reports/stage_zero_local_plasticity_lock.json`. Review `DESIGN.md` for exact stimulus, timing and learning boundaries. Raw logs and full checkpoints are local only. `archive/stage_zero_legacy/` is historical material and is never imported by this package.

