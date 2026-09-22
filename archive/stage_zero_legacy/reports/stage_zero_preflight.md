# Stage Zero preflight, 2026-09-20

- Clean starting main and GitHub-resolved remote main:
  `be9c1cf604b5e3e66d57774cd1ab1704df68fdd5`.
- First required `pytest -q`: 151 passed, 2 errors caused by inaccessible
  default Windows pytest temporary directory. A fresh local basetemp and cache
  directory gave **153 passed**; no production fix was needed.
- After Stage Zero implementation: **201 passed, 0 skipped, 0 failed**
  (`pytest -q --basetemp=.test-tmp/stage-zero-final-20260920
  -o cache_dir=.test-tmp/stage-zero-pytest-cache`). CUDA test ran, not skipped.
- Production `acnt/*.py` unchanged. Real 10x100 integration tests verify Eye
  shape, distinct bindings, 27/0 Hand, all-active set, forbidden ReadOuts,
  no physical input, current Logistic sample, zero-frame forced pulse and its
  consumption, shared old snapshot, fixed feedback VJP, no supervised loss,
  250 ms delayed exp(-1) decay, exact parameter proposal, baseline, and frozen
  weight immutability. Positive scalar injection is an isolated wiring test.

## Completed short runs

Each used seed 11, 27 initial +27 training +27 frozen episodes, strict reward.
All rewards were zero; all parameter/eligibility checks finite; initial/frozen
parameters unchanged and fixed feedback unchanged.

| Device | Initial seconds | Training seconds | Frozen seconds |
|---|---:|---:|---:|
| CPU, one thread | 3.875 | 5.238 | 3.792 |
| CUDA RTX 5080 | 2.742 | 4.670 | 2.458 |

Raw data: `runs/stage_zero/smoke-cpu-20260920/` and
`runs/stage_zero/smoke-cuda-20260920/`. CPU and CUDA have distinct action RNG
streams; identical seed values do not promise identical actions across devices.

## Fixed formal budget (before formal training)

Use CUDA FP32; seeds 11,22,33,44,55; 1,080 training episodes per seed;
270 initial and 270 frozen evaluation episodes per seed; window=270.
This gives 5,400 training actions and 2,700 no-update evaluation actions.
Budget selected from wall-clock smoke throughput, about 20 minutes total,
without changing any learning hyperparameter, reward, architecture or seed.
Every class appears 40 times in training and 10 times in each evaluation.
No early stopping or selection based on reward. Preserve all seeds.

Command:

```powershell
python -m experiments.stage_zero --output runs/stage_zero/formal-20260920 --device cuda --seeds 11 22 33 44 55 --train-episodes 1080 --evaluation-episodes 270 --window 270
```
