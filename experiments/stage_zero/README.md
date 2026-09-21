# ACNT Stage Zero

Run from the repository root (Python 3.11+, existing torch/pytest dependencies):

```powershell
python -m experiments.stage_zero --output runs/stage_zero/example --device auto --seeds 11 22 33 44 55 --train-episodes 2700 --evaluation-episodes 270
```

The output directory must not already exist. Every phase count must be a
multiple of 27. For a short full-protocol smoke use `--seeds 11
--train-episodes 27 --evaluation-episodes 27 --window 27`. `--device cpu`
and `--device cuda` select the execution device explicitly. There is no real
keyboard/mouse execution. The CLI changes no official Runtime behavior.

Read [DESIGN.md](DESIGN.md) for the fixed architecture, fractional environment Goodness and time
protocol. Each episode uses three explicit 1080p frames, then one independent
27-bit Logistic sample. The environment computes `G = correct_pressed / pressed_count` (or 0 when
nothing is pressed) and delivers it to current Plasticity 250 logical
milliseconds later. Logical time is simulated, not a wall-clock sleep.

Initial sanity and frozen evaluation use the same phase-reset protocol and
independent fresh balanced sequences. A bitwise parameter comparison fails the
run if either phase changes weights. All five seeds are retained; weights are
randomly initialized with existing module initializers and no pretrained data.

Artifacts include config/environment/source/runtime metadata, full episode CSV
(including all 27 q/p/action coordinates), learning curves, per-class results,
per-parameter changes, final state files and aggregate report. Final states are
for audit; reset dynamics before reevaluation and load weights into a model on
the same device. Only scalar goodness crosses the environment/learner boundary.

`visual_diagnostics.json` measures pairwise Eye encodings under no-grad before
training; it has no role in learning or action selection. Unit tests inject a
positive scalar solely to verify the existing delayed update wiring. The
current run computes its learning scalar directly from sampled actions.

Bulk raw results and checkpoints stay under ignored `runs/stage_zero/`; the
compact formal report is kept under `reports/`. The repository version remains
ACNT Runtime v0. Neither a parameter change nor a completed run by itself
establishes successful learning.

Historical binary-reward formal run: [report](../../reports/stage_zero_report.md), five seeds,
1,080 training +270 initial +270 frozen episodes per seed, conclusion
**NOT SUPPORTED**. All raw results remain local under
`runs/stage_zero/formal-20260920/`.

Recompute the raw-log audit independently (does not run learning):

```powershell
python -m experiments.stage_zero.audit runs/stage_zero/fractional-smoke-20260921
```


Current fractional environment Goodness revision:

```powershell
pytest -q
python -m experiments.stage_zero --output runs/stage_zero/fractional-smoke-20260921 --device cuda --seeds 11 --train-episodes 27 --evaluation-episodes 27 --window 27
python -m experiments.stage_zero.audit runs/stage_zero/fractional-smoke-20260921
```

Mean goodness is diagnostic; exact one-hot success, target and non-target
probabilities, and frozen retention determine behavioral evidence. The current
runner requires no Teacher table or API.

Completed five-seed fractional run: [report](../../reports/stage_zero_fractional_report.md),
270 initial + 1080 training + 270 frozen episodes per seed, **NOT SUPPORTED**.
Raw data remain local under `runs/stage_zero/fractional-formal-20260921/`.

Previous DeepSeek frozen Teacher Goodness experiment: **NOT SUPPORTED**.
The following commands and calibration artifacts are retained as historical
evidence; they are not part of the current runner:

```powershell
pytest -q
python -m experiments.stage_zero.calibrate_teacher --api-key-file D:/EVE/APIKey.txt --samples-per-cell 1 --output runs/stage_zero/teacher-smoke-20260921/teacher_goodness.json
python -m experiments.stage_zero.calibrate_teacher --api-key-file D:/EVE/APIKey.txt
# Inspect all means/std in experiments/stage_zero/teacher_goodness.md before training.
python -m experiments.stage_zero --teacher-table experiments/stage_zero/teacher_goodness.json --output runs/stage_zero/teacher-formal-20260921 --device cuda --seeds 11 22 33 44 55 --train-episodes 1080 --evaluation-episodes 270 --window 270
python -m experiments.stage_zero.audit runs/stage_zero/teacher-formal-20260921
```

The key file is read-only and never copied. Alternatively set DEEPSEEK_API_KEY
and omit --api-key-file. No key -> explicit error, no fake table. For that historical revision, the runner defaulted to
experiments/stage_zero/teacher_goodness.json and its audit required Teacher
artifacts. Use the recorded historical commit to audit those old logs.

Completed frozen-Teacher run: [report](../../reports/stage_zero_teacher_report.md), five seeds, same 1080/270/270 budget, **NOT SUPPORTED**. Raw data: `runs/stage_zero/teacher-formal-20260921/`.
