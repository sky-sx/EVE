# Stage Zero Teacher preflight, 2026-09-21

Base and remote HEAD: 43677b0bce2d10dfdbe217fc8c7670583054644e.
Initial worktree clean. No production acnt/*.py edits or dependency changes.

Required first `pytest -q`: 215 passed, 1 failed, 30 fixture errors, 2 warnings.
The failure was an overbroad new test matching the fixed system prompt phrase
"reward shaping"; it now checks the actual user I/O text. Fixture errors and
cache warnings were Windows access denial on existing temporary/cache directories.
No production source was changed to work around these filesystem errors.

Complete rerun:
`pytest -q --basetemp=.test-tmp/teacher-20260921-a -o cache_dir=.test-tmp/teacher-cache-20260921`
Result: **246 passed in 12.12s**, no failures/errors/skips. CUDA integration ran.
Full output: runs/stage_zero/teacher-validation-20260921/pytest.txt.

Coverage includes all 54 action buckets and finite means, strict bool actions,
PNG roundtrip versus rendered Eye pixels, Teacher request I/O boundary, missing
key failure, bounded invalid-response retries, continuous delayed update and
exact e-prop equations, rejected nonfinite/out-of-range feedback, frozen phases,
independent exact-success metrics, table/value/timing tampering, and byte-equivalent
acnt/*.py versus the starting commit (newline normalization only).

Smoke: 54 real DeepSeek requests, one sample per cell. All completed with numeric
scores and response model deepseek-flash. Table SHA256:
14bd11a0db7b5724578f143ac15995d4a33a32d3473e4129007078e23054ee0b.
All 54 image hashes and serialized request hashes independently reconstructed.
The exact-only smoke instance was Y -> only Y; Teacher returned 0. Its actual
PNG was visually inspected and shows Y. Scores are retained, not corrected.
The sample standard deviations are zero by construction for one sample; formal
calibration uses the predeclared five samples per cell and population std.

Formal protocol remains CUDA FP32, seeds 11/22/33/44/55, 270 initial +1080 training
+270 frozen actions per seed, window 270. No tuning based on calibration or training
outcomes. Fixed prompt, model and scalar bounds are unchanged after smoke.

The API key is read only from the user-specified file into the calibration process;
it is never printed, copied to another file, put in command arguments or saved
in metadata. Training reads only the frozen local table.


Formal calibration completed: 54 cells, 5 real responses each, 270 scores.
Table SHA256: 7c40b1d32f4cc67a5ab64341bfc29c58a3297878395274dd7133a0792a421435.
Independent audit regenerated all 270 RNG instances, actual images and full request
hashes. All matched; response model was deepseek-flash throughout. Raw mean,
population variance/std and [0,1] finite bounds verified. Full audit:
`runs/stage_zero/teacher-validation-20260921/calibration-audit.json`.

All 54 means/std were displayed and inspected before formal training. M[1,0]=0.58
(std 0.4749736834815167); M[0,0]=0. Inversions M[1,k]<M[0,k] at k=14,16,22,23,25
and nonmonotonic rows are retained unchanged. This is a material limitation of
this Teacher realization. No samples were discarded or used to tune the table.
Full table: [teacher_goodness.md](../experiments/stage_zero/teacher_goodness.md).

Formal run launched only after the above checks. Same five seeds, CUDA device,
learning settings and 1080/270/270 budget as the historical binary run.
