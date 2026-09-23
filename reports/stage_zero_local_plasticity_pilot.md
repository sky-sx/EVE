# Local Plasticity Stage 0 Pilot Report

Run date: 2026-09-23  
Status: pilot trend analysis only; no formal SUPPORTED/PARTIAL/NOT SUPPORTED classification

## Scope

This pilot evaluates the production `CorrelationRule` under the current inverse-active-count external Goodness definition. It uses CUDA, seeds 11 and 22, 135 initial + 540 training + 135 frozen episodes per seed, `tau_G=5 s`, and an eligibility-timescale scan over `tau_e` in `{0.5, 1, 2} s`. The network, stimulus, timing, and learning boundaries are those in [`experiments/stage_zero/DESIGN.md`](../experiments/stage_zero/DESIGN.md).

Raw JSONL logs remain local under ignored `runs/stage_zero_local/tau_scan/`. This report records the reviewed, reproducible summary without publishing bulk logs or model state.

## Integrity and reproducibility

- All three group audits passed: 1,620 rows per group, 4,860 rows total.
- Each audit regenerated and checked all 53 stimulus hashes.
- The three groups used identical locked source hashes and the declared configuration.
- All 4,860 logged episodes had zero NaN and zero Inf values.
- No checkpoints were requested or written.
- Python 3.11.9, PyTorch 2.10.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5080.

The raw log SHA256 values are:

| `tau_e` | Seed 11 | Seed 22 |
| --- | --- | --- |
| 0.5 s | `2d5ad48bb7a784033354b4fc54949b872c3ea543ed344855b22771434ff56529` | `a13bba4058fab0ac9da06e5e5707c96313be112c558b84dc3a34429f99a4bec0` |
| 1.0 s | `e0145bb11ebf82b62c184e666c17a0976067c66afec99cc1d09703512697c62d` | `6e5063a5c248192016bb34c92cdd343fe07a051016f5e3a7c4274773298ff04c` |
| 2.0 s | `4513f476e4f96a5aeba325ae02e1455e3e679df1c7ed2d0e4571f29772e68444` | `c6584bb4668067c378abcb56203e3c768762fb11cf10eebf05c1b41ff506a47a` |

## Cross-group behavior

The initial phase is identical across groups: exact success 0, target-hit rate 0.511111, 14.2222 active bits per action, target probability 0.520561, and non-target probability 0.515059.

| `tau_e` | Frozen exact | Frozen target hit | Frozen active bits | Frozen target p | Delta target p | Frozen non-target p | Delta non-target p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 s | 0 | 0.511111 | 13.6074 | 0.525695 | +0.005135 | 0.513176 | -0.001884 |
| 1.0 s | 0 | 0.503704 | 13.8000 | 0.504497 | -0.016064 | 0.514036 | -0.001024 |
| 2.0 s | 0 | 0.503704 | 13.8556 | 0.511436 | -0.009125 | 0.514192 | -0.000868 |

Exact one-hot success was zero in initial, training, and frozen phases for every group and both seeds. The action remained dense after training, with roughly half of the 27 bits active. A positive fractional `g*` therefore indicates that the target bit happened to be among many active bits; it is not evidence of a learned visual-to-action mapping.

`tau_e=0.5 s` is the only group with the desired directional probability changes in both seeds: target p rose and non-target p fell. The aggregate changes are small, exact success did not improve, and the two-seed pilot is not large enough to establish reproducibility. It is at most the least-unfavorable setting in this scan, not a supported result.

## Eligibility and delay behavior

Longer `tau_e` produced larger eligibility states and parameter updates, as expected:

| `tau_e` | Mean training e L2 | Mean e L2 at 250/500/1000 ms delivery | Largest per-episode parameter delta |
| --- | --- | --- | --- |
| 0.5 s | 428.485 | 704.012 / 425.479 / 155.964 | 0.307322 |
| 1.0 s | 808.313 | 1072.81 / 839.733 / 512.395 | 0.409979 |
| 2.0 s | 1306.73 | 1504.63 / 1347.15 / 1068.42 | 0.578723 |

All training deliveries had nonzero local state, and eligibility magnitude decreased with delivery delay within each group. This validates the intended timed persistence mechanism mechanically. However, larger and longer-lived eligibility did not improve frozen behavior: both 1 s and 2 s reduced target p relative to their identical initial baseline. Parameter movement and nonzero local state are therefore not behavioral learning evidence.

Frozen target p by delay was non-monotonic in every group:

| `tau_e` | 250 ms | 500 ms | 1000 ms |
| --- | --- | --- | --- |
| 0.5 s | 0.532774 | 0.502074 | 0.542237 |
| 1.0 s | 0.487634 | 0.517256 | 0.508600 |
| 2.0 s | 0.510942 | 0.507926 | 0.515439 |

These small pilot strata do not justify changing the architecture's time definition.

## Conclusion and next decision

The pilot is mechanically valid but provides no evidence that the current `CorrelationRule` learns a reproducible visual-to-action mapping. The decisive behavioral measure, frozen exact success, stayed at zero throughout all six seed/group runs. Increasing the eligibility timescale amplified state and parameter changes without improving the mapping.

Do not describe this candidate as supported. If a formal five-seed run must be selected from this scan, `tau_e=0.5 s` is the only defensible choice because both seeds moved in the desired probability directions. Scientifically, the stronger next step is to revise or replace the candidate `F_e/F_w` rule (or first run a targeted diagnostic of the persistently dense Hand output) before spending the full formal budget.
