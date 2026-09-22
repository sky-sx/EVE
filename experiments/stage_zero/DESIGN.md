# Local Plasticity Stage 0

This experiment freezes production `CorrelationRule` as the first candidate. It is separate from the archived e-prop Stage 0.

## Boundary

- Ten active Blocks, 100 neurons each, `hold_tick=4`, `ticktime=0.25 s`.
- Block 0 receives 1080p RGB Eye frames; Block 1 drives a 27-bit Hand adapter. Blocks 2–9 are ordinary Core Blocks. No other organ or Adapter is constructed.
- A target is presented at logical times 0, 250 and 500 ms through three new EyeAdapter forward events. Core uses committed old-z snapshots. One Hand action follows the third frame.
- Each Hand bit independently samples Logistic noise with `tau=.25`, threshold zero. All-zero and multi-bit actions are retained.
- External Goodness is zero if the target bit is off. If it is on, `g*=1/n` where `n` is the total number of active Hand bits. Exact one-hot success is logged for evaluation only. The environment alone knows targets; no target, color or correctness label reaches ACNT.
- A separately seeded balanced delay is 250, 500 or 1000 ms. No local event occurs in the idle interval. Delivery invokes production `F_w` and does not invoke `F_e`.
- Training calls `F_e` for observed local operations and `F_w` when `g*` arrives. Initial and frozen evaluation call neither.
- `CorrelationRule` defaults are frozen: learning rate .001, retention .95, no clipping. No other plasticity rule, gradient, baseline or Goodness term is used.

## Phases and artifacts

Each of seeds 11, 22, 33, 44, 55 runs 270 initial, 1080 training, 270 frozen episodes. Targets and delay values are balanced per phase. Each letter alternates red and blue appearances; the phase-specific first color is drawn independently. Target, color, delay, and Hand RNG streams have distinct fixed seeds documented in the run-local config.

Each run writes its frozen configuration and source hashes to `config.json` in the selected output directory. Use `runs/stage_zero_local/potential_goodness/` for the next formal run so the earlier strict-reward raw data is preserved. The old lock, preflight and formal report in `reports/` document that earlier run only. Raw JSONL and sparse checkpoints remain ignored by Git; the audit regenerates stimulus hashes and verifies every row.

The success classification is conservative: parameter changes and nonzero local states alone do not support visual→action learning. Multiple seeds, positive target tendency, reduced non-target tendency, frozen retention and actual exact success improvement are needed for SUPPORTED. Delay differences are observational; this experiment cannot infer a required change to the formal time definition.

