# ACNT Stage Zero protocol (written before implementation / training)

Baseline: `be9c1cf604b5e3e66d57774cd1ab1704df68fdd5`, clean main;
GitHub commit API resolved remote main to the same SHA on 2026-09-20.
Initial `pytest -q`: 151 passed, 2 fixture errors from Windows temporary
directory permissions. Full rerun using a fresh repository-local basetemp:
153 passed. This is an environment workaround, not a source fix.

## Boundary and reuse

Ten canonical Blocks of 100 neurons, all active throughout; Block 0 is Eye,
Block 1 is Hand, Blocks 2-9 are ordinary. Only EyeAdapter and HandAdapter
(27 independent discrete controls, zero continuous controls) are constructed.
No Runtime.step, Route, Speak, Goodness readout, Ear, executors, LLM, replay,
sleep, or operating-system input. No target is passed to the network.

The thin module borrows Runtime's existing encode_readin, decode_readout,
enable_plasticity and _observe_discrete methods. Its scheduler captures one
Core.source_snapshot per round and calls Core.update_block for due Blocks,
forcing Eye on each new frame. This preserves Core's pulse consumption and
committed snapshot rule. Runtime.enable_plasticity initializes unused route
feedback buffers as well as hand feedback; these inert buffers never enter a
forward, sampling, eligibility or update path. No production source changes.

## Stimulus and timing

Render fixed centered white uppercase bitmap glyphs on black at RGB FP32
[3,1080,1920], values 0..1; green (0,1,0) is MOUSE_LEFT. A fixed built-in
5x7 glyph font scaled by 64 pixels avoids external fonts/dependencies.
Record the font specification and SHA256 of every rendered image.

Each episode has THREE explicit camera frames of the same target, spaced
250 ms apart. Each frame is a fresh Eye event, each pulse is consumed once.
Three rounds allow Eye -> ordinary Block -> Hand propagation through OLD
snapshots. This is repeated stimulus presentation, not a retained pulse.
Only the final round samples Hand and generates local eligibility, with a
current Eye local graph. Earlier graphs are discarded; no temporal backprop.
Action time is episode start +500 ms. The environment computes strict binary
exact-one-hot goodness, delivered at action time +250 ms. No further actions
intervene. The next episode starts 250 ms after delivery. Logical time is
simulated without wall-clock sleep. Core state/history persist across samples.

## Fixed learning settings

FP32, ticktime=0.25 seconds (also existing sampling/eligibility tau), hold_tick=4,
threshold=0, learning_rate=0.001, rho=0.9, ema_alpha=0.1, initial_g_bar=0.5,
parameter_clip=None. Defaults match current Plasticity; ticktime matches the
existing mock CLI. No tuning, classifier loss, teacher gradients, reward
shaping, argmax or categorical sampling. Feedback uses the existing seeded
fixed-random initialization and VJP. Goodness calls Plasticity.apply_goodness
at the delivery timestamp; that method alone handles decay, updates and EMA.

## Phases and audit

Tests precede smoke, which precedes formal training. Preflight includes CUDA
smoke if available. A timing smoke determines a practical episode budget;
the formal configuration is saved before running, never chosen from accuracy.
Aim for 5 seeds (at least 3 if runtime is prohibitive), balanced shuffled
27-class cycles, separate target-order and action RNG streams for each phase.
Initial sanity and final evaluation use no updates. Reset dynamic state and
eligibility between phases, preserve learned parameters and baseline; use the
same reset protocol before initial and frozen evaluation. Do not reset within
a phase. Frozen evaluation uses a new sequence and separate action RNG.

Every episode records all 27 q/p values and actions, target, exact reward,
target-bit hit, non-target false activation rate, active count, logical times,
parameter norm/change, eligibility norm, baseline, NaN/Inf counts and expected
exact probability (diagnostic product only, never a loss). Preserve every seed,
per-class summaries, windowed curves, per-tensor parameter changes, source
hashes, config, environment and runtime metadata. Raw data remain local under
runs/stage_zero; publish a compact report consistent with repository policy.

Report SUPPORTED only for reproducible reward/tendency/frozen improvement;
PARTIAL for mixed evidence; NOT SUPPORTED if no learning is observed under
this protocol. Random parameter drift alone is not evidence of task learning.
If strict reward starves the learner, retain that outcome without changing it.
