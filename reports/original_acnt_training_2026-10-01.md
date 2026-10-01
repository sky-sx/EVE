# Original ACNT end-to-end learning validation

Date: 2026-10-01. Local CPU runs; no G2 neural dynamics and no old delta-W.

Algorithm: bounded BPTT + on-policy scalar-Goodness REINFORCE; Adam lr=0.003, gradient clip=1, entropy weight=0.001, past-only EMA baseline decay=0.95. Three seeds, 200 updates each, 8 independent episodes per update (1600 episodes per seed).

Original EarAdapter receives one signed cue; four original Core events preserve and process it; one original Hand coordinate chooses the remembered bit. Only terminal correct/incorrect Goodness enters training. Six original organ bindings and private neuron NLMs are present. Route is fixed all-active; no real mechanical action is executed.

| Seed | Initial expected Goodness | Trained expected Goodness | Trained accuracy | Freeze-Core expected Goodness | Freeze-Core accuracy |
|---|---:|---:|---:|---:|---:|
| 11 | 0.499711 | 0.991494 | 100.00% | 0.515684 | 50.00% |
| 22 | 0.500442 | 0.986264 | 100.00% | 0.512732 | 50.00% |
| 33 | 0.501040 | 0.995481 | 100.00% | 0.677539 | 66.67% |

Accuracy uses a deterministic 0.5 threshold on 24 fixed diagnostic cases per seed: both cue bits, four amplitudes and three event intervals. Expected Goodness uses the actual Bernoulli correct-action probability. These are distinct metrics; 100% threshold accuracy does not mean every stochastic action succeeds. The probe grid was evaluated during training and is not a blind test set.

Full training changes the Ear input Adapter, original Core synapses/private NLMs and Hand output Adapter. Freeze-Core preserves every Core parameter; input/output Adapters remain trainable. This supports the usefulness of training Core on this task at this budget, not a claim that a fixed reservoir can never learn it.

## Verification

188 tests passed in an isolated export of the published files, and the training CLI also ran there. The full local workspace had 203 passing tests including separate G0-G2 reference tests that are not part of this upload. Original asynchronous and forced-ReadIn forward values match inference exactly, including all four ReadOuts. New tests verify cross-history gradients, a finite-difference derivative, actual Eye convolution/projection gradients, role-masked Route score, continuous Hand/Speak credit, isolated Goodness calibration, truncation and checkpoint restoration with identical next optimizer update.

## Artifacts

Published per-update curves, evaluation rows, parameter-change norms, source hashes and regression stdout are in [the experiment records](original_acnt_training_2026-10-01/summary.json). Six complete numerical runs and SHA256SUMS.json are in that directory. Source-hash keys are normalized to repository-relative paths; hashes describe sources recorded by each run.

Weights and full resumable trainer checkpoints remain local under `runs/original_acnt_training/final/` according to the repository artifact policy. Model-only `.pt` files are for parameters/inference buffers; `*_training.pt` also carries neural histories, optimizer/baseline and action RNG. Runtime weights are shared with the original inference API.

Reproduce using [the algorithm guide](../docs/training/ORIGINAL_ACNT_TRAINING.md). Final regression stdout is `runs/original_acnt_training/final/pytest_stdout.txt`.

## Limits

This validates a four-event synthetic memory task, not unrestricted ACNT learning. Eye has derivative evidence only; Speak, all Hand coordinates and learned Route have not been behaviorally trained by this benchmark. Credit before a detached window boundary is truncated, and full-resolution visual windows may be memory-expensive. The original Core is unbatched; each batch accumulates separate on-policy episodes. Training performs episode resets here, not lifelong uninterrupted adaptation. Goodness organ training is external local calibration, not self-reward maximization.

Default Runtime.step still exposes legacy delta-W when learn=True. Use the explicit new trainer for learning and learn=False for learned inference. The finite-window method intentionally permits autograd/BPTT in this training API while preserving original forward equations.
