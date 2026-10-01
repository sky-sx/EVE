# Training the original ACNT

This implementation trains the original Adapter -> Core -> Adapter equations.
It does not use EventFlowBlock, exp target dynamics, or persistent delta-W.
The original inference API stays no-grad; training is an explicit separate API.

## Algorithm

1. Roll out original Block dynamics with the original active mask, old-state
   snapshot, forced ReadIns, FIFO pre-activation history and timestamp ages.
2. Keep z and every history activation differentiable within a bounded window.
   Share the exact same Adapter mapping and Block.transition with inference.
3. Sample each discrete output independently. For raw tendency q, threshold h
   and logistic scale tau, p = sigmoid((q-h)/tau). This is the same marginal
   distribution as original logistic-noise thresholding; it is not Softmax.
4. Store log probability of the actually sampled action. Environment rewards
   are detached scalar Goodness, not a differentiable oracle through the world.
5. Use reward-to-go R_t = G_t + discount(dt)*R_next. Optional physical-time
   discount is exp(-dt/discount_tau_ms); this only weights reward credit and
   does not introduce exponential flow into ACNT neural dynamics.
6. Minimize L = -sum_t (R_t - baseline)*log pi(a_t) - entropy_weight*H(pi_t).
   The baseline uses past batches only and is detached. There is no value head
   or second evaluation scalar. Backpropagate through all selected actions,
   original recurrent communication, private NLMs, retained A history, and
   upstream ReadIn Adapters. Adam with norm clipping updates the parameters.
7. Only after consuming the batch graphs and updating parameters, detach the
   current state and all history activations. Preserve their values/timestamps
   across windows. Episode resets are an environment protocol, not required
   between successive windows on a continuing trajectory.

This is bounded BPTT + on-policy REINFORCE. It intentionally relaxes the old
training implementation's no-autograd/no-BPTT constraints, following the user's
request to design a new algorithm without being bound by delta-W. It preserves
original forward architecture. It is a working baseline, not an unlimited-time
online RTRL algorithm. Its credit horizon is the retained training window.

## Whole runtime interfaces

- OriginalTrainingRuntime(runtime) shares all original parameters, while its
  differentiable neural state is separate from production buffers.
- step(now_ms=..., readins=...) follows original scheduling and forced inputs.
- decode(name) returns a differentiable original ReadOut.
- sample_hand() uses original discrete probabilities. Continuous coordinates
  require explicit positive continuous_std for training exploration.
- sample_speak(std=...) uses an explicit Gaussian exploration distribution.
  Gaussian exploration is a training policy choice; original deployed raw
  continuous outputs are the means, with no added noise.
- sample_route() samples independent gates and applies them for the next event.
  Route and Goodness role overrides receive zero policy score. No score is
  credited when Route execution is disabled. Masks/actions are detached; their
  discrete consequences get likelihood-ratio credit, not surrogate gradients.
- joint_sample(hand, speak, route) combines concurrent decisions into one event.
- GoodnessTrainer.update receives (samples, scalar Goodness values, times_ms).
  For a terminal-only score use zeros before the final score. Multiple delayed
  action credits within the retained window are supported; reward event/action
  pairing remains the environment caller's responsibility.
- Goodness Block and Adapter are excluded from actor optimization so the agent
  cannot improve the evaluation output by directly optimizing that output.
  calibrate_goodness(external_teacher) fits only their parameters in a separate
  fresh rollout, after actor graphs have been consumed. It requires Goodness
  enabled. The original inference clamp is preserved; calibration fits raw g.
- update_loss supports supervised calibration or pretraining objectives when
  externally justified. The validated task below uses no supervised loss.
- commit() copies neural values and histories back to Runtime and disables old
  plasticity. Run Runtime.step(..., learn=False) for learned inference; learn=True
  deliberately opts back into the old delta-W update and must not be mixed with
  this trainer.
- save_checkpoint/load_checkpoint preserve parameters, Adam state, neural
  histories/timestamps, active mask, baseline, policy settings and an optional
  supplied action RNG. Restore into the same architecture. Call only after
  pending graphs are consumed. Checkpoint tests verify identical next update.

## Minimal training pattern

```python
from acnt import OriginalTrainingRuntime, GoodnessTrainer
from acnt.__main__ import build_mock_runtime

model = OriginalTrainingRuntime(build_mock_runtime(), max_window=32)
trainer = GoodnessTrainer(model, lr=0.003)
# At each environment event:
model.step(now_ms=0, readins={"ear": audio_tensor})
hand = model.sample_hand(continuous_std=0.2)
speak = model.sample_speak(std=0.2)
route = model.sample_route()
action = model.joint_sample(hand, speak, route)
# The environment receives detached actions and supplies one scalar Goodness.
# Append (action, goodness, timestamp) to your on-policy rollout.
trainer.update([([action], [external_goodness], [0])])
trainer.save_checkpoint("runs/my_training/checkpoint.pt")
model.commit()
```

Use a window long enough to contain the input-to-action causal path. The first
Core event sees old source states; immediate one-tick supervision cannot see
new information cross a Block connection. Do not update weights while other
batch episode graphs remain outstanding. A batch must use one frozen policy.
Do not move model device/dtype after capturing training state; move Runtime
before constructing OriginalTrainingRuntime. Original adapters require FP32.

## Reproduce the actual learning validation

```powershell
python -m experiments.original_acnt_train --seeds 11 22 33 --updates 200 --batch-size 8 --output runs/original_acnt_training/final
python -m experiments.original_acnt_train --seeds 11 22 33 --updates 200 --batch-size 8 --freeze-core --output runs/original_acnt_training/final
New-Item -ItemType Directory -Force .test-tmp | Out-Null
python -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/original-training-check
```

Task: six original organ bindings, six Blocks of width 6, private NLMs and
hold_tick=4. The original EarAdapter receives a signed four-sample synthetic
cue at the first event. Three further original Core events occur with no new
input. The first Hand coordinate is sampled at the final event; Goodness=1 iff
it matches the remembered bit, otherwise 0. Labels are used by the environment
only to produce Goodness, never by a supervised optimizer or ReadIn leak.
All Blocks are active; routing is fixed in this benchmark. The full 85+2 Hand
Adapter exists, but only one binary action affects this task. Eye, Speak and
Route are present but do not get task learning objectives in this benchmark.

Training amplitude is uniform in [0.5,1), event interval 2 ms, action time 6 ms.
Evaluation uses balanced bits, amplitudes {0.35,0.65,0.85,1.15} and intervals
{1,2,3} ms, giving 24 cases. Accuracy is deterministic threshold accuracy;
expected Goodness averages the sampled policy's correct-action probability.
These probe cases are evaluated periodically, so this is a fixed diagnostic
validation set, not a blind test set. Hyperparameters were not selected by
searching this grid, and training always runs the fixed 200-update budget.

Raw per-update curves, all evaluation rows, parameter changes, weights and
resumable checkpoints are written under runs/. Reviewed numerical results
are in reports/original_acnt_training_2026-10-01.md.

## Evidence and limits

Forward equality is checked exactly against original Runtime with async due
checks, dormant/forced ReadIn and all four ReadOuts. Tests check real 1080p Eye
convolution/projection credit, Ear-to-Core-to-Hand gradients, a finite-difference
derivative through history, role-overridden Route credit, continuous Hand and
Speak scores, Goodness-only calibration, window truncation and checkpoint
continuation. Three-seed learning and Freeze-Core are separate experiments.

This proves a small original-ACNT task can learn end to end. It does not prove
all six organs learn useful real-world behavior, long-horizon credit survives
truncation, routing is robust, or visual training is economical at full scale.
BPTT memory grows with window length and retained input-adapter activations.
The current Core remains unbatched; the validation accumulates independent
rollouts into a batch rather than changing its architecture.

## Method references

PyTorch describes the score-function estimator and sample/log_prob pattern:
https://docs.pytorch.org/docs/stable/distributions.html
Its autograd documentation describes graph recording and detach boundaries:
https://docs.pytorch.org/docs/stable/notes/autograd.html
