# Original ACNT: forward sensitivity and delayed policy credit

Research design and reference implementation, 2026-10-01. The original
Adapter -> Core -> Adapter forward is the starting definition. No legacy
delta-W rule, EventFlowBlock, replacement time dynamics, or episode BPTT is
used by this algorithm. This is a candidate, not a lifelong stability theorem.

## 1. Differentiate the actual original forward

For an active destination Block i, with the OLD committed source snapshot:

\[
r_i=o_i+b_i+\sum_{j\in active}W_{ij}z_j,
\quad u_i=l_i\odot\sigma(g_i),\quad a_i=LN(u_i).
\]

Append a_i to its FIFO pre-activation history. For each private neuron NLM,
construct x=[activation history | physical ages | validity], then:

\[
h=GLU(W_1x+b_1),\qquad z=GLU(W_2h+b_2).
\]

The original forward is not a z-only recurrence. Its differentiable Markov
state is s=(all committed z, all retained activation-history slots). The
fixed-size packed representation right-aligns genuine history and pads unused
slots. Logical timestamps, validity lengths, due checks, role masks and sampled
routing are conditioned-on event metadata. Consumed input o is zero; an initial
pending o is an external constant. Diagnostic r/a are not additional recurrent
state because all future access to a is through its history slot.

For GLU([l,g]), its Jacobian is:

\[
D_{GLU}=[diag(\sigma(g)),\ diag(l\odot\sigma(g)\odot(1-\sigma(g)))].
\]

For original non-affine LayerNorm, c=u-mean(u), scale=sqrt(mean(c^2)+eps),
and a=c/scale:

\[
D_{LN}=scale^{-1}(I-\mathbf1\mathbf1^T/n-aa^T/n).
\]

Each private NLM has input Jacobian D_GLU2 W2 D_GLU1 W1. History contributes
a FIFO shift plus insertion of the new a. Thus ordinary chain rule constructs
both the state Jacobian A_t and direct parameter Jacobian B_t:

\[
s_{t+1}=F_t(s_t,\theta_t),\quad
A_t=\partial F_t/\partial s_t,\quad B_t=\partial F_t/\partial\theta_t.
\]

An unrun Block leaves its state AND history untouched; its corresponding state
map is identity and its direct parameter term is zero. All destinations use
the same OLD snapshot. ReadIn-forced execution has the same source inclusion
and input consumption semantics as Runtime.update_blocks.

## 2. Carry sensitivity forward, not a history computation graph

\[
J_{t+1}=A_tJ_t+B_t.
\]

For fixed theta and a fixed initial state, J is exactly ds/dtheta. This follows
by induction from J_0=0 and the chain rule. The reference computes A/B using
autograd for ONE event, then detaches state, history and J. No previous event
graph survives, and there is no horizon-dependent tape. Local autograd is an
implementation choice, not backward propagation through the lifetime.

For changing weights, define an exogenous realized schedule theta_t^0 and a
common offset epsilon: theta_t=theta_t^0+epsilon. Holding the schedule, observed
inputs and realized masks fixed, the same recurrence differentiates this
perturbed neural trajectory exactly. It does NOT differentiate how an earlier
update changes later optimizer state, learning decisions, observation
distributions or the actual world. In the learner this is a stop-update tangent
and a small-step approximation, not an exact meta-gradient of a learning life.

OriginalOnlineRuntime defaults to no sensitivity clipping for derivative
verification. The experiment explicitly clips ||J||_F at 1000 and counts clips.
After a clip, even the fixed-weight sensitivity is approximate.

## 3. Derive the action score from original output semantics

For original discrete Hand/Route coordinate q, threshold h and logistic-noise
scale tau:

\[
p=\sigma((q-h)/\tau),\qquad a\sim Bernoulli(p).
\]

Therefore the score for the actually sampled action is:

\[
e_t=\nabla_\theta\log\pi(a_t|s_t)
=\partial_\theta\log\pi+(\partial_s\log\pi)J_t.
\]

The scalar derivative with respect to q is (a-p)/tau. Independent simultaneous
coordinates add their scores. The direct term trains the output Adapter; J
carries credit into original Core connections/private NLMs and input Adapters.
Actions and their scores are detached immediately after this local calculation.

For stochastic Route, role-overridden coordinates receive zero score, and
disabled routing receives zero score. Realized mask changes condition subsequent
neural Jacobians; no straight-through derivative is applied to Boolean masks.
The benchmark fixes Route; learned Route behavior is not validated here.

For continuous outputs a future extension can use an explicit exploration
distribution N(mu,std^2) and score (a-mu)/std^2 times the derivative of mu.
The reference currently exposes discrete Hand/Route sampling only; continuous
behavior and real Eye training are not claimed.

## 4. Preserve delayed action credit in bounded learning state

For a small fixed bank of physical-time constants tau_k, carry:

\[
Q_k(t+dt)=e^{-dt/\tau_k}Q_k(t)+e_t,
\quad C_k(t+dt)=e^{-dt/\tau_k}C_k(t)+b_t e_t.
\]

Injection happens only at an actual action event. b_t is the scalar Goodness
baseline available BEFORE that action. C preserves that original baseline;
using a changed baseline at late feedback would silently recenter old actions.
Here b is an EMA of prior received Goodness, initialized to 0.5. It is not a
second reward or learned critic.

At a feedback event only the one scalar G in [0,1] enters the learner:

\[
d_t=\frac1K\sum_k(G_tQ_k-C_k).
\]

For an unclipped fixed-policy finite trajectory, the likelihood-ratio identity
gives E[G sum(scores)] as a reward gradient when the world law is independent
of policy parameters. Pre-action baselines remove zero-mean score terms under
the usual conditional-independence assumptions. Exponential kernels deliberately
favor recent causes and bias long-range credit. With overlapping feedback, d
mixes the scores of several earlier actions; it never receives a matching
action ID. Policy drift, clipping, adaptive moments, finite trace constants,
endogenous observation distributions and reward timing limit any exact-gradient
claim. The experiment's world delay schedule is exogenous.

The scalar-Goodness trace approach is related to GPOMDP (Baxter & Bartlett,
2001), which explicitly studies biased infinite-horizon gradient estimation:
https://arxiv.org/abs/1106.0665 . Forward sensitivity is the RTRL family; UORO
studies compressed online estimation: https://arxiv.org/abs/1702.05043 . The
present work adapts those ingredients to the original ACNT state semantics.
It does not claim a new theorem or a previously unknown algorithm family.

## 5. Apply a bounded update without resetting actual state

The implementation caps action-score norm at 10, trace norm at 100, and
normalizes d to norm <=1. It uses Adam-style first/second moments on this ascent
direction, computes a PROPOSED displacement, caps its L2 norm at 0.02, validates
it, and commits it once. No candidate policy is executed and then rolled back.
The old History, neural outputs and timestamps stay exactly as they occurred;
the next event uses the new weights with those old values. J and Q/C also
persist. Sensitivity, scores and traces have no live autograd graphs.

Clipping makes the update biased. A bounded per-update displacement does not
bound total weight drift or prove stable action behavior. A finite sensitivity
and trace cap bounds numerical learning state, not the irreversible world.
Goodness Block/Adapter are excluded from actor updates. Internal Goodness
calibration and its validity remain a separate unsolved requirement.

## 6. Resource and stability claims

With S packed state elements, P selected actor parameters and K time scales,
persistent learning storage is (S+2K+2)P FP32 elements: J, Q, C, first and second
moments. It is constant with lifetime length. Current local Jacobians need
temporary S*P and S*S storage; dense propagation costs O(S*S*P) per event.
This is a tiny-model derivative oracle, not a large-network production kernel.
The budget guard rejects oversized selected parameter sets.

A future scalable implementation should calculate local JVP/VJP operations and
compress cross-Block influence. Neither low-rank unbiasedness nor lifetime
learning success follows merely from a small oracle's success.

Even sensitivity stability needs assumptions. For example, uniformly bounded
B and uniformly contracting products of A over sufficiently long blocks give
a bound on the linearized recurrence. Original FIFO shifts need not contract
at each individual event. Those conditions have not been established for
arbitrary original ACNT weights, variable ages or online policy drift.

## 7. Experiment protocol and falsification

experiments/original_acnt_online.py creates all six original organ bindings,
width 4, private NLM hidden width 2, hold_tick 4. Eye/Speak remain present but
unused by the behavioral objective. All Blocks are active and Route is fixed.
At phase 0 a random bit becomes a four-sample Ear cue. At phase gap one Hand
coordinate is sampled. The world computes correctness, hides it, and emits
only that scalar after randomly 1, 2 or 3 further cue periods plus two events.
Several actions/rewards overlap; rewards arrive at the next cue phase.

There is exactly one continuous world event sequence per run. No reset is
called at cue or reward boundaries; timestamps only advance. Each arrived
feedback updates parameters immediately, including while other older feedback
is pending. The world keeps labels and action association privately; the
learner gets no label, no teacher gradient and no action ID.

Record online expected Goodness, actual sampled correctness, threshold
diagnostics, pending feedback, intervening update counts, module parameter
changes, clipping, resource bytes, source hashes and timing. The lifetime stops
at its prescribed unknown-to-policy budget, without flushing future feedback.
Exogenous cues are shared by matched seeds; sampled actions and consequences
are allowed to differ between learners.

Controls:

- full: persistent state sensitivity and delayed score traces;
- cut_state: J=B at every event, removing propagated state credit;
- cut_delay: clear action traces when time advances, removing delayed credit;
- freeze_core: train Ear/Hand only, preserving Core parameters;
- frozen: no learning, same original neural forward and stochastic actions.

### Diagnostic fixed-delay ledger

FixedDelayLedgerTrainer is a stronger timing-assumption variant. Given a known
constant physical feedback latency D, a feedback arriving at t is associated
with the learner's own action timestamp t-D. The finite ledger stores the
detached policy score and PRE-ACTION baseline for that timestamp. It consumes
the matched entry and uses d=(G-b_at_action)*e_at_action for the bounded update.
No task target, matching action ID or hindsight recurrent graph is delivered.
Association is possible because the latency is known, NOT because anonymous
Goodness magically identifies its cause. If timing cannot identify an entry,
it rejects the feedback rather than silently crediting the newest action.

Ledger capacity is fixed at 64 action events by default; unresolved entries
cannot be overwritten. Extra numerical storage is capacity*(P+1) FP32 elements
plus bounded timestamp metadata. Scores persist while weights and neural state
change. This tests stale numerical policy credit with small bounded updates;
it is not an exact on-policy gradient at the later delivery weights.

The experiment separates feedback_protocol (fixed/variable) from credit_mode
(trace/fixed_delay), so ledger and trace can be compared in the SAME fixed-delay
world. Fixed-delay experiments use three full cue periods plus two events:
gap=3 gives 17 events =34ms, with several intervening decisions/updates.

```powershell
python -m experiments.original_acnt_online --seeds 11 22 33 --modes full --decisions 2000 --gap 3 --credit-mode fixed_delay --output runs/original_acnt_online/fixed
python -m experiments.original_acnt_online --seeds 11 --modes full cut_delay --decisions 1000 --gap 3 --credit-mode trace --feedback-protocol fixed --output runs/original_acnt_online/fixed_trace_controls
```

The fixed-delay ledger is an attribution diagnostic and a usable algorithm
where that feedback timing contract really holds. Its success cannot establish
learning with unknown overlapping real-world causes.

This synthetic world has repeated cue opportunities and no mechanical effects.
It tests chronological no-reset learning, not survival/resource constraints in
a non-ergodic real world. Short successful lifetimes do not establish continual
no-forgetting, retained learning speed or safe irreversible exploration.

## 8. Verification and use

The tests compare original async/forced-input forward values exactly, compare
streaming J to full short-history autograd, check a common-offset finite
difference under changing exogenous weights, check state preservation across
delayed updates, check constant resource sizes and pre-action baseline capture.

```powershell
python -m pytest tests/test_original_online.py -q -p no:cacheprovider --basetemp=.test-tmp/original-online
python -m experiments.original_acnt_online --seeds 11 22 33 --modes full --decisions 2000 --gap 3 --output runs/original_acnt_online/full
```

Read the paired experiment report before interpreting a PASS in the derivative
tests as behavioral learning success.

Completed results: [2026-10-01 report](../../reports/original_acnt_online_2026-10-01.md).
Proposed learned self-write architecture (not yet implemented or validated):
[SELF_WRITE_DESIGN.md](SELF_WRITE_DESIGN.md).
