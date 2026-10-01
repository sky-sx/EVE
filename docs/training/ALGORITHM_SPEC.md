# ACNT Candidate Training Algorithm — Current Reference

This is the algorithmic state reached after G0–G2 in the 2026-10-01 research session.

## 1. Time semantics

The old idea `sigmoid(a*dt+b)` as the sole time mechanism is superseded in the current candidate.

### Pure physical-time flow

For Block `i` between semantic events:

```math
z_i(t+\Delta t)=\bar z_i + \alpha_i(\Delta t)\odot(z_i(t)-\bar z_i)
```

```math
\alpha_i(\Delta t)=\exp(-\lambda_i\Delta t),\qquad \lambda_i>0
```

A convenient bounded parameterization is:

```math
\lambda_i=\lambda_{min}+(\lambda_{max}-\lambda_{min})\sigma(a_i)
```

This gives exact semigroup composition while `target` and `lambda` are unchanged:

```math
\Phi_{t_2}\circ\Phi_{t_1}=\Phi_{t_1+t_2}
```

### Semantic event

At event time `t_k`:

1. materialize pre-event state `z^- = Phi_dt(z)`;
2. shift semantic history once: `H+ = shift(H,z-)`;
3. build event content `q` from history representation, Block messages and ReadIn;
4. immediate event jump:

```math
u=\tanh(W_uq+b_u)
```

```math
g=\sigma(W_gq+b_g)
```

```math
z^+=(1-g)\odot z^-+g\odot u
```

5. set the next inter-event target and rate:

```math
\bar z^+=\tanh(W_cq+b_c)
```

```math
\lambda^+=\lambda_{min}+(\lambda_{max}-\lambda_{min})\sigma(W_\lambda q+b_\lambda)
```

## 2. Non-negotiable event/time invariant

```text
scheduler tick != semantic event != physical time
```

A scheduler-only slice must not:

- shift History;
- create a new local credit injection;
- trigger stochastic low-rank merge;
- change target/lambda merely because the runtime woke up.

It may lazily materialize a Block from `t_last` using the closed-form exp flow.

## 3. Online recurrent influence

Exact RTRL recurrence:

```math
J_{t+1}=A_tJ_t+B_t
```

with:

```math
A_t=\partial S_{t+1}/\partial S_t
```

```math
B_t=\partial S_{t+1}/\partial\theta
```

The candidate approximation splits influence into:

```math
\hat J=E_{local}+UV^T
```

### `E_local`

Exact Block-local eligibility / forward sensitivity. In production this must be stored block-sparsely, not as a dense global matrix.

### `UV^T`

Cross-Block residual influence. It is propagated indefinitely but compressed to a small rank.

`r_cross` is a **capacity parameter**. The session observed that rank needs depended on dynamics; do not hard-code `r=4` as an architectural law.

## 4. Stochastic streaming merge

If a cross residual is expressed as rank-one components:

```math
R=\sum_k u_kv_k^T
```

an unbiased rank-one merge uses random signs `nu_k`:

```math
\tilde u=\sum_k\nu_k\rho_ku_k
```

```math
\tilde v=\sum_k\frac{\nu_k}{\rho_k}v_k
```

so:

```math
\mathbb E[\tilde u\tilde v^T]=R
```

For rank `r`, stream components into `r` independent/randomized buckets/channels. This is UORO-style in spirit; the ACNT-specific part is keeping Block-local influence exact and compressing only the cross-Block residual.

## 5. Action/readout credit

For action distribution `pi(a|S)`:

```math
s_t=\nabla_S\log \pi(a_t|S_t)
```

then:

```math
q_t=s_t\hat J_t
```

For Bernoulli logits, the scalar logit score is `a-p`.

## 6. Delayed scalar Goodness

Goodness remains the only final evaluation scalar.

Maintain one or a small constant number of real-time delayed traces:

```math
Q_k(t+\Delta t)=e^{-\Delta t/\tau_k}Q_k(t)+q_t
```

and update slowly:

```math
\dot\theta=\eta (G-\bar G)\sum_k\alpha_kQ_k
```

`G_bar` is only a running baseline for variance reduction, not a second value/confidence signal.

## 7. Pure time flow and credit

If the pure-time state Jacobian is diagonal/structured by:

```math
D(\Delta t)=diag(exp(-lambda*dt))
```

then low-rank cross factors propagate as:

```math
U <- D(\Delta t) U
```

with `V` unchanged for that pure-time segment.

**Do not stochastic-merge just because the scheduler subdivided time.** Otherwise training noise becomes scheduler-frequency dependent, violating G2.

## 8. Birth condition (G3 frontier)

Current unresolved issue: some asynchronous random births have distant cue/output sensitivity around `1e-12..1e-9`, making even exact online gradients practically unable to bootstrap.

The next design target is a task-agnostic neutral initialization ensuring nonzero, bounded signal and credit transmissibility across reasonable active Block paths without pre-solving the task.
