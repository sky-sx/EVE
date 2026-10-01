# Validation Status

Scope: imported G0–G3 event-flow reference gates. Historical chat claims below
are separate from the local smoke in [INTEGRATION](INTEGRATION.md) and the
subsequent [online self-write experiments](../../reports/acnt_self_write_session_2026-10-02.md).
These gates do not certify the latest candidate's lifelong learning.

## G0 — PASS

Goal: prove implementation-level local/online derivative correctness before training claims.

Session evidence included:
- local eligibility recursion matching full BPTT/autograd to machine precision on tiny Block models;
- finite-difference agreement around `1e-11` scale;
- after the G2 time upgrade, scheduler partition changes leave forward and parameter gradients unchanged to floating-point error.

Current package re-runs the post-G2 partition/gradient regression in `experiments/g0_math_check.py`.

## G1 — PASS at mechanism/POC level

Problem discovered:
- local-only Block eligibility can be initialization-dependent and can miss causal paths that leave one Block, traverse others, and affect the ReadOut.

Upgrade:

```math
J \approx E_{local}+UV^T
```

with streaming low-rank cross influence.

Session diagnostics included a chain task where an upstream edge could receive no local-only output credit; local-only stayed at chance while streaming low-rank matched exact RTRL near the same learned score.

Caveat:
- the reference package is intentionally small and transparent; production kernels must avoid dense toy matrices and global autograd-Jacobian construction.

## G2 — PASS at mechanism/POC level

The old time semantics were found insufficiently partition-invariant.

Current rule:

```text
semantic event jump + exponential continuous flow
```

Key invariant:

```text
scheduler tick != semantic event != physical time
```

Session checks reported:
- physical-time classification learned;
- falsifying dt returns performance to chance;
- scheduler partition changes alter forward/gradient only at floating-point error;
- mixed histories `{2,4,8,16}` remain compatible;
- pure-time flow does not consume cross-credit rank.

## G3 — PARTIAL / FRONTIER

Already passed:
- `t_last` lazy materialization agrees with eager continuous advancement;
- asynchronous runtime semantics are compatible with the exp flow.

Not passed:
- developmental robustness across random initializations.

Observed failure signature:
- some births have extremely small distant input->output sensitivity (`~1e-12..1e-9`), so even exact policy-gradient/RTRL has almost no usable bootstrap signal.

Next question:
- derive and test a neutral birth initialization that preserves bounded signal/credit transmissibility without task knowledge.
