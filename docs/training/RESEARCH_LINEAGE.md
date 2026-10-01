# Research Lineage / What Was Tried in This Conversation

This is the imported G0–G3 event-flow research lineage. Its current/canonical
labels refer to that historical reference line. For the subsequent original-
forward self-write experiments, use the [2026-10-02 session report](../../reports/acnt_self_write_session_2026-10-02.md).

This file exists so future Codex work does not confuse historical experiments with the current candidate.

## Stage A — real-trajectory node perturbation

Used actual neural perturbations plus local sensitivity/eligibility and scalar Goodness. It established that:

- scalar Goodness is not an information-theoretic blocker by itself;
- signed local credit must exist before delayed Goodness arrives;
- long reward delay stresses eligibility time scales;
- dense hidden perturbation has scaling/variance problems.

Status: **useful ancestor / benchmark, not current Core training rule**.

## Stage B — online activity conditioning / decorrelation

ReadOut REINFORCE became much more stable when hidden activity was standardized/decorrelated online. This highlighted conditioning as a separate problem from credit assignment.

Status: **useful optional conditioning mechanism; not a substitute for cross-Block credit**.

## Stage C — local e-prop-style credit

Block-local eligibility plus low-dimensional output feedback worked in some simple tasks but failed the first strict multi-Block G1 setup. The failure was traced to missing cross-Block recurrent influence.

Status: **local eligibility retained, but local-only spatial credit is insufficient as the canonical rule**.

## Stage D — Block-local exact + low-rank cross RTRL

Current G1 candidate:

```math
J \approx E_{local}+UV^T
```

Cross influence is propagated online and stochastically compressed; local influence stays exact.

Status: **current canonical training candidate**.

## Stage E — time semantics upgrade

Old full-CfC-style `sigmoid(a*dt+b)` interpolation did not reliably separate physical time from event count/scheduler partition.

Current G2 rule:

```math
z(t+dt)=target+exp(-lambda*dt)(z(t)-target)
```

with discrete semantic-event jumps and History shifts only on semantic events.

Status: **current canonical time candidate**.

## Current frontier — G3

Asynchronous lazy/eager semantics are consistent. Remaining blocker is task-agnostic developmental bootstrap / neutral birth initialization.
