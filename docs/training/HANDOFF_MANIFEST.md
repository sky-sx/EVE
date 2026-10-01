# Handoff Manifest

This manifest describes the original imported archive layout; paths below are
historical, not current checkout paths. Integrated implementations are in
`acnt/`, experiments in `experiments/training/`, prototypes in
`archive/training_prototypes/`, and original manifests/results in
`reports/training_handoff_2026-10-01/`. See [INTEGRATION](INTEGRATION.md).
The subsequent self-write evidence has its own [verified manifest](../../reports/acnt_self_write_session_2026-10-02/manifest.json).

## Canonical/current

- `src/acnt/event_flow.py` — post-G2 event jump + exp physical-time Block.
- `src/acnt/rtrl.py` — exact tiny oracle + streaming low-rank cross-credit reference.
- `src/acnt/policy.py` — Bernoulli policy score helper.
- `experiments/g0_math_check.py` — current mathematical/partition regression.
- `experiments/g1_cross_credit.py` — statistical low-rank-vs-exact smoke.
- `experiments/g2_event_flow_time.py` — physical-time observability and fake-dt ablation.
- `experiments/g3_async_bootstrap.py` — eager/lazy async semantic regression; G3 not passed.
- `ALGORITHM_SPEC.md` — current equations and invariants.
- `STATUS_AND_GATES.md` — gate status.

## Historical / archived

- `archive_prototypes/node_perturbation_reference.py`
- `archive_prototypes/online_whitening_reference.py`
- `archive_prototypes/legacy_full_cfc_time_gate.py`
- `RESEARCH_LINEAGE.md`

## Evidence/provenance

- `results/fresh_smoke_results.txt` — actually re-run during packaging.
- `results/reported_chat_results.md` — numbers reported earlier in chat; NOT raw persisted logs.

## Continuation

- `CODEX_HANDOFF.md` — G3 continuation checklist.
- `configs/gates.yaml` — compact human-readable gate state.
