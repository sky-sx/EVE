# ACNT integration, 2026-10-01

User scope: organize the handoff, update ACNT, and preserve the validated G0-G2 reference including exp time flow. The handoff's suggested G3 sweep is future research, not an additional user instruction.

## Placement

- src/acnt modules -> acnt; original package initializer did not overwrite existing exports.
- Experiments -> experiments/training, run as Python modules.
- Tests -> tests/test_training_* with project imports.
- Specifications -> docs/training.
- Original results and manifests -> reports/training_handoff_2026-10-01.
- Prototypes -> archive/training_prototypes; config -> experiments/training/configs.

Original SHA256SUMS and manifests describe the source archive, not modified integrated files. Reported chat results are historical claims, not fresh local evidence.

## Runtime boundaries

EventFlowCore accepts explicit events with caller-supplied snapshot messages. Materialization only advances physical time; histories shift only for selected semantic events. Use consistent caller-selected physical time units, with lambda in inverse units.

Existing Block/Core/Runtime preserve legacy A/At, ticktime, organs, Route and delta-W behavior. The default CLI has not migrated to event flow. canonical_architecture.txt specifies the legacy path; ALGORITHM_SPEC specifies the new reference. Full production migration still requires organ-event mapping, complete recurrent state layout and online Goodness credit integration.

RTRL is a tiny-model reference: E is dense, not production block-sparse. Pure time propagation requires a correct full-state Jacobian including event-generated target and lambda when these depend on learned parameters. A z-only decay Jacobian is insufficient in that case. G2 uses a supervised oracle to isolate physical-time observability; its success is not a complete Goodness online training claim.

## Fixes

Reject invalid dimensions, rates, rank, nonfinite or decreasing time. Initial states follow model device/dtype conversions. Pure-time RTRL propagation preserves compression RNG state. Replace the vacuous history-shift assertion with exact expected history verification. EventFlowCore commits explicit events transactionally and never maps a scheduler wakeup to a semantic event.

## Fresh validation

193 tests passed, including the existing runtime suite and new regressions. Pytest uses a fresh workspace-local basetemp because the host default temp/cache directories deny access.

G0 forward max error 1.735e-18, gradient max error 3.469e-18.
G1 mean of 128 rank-4 estimators: cosine 0.987253, relative error 0.194762. This is a statistical influence check, not full online task training.
G2 time accuracy 0.9993, fake-dt accuracy 0.5100, maximum partition error 1.561e-17.
G3 robustness remains unresolved; no completed G3 gate is claimed.
