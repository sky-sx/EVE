# Results Reported in the Chat (NOT RAW PERSISTED LOGS)

**Warning:** these are numbers stated during the exploratory conversation. The transient executions were not saved as a formal raw-results directory. They are included only so Codex knows what behavior the reconstructed package is intended to reproduce/challenge.

## G0 reported

- local eligibility recursion vs full BPTT Jacobian: max abs error around `2.22e-16`.
- finite difference vs BPTT: around `3.96e-11`.
- policy-gradient local update vs autograd oracle: around `2.05e-17` max update difference.

## Early recurrent / perturbation research reported

- node perturbation + eligibility learned a recurrent delayed-memory toy; N=32 about `0.981`, N=64 reported `1.000` in one setup.
- longer reward delay exposed temporal-trace sensitivity.
- naive Python learning overhead reported ~`5.45x` forward in one prototype.
- online whitening/decorrelation dramatically stabilized scalar-Goodness ReadOut learning.

These experiments were exploratory predecessors; the final G1 candidate shifted to explicit online recurrent influence rather than pure node perturbation.

## G1 reported

A 4-Block chain diagnostic was built so local-only credit could not reach an upstream causal edge.

Reported 5-seed averages:
- local-only: `50.000%`
- deterministic incremental rank-4: `83.926%`
- exact RTRL: `83.927%`

A no-SVD stochastic streaming merge was then reported around:
- rank-1 mean `83.937%`
- rank-4 mean `83.952%`
- exact mean `83.949%`

A richer event-flow gradient-geometry diagnostic later reported average gradient cosine vs exact roughly:
- rank 1: `0.660`
- rank 2: `0.780`
- rank 4: `0.912`
- rank 6: `0.977`
- rank 8: `1.000`

Conclusion: rank is a capacity parameter, not a fixed constant.

## G2 reported

Old time rule physical-time oracle was reported around `59.1%` in a minimal time task and strongly entangled with event count.

After replacing the time rule with event jump + exponential flow:
- 4-Block physical-time oracle: roughly `90.9%, 94.3%, 94.0%` in one version.
- fake/fixed dt returned to about chance.
- sampled-Goodness versions reported around `95.2%, 93.3%, 94.7%` in one setup.
- after fully separating scheduler slices from semantic events, partition differences were reported at ~`1e-14` or smaller forward and `1e-16..1e-15` gradient scale.
- later mixed-history scalar-Goodness run reported N=8 accuracies `99.00%, 98.85%, 96.25%`; fake dt around `50%`.

## G3 reported so far

- eager vs lazy async state differences: around `1e-24..1e-22` in one tiny prototype.
- parameter gradient difference around `1.39e-17`.
- first async Goodness training run: `100%, 100%, 47.05%` across three seeds, indicating bootstrap instability.
- observed birth cue->output sensitivity sometimes around `1e-12..1e-9`.

Therefore G3 is NOT passed.
