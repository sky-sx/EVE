# Codex Continuation Checklist

This file is deliberately short. The code + algorithm spec are the primary handoff.

## Do not regress these decisions

1. Goodness is the only final evaluation scalar.
2. No BPTT tape in the target online algorithm.
3. Cross-Block credit is not allowed to collapse back to local-only `L_i e_i` without evidence.
4. Do not hard-code cross rank = 4.
5. Do not restore scheduler-tick History shifts.
6. Do not restore the old CfC `sigmoid(a*dt+b)` interpolation as the sole physical-time mechanism.
7. `exp` is allowed specifically as `exp_time` for continuous flow.
8. Pure time passage must not trigger stochastic cross-credit recompression.

## Immediate task: G3 neutral birth

Build a parameterized birth initializer and sweep, at minimum:

- Block message path gain / near-identity strength;
- event-jump message subpath gain;
- recurrent/state Jacobian spectral radius;
- initial lambda / time-constant distribution;
- ReadOut initial scale;
- cross-credit rank;
- active path length and Block width.

For every seed log:

```text
birth distant input->output sensitivity
birth state Jacobian singular-value summary
birth cross/local influence norm ratio
Goodness curve
accuracy/task metric
exact-RTRL oracle curve
low-rank curve
Freeze-Core curve
runtime/VRAM
```

Decision order:

1. If exact RTRL cannot bootstrap -> birth/dynamics issue.
2. If exact bootstraps but low-rank does not -> cross-rank/variance/compression issue.
3. Only after G3 is stable proceed to learnable Edge Gate / Route gates.

Persist raw outputs. Do not report only final percentages.
