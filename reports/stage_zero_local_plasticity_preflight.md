# Local Plasticity Stage 0 preflight

- Baseline main: `a7c72597bdbac10819a88c802b3cb25b1b9a6c83`
- Candidate: unchanged `CorrelationRule` F_e/F_w; learning_rate 0.001, retention 0.95, no clipping.
- Lock: `reports/stage_zero_local_plasticity_lock.json` records formal budgets, seeds, CPU device, RNG rule, source SHA256, environment/font and versions before formal results.
- Formal seed budgets: each 270 initial, 1080 training, 270 frozen; five seeds 11/22/33/44/55.
- Independent smoke: CPU and RTX 5080 CUDA both completed 1/1/1; timing smoke 27/9/9 on seed 99 completed on both. CPU was faster (7.68 s versus 17.43 s total).
- Initial probability precheck: independent timing smoke initial target p 0.48179, non-target p 0.47345; mean exact event probability 1.6361e-8. For a 27-bit independent near-0.5 baseline, 2^-27 = 7.45058e-9. Multiplying the measured precheck mean by 5400 formal training episodes gives only 8.84e-5 expected positive hits; multiplying 2^-27 gives 4.02e-5. These are estimates, not observed formal outcomes.
- No Goodness shaping, parameter search, delta-time rule, or extra plasticity mechanism is permitted after the lock.
- Full active pytest before smoke: 143 passed.
