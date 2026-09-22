# Current Local Plasticity Stage 0 formal report

This is the new Local Plasticity experiment. The archived e-prop Stage 0 is a separate historical result.

- Baseline main commit: `a7c72597bdbac10819a88c802b3cb25b1b9a6c83`.
- Final main commit: see release response; a tracked report cannot embed its own content-addressed commit SHA.
- Git status at start: clean. Final status checked after commit and push.
- Python 3.11.9; PyTorch 2.10.0+cu130; CUDA 13.0; GPU NVIDIA GeForce RTX 5080. Formal execution used CPU.
- Frozen candidate: production CorrelationRule, learning_rate=.001, retention=.95, parameter_clip=None.
- Formal seeds 11, 22, 33, 44, 55; per seed 270 initial, 1080 training, 270 frozen episodes.
- Independent audit: passed, 8100 rows and 53 stimulus hashes.
- Pre-formal full active pytest: 143 passed; final full active pytest after report: 146 passed. CPU and RTX 5080 CUDA 1/1/1 smoke passed; 27/9/9 timing smoke passed on both.
- Formal command: `python -m experiments.stage_zero --device cpu --seeds 11 22 33 44 55 --initial 270 --training 1080 --frozen 270 --checkpoint-interval 270 --output runs/stage_zero_local`.
- Audit command: `python -m experiments.stage_zero.audit --directory runs/stage_zero_local`.
- Local only: raw JSONL and checkpoints under ignored `runs/stage_zero_local/`.

## Overall behavior

| Phase | Episodes | Exact | Exact rate | Target p | Non-target p | Target-bit hit | False rate | Active bits | Positive g* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | 1350 | 0 | 0 | 0.505055 | 0.501505 | 0.487407 | 0.504074 | 13.5933 | 0 |
| training | 5400 | 0 | 0 | 0.499786 | 0.499648 | 0.514259 | 0.500826 | 13.5357 | 0 |
| frozen | 1350 | 0 | 0 | 0.496885 | 0.496812 | 0.512593 | 0.50037 | 13.5222 | 0 |

## Per seed

| Seed | Initial exact | Train exact | Frozen exact | Initial target p | Frozen target p | Initial non-target p | Frozen non-target p | Elapsed s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0 | 0 | 0 | 0.51977 | 0.492598 | 0.515661 | 0.492177 | 309.121 |
| 22 | 0 | 0 | 0 | 0.515756 | 0.501408 | 0.513608 | 0.501602 | 325.602 |
| 33 | 0 | 0 | 0 | 0.503827 | 0.492035 | 0.508562 | 0.491423 | 323.456 |
| 44 | 0 | 0 | 0 | 0.488569 | 0.506171 | 0.47544 | 0.506383 | 322.591 |
| 55 | 0 | 0 | 0 | 0.497351 | 0.492215 | 0.494253 | 0.492474 | 325.29 |

## Per class

| Class | Initial exact | Training exact | Frozen exact | Initial target p | Frozen target p | Initial non-target p | Frozen non-target p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 0.588768 | 0.48669 | 0.499158 | 0.497199 |
| B | 0 | 0 | 0 | 0.437416 | 0.482136 | 0.506346 | 0.497308 |
| C | 0 | 0 | 0 | 0.544657 | 0.510003 | 0.499582 | 0.496231 |
| D | 0 | 0 | 0 | 0.468693 | 0.496619 | 0.503469 | 0.496668 |
| E | 0 | 0 | 0 | 0.498433 | 0.503142 | 0.500468 | 0.496477 |
| F | 0 | 0 | 0 | 0.437921 | 0.495324 | 0.502083 | 0.496836 |
| G | 0 | 0 | 0 | 0.561888 | 0.494854 | 0.501092 | 0.496838 |
| H | 0 | 0 | 0 | 0.439637 | 0.493151 | 0.504498 | 0.496911 |
| I | 0 | 0 | 0 | 0.492138 | 0.465095 | 0.502618 | 0.497885 |
| J | 0 | 0 | 0 | 0.512246 | 0.47845 | 0.498598 | 0.497465 |
| K | 0 | 0 | 0 | 0.488662 | 0.504193 | 0.49934 | 0.496542 |
| L | 0 | 0 | 0 | 0.463408 | 0.51927 | 0.506061 | 0.49583 |
| M | 0 | 0 | 0 | 0.526134 | 0.516889 | 0.506612 | 0.496062 |
| N | 0 | 0 | 0 | 0.550459 | 0.515818 | 0.498435 | 0.495961 |
| O | 0 | 0 | 0 | 0.480173 | 0.492655 | 0.49778 | 0.496953 |
| P | 0 | 0 | 0 | 0.553792 | 0.471044 | 0.499472 | 0.497806 |
| Q | 0 | 0 | 0 | 0.554521 | 0.502229 | 0.495521 | 0.496537 |
| R | 0 | 0 | 0 | 0.566902 | 0.483897 | 0.498581 | 0.497731 |
| S | 0 | 0 | 0 | 0.482992 | 0.49732 | 0.503732 | 0.496773 |
| T | 0 | 0 | 0 | 0.472673 | 0.506549 | 0.506073 | 0.496407 |
| U | 0 | 0 | 0 | 0.495226 | 0.504628 | 0.502642 | 0.496435 |
| V | 0 | 0 | 0 | 0.449431 | 0.490354 | 0.505138 | 0.497041 |
| W | 0 | 0 | 0 | 0.521568 | 0.520718 | 0.499108 | 0.495849 |
| X | 0 | 0 | 0 | 0.545803 | 0.512249 | 0.50225 | 0.496252 |
| Y | 0 | 0 | 0 | 0.572758 | 0.478657 | 0.496946 | 0.498304 |
| Z | 0 | 0 | 0 | 0.423041 | 0.485439 | 0.501398 | 0.497222 |
| MOUSE_LEFT | 0 | 0 | 0 | 0.507137 | 0.508533 | 0.503623 | 0.496391 |

## Per delay

| Delay ms | Phase | Episodes | Exact | Target p | Non-target p | Target-bit hit | False rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | initial | 450 | 0 | 0.504972 | 0.502579 | 0.52 | 0.502735 |
| 500 | initial | 450 | 0 | 0.502431 | 0.500788 | 0.48 | 0.50188 |
| 1000 | initial | 450 | 0 | 0.50776 | 0.501147 | 0.462222 | 0.507607 |
| 250 | training | 1800 | 0 | 0.499267 | 0.499612 | 0.52 | 0.505491 |
| 500 | training | 1800 | 0 | 0.500065 | 0.499677 | 0.499444 | 0.496581 |
| 1000 | training | 1800 | 0 | 0.500027 | 0.499654 | 0.523333 | 0.500406 |
| 250 | frozen | 450 | 0 | 0.496042 | 0.496807 | 0.508889 | 0.505897 |
| 500 | frozen | 450 | 0 | 0.497009 | 0.496768 | 0.493333 | 0.494274 |
| 1000 | frozen | 450 | 0 | 0.497604 | 0.49686 | 0.535556 | 0.50094 |

## Per color

| Color | Phase | Episodes | Exact | Target p | Non-target p |
| --- | --- | --- | --- | --- | --- |
| red | initial | 650 | 0 | 0.505271 | 0.501591 |
| blue | initial | 650 | 0 | 0.504678 | 0.501255 |
| green | initial | 50 | 0 | 0.507137 | 0.503623 |
| red | training | 2600 | 0 | 0.499678 | 0.499498 |
| blue | training | 2600 | 0 | 0.500141 | 0.499746 |
| green | training | 200 | 0 | 0.49658 | 0.500307 |
| red | frozen | 650 | 0 | 0.496311 | 0.496781 |
| blue | frozen | 650 | 0 | 0.496564 | 0.496875 |
| green | frozen | 50 | 0 | 0.508533 | 0.496391 |

## Training windows

| Window | Episodes | Exact | Target p | Non-target p | Mean e L2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1350 | 0 | 0.500991 | 0.500319 | 3178.61 |
| 1 | 1350 | 0 | 0.496902 | 0.497426 | 4058.54 |
| 2 | 1350 | 0 | 0.502235 | 0.502025 | 3926.27 |
| 3 | 1350 | 0 | 0.499018 | 0.498821 | 3594.4 |

## Local state and parameters

| Seed | Train mean e L2 | Train max e L2 | End train e L2 | Frozen e L2 | Changed parameter tensors | Max per-episode delta |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 3793.57 | 7106.21 | 3896.43 | 0 | 200 | 3.55311 |
| 22 | 3740.23 | 5448.61 | 3109.89 | 0 | 200 | 2.7243 |
| 33 | 3618.95 | 5511.96 | 3599.25 | 0 | 200 | 2.75598 |
| 44 | 3488.46 | 5250.42 | 3443.41 | 0 | 200 | 2.62521 |
| 55 | 3806.05 | 6390.55 | 4558.48 | 0 | 200 | 3.19528 |

Per-group local-state L2 at the last training Goodness delivery:

| Seed | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Block 7 | Block 8 | Block 9 | Eye Adapter | Hand Adapter | Ordinary Blocks | Terminal Hand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 1412.12 | 1141.93 | 1311.6 | 1323.98 | 1072.34 | 1225.57 | 1294.76 | 1085 | 1209.98 | 1200.52 | 0.241475 | 5.46811 | 3447.33 | 5.14882 |
| 22 | 1034.4 | 965 | 973.802 | 1001.79 | 1026.19 | 972.064 | 948.334 | 987.496 | 968.266 | 953.058 | 0.137066 | 7.10902 | 2769.51 | 6.73349 |
| 33 | 1150.1 | 1108.12 | 1168.8 | 1119.8 | 1159.57 | 1184.24 | 1103.73 | 1064.51 | 1197.93 | 1118.21 | 0.128566 | 6.36068 | 3225.51 | 6.1591 |
| 44 | 1229.24 | 946.303 | 1030.27 | 1061.03 | 1132.76 | 1097.95 | 1158.58 | 989.258 | 1061.9 | 1151.56 | 0.131663 | 8.23615 | 3074.17 | 7.91953 |
| 55 | 1311.78 | 1564.55 | 1369.73 | 1575.08 | 1404.46 | 1416.14 | 1492.76 | 1301.05 | 1475.64 | 1475.65 | 0.134185 | 7.15341 | 4075.68 | 6.86429 |

Local-state distribution at the final training delivery:

| Seed | e mean | e std | e max abs | e nonzero fraction |
| --- | --- | --- | --- | --- |
| 11 | -1.01233e-05 | 1.80367 | 64.663 | 0.958558 |
| 22 | 9.20494e-06 | 1.43958 | 40.3235 | 0.975744 |
| 33 | 3.19326e-06 | 1.6661 | 54.3826 | 0.977715 |
| 44 | 1.82562e-05 | 1.59396 | 51.1158 | 0.933831 |
| 55 | 1.62735e-05 | 2.11013 | 64.6554 | 0.95783 |

The candidate has no bounded e state, so saturation is not a defined failure criterion; finite values, magnitude, and nonzero fraction are reported instead.

At Goodness delivery, local-state persistence by delay:

| Delay ms | Train rows | Nonzero e | Mean e L2 | Mean parameter delta |
| --- | --- | --- | --- | --- |
| 250 | 1800 | 1800 | 3696.91 | 1.84845 |
| 500 | 1800 | 1800 | 3682.34 | 1.84117 |
| 1000 | 1800 | 1800 | 3689.11 | 1.84455 |

F_w coefficient is 0.001*(g* - 0.5): -0.0005 when g*=0 and +0.0005 when g*=1. Numerical tests verify that update direction. All 10 Block groups, including both Adapters, changed in the formal summaries.
NaN / Inf counts across all logged episodes: 0 / 0.
Seeds without joint required behavior improvement: [11, 22, 33, 44, 55].
Total formal elapsed seed time: 1606.060 s.

## Local-only raw inventory

| Seed | Raw JSONL bytes | Raw SHA256 | Checkpoint count | Checkpoint bytes |
| --- | --- | --- | --- | --- |
| 11 | 7843589 | 931c3f4247bac8afa5980188a942e9dfa8a878f0331017f0e8dff0833117190a | 4 | 149952959 |
| 22 | 7837973 | a41cdb87b003b618a0decbd66ee61b02ad610a747383d698e2e87a53845fb72d | 4 | 149952959 |
| 33 | 7839970 | ff9de1338a7274b38e0b6f16d4b767ca8eaa4e26dfe6e3343766342dd037b1c8 | 4 | 149952959 |
| 44 | 7843114 | 2714e1cf77623eae1be27313d7014c97b4a9ab83f4383bc205b7b0eed48c2815 | 4 | 149952959 |
| 55 | 7840159 | 73607ac82991ae66c85e35653e3de0b4584411d47f3caace652303a0f859bd05 | 4 | 149952959 |

These bulk raw logs and full state checkpoints are excluded by repository Git rules.

## Locked source SHA256

| Source | SHA256 |
| --- | --- |
| acnt/plasticity.py | 22fcb9f22d3690e6dcb6f62dd8dbc5ba201e6e38a2dd7f8303049641e3ff362c |
| acnt/block.py | ff45af761a075d1f4d61a068c3af79e1041b0ace0f8c6e1078a14d9a01b30d94 |
| acnt/core.py | cf5ecaff7d804ce72aed3bbbed8f203e8dec18e06b0733a3f1175a7c27e2d091 |
| acnt/adapters.py | d81d1d8c7033b295644916deea617907bae066fb3a5880a365705787faf7c286 |
| acnt/control.py | fb8c52a5696f461133723343219b3f15a022fb587d0761d0f4945582b5d23841 |
| experiments/stage_zero/environment.py | f9d3d0ecd9696fa43a9b09f2cb4b8fa80d6214f7bada2a7fe81820e7e0baed56 |
| experiments/stage_zero/harness.py | 91b8c6da937f67b944bfbef0a9d7e13cb47d923460268574da5aae56f50a918d |
| experiments/stage_zero/runner.py | 9e5850a7108d4317e66bf730dad6f50fdf626c3942d3756b7da6c1e7cc379558 |

## Conclusion

Classification: **NOT SUPPORTED**. Independent frozen behavior is the strongest evidence. Exact-one-hot reward sparsity is the largest failure point when no positive g* occurs. Parameter or e changes alone are not learning evidence.

Q1: The current local correlation state plus delayed scalar Goodness did not demonstrate a reproducible visual-to-action mapping under this protocol.
Q2: All three delay strata have zero exact success. Frozen target p values are 0.496042 (250 ms), 0.497009 (500 ms), and 0.497604 (1000 ms); this observed spread gives no evidence of a clear delay-dependent mapping difference. This does not imply a required delta-time rule.

Limitations: These results assess the frozen candidate under the strict 27-bit Stage 0 reward. They do not resolve which future F_e/F_w rule the architecture should use.
