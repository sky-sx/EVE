"""Representation diagnostics, NOT ACNT learning or real eligibility tests.

Synthetic oracle gradients quantify grouping constraints; FP32 codewords test
packing robustness. Neither labels nor an oracle are supplied to an ACNT actor.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(11)
    count = 65536
    words = np.arange(count, dtype=np.int64)
    carriers = ((words + .5) / count).astype(np.float32)
    packing = []
    for noise in (0., 1e-7, 1e-6, 1e-5, 1e-4):
        observed = (carriers + rng.uniform(-noise, noise, count)).astype(np.float32)
        decoded = np.clip(np.floor(observed.astype(np.float64) * count), 0, count-1).astype(np.int64)
        differences = np.bitwise_xor(decoded, words)
        wrong_bits = sum(np.count_nonzero((differences >> bit) & 1) for bit in range(16))
        packing.append(dict(noise_bound=noise, word_accuracy=float(np.mean(decoded == words)),
                            bit_error_rate=float(wrong_bits / (16 * count))))

    # a_p = gradient_p * eligibility_p. Values are SYNTHETIC, not actual ACNT E.
    n = 65536
    samples = {
        "mixed": rng.standard_normal(n),
        "group_aligned": np.abs(rng.standard_normal(n)),
        "opposing_pairs": np.tile([1., -1.], n // 2),
    }
    grouping = []
    for name, products in samples.items():
        for size in (1, 2, 8, 16, 32):
            achievable = np.abs(products.reshape(-1, size).sum(axis=1)).sum()
            independent = np.abs(products).sum()
            grouping.append(dict(synthetic_case=name, group_size=size,
                                 oracle_gain_ratio=float(achievable / independent)))

    result = dict(status="structural diagnostic only; no training conclusion", seed=11,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  packing=packing, grouping=grouping)
    output = Path(__file__).resolve().parents[1] / "reports/grouped_write_feasibility_2026-10-01.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
