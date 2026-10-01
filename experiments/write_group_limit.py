"""Algebraic group-size limits under an explicit Gaussian credit model.

This is neither an ACNT training run nor a measurement of its credit statistics.
"""
import json
import math
from pathlib import Path


def mean_gain_ratio(group_size, correlation):
    return math.sqrt(correlation+(1-correlation)/group_size)


def conditional_upper(correlation, retention):
    if not 0 <= correlation <= 1 or not 0 < retention <= 1:
        raise ValueError('rho in [0,1] and retention in (0,1] required')
    if correlation >= retention**2:
        return None
    bound = (1-correlation)/(retention**2-correlation)
    candidate = math.floor(bound)
    # Avoid an off-by-one from binary floating point at an exact integer bound.
    while mean_gain_ratio(candidate+1, correlation) >= retention:
        candidate += 1
    while mean_gain_ratio(candidate, correlation) < retention:
        candidate -= 1
    assert candidate >= 1
    assert mean_gain_ratio(candidate, correlation) >= retention
    assert mean_gain_ratio(candidate+1, correlation) < retention
    return candidate


def main():
    retention = .5
    rows = []
    for neurons in (2, 4, 8):
        for rho in (0., .1, .2, .25):
            upper = conditional_upper(rho, retention)
            lower = neurons+2
            rows.append(dict(neurons=neurons, rho_assumption=rho,
                required_mean_retention=retention, structural_min_g=lower,
                conditional_max_g=upper, nonempty=upper is None or lower <= upper))
    result = dict(kind='algebra under assumed joint Gaussian credits; not measured ACNT',
                  gain_ratio='sqrt(rho + (1-rho)/g)',
                  criterion='ratio of expected absolute first-order gains',
                  rows=rows,
                  rho_needed_g32_retention_half=(32*retention**2-1)/31)
    Path('reports/write_group_limit_2026-10-01.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
