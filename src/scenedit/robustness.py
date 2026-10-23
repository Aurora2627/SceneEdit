"""Paired evaluation without frame-wise pseudo-replication or missing-pair selection."""

import math


def paired_results(records, expected_pairs):
    groups = {}
    for row in records:
        key = (row["case"], row["seed"])
        if key not in expected_pairs:
            raise ValueError("Unexpected case/seed")
        radius = row["radius"]
        if radius not in (0, 3) or radius in groups.setdefault(key, {}):
            raise ValueError("Invalid or duplicate condition")
        value = row["metrics"]["foreground_pixel_mae"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid pixel metric")
        groups[key][radius] = row
    if set(groups) != set(expected_pairs) or any(
        set(pair) != {0, 3} for pair in groups.values()
    ):
        raise ValueError("Incomplete paired evaluation")
    results = []
    for case, seed in expected_pairs:
        pair = groups[(case, seed)]
        baseline = pair[0]["metrics"]["foreground_pixel_mae"]
        candidate = pair[3]["metrics"]["foreground_pixel_mae"]
        results.append(
            {
                "case": case,
                "seed": seed,
                "baseline_foreground_mae": baseline,
                "candidate_foreground_mae": candidate,
                "absolute_delta_candidate_minus_baseline": candidate - baseline,
                "relative_reduction_percent": (
                    100 * (baseline - candidate) / baseline if baseline > 0 else None
                ),
            }
        )
    return results
