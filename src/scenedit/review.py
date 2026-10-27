"""Blinded review contracts; real human ratings are never synthesized or inferred."""

from collections import Counter
from itertools import combinations
import hashlib
import json

import torch

DIMENSIONS = ("subject", "boundary", "scene", "temporal", "overall")
CHOICES = ("A", "B", "tie", "neither")


def protocol_id(tasks):
    payload = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def assignments(pair_ids, seed):
    """Balance left/right assignments within each reviewer, with stable opaque IDs."""
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Empty or duplicate paired tasks")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pair_ids), generator=generator).tolist()
    flips = torch.randperm(len(pair_ids), generator=generator).tolist()
    candidate_left = set(flips[: len(pair_ids) // 2])
    return [
        dict(
            task_id=hashlib.sha256(f"{seed}:{pair_ids[i]}".encode()).hexdigest()[:16],
            pair_id=pair_ids[i],
            candidate_side="A" if i in candidate_left else "B",
        )
        for i in order
    ]


def validate_ratings(document, key):
    """Require the full predeclared task set, one pseudonymous reviewer, no blank defaults."""
    if document.get("protocol_id") != key["protocol_id"]:
        raise ValueError("Review belongs to a different package")
    tasks = key.get("tasks", [])
    if (
        not tasks
        or len({t["task_id"] for t in tasks}) != len(tasks)
        or len({t["pair_id"] for t in tasks}) != len(tasks)
        or any(t["candidate_side"] not in ("A", "B") for t in tasks)
    ):
        raise ValueError("Invalid or empty review key")
    rater = document.get("rater_id")
    if not isinstance(rater, str) or not rater.strip():
        raise ValueError("Reviewer ID required")
    rows = document.get("ratings", [])
    names = [row["task_id"] for row in rows]
    expected = {row["task_id"] for row in key["tasks"]}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Incomplete, unexpected or duplicate review tasks")
    mapping = {row["task_id"]: row for row in key["tasks"]}
    result = {}
    for row in rows:
        if set(row["preferences"]) != set(DIMENSIONS):
            raise ValueError("Missing or unexpected rating dimension")
        task = mapping[row["task_id"]]
        converted = {}
        for dim, choice in row["preferences"].items():
            if choice not in CHOICES:
                raise ValueError("Blank or invalid preference")
            converted[dim] = (
                ("candidate" if choice == task["candidate_side"] else "baseline")
                if choice in ("A", "B")
                else choice
            )
        result[task["pair_id"]] = dict(case=task["case"], preferences=converted)
    return rater.strip(), result


def summarize_reviews(reviews):
    if not reviews:
        raise ValueError("No real reviews provided")
    if len({rater for rater, _ in reviews}) != len(reviews):
        raise ValueError("Duplicate reviewer ID")
    ids = set(reviews[0][1])
    if any(set(rows) != ids for _, rows in reviews):
        raise ValueError("Reviewers evaluated different task sets")
    summaries = []
    for rater, rows in reviews:
        counts = {
            dim: dict(Counter(row["preferences"][dim] for row in rows.values()))
            for dim in DIMENSIONS
        }
        summaries.append(dict(rater_id=rater, tasks=len(rows), counts=counts))
    agreements = []
    for (a, left), (b, right) in combinations(reviews, 2):
        values = {}
        for dim in DIMENSIONS:
            x = [left[i]["preferences"][dim] for i in sorted(ids)]
            y = [right[i]["preferences"][dim] for i in sorted(ids)]
            observed = float(
                torch.tensor([u == v for u, v in zip(x, y)], dtype=torch.float64).mean()
            )
            expected = (
                sum(
                    x.count(c) * y.count(c)
                    for c in ("baseline", "candidate", "tie", "neither")
                )
                / len(ids) ** 2
            )
            values[dim] = dict(
                observed_agreement=observed,
                cohen_kappa=(
                    (observed - expected) / (1 - expected) if expected < 1 else None
                ),
            )
        agreements.append(dict(reviewers=[a, b], dimensions=values))
    return dict(
        status="single_reviewer_only" if len(reviews) == 1 else "reviewed",
        independent_source_sequences=len(
            {row["case"] for row in reviews[0][1].values()}
        ),
        reviewer_summaries=summaries,
        pairwise_agreement=agreements,
        interpretation="Descriptive results only. Seeds/prompts from one source are repeated measures. Ties and neither are reported, never silently excluded. Multiple reviewer IDs do not prove independent human raters. No population significance or calibrated perceptual metric claimed.",
    )
