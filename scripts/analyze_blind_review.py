"""Validate actual human exports and report preferences alongside pixel diagnostics."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha
from scenedit.review import validate_ratings, summarize_reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        nargs=2,
        action="append",
        required=True,
        metavar=("PRIVATE_KEY", "RATINGS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new report path")
    reviews, provenance, comparisons = [], [], []
    for key_path, ratings_path in args.review:
        key = read(key_path)
        document = read(ratings_path)
        reviewer, rows = validate_ratings(document, key)
        reviews.append((reviewer, rows))
        provenance.append(
            dict(key_sha256=sha(key_path), ratings_sha256=sha(ratings_path))
        )
        for task in key["tasks"]:
            for dimension, metric in (
                ("subject", "foreground_pixel_mae"),
                ("boundary", "boundary_pixel_mae"),
            ):
                baseline = task["metrics"]["0"][metric]
                candidate = task["metrics"]["3"][metric]
                diagnostic = (
                    "candidate"
                    if candidate < baseline
                    else "baseline" if candidate > baseline else "tie"
                )
                human = rows[task["pair_id"]]["preferences"][dimension]
                comparisons.append(
                    dict(
                        rater_id=reviewer,
                        pair_id=task["pair_id"],
                        case=task["case"],
                        dimension=dimension,
                        metric=metric,
                        baseline=baseline,
                        candidate=candidate,
                        metric_preference=diagnostic,
                        human_preference=human,
                        directional_agreement=(
                            human == diagnostic if human != "neither" else None
                        ),
                    )
                )
    report = summarize_reviews(reviews)
    report.update(
        input_hashes=provenance,
        per_task_metric_comparison=comparisons,
        comparison_note="Directional diagnostic checks, not proof of perceptual validity. The full result includes ties and neither. Small correlated samples do not justify a population correlation claim.",
    )
    save(args.output, report)
    print("HUMAN_REVIEW_REPORT_SAVED", args.output, report["status"])


if __name__ == "__main__":
    main()
