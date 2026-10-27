"""Contract tests for paired comparisons and blinded human review."""

import copy
import unittest
from scenedit.benchmark import planned_jobs, paired_summary
from scenedit.review import assignments, validate_ratings, summarize_reviews, DIMENSIONS


class ValidationContracts(unittest.TestCase):
    def setUp(self):
        self.config = dict(
            cases=["new-case"],
            scenes={"forest": "green forest"},
            seeds=[23],
            radii=[0, 3],
            scale=1,
        )
        self.rows = [
            dict(
                j,
                source_sha256="source",
                reference_sha256="reference",
                prompt="forest",
                metrics={
                    k: 0.2 if j["radius"] == 0 else 0.1
                    for k in (
                        "foreground_pixel_mae",
                        "boundary_pixel_mae",
                        "far_background_pixel_change",
                    )
                },
            )
            for j in planned_jobs(self.config)
        ]

    def test_complete_pairs_and_exact_effect(self):
        result = paired_summary(self.rows, self.config)
        self.assertEqual(result["source_sequences"], 1)
        self.assertEqual(
            result["pairs"][0]["metrics"]["foreground_pixel_mae"]["reduction_percent"],
            50,
        )

    def test_missing_pair_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            paired_summary(self.rows[:1], self.config)

    def test_duplicate_output_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary(self.rows + self.rows[:1], self.config)

    def test_changed_reference_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["reference_sha256"] = "different"
        with self.assertRaises(ValueError):
            paired_summary(rows, self.config)

    def test_nonfinite_metric_rejected(self):
        self.rows[1]["metrics"]["foreground_pixel_mae"] = float("nan")
        with self.assertRaises(ValueError):
            paired_summary(self.rows, self.config)

    def test_unsafe_and_repeated_tasks_rejected(self):
        for cases in (["../escape"], ["new-case", "new-case"]):
            config = dict(self.config, cases=cases)
            with self.assertRaises(ValueError):
                planned_jobs(config)


class HumanReviewContracts(unittest.TestCase):
    def fixture(self, rater="reviewer-1", side="A"):
        key = dict(
            protocol_id="frozen",
            tasks=[
                dict(task_id="opaque", pair_id="pair", case="case", candidate_side=side)
            ],
        )
        document = dict(
            protocol_id="frozen",
            rater_id=rater,
            ratings=[
                dict(task_id="opaque", preferences={dim: "A" for dim in DIMENSIONS})
            ],
        )
        return document, key

    def test_balanced_reproducible_assignment(self):
        a = assignments(["one", "two", "three", "four"], 17)
        self.assertEqual(a, assignments(["one", "two", "three", "four"], 17))
        self.assertEqual(sum(row["candidate_side"] == "A" for row in a), 2)
        self.assertEqual(len({row["task_id"] for row in a}), 4)

    def test_side_is_unblinded_correctly(self):
        document, key = self.fixture(side="B")
        self.assertEqual(
            validate_ratings(document, key)[1]["pair"]["preferences"]["overall"],
            "baseline",
        )

    def test_incomplete_or_blank_ratings_rejected(self):
        for mode in ("missing", "blank", "different_protocol"):
            document, key = self.fixture()
            if mode == "missing":
                document["ratings"] = []
            elif mode == "blank":
                document["ratings"][0]["preferences"]["overall"] = ""
            else:
                document["protocol_id"] = "wrong"
            with self.assertRaises(ValueError):
                validate_ratings(document, key)

    def test_ties_and_neither_remain_in_counts(self):
        document, key = self.fixture()
        document["ratings"][0]["preferences"]["overall"] = "neither"
        result = summarize_reviews([validate_ratings(document, key)])
        self.assertEqual(result["status"], "single_reviewer_only")
        self.assertEqual(
            result["reviewer_summaries"][0]["counts"]["overall"]["neither"], 1
        )

    def test_agreement_handles_constant_categories(self):
        a, key = self.fixture()
        b, _ = self.fixture("reviewer-2")
        result = summarize_reviews([validate_ratings(a, key), validate_ratings(b, key)])
        value = result["pairwise_agreement"][0]["dimensions"]["overall"]
        self.assertEqual(value["observed_agreement"], 1)
        self.assertIsNone(value["cohen_kappa"])

    def test_duplicate_rater_rejected(self):
        document, key = self.fixture()
        review = validate_ratings(document, key)
        with self.assertRaises(ValueError):
            summarize_reviews([review, review])


if __name__ == "__main__":
    unittest.main()
