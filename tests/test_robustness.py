import unittest
from scenedit.robustness import paired_results


class RobustnessTests(unittest.TestCase):
    def row(self, seed, radius, value=0.1):
        return {
            "case": "dev",
            "seed": seed,
            "radius": radius,
            "metrics": {"foreground_pixel_mae": value},
        }

    def test_missing_seed_or_condition_blocks_reporting(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_results([self.row(7, 0), self.row(7, 3)], [("dev", 7), ("dev", 123)])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_results([self.row(7, 0)], [("dev", 7)])

    def test_duplicate_and_unplanned_seed_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_results([self.row(7, 0), self.row(7, 0)], [("dev", 7)])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            paired_results([self.row(42, 0)], [("dev", 7)])

    def test_zero_baseline_does_not_invent_percentage_gain(self):
        result = paired_results([self.row(7, 0, 0), self.row(7, 3, 0.1)], [("dev", 7)])[
            0
        ]
        self.assertIsNone(result["relative_reduction_percent"])
        self.assertGreater(result["absolute_delta_candidate_minus_baseline"], 0)
