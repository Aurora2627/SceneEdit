import unittest
import numpy as np
from scenedit.diagnostics import edge_error, temporal_error, flow_consistency_mask


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((12, 12, 3), dtype=np.uint8)
        self.mask = np.ones((12, 12), dtype=bool)
        self.flow = np.zeros((12, 12, 2), dtype=np.float32)

    def temporal(self, previous, current, flow=None):
        return temporal_error(
            self.image,
            self.image,
            previous,
            current,
            self.mask,
            self.mask,
            self.flow if flow is None else flow,
            self.mask,
        )

    def test_identical_edge_is_zero(self):
        image = np.random.default_rng(7).integers(0, 256, (12, 12, 3), dtype=np.uint8)
        result = edge_error(image, image, self.mask)
        self.assertEqual(result["edge_gradient_mae"], 0)
        self.assertEqual(result["edge_pixels"], 100)

    def test_local_structure_change_is_detected(self):
        output = self.image.copy()
        output[4:8, 4:8] = 255
        self.assertGreater(
            edge_error(self.image, output, self.mask)["edge_gradient_mae"], 0
        )

    def test_known_flicker_has_known_residual(self):
        result = self.temporal(self.image, self.image + 20)
        self.assertAlmostEqual(result["temporal_residual_mae"], 20 / 255, places=6)

    def test_constant_bias_is_not_temporal_flicker(self):
        result = self.temporal(self.image + 20, self.image + 20)
        self.assertEqual(result["temporal_residual_mae"], 0)

    def test_known_translation_is_aligned(self):
        previous = np.random.default_rng(9).integers(
            0, 256, (12, 12, 3), dtype=np.uint8
        )
        current = np.roll(previous, 2, axis=1)
        back = self.flow.copy()
        back[..., 0] = -2
        forward = -back
        confidence = flow_consistency_mask(back, forward)
        result = temporal_error(
            previous, current, previous, current, self.mask, self.mask, back, confidence
        )
        self.assertEqual(result["temporal_pixels"], 120)
        self.assertLess(result["source_warp_mae"], 1e-6)
        self.assertEqual(result["temporal_residual_mae"], 0)

    def test_no_correspondence_is_missing_not_perfect(self):
        flow = self.flow + 100
        result = self.temporal(self.image, self.image, flow)
        self.assertEqual(result["temporal_pixels"], 0)
        self.assertIsNone(result["temporal_residual_mae"])

    def test_background_change_does_not_affect_subject_edge(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:9, 3:9] = True
        output = self.image.copy()
        output[~mask] = 255
        self.assertEqual(edge_error(self.image, output, mask)["edge_gradient_mae"], 0)

    def test_invalid_flow_is_rejected(self):
        flow = self.flow.copy()
        flow[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.temporal(self.image, self.image, flow)


if __name__ == "__main__":
    unittest.main()
