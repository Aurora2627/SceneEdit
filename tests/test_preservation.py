import unittest
import numpy as np
import torch
from scenedit.preservation import (
    conditioning,
    dilate,
    fixed_region_changes,
    valid_pixels,
)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((20, 20, 3), 80, dtype=np.uint8)
        self.edit = np.ones((20, 20), dtype=bool)
        self.edit[7:13, 7:13] = False
        self.valid = valid_pixels(
            (20, 20), {"left": 0, "top": 2, "width": 20, "height": 16}
        )
        self.edit[~self.valid.numpy()] = False

    def test_zero_radius_and_kept_pixels(self):
        control, edit = conditioning(self.source, self.edit, self.valid, 0)
        self.assertTrue(torch.equal(edit, torch.as_tensor(self.edit)))
        self.assertTrue(torch.all(control[~edit] == 80))
        self.assertTrue(torch.all(control[edit] == 127))

    def test_protection_expands_and_padding_stays_noneditable(self):
        control, edit = conditioning(self.source, self.edit, self.valid, 3)
        self.assertLess(int(edit.sum()), int(self.edit.sum()))
        self.assertFalse(edit[~self.valid].any())
        self.assertTrue(torch.all(control[~edit] == 80))

    def test_fixed_regions_exclude_padding_and_do_not_move_with_variant(self):
        output = self.source.copy()
        output[:2] = 255
        result = fixed_region_changes(self.source, output, self.edit, self.valid)
        self.assertTrue(all(value == 0 for value in result.values()))
        output[8:12, 8:12] = 160
        result = fixed_region_changes(self.source, output, self.edit, self.valid)
        self.assertGreater(result["foreground_pixel_mae"], 0)
        self.assertEqual(result["far_background_pixel_change"], 0)

    def test_bad_radius_rejected(self):
        for radius in (-1, 1.5, 17):
            with self.assertRaises(ValueError):
                dilate(self.edit, radius)
