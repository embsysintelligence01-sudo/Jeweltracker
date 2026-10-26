import unittest

import numpy as np

from new_rub_detection.core import merged_mark_regions, new_mark_locations, new_mark_regions
from new_rub_detection.geometry import SIZE
from new_rub_detection.rubbing_zones import analyze_zones


class FirstPhotoGateTests(unittest.TestCase):
    def setUp(self):
        self.stone = np.ones((SIZE, SIZE), bool)
        self.mask = np.zeros((SIZE, SIZE), np.uint8)

    def test_overlapping_predictions_count_once(self):
        self.mask[40:90, 40:90] = 255
        self.mask[65:115, 65:115] = 255
        _, count, coverage = merged_mark_regions(self.mask, self.stone)
        self.assertEqual(count, 1)
        self.assertGreater(coverage, 0)

    def test_five_old_marks_still_allow_a_clear_rubbing_target(self):
        for x in (30, 120, 210, 300):
            self.mask[40:80, x:x + 30] = 255
        _, count, coverage = merged_mark_regions(self.mask, self.stone)
        self.assertEqual(count, 4)
        self.mask[130:170, 390:420] = 255
        _, count, coverage = merged_mark_regions(self.mask, self.stone)
        self.assertEqual(count, 5)
        _, selected = analyze_zones(self.mask, self.stone)
        self.assertIsNotNone(selected)

    def test_more_than_six_percent_coverage_still_allows_clear_zones(self):
        self.mask[80:220, 80:220] = 255
        _, count, coverage = merged_mark_regions(self.mask, self.stone)
        self.assertEqual(count, 1)
        self.assertGreater(coverage, 0.06)
        _, selected = analyze_zones(self.mask, self.stone)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.old_marks, 0)

    def test_new_location_follows_current_stone_position(self):
        self.mask[200:220, 300:320] = 255
        frame_to_stone = np.float32([[1, 0, -40], [0, 1, -30], [0, 0, 1]])
        locations = new_mark_locations(self.mask, frame_to_stone)
        self.assertEqual(locations, [((310, 210), (350, 240))])
        self.assertEqual(new_mark_locations(np.zeros_like(self.mask), frame_to_stone), [])

    def test_shifted_old_marks_are_not_reported_as_new(self):
        old = np.zeros_like(self.mask)
        old[70:100, 70:130] = 255
        old[70:100, 300:360] = 255
        current = np.zeros_like(self.mask)
        current[76:106, 82:142] = 255
        current[74:104, 294:354] = 255
        self.assertEqual(np.count_nonzero(new_mark_regions(current, old, self.stone)), 0)
        current[170:200, 90:150] = 255
        new = new_mark_regions(current, old, self.stone)
        self.assertEqual(np.count_nonzero(new), 30 * 60)
        self.assertEqual(new[185, 120], 255)
        self.assertEqual(new[90, 100], 0)


if __name__ == "__main__":
    unittest.main()
