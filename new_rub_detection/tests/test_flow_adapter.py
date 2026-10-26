import unittest

import cv2
import numpy as np

from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import DEST, SIZE


class PortableFlowTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.check = RubCheck()
        self.image = np.random.default_rng(8).integers(60, 130, (SIZE, SIZE, 3), dtype=np.uint8)
        self.stone = np.full((SIZE, SIZE), 255, np.uint8)
        self.old = np.zeros((SIZE, SIZE), np.uint8)
        for point in ((76, 76), (332, 76), (76, 332), (332, 332)):
            cv2.circle(self.old, point, 12, 255, -1)

    def test_all_marked_zones_are_ready_when_a_clear_target_exists(self):
        result = self.check.before(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "READY")
        self.assertIn("All zones contain old marks", result["message"])
        self.assertEqual([zone.old_marks for zone in result["zones"]], [1, 1, 1, 1])
        self.assertEqual(result["photo"].shape, self.image.shape)

    def test_new_mark_follows_reference_zone_after_actual_image_rotation(self):
        for turns in (0, 1, 2):
            with self.subTest(turns=turns):
                before = self.check.before(self.image, self.stone, self.old, DEST)
                target = before["recommendation"].target
                after = self.image.copy()
                masks = self.old.copy()
                cv2.circle(after, target, 12, (220, 220, 220), -1)
                cv2.circle(masks, target, 12, 255, -1)
                result = self.check.after(np.rot90(after, turns).copy(), self.stone,
                                          np.rot90(masks, turns).copy(), DEST)
                self.assertEqual(result["status"], "NEW_RUB", result["message"])
                self.assertIn(before["recommendation"].name, result["message"])
                point, _ = result["locations"][0]
                self.assertLess(np.linalg.norm(np.subtract(point, target)), 3)
                self.assertGreater(np.count_nonzero(result["new_mask"]), 0)

    def test_other_zone_is_accepted(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        after, masks = self.image.copy(), self.old.copy()
        cv2.circle(after, (400, 400), 12, (220, 220, 220), -1)
        cv2.circle(masks, (400, 400), 12, 255, -1)
        result = self.check.after(after, self.stone, masks, DEST)
        self.assertEqual(result["status"], "NEW_RUB")
        self.assertIn("Z4", result["message"])
        self.assertIn("rubbing accepted", result["message"])

    def test_unchanged_masks_do_not_confirm_new_rubbing(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "RETAKE")
        self.assertIsNone(result["new_mask"])
        self.assertEqual(result["locations"], [])

    def test_cleaning_request_clears_an_older_reference(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        result = self.check.before(self.image, self.stone, self.stone, DEST)
        self.assertEqual(result["status"], "CLEAN_THE_STONE")
        self.assertIsNone(self.check.reference)
        result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "RETAKE")

    def test_invalid_frame_mask_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "match the captured frame"):
            self.check.before(self.image, self.stone[:50], self.old, DEST)

    def test_reset_requires_a_new_before_photo(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        self.check.reset()
        result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "RETAKE")


if __name__ == "__main__":
    unittest.main()
