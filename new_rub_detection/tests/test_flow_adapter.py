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

    def test_reference_saves_marks_without_recommending_or_drawing_a_zone(self):
        result = self.check.before(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "READY")
        self.assertNotIn("zone", result["message"])
        self.assertNotIn("recommendation", result)
        np.testing.assert_array_equal(self.image, result["preview"])
        np.testing.assert_array_equal(self.old, result["preview_old_mask"])
        self.assertEqual(result["photo"].shape, self.image.shape)

    def test_new_mark_follows_reference_after_actual_image_rotation(self):
        for turns in (0, 1, 2):
            with self.subTest(turns=turns):
                self.check.before(self.image, self.stone, self.old, DEST)
                target = (180, 180)
                after = self.image.copy()
                masks = self.old.copy()
                cv2.circle(after, target, 12, (220, 220, 220), -1)
                cv2.circle(masks, target, 12, 255, -1)
                result = self.check.after(np.rot90(after, turns).copy(), self.stone,
                                          np.rot90(masks, turns).copy(), DEST)
                self.assertEqual(result["status"], "NEW_RUB", result["message"])
                self.assertEqual("New rub detected on the stone.", result["message"])
                point, _ = result["locations"][0]
                self.assertLess(np.linalg.norm(np.subtract(point, target)), 3)
                self.assertGreater(np.count_nonzero(result["new_mask"]), 0)

    def test_clear_mark_anywhere_on_stone_is_accepted(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        after, masks = self.image.copy(), self.old.copy()
        cv2.circle(after, (400, 400), 12, (220, 220, 220), -1)
        cv2.circle(masks, (400, 400), 12, 255, -1)
        result = self.check.after(after, self.stone, masks, DEST)
        self.assertEqual(result["status"], "NEW_RUB")
        self.assertEqual("New rub detected on the stone.", result["message"])

    def test_unchanged_masks_do_not_confirm_new_rubbing(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "RETAKE")
        self.assertIn("No new rub detected", result["message"])
        self.assertIsNone(result["new_mask"])
        self.assertEqual(result["locations"], [])

    def test_full_stone_requires_cleaning_and_invalidates_an_older_reference(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        result = self.check.before(self.image, self.stone, self.stone, DEST)
        self.assertEqual(result["status"], "CLEAN_THE_STONE")
        self.assertIn('CLEAN THE STONE', result['message'])
        self.assertNotIn('zone', result['message'])
        self.assertNotIn('recommendation', result)
        self.assertIsNone(self.check.reference)
        self.assertIsNone(self.check.old_mask)
        np.testing.assert_array_equal(self.image, result['preview'])
        result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual(result["status"], "RETAKE")

    def test_fragmented_clear_space_requires_cleaning_despite_low_coverage(self):
        stripes = np.zeros_like(self.old)
        for x in range(0, SIZE, 80):
            stripes[:, x:x + 20] = 255
        result = self.check.before(self.image, self.stone, stripes, DEST)
        self.assertLess(result['coverage'], .3)
        self.assertEqual('CLEAN_THE_STONE', result['status'])
        self.assertIsNone(self.check.reference)

    def test_mark_outside_stone_does_not_confirm_rubbing(self):
        stone = np.zeros_like(self.stone)
        stone[50:462, 50:462] = 255
        self.check.before(self.image, stone, self.old, DEST)
        after, masks = self.image.copy(), self.old.copy()
        cv2.circle(after, (20, 250), 12, (220, 220, 220), -1)
        cv2.circle(masks, (20, 250), 12, 255, -1)
        result = self.check.after(after, stone, masks, DEST)
        self.assertEqual('RETAKE', result['status'])
        self.assertIn('No new rub detected', result['message'])

    def test_old_mark_shift_is_not_a_new_rub(self):
        self.check.before(self.image, self.stone, self.old, DEST)
        shifted = np.roll(self.old, 5, axis=1)
        after = self.image.copy()
        after[shifted > 0] = 220
        result = self.check.after(after, self.stone, shifted, DEST)
        self.assertEqual('RETAKE', result['status'])
        self.assertIn('No new rub detected', result['message'])

    def test_unrecognizable_stone_alignment_cannot_be_reported_as_no_rub(self):
        from unittest.mock import patch
        self.check.before(self.image, self.stone, self.old, DEST)
        with patch('new_rub_detection.flow.align_to_reference',
                   return_value=(None, None, 'stone alignment failed')):
            result = self.check.after(self.image, self.stone, self.old, DEST)
        self.assertEqual('RETAKE', result['status'])
        self.assertIn('stone alignment failed', result['message'])
        self.assertNotIn('No new rub detected', result['message'])

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
