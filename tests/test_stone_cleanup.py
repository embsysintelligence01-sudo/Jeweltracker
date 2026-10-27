import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from acid_test.onnx_detector import Detection
from rub_preparation import RubCycle, StoneCleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((300, 400, 3), 210, np.uint8)
        self.frame[75:225, 125:275] = 60
        self.mask = np.zeros((300, 400), np.uint8)
        self.mask[75:225, 125:275] = 255
        self.quad = np.float32([[125, 75], [274, 75], [274, 224], [125, 224]])
        self.old = np.zeros_like(self.mask)
        cv2.circle(self.old, (160, 110), 6, 255, -1)
        self.models = Mock(surface=Mock(return_value=(self.mask, self.quad)),
                           marks=Mock(return_value=self.old))
        self.detector = Mock(predict=Mock(return_value=([], 1.)))
        self.cache = Mock(get=Mock(return_value=(self.detector, Mock())))
        self.cleanup = StoneCleanup(self.models, self.cache)
        self.cycle = RubCycle(self.models)

    def check(self, ts, frame=None):
        return self.cleanup.before(self.cycle, self.frame if frame is None else frame, ts)

    def test_acid_present_prevents_reference_and_does_not_run_rubbed_model(self):
        # Far from the old rub at (160, 110): cleanup checks the entire stone.
        self.detector.predict.return_value = ([Detection((220, 170, 250, 200), .95)], 1.)
        for i in range(15):
            result = self.check(i*.25)
            self.assertEqual('WAIT_FOR_CLEANUP', result['status'])
            self.assertIn('Acid is still detected', result['message'])
        self.models.marks.assert_not_called()
        self.assertIsNone(self.cycle.before_result)

    def test_clear_stable_stone_saves_same_photo_with_remaining_old_marks(self):
        # Acid elsewhere on the bench must not block this stone's cleanup.
        self.detector.predict.return_value = ([Detection((10, 10, 40, 40), .95)], 1.)
        for i in range(8):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(i*.25)['status'])
        result = self.check(2.)
        self.assertEqual('READY', result['status'])
        self.assertIs(result, self.cycle.before_result)
        self.models.photo.assert_not_called()
        self.models.marks.assert_called_once()
        np.testing.assert_array_equal(self.frame, self.models.marks.call_args.args[0])
        self.assertGreater(np.count_nonzero(self.cycle.check.old_mask), 0)

    def test_motion_restarts_two_seconds_even_when_acid_is_absent(self):
        for i in range(6):
            self.check(i*.25)
        moved = cv2.warpAffine(self.frame, np.float32([[1, 0, 12], [0, 1, 0]]),
                               (400, 300), borderValue=(210, 210, 210))
        for i in range(8):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(1.5+i*.25, moved)['status'])
        self.assertEqual('READY', self.check(3.5, moved)['status'])

    def test_mask_boundary_jitter_does_not_move_the_settling_crop(self):
        for i in range(9):
            jitter = i % 3-1
            quad = self.quad + np.float32([[-2, -2], [2, -2], [2, 2], [-2, 2]])*jitter
            self.models.surface.return_value = (self.mask, quad)
            result = self.check(i*.25)
            self.assertEqual('READY' if i == 8 else 'WAIT_FOR_CLEANUP', result['status'])

    def test_stationary_hand_or_cloth_is_not_accepted_as_acid_removal(self):
        hand = self.frame.copy()
        hand[80:220, 130:270] = (55, 100, 180)
        for i in range(12):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(i*.25, hand)['status'])
        self.detector.predict.assert_not_called()
        partial = self.mask.copy()
        partial[90:210, 140:260] = 0
        self.models.surface.return_value = (partial, self.quad)
        for i in range(12, 24):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(i*.25)['status'])
        self.models.marks.assert_not_called()

    def test_surface_missing_is_normal_wait_and_resets_quiet_credit(self):
        for i in range(7):
            self.check(i*.25)
        self.models.surface.side_effect = RuntimeError('RETAKE - expose one complete stone')
        result = self.check(1.75)
        self.assertEqual('WAIT_FOR_CLEANUP', result['status'])
        self.assertNotIn('RETAKE', result['message'])
        self.assertNotIn('not detected', result['message'])
        self.models.surface.side_effect = None
        for i in range(8):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(2.+i*.25)['status'])
        self.assertEqual('READY', self.check(4.)['status'])

    def test_acid_reappearing_restarts_all_clear_observations(self):
        for i in range(7):
            self.check(i*.25)
        self.detector.predict.return_value = ([Detection((150, 100, 180, 130), .9)], 1.)
        self.assertEqual('WAIT_FOR_CLEANUP', self.check(1.75)['status'])
        self.detector.predict.return_value = ([], 1.)
        for i in range(8):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(2.+i*.25)['status'])
        self.assertEqual('READY', self.check(4.)['status'])

    def test_duplicate_timestamps_or_long_gap_cannot_finish_cleanup(self):
        for _ in range(20):
            self.assertEqual('WAIT_FOR_CLEANUP', self.check(0.)['status'])
        self.assertEqual('WAIT_FOR_CLEANUP', self.check(5.)['status'])
        self.models.marks.assert_not_called()

    def test_crowded_stone_stays_blocked_then_recovers_after_more_cleaning(self):
        self.models.marks.return_value = self.mask.copy()
        for i in range(8):
            self.check(i*.25)
        self.assertEqual('CLEAN_THE_STONE', self.check(2.)['status'])
        self.assertIsNone(self.cycle.before_result)
        self.models.marks.return_value = self.old
        for i in range(8):
            waiting = self.check(2.25+i*.25)
            self.assertEqual('WAIT_FOR_CLEANUP', waiting['status'])
            self.assertIn('CLEAN THE STONE', waiting['message'])
        self.assertEqual('READY', self.check(4.25)['status'])

    def test_rub_model_failure_does_not_latch_cleanup_forever(self):
        self.models.marks.side_effect = RuntimeError('model error')
        for i in range(8):
            self.check(i*.25)
        with self.assertRaisesRegex(RuntimeError, 'model error'):
            self.check(2.)
        self.models.marks.side_effect = None
        for i in range(8):
            self.check(2.25+i*.25)
        self.assertEqual('READY', self.check(4.25)['status'])


if __name__ == '__main__':
    unittest.main()
