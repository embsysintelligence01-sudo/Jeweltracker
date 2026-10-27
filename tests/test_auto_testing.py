import unittest

import cv2
import numpy as np

from auto_testing import StoneSettling


def scene():
    image = np.full((300, 400, 3), 210, np.uint8)
    image[75:225, 125:275] = 60
    return image


class SettlingTests(unittest.TestCase):
    def test_small_center_jitter_and_exposure_noise_do_not_reset_two_seconds(self):
        gate = StoneSettling((125, 75, 275, 225))
        frame = scene()
        triggered = []
        for i in range(13):
            # Same returned jewel, with normal centre jitter and light changes.
            image = np.clip(frame.astype(int)+(i % 3)*3, 0, 255).astype(np.uint8)
            if gate.update(image, i*.2, center=(40+i % 3, 40-i % 3)):
                triggered.append(i)
        self.assertEqual([10], triggered)

    def test_real_image_movement_restarts_settling(self):
        gate = StoneSettling((125, 75, 275, 225))
        for i in range(8):
            self.assertFalse(gate.update(scene(), i*.2))
        moved = cv2.warpAffine(scene(), np.float32([[1, 0, 12], [0, 1, 0]]),
                               (400, 300), borderValue=(210, 210, 210))
        self.assertFalse(gate.update(moved, 1.6))
        self.assertEqual(0., gate.elapsed)
        for i in range(9):
            self.assertFalse(gate.update(moved, 1.8+i*.2))
        self.assertTrue(gate.update(moved, 3.6))

    def test_real_jewel_movement_restarts_the_timer(self):
        gate = StoneSettling((125, 75, 275, 225))
        for i in range(8):
            self.assertFalse(gate.update(scene(), i*.2, center=(40, 40)))
        self.assertFalse(gate.update(scene(), 1.6, center=(70, 40)))
        self.assertEqual(0., gate.elapsed)

    def test_brief_misses_pause_without_erasing_observed_credit(self):
        gate = StoneSettling((125, 75, 275, 225))
        for i in range(4):
            gate.update(scene(), i*.2)
        self.assertAlmostEqual(.6, gate.elapsed)
        gate.update(scene(), .8, visible=False)
        gate.update(scene(), 1., visible=False)
        self.assertFalse(gate.update(scene(), 1.2))
        self.assertAlmostEqual(.6, gate.elapsed)
        for i in range(6):
            self.assertFalse(gate.update(scene(), 1.4+i*.2))
        self.assertTrue(gate.update(scene(), 2.6))

    def test_long_absence_or_camera_gap_cannot_count_as_quiet_time(self):
        gate = StoneSettling((125, 75, 275, 225))
        gate.update(scene(), 0.)
        gate.update(scene(), .2)
        self.assertFalse(gate.update(scene(), 3.))
        self.assertEqual(0., gate.elapsed)
        gate.update(scene(), 3.2, visible=False)
        gate.update(scene(), 4.2, visible=False)
        self.assertFalse(gate.update(scene(), 4.4))
        self.assertEqual(0., gate.elapsed)

    def test_repeated_frame_timestamp_does_not_advance_settling(self):
        gate = StoneSettling((125, 75, 275, 225))
        for _ in range(30):
            self.assertFalse(gate.update(scene(), 10.))
        self.assertEqual(0., gate.elapsed)

    def test_failed_attempt_is_latched_through_a_processing_gap_until_movement(self):
        gate = StoneSettling((125, 75, 275, 225))
        for i in range(11):
            triggered = gate.update(scene(), i*.2)
        self.assertTrue(triggered)
        for i in range(11):
            self.assertFalse(gate.update(scene(), 5.+i*.2))
        self.assertTrue(gate.latched)
        moved = scene()
        moved[90:200, 140:250] = 180
        self.assertFalse(gate.update(moved, 7.2))
        self.assertFalse(gate.latched)
