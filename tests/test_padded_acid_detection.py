import threading
import cv2
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from continuous_testing import ContinuousMonitor, surface_ready_at
from acid_test.onnx_detector import ACID_DETECTION_CONFIDENCE
from acid_application import AcidApplicationVerifier
from acid_freshness import normalized
from test_continuous_testing import scene, QUAD, STONE
from test_sop_revision import monitor_case, snap, contact


class PaddedAcidDetectionTests(unittest.TestCase):
    def test_crop_maps_boxes_back_and_filters_padding_only_detections(self):
        monitor = ContinuousMonitor.__new__(ContinuousMonitor)
        monitor.detector_lock = threading.Lock()
        monitor.detector = Mock()
        monitor.detector.predict.return_value = ([
            SimpleNamespace(bbox=(27, 37, 47, 67), confidence=.84),
            SimpleNamespace(bbox=(0, 0, 4, 4), confidence=.91)], 0.)
        frame = scene()
        _, _, found = monitor._detect(frame, 10, STONE, 'cycle')
        # 16 px padding for the 140 px stone, with integer-bound rounding.
        cropped, threshold = monitor.detector.predict.call_args.args
        np.testing.assert_array_equal(frame[43:216, 63:236], cropped)
        self.assertEqual(ACID_DETECTION_CONFIDENCE, threshold)
        self.assertEqual([dict(bbox=(90, 80, 110, 110), confidence=.84)], found)

    def test_empty_crop_does_not_call_model(self):
        monitor = ContinuousMonitor.__new__(ContinuousMonitor)
        monitor.detector = Mock()
        self.assertEqual([], monitor._detect(scene(), 10, (600, 500, 650, 550), 'cycle')[2])
        monitor.detector.predict.assert_not_called()

    def test_clear_stone_can_arm_baseline_without_claiming_return(self):
        with monitor_case() as (monitor, events):
            monitor.step(snap(100, held=1, contact=contact(100, confirmed=True)))
            attempt = monitor.active
            for ts in (100.8, 101.0, 101.3, 102.2):
                monitor.step(snap(ts, held=1, frame=scene(mark=True)))
            self.assertIsNone(attempt['returned'])
            self.assertIsNotNone(attempt.get('surface_ready_at'))
            self.assertIsNotNone(attempt['baseline_at'])
            self.assertEqual('UNVERIFIED_CLEAR_STONE_AFTER_RUB', attempt['return_status'])
            record = monitor._record(attempt)
            self.assertEqual(attempt['surface_ready_at'], record['surface_ready_at'])
            self.assertIsNone(record['returned'])

    def test_readiness_keeps_original_baseline_time_after_late_return(self):
        self.assertEqual(10, surface_ready_at(dict(surface_ready_at=10, returned=12)))

    def test_unchanged_dark_mark_is_rejected_but_new_patch_can_confirm(self):
        before = scene()
        before[110:130, 130:140] = 25
        verifier = AcidApplicationVerifier(normalized(before, QUAD), np.zeros((256, 256), np.uint8))
        detection = dict(bbox=(125, 100, 148, 145), confidence=.88)
        verifier.offer(before, QUAD, detection, 10)
        for ts in (10, 10.1, 10.2):
            self.assertIsNone(verifier.observe(before, QUAD, ts))
        self.assertEqual('NO_NEW_PATCH_SINCE_BASELINE', verifier.pending[0]['reason'])
        after = scene(acid=True)
        self.assertIsNone(verifier.observe(after, QUAD, 10.3))
        self.assertIsNotNone(verifier.observe(after, QUAD, 10.4))

    def test_hand_at_stone_edge_blocks_confirmation_away_from_patch(self):
        before = scene()
        after = scene(acid=True)
        skin = cv2.cvtColor(np.uint8([[[10, 90, 180]]]), cv2.COLOR_HSV2BGR)[0, 0]
        covered = after.copy()
        covered[65:110, 185:215] = skin
        verifier = AcidApplicationVerifier(normalized(before, QUAD), np.zeros((256, 256), np.uint8))
        detection = dict(bbox=(125, 100, 148, 145), confidence=.88)
        verifier.offer(covered, QUAD, detection, 10)
        for ts in (10, 10.1):
            self.assertIsNone(verifier.observe(covered, QUAD, ts))
        self.assertEqual('HAND_ON_STONE', verifier.pending[0]['reason'])
        self.assertIsNone(verifier.observe(after, QUAD, 10.2))
        self.assertIsNotNone(verifier.observe(after, QUAD, 10.3))

    def test_bright_patch_outside_candidate_box_does_not_block_confirmation(self):
        before = scene()
        after = scene(acid=True)
        covered = after.copy()
        covered[65:90, 170:215] = 190
        verifier = AcidApplicationVerifier(normalized(before, QUAD), np.zeros((256, 256), np.uint8))
        detection = dict(bbox=(125, 100, 148, 145), confidence=.88)
        verifier.offer(covered, QUAD, detection, 10)
        self.assertIsNone(verifier.observe(covered, QUAD, 10))
        self.assertIsNotNone(verifier.observe(covered, QUAD, 10.1))
