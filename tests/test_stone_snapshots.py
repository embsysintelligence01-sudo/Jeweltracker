import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets

from acid_test.integration import AcidTestThread
from acid_test.onnx_detector import Detection
from acid_test.sequence import AcidSequence
from new_rub_detection.geometry import stone_surface
from rub_preparation import RubCycle, RubPhotoModels


def scene():
    image = np.full((900, 900, 3), 230, np.uint8)
    texture = np.random.default_rng(4).integers(65, 100, (450, 450), dtype=np.uint8)
    image[200:650, 200:650] = texture[:, :, None]
    stone = np.zeros(image.shape[:2], np.uint8)
    stone[200:650, 200:650] = 255
    quad = np.float32([[200, 200], [649, 200], [649, 649], [200, 649]])
    marks = np.zeros_like(stone)
    cv2.circle(marks, (350, 350), 12, 255, -1)
    cv2.circle(image, (350, 350), 12, (0, 180, 220), -1)
    return image, stone, marks, quad


def confirmed_target():
    frame, stone, old, quad = scene()
    models = Mock(photo=Mock(return_value=(stone, old, quad)))
    cycle = RubCycle(models)
    cycle.before(frame)
    after, marks = frame.copy(), old.copy()
    cv2.circle(after, (450, 450), 14, (180, 210, 220), -1)
    cv2.circle(marks, (450, 450), 14, 255, -1)
    models.photo.return_value = (stone, marks, quad)
    _, target = cycle.after(after)
    return after, target, models


class OrientedSurfaceTests(unittest.TestCase):
    def test_testbed_marks_inside_detector_box_but_outside_stone_are_rejected_without_tags(self):
        frame, stone, old, quad = scene()
        motion = cv2.getRotationMatrix2D((450, 450), 30, 1.)
        frame = cv2.warpAffine(frame, motion, (900, 900), borderValue=(230, 230, 230))
        stone = cv2.warpAffine(stone, motion, (900, 900), flags=cv2.INTER_NEAREST)
        bbox = cv2.boundingRect(stone)
        x, y, w, h = bbox
        clipped, quad = stone_surface(stone, (x, y, x+w, y+h))
        self.assertGreater(cv2.contourArea(quad), .9*450*450)
        self.assertEqual(0, clipped[y+25, x+25])
        old = np.zeros_like(stone)
        models = Mock(photo=Mock(return_value=(clipped, old, quad)))
        cycle = RubCycle(models)
        cycle.before(frame)
        after, marks = frame.copy(), old.copy()
        # A new mark lies on the bench, inside the axis-aligned detector box.
        cv2.circle(after, (x+25, y+25), 14, (20, 20, 20), -1)
        cv2.circle(marks, (x+25, y+25), 14, 255, -1)
        models.photo.return_value = (clipped, marks, quad)
        with self.assertRaisesRegex(RuntimeError, 'No new rub detected'):
            cycle.after(after)

    def test_highest_confidence_surface_inside_roi_uses_shared_model(self):
        frame, stone, _, _ = scene()
        other = np.zeros_like(stone)
        other[10:180, 10:180] = 255
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=torch.tensor([[10, 10, 180, 180], [200, 200, 650, 650]]),
                                  conf=torch.tensor([.99, .8])),
            masks=SimpleNamespace(data=torch.from_numpy(np.stack((other, stone)) > 0)))
        model = Mock()
        model.predict.return_value = [result]
        models = RubPhotoModels()
        models.use_stone_model(model, np.float32([[190,190], [660,190], [660,660], [190,660]]))
        mask, quad = models.surface(frame)
        self.assertGreater(np.count_nonzero(mask[200:650, 200:650]), 200000)
        self.assertEqual(0, np.count_nonzero(mask[:180, :180]))
        model.predict.assert_called_once_with(frame, conf=.20, return_masks=True)
        models.close()
        model.close.assert_not_called()

    def test_unclear_surface_does_not_fall_back_to_filled_detector_box(self):
        mask = np.zeros((900, 900), np.uint8)
        cv2.circle(mask, (450, 450), 200, 255, -1)
        with self.assertRaisesRegex(RuntimeError, 'stone corners unclear'):
            stone_surface(mask, (200, 200, 700, 700))


class FrozenTargetTests(unittest.TestCase):
    def test_confirmed_target_survives_hand_without_loading_or_calling_any_model(self):
        frame, target, models = confirmed_target()
        frozen = target.roi_mask.copy()
        models.photo.side_effect = AssertionError('No further surface inference after confirmation')
        hidden = np.full_like(frame, 80)
        located = target.locate(hidden)
        self.assertEqual(target.stone, located['stone'])
        self.assertEqual(target.streak, located['streak'])
        np.testing.assert_array_equal(frozen, located['roi_mask'])
        located['roi_mask'][:] = 0
        np.testing.assert_array_equal(frozen, target.roi_mask)
        models.surface.assert_not_called()
        self.assertEqual(2, models.photo.call_count)

    def test_camera_resolution_change_still_requires_a_new_reference(self):
        frame, target, _ = confirmed_target()
        with self.assertRaisesRegex(RuntimeError, 'resolution changed'):
            target.locate(frame[:800])


class FrozenAcidFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def run_frozen_capture(self, automatic=False):
        frame, target, models = confirmed_target()
        models.photo.side_effect = AssertionError('Stone model must not be rerun at Ready')
        hidden = np.full_like(frame, 80)
        live_stone = Mock(side_effect=AssertionError('Hand occlusion must not query live boxes'))
        context = dict(token='fixed', jewel_id=1, batch_id='test', stone=target.stone,
                       streak=target.streak, roi_mask=target.roi_mask.copy(), rub_target=target,
                       stone_getter=live_stone, ph_verification_enabled=False, auto_test=automatic,
                       grabber=SimpleNamespace(latest=lambda: (hidden.copy(), time.time())))
        detector = Mock()
        detector.predict.return_value = ([Detection((435, 435, 465, 465), .95)], 1.)
        classifier = Mock()
        classifier.predict.return_value = dict(prediction='non_gold', confidence=.9)
        cache = SimpleNamespace(get=lambda: (detector, classifier), detector_sha256='test')
        thread = AcidTestThread(context, .12, models=cache)
        completed, failed, events = [], [], []
        def progress(payload):
            events.append(payload)
            if payload['phase'] == 'WAIT_FOR_STONE_READY' and not automatic:
                thread.confirm_stone_ready()
        thread.progress.connect(progress)
        thread.completed.connect(completed.append)
        thread.failed.connect(failed.append)
        with tempfile.TemporaryDirectory() as directory, \
             patch('acid_test.integration.LAB_ROOT', Path(directory)), \
             patch('acid_test.integration.AcidSequence',
                   side_effect=lambda a, b, c: AcidSequence(a, b, .12, .1)):
            thread.start()
            deadline = time.monotonic()+5
            while thread.isRunning() and time.monotonic() < deadline:
                self.qt.processEvents()
                time.sleep(.01)
            thread.cancel()
            self.assertTrue(thread.wait(2000))
            self.qt.processEvents()
        self.assertFalse(failed)
        self.assertEqual(1, len(completed))
        classifier.predict.assert_called_once()
        live_stone.assert_not_called()
        models.surface.assert_not_called()
        self.assertFalse(any('not visible' in p['message'] or 'marker' in p['message'].lower()
                             for p in events))
        np.testing.assert_array_equal(target.roi_mask, thread.context['roi_mask'])
        return events

    def test_ready_and_entire_acid_capture_ignore_live_stone_loss(self):
        self.run_frozen_capture()

    def test_automatic_ready_uses_fixed_target_without_live_stone_or_a_ready_click(self):
        events = self.run_frozen_capture(automatic=True)
        self.assertIn('AUTO_STONE_SETTLE', [p['phase'] for p in events])
        self.assertNotIn('WAIT_FOR_STONE_READY', [p['phase'] for p in events])

