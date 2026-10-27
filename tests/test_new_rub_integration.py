import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from collections import deque
from concurrent.futures import Future
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import jewel_tracker_5 as app
from acid_test.integration import AcidTestThread
from acid_test.onnx_detector import Detection
from acid_test.sequence import AcidSequence
from new_rub_detection.geometry import DEST, SIZE
from rub_preparation import RubCycle, RubPhotoModels, render_rub_preview


def photos():
    frame = np.random.default_rng(8).integers(60, 130, (SIZE, SIZE, 3), dtype=np.uint8)
    stone = np.full((SIZE, SIZE), 255, np.uint8)
    old = np.zeros((SIZE, SIZE), np.uint8)
    for point in ((76,76), (332,76), (76,332), (332,332)):
        cv2.circle(old, point, 12, 255, -1)
    return frame, stone, old


def pending_track():
    return SimpleNamespace(id=1, rubbed=True, acid_done=False, new_rub_target=None,
                           rub_boxes=deque([(10,10,30,30)]), rub_masks=deque(), rub_marks=deque(),
                           rub_time=.7, rub_last=100., acid_roi_mask=None, acid_roi_stone=None)


class RubSnapshotTests(unittest.TestCase):
    def test_before_reference_and_new_only_mask_survive_rotation(self):
        frame, stone, old = photos()
        models = Mock()
        models.photo.return_value = (stone, old, DEST)
        cycle = RubCycle(models)
        before = cycle.before(frame)
        self.assertEqual('READY', before['status'])
        target_point = (180, 180)
        after, marks = frame.copy(), old.copy()
        cv2.circle(after, target_point, 12, (220,220,220), -1)
        cv2.circle(marks, target_point, 12, 255, -1)
        models.photo.return_value = (stone, np.rot90(marks).copy(), DEST)
        result, target = cycle.after(np.rot90(after).copy())
        self.assertEqual('NEW_RUB', result['status'])
        self.assertEqual(0, np.count_nonzero(target.new_mask & old))
        self.assertGreater(np.count_nonzero(target.roi_mask), 100)
        models.surface.side_effect = RuntimeError('live model missed under the hand')
        located = target.locate(np.rot90(after).copy())
        expected = np.zeros_like(stone)
        cv2.circle(expected, target_point, 12, 255, -1)
        expected = np.rot90(expected).copy()
        models.surface.assert_not_called()
        intersection = np.count_nonzero((located['roi_mask'] > 0) & (expected > 0))
        union = np.count_nonzero((located['roi_mask'] > 0) | (expected > 0))
        self.assertGreater(intersection / union, .8)
        json.dumps(target.metadata)
        self.assertEqual(2, models.photo.call_count)

    def test_unchanged_photo_cannot_create_an_acid_target(self):
        frame, stone, old = photos()
        models = Mock(photo=Mock(return_value=(stone, old, DEST)))
        cycle = RubCycle(models)
        cycle.before(frame)
        with self.assertRaisesRegex(RuntimeError, 'RETAKE'):
            cycle.after(frame)

    def test_fully_marked_stone_requires_cleaning_and_blocks_after_check(self):
        frame, stone, _ = photos()
        models = Mock(photo=Mock(return_value=(stone, stone, DEST)))
        cycle = RubCycle(models)
        result = cycle.before(frame)
        self.assertEqual('CLEAN_THE_STONE', result['status'])
        self.assertIn('Clean the stone', result['message'])
        self.assertIsNone(cycle.before_result)
        with self.assertRaisesRegex(RuntimeError, 'before-rubbing reference'):
            cycle.after(frame)

    def test_real_snapshot_models_use_existing_frame_without_camera(self):
        models = RubPhotoModels()
        self.addCleanup(models.close)
        frame = cv2.imread(str(Path(app.SCRIPT_DIR) / 'diagnostics/antiflicker_20261021/final_verified_00.jpg'))
        stone, marks, quad = models.photo(frame)
        self.assertEqual(frame.shape[:2], stone.shape)
        self.assertEqual(frame.shape[:2], marks.shape)
        self.assertEqual((4,2), quad.shape)
        self.assertGreater(np.count_nonzero(stone), 1000)

    def test_rubbing_over_old_mark_cannot_confirm_even_when_mask_grows(self):
        frame, stone, old = photos()
        models = Mock(photo=Mock(return_value=(stone, old, DEST)))
        cycle = RubCycle(models)
        cycle.before(frame)
        after, marks = frame.copy(), old.copy()
        # Most of this component is outside the old mask; it still touches it.
        cv2.rectangle(after, (70, 70), (200, 90), (220, 220, 220), -1)
        cv2.rectangle(marks, (70, 70), (200, 90), 255, -1)
        models.photo.return_value = (stone, marks, DEST)
        with self.assertRaisesRegex(RuntimeError, 'No new rub detected'):
            cycle.after(after)

    def test_hiding_old_marks_keeps_new_mark_visible_and_masks_unchanged(self):
        frame, stone, old = photos()
        models = Mock(photo=Mock(return_value=(stone, old, DEST)))
        cycle = RubCycle(models)
        before = cycle.before(frame)
        np.testing.assert_array_equal(frame, render_rub_preview(before, False))
        self.assertFalse(np.array_equal(frame, render_rub_preview(before, True)))
        after, marks = frame.copy(), old.copy()
        cv2.circle(after, (180, 180), 12, (220, 220, 220), -1)
        cv2.circle(marks, (180, 180), 12, 255, -1)
        models.photo.return_value = (stone, marks, DEST)
        result, target = cycle.after(after)
        saved_mask = target.roi_mask.copy()
        hidden = render_rub_preview(result, False)
        shown = render_rub_preview(result, True)
        np.testing.assert_array_equal(hidden[180, 180], shown[180, 180])
        self.assertGreater(int(hidden[180, 180, 0]), int(hidden[180, 180, 1]))
        self.assertFalse(np.array_equal(hidden[76, 76], shown[76, 76]))
        np.testing.assert_array_equal(saved_mask, target.roi_mask)
        np.testing.assert_array_equal(old, cycle.check.old_mask)
        self.assertEqual(2, models.photo.call_count)


class AcidMaskTests(unittest.TestCase):
    def sequence(self, mask):
        seq = AcidSequence((0,0,160,100), (30,20,120,80), .1, baseline_seconds=0,
                           clock=lambda:1., streak_mask=mask)
        seq.push(np.full((100,160,3), 80, np.uint8), 0.)
        return seq

    def test_acid_between_separate_new_marks_inside_their_box_is_rejected(self):
        mask = np.zeros((100,160), np.uint8)
        mask[25:45,35:55] = 255
        mask[60:80,100:120] = 255
        seq = self.sequence(mask)
        wrong = Detection((65,40,85,60), .95)
        for i in range(4):
            seq.accept_detections([wrong], .1*i)
        self.assertEqual('WAIT_FOR_ACID', seq.state)
        self.assertEqual(0, seq.hits)
        self.assertIn('outside', seq.instruction(.5))

    def test_correct_mask_requires_three_consistent_detections(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:60,40:70] = 255
        seq = self.sequence(mask)
        correct = Detection((42,32,68,58), .95)
        for i in range(2):
            seq.accept_detections([correct], .1*i)
            self.assertEqual('WAIT_FOR_ACID', seq.state)
        seq.accept_detections([correct], .2)
        self.assertEqual('OBSERVING', seq.state)

    def test_spreading_acid_can_contain_mark_with_center_outside_padded_streak(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:50,40:60] = 255
        seq = AcidSequence((0,0,160,100), (40,30,60,50), .1, baseline_seconds=0,
                           clock=lambda:1., streak_mask=mask)
        seq.push(np.full((100,160,3), 80, np.uint8), 0.)
        spreading = Detection((40,20,150,80), .95)
        for i in range(2):
            seq.accept_detections([spreading], .1*i)
            self.assertEqual(i+1, seq.hits)
            self.assertEqual('WAIT_FOR_ACID', seq.state)
        seq.accept_detections([spreading], .2)
        self.assertEqual('OBSERVING', seq.state)
        self.assertEqual(spreading.bbox, seq.detector_bbox)

    def test_spreading_acid_uses_smaller_area_overlap_threshold(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:50,40:60] = 255
        for left, expected_hits in ((50,1), (51,0), (60,0)):
            with self.subTest(left=left):
                seq = self.sequence(mask)
                seq.accept_detections([Detection((left,20,150,80), .95)], .1)
                self.assertEqual(expected_hits, seq.hits)

    def test_acid_covering_separate_marks_can_have_center_in_gap(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:50,30:50] = 255
        mask[30:50,90:110] = 255
        seq = self.sequence(mask)
        spreading = Detection((25,25,115,55), .95)
        self.assertEqual(0, mask[40,70])
        for i in range(3):
            seq.accept_detections([spreading], .1*i)
        self.assertEqual('OBSERVING', seq.state)

    def test_spreading_acid_still_requires_confidence_and_stone_location(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:50,40:60] = 255
        for stone, confidence in (((0,0,160,100), .59), ((30,20,80,70), .95)):
            with self.subTest(stone=stone, confidence=confidence):
                seq = self.sequence(mask)
                seq.stone = stone
                for i in range(3):
                    seq.accept_detections([Detection((40,20,150,80), confidence)], .1*i)
                self.assertEqual(0, seq.hits)
                self.assertEqual('WAIT_FOR_ACID', seq.state)

    def test_center_inside_but_insufficient_mask_overlap_is_rejected(self):
        mask = np.zeros((100,160), np.uint8)
        mask[30:60,40:70] = 255
        seq = self.sequence(mask)
        seq.accept_detections([Detection((0,44,110,46), .95)], .1)
        self.assertEqual(0, seq.hits)

    def test_empty_or_wrong_size_roi_cannot_start_baseline(self):
        with self.assertRaisesRegex(RuntimeError, 'new-rub mask'):
            self.sequence(np.zeros((100,160), np.uint8))
        with self.assertRaisesRegex(RuntimeError, 'new-rub mask'):
            self.sequence(np.ones((20,20), np.uint8))


class MainRubFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QtWidgets.QApplication.instance() or app.QtWidgets.QApplication([])

    def setUp(self):
        with patch.object(app.QtCore.QTimer, 'singleShot'), patch.object(app, 'write_log'):
            self.window = app.MainWindow()
        self.window.rub_timer.stop()
        self.window.preview_timer.stop()
        self.window.chk_auto_test.setChecked(False)
        self.window.add_log = Mock()
        with patch.object(app, 'JewelIdModel', return_value=None):
            self.worker = app.Worker(0, [(0,0),(511,0),(511,511),(0,511)])
        self.worker.mode = 'LOCKED'
        self.worker.emit_jewel_table = Mock()
        self.frame, self.stone, self.old = photos()
        self.worker.stone_box = Mock(return_value=(0,0,512,512))
        self.worker.grabber = SimpleNamespace(latest=lambda:(self.frame.copy(),time.time()))
        self.window.worker = self.worker
        self.window.rub_models = Mock(photo=Mock(return_value=(self.stone,self.old,DEST)))
        self.window.rub_pool.shutdown(wait=True)
        self.window.rub_pool = Mock()
        def submit(fn, frame):
            future = Future()
            try:
                future.set_result(fn(frame))
            except Exception as exc:
                future.set_exception(exc)
            return future
        self.window.rub_pool.submit.side_effect = submit
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.window.worker = None
        self.window.rub_job = None
        self.window.acid_thread = None
        self.window.close()

    def prepare(self):
        self.window.rub_guidance_tick()
        self.assertFalse(self.worker.rub_reference_ready)
        self.window.rub_guidance_tick()
        self.assertTrue(self.worker.rub_reference_ready)
        self.assertIsNotNone(self.window.rub_preview.pixmap())

    def test_auto_reference_and_start_acid_after_check(self):
        self.prepare()
        cycle = self.window.rub_cycle
        point = (180, 180)
        after, marks = self.frame.copy(), self.old.copy()
        cv2.circle(after, point, 12, (220,220,220), -1)
        cv2.circle(marks, point, 12, 255, -1)
        self.worker.grabber = SimpleNamespace(latest=lambda:(after.copy(),time.time()))
        self.window.rub_models.photo.return_value = (self.stone, marks, DEST)
        tr = pending_track()
        rubbing_photo = self.frame[40:140, 40:140].copy()
        tr.rub_img = rubbing_photo
        self.worker.tracks, self.worker.acid_pending_id = [tr], 1
        self.window.launch_acid_test = Mock()
        self.window.start_acid_test()
        self.window.launch_acid_test.assert_not_called()
        self.assertIsNone(tr.new_rub_target)
        self.window.rub_guidance_tick()
        self.assertIsNotNone(tr.new_rub_target)
        self.assertIs(rubbing_photo, tr.rub_img)
        np.testing.assert_array_equal(tr.new_rub_target.roi_mask, tr.acid_roi_mask)
        self.window.launch_acid_test.assert_called_once()
        self.assertEqual(2, self.window.rub_models.photo.call_count)
        self.worker.stone_box.return_value = None
        context = self.worker.begin_acid_test(False)
        self.assertIs(tr.new_rub_target, context['rub_target'])
        self.assertEqual(tr.new_rub_target.stone, context['stone'])
        np.testing.assert_array_equal(tr.acid_roi_mask, context['roi_mask'])

    def completed_test_for_cleanup(self):
        self.prepare()
        self.window.rub_models.surface.return_value = (self.stone, DEST)
        self.window.rub_models.marks.return_value = self.old
        detector, classifier = Mock(), Mock()
        detector.predict.return_value = ([], 1.)
        self.window.acid_models = Mock(get=Mock(return_value=(detector, classifier)))
        tr = pending_track()
        self.worker.tracks, self.worker.acid_pending_id = [tr], tr.id
        self.worker.acid_token, self.worker.acid_ph_required = 'finished', False
        record = dict(detector_bbox=(160, 160, 200, 200),
                      prediction=dict(prediction='non_gold', confidence=.94),
                      ph_verification=dict(result='SKIPPED_BY_OPERATOR'))
        self.assertTrue(self.worker.finish_acid_test(dict(token='finished', jewel_id=1, result=record)))
        self.window.refresh_acid_controls()
        self.assertTrue(self.worker.stone_cleanup_pending)
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertFalse(self.window.btn_rub_reference.isEnabled())
        self.assertFalse(self.window.btn_start_acid.isEnabled())
        return tr, record, detector, classifier

    def cleanup_tick(self, ts):
        with patch.object(app.time, 'time', return_value=ts):
            self.window.rub_guidance_tick()
            self.window.rub_guidance_tick()

    def test_completion_waits_for_acid_removal_then_captures_next_reference(self):
        tr, record, detector, classifier = self.completed_test_for_cleanup()
        detector.predict.return_value = ([Detection((160, 160, 200, 200), .95)], 1.)
        # Live detector misses do not erase the result or interrupt cleaning.
        self.worker.stone_box.return_value = None
        for i in range(12):
            self.cleanup_tick(100.+i*.25)
            self.assertFalse(self.worker.rub_reference_ready)
        self.assertIn('Acid is still detected', self.window.rub_instruction.text())
        self.assertTrue(self.window.chk_ph_verification.isEnabled())
        self.assertIs(record, tr.acid_result)
        self.assertTrue(tr.acid_done)
        self.window.rub_models.marks.assert_not_called()
        # The manual capture control also cannot bypass the cleaning wait.
        self.window.capture_rub_reference(manual=True)
        self.assertFalse(self.worker.rub_reference_ready)
        detector.predict.return_value = ([], 1.)
        for i in range(8):
            self.cleanup_tick(103.+i*.25)
            self.assertFalse(self.worker.rub_reference_ready)
        self.cleanup_tick(105.)
        self.assertTrue(self.worker.rub_reference_ready)
        self.assertFalse(self.worker.stone_cleanup_pending)
        self.assertIsNotNone(self.window.rub_cycle.before_result)
        self.window.rub_models.marks.assert_called_once()
        self.assertEqual(1, self.window.rub_models.photo.call_count)
        self.assertIs(record, tr.acid_result)
        classifier.predict.assert_not_called()
        # No extra idle acid inference once the next reference is saved.
        calls = detector.predict.call_count
        self.cleanup_tick(106.)
        self.assertEqual(calls, detector.predict.call_count)

    def test_cleanup_keeps_waiting_for_surface_without_a_stone_loss_error(self):
        tr, record, detector, _ = self.completed_test_for_cleanup()
        self.window.rub_models.surface.side_effect = RuntimeError('RETAKE - expose one complete stone')
        for i in range(12):
            self.cleanup_tick(100.+i*.25)
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertIn('Finish cleaning', self.window.acid_instruction.text())
        self.assertNotIn('RETAKE', self.window.rub_instruction.text())
        self.assertIs(record, tr.acid_result)
        detector.predict.assert_not_called()

    def test_reset_discards_inflight_cleanup_reference(self):
        self.completed_test_for_cleanup()
        for i in range(8):
            self.cleanup_tick(100.+i*.25)
        with patch.object(app.time, 'time', return_value=102.):
            self.window.rub_guidance_tick()
        self.assertEqual('READY', self.window.rub_job['future'].result()['status'])
        self.worker.arm()
        self.worker.stone_box.return_value = None
        self.window.rub_guidance_tick()
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertFalse(self.worker.stone_cleanup_pending)
        self.assertIsNone(self.window.rub_cleanup)
        self.assertIsNone(self.window.rub_cycle.before_result)

    def test_stale_camera_does_not_check_acid_or_capture_cleanup_reference(self):
        self.completed_test_for_cleanup()
        self.worker.grabber = SimpleNamespace(latest=lambda: (self.frame.copy(), 90.))
        self.cleanup_tick(100.)
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertIsNone(self.window.rub_job)
        self.assertIn('fresh camera', self.window.acid_instruction.text())
        self.window.rub_models.surface.assert_not_called()

    def prepare_automatic_return(self):
        self.prepare()
        self.window.chk_auto_test.setChecked(True)
        # Use a fixed reference crop and a returned jewel outside that surface.
        self.window.rub_cycle.stone = (220, 220, 470, 470)
        tr = pending_track()
        tr.pos = np.array((80., 80.))
        tr.state = 'OK'
        tr.det_box = (60, 60, 100, 100)
        self.worker.tracks, self.worker.acid_pending_id = [tr], 1
        return tr

    def test_automatic_return_ignores_jewel_and_stone_box_shape_jitter(self):
        tr = self.prepare_automatic_return()
        self.window.start_acid_test = Mock()
        for i in range(16):
            ts = 100.+i*.2
            tr.det_seen = ts
            tr.pos = np.array((80.+i % 3, 80.-i % 3))
            tr.det_box = (40-i, 45-i, 120+i, 115+i)
            self.worker.stone_box.return_value = (210-i, 210-i, 470+i, 470+i)
            with patch.object(app.time, 'time', return_value=ts):
                self.window.auto_test_tick()
            if i < 10:
                self.window.start_acid_test.assert_not_called()
        self.window.start_acid_test.assert_called_once_with(automatic=True)
        self.assertEqual((220., 220., 470., 470.), self.window.auto_gate.stone)

    def test_another_visible_jewel_cannot_satisfy_the_pending_jewels_return(self):
        tr = self.prepare_automatic_return()
        tr.det_seen = 0.
        other = pending_track()
        other.id, other.pos, other.det_seen, other.state = 2, np.array((100., 80.)), 101., 'OK'
        self.worker.tracks.append(other)
        self.window.start_acid_test = Mock()
        for i in range(15):
            ts = 100.+i*.2
            other.det_seen = ts
            with patch.object(app.time, 'time', return_value=ts):
                self.window.auto_test_tick()
        self.window.start_acid_test.assert_not_called()

    def test_automatic_start_checks_new_rub_then_preserves_ph_first_setting(self):
        tr = self.prepare_automatic_return()
        after, marks = self.frame.copy(), self.old.copy()
        cv2.circle(after, (400, 400), 12, (220, 220, 220), -1)
        cv2.circle(marks, (400, 400), 12, 255, -1)
        self.worker.grabber = SimpleNamespace(latest=lambda: (after.copy(), time.time()))
        self.window.rub_models.photo.return_value = (self.stone, marks, DEST)
        self.window.launch_acid_test = Mock()
        for i in range(11):
            ts = 100.+i*.2
            tr.det_seen = ts
            with patch.object(app.time, 'time', return_value=ts):
                self.window.auto_test_tick()
        self.assertIsNotNone(self.window.rub_job)
        self.window.launch_acid_test.assert_not_called()
        self.assertTrue(self.window.rub_job['ph_enabled'])
        self.assertTrue(self.window.rub_job['auto_test'])
        self.window.rub_guidance_tick()
        self.assertIsNotNone(tr.new_rub_target)
        args, kwargs = self.window.launch_acid_test.call_args
        self.assertTrue(args[0])
        self.assertTrue(kwargs['auto_test'])

    def test_cancel_holds_automatic_start_until_explicit_resume(self):
        tr = self.prepare_automatic_return()
        self.window.start_acid_test = Mock()
        self.window.cancel_acid_test()
        for i in range(15):
            ts = 100.+i*.2
            tr.det_seen = ts
            with patch.object(app.time, 'time', return_value=ts):
                self.window.auto_test_tick()
        self.window.start_acid_test.assert_not_called()
        self.window.chk_auto_test.setChecked(False)
        self.window.chk_auto_test.setChecked(True)
        for i in range(11):
            ts = 104.+i*.2
            tr.det_seen = ts
            with patch.object(app.time, 'time', return_value=ts):
                self.window.auto_test_tick()
        self.window.start_acid_test.assert_called_once_with(automatic=True)

    def test_retake_does_not_launch_ph_or_acid(self):
        self.prepare()
        tr = pending_track()
        self.worker.tracks, self.worker.acid_pending_id = [tr], 1
        self.window.launch_acid_test = Mock()
        self.window.start_acid_test()
        self.window.rub_guidance_tick()
        self.assertIsNone(tr.new_rub_target)
        self.assertIsNone(self.worker.acid_token)
        self.window.launch_acid_test.assert_not_called()
        self.assertIn('RETAKE', self.window.rub_instruction.text())
        self.assertIn('No new rub detected', self.window.rub_instruction.text())
        self.assertTrue(self.window.btn_start_acid.isEnabled())

    def test_cleaning_prompt_blocks_rubbing_until_clean_stone_reference_is_retaken(self):
        self.prepare()
        self.window.rub_models.photo.return_value = (self.stone, self.stone, DEST)
        self.window.capture_rub_reference(manual=True)
        self.window.rub_guidance_tick()
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertIsNone(self.window.rub_cycle.before_result)
        self.assertIn('CLEAN THE STONE', self.window.rub_instruction.text())
        self.assertIn('CLEAN THE STONE', self.window.acid_instruction.text())
        self.assertFalse(self.window.btn_start_acid.isEnabled())
        self.assertTrue(self.window.btn_rub_reference.isEnabled())
        for enabled in (True, False):
            self.window.chk_existing_rub_marks.setChecked(enabled)
            self.assertIn('CLEAN THE STONE', self.window.rub_instruction.text())
            self.assertFalse(self.worker.rub_reference_ready)
        # Live movement must not receive RUBBED credit while cleaning is required.
        tr = pending_track()
        tr.rubbed = False
        self.worker.tracks, self.worker.rub_track = [tr], tr
        self.worker.rubber = SimpleNamespace(snapshot=lambda: ({}, True, 100.), active=True)
        self.worker.update_rubbing(100.)
        self.assertFalse(tr.rubbed)
        self.assertIsNone(self.worker.acid_pending_id)
        # Cleaning and a manual retake restore readiness without choosing a zone.
        self.window.rub_models.photo.return_value = (self.stone, self.old, DEST)
        self.window.capture_rub_reference(manual=True)
        self.window.rub_guidance_tick()
        self.assertTrue(self.worker.rub_reference_ready)
        self.assertIsNotNone(self.window.rub_cycle.before_result)
        self.assertNotIn('CLEAN THE STONE', self.window.acid_instruction.text())
        self.assertNotIn('zone', self.window.rub_instruction.text())

    def test_failed_manual_retake_retains_saved_reference_and_keeps_credit_blocked(self):
        self.prepare()
        cycle = self.window.rub_cycle
        before, old = cycle.before_result, cycle.check.old_mask.copy()
        self.window.rub_models.photo.side_effect = RuntimeError('stone surface is temporarily unclear')
        self.window.capture_rub_reference(manual=True)
        self.window.rub_guidance_tick()
        self.assertIs(cycle, self.window.rub_cycle)
        self.assertIs(before, cycle.before_result)
        np.testing.assert_array_equal(old, cycle.check.old_mask)
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertIn('temporarily unclear', self.window.rub_instruction.text())

    def test_existing_mark_toggle_redraws_without_recapture_or_changing_reference(self):
        self.prepare()
        self.assertFalse(self.window.chk_existing_rub_marks.isChecked())
        hidden = self.window.rub_preview.pixmap().toImage()
        old = self.window.rub_cycle.check.old_mask.copy()
        self.window.chk_existing_rub_marks.setChecked(True)
        self.assertIn('(ON)', self.window.chk_existing_rub_marks.text())
        self.assertNotEqual(hidden, self.window.rub_preview.pixmap().toImage())
        self.window.rub_instruction.setText('No new rub detected')
        self.window.chk_existing_rub_marks.setChecked(False)
        self.assertIn('(OFF)', self.window.chk_existing_rub_marks.text())
        self.assertEqual(hidden, self.window.rub_preview.pixmap().toImage())
        self.assertEqual('No new rub detected', self.window.rub_instruction.text())
        np.testing.assert_array_equal(old, self.window.rub_cycle.check.old_mask)
        self.assertTrue(self.worker.rub_reference_ready)
        self.assertEqual(1, self.window.rub_models.photo.call_count)

    def test_reset_clears_cached_mark_display_even_if_toggle_is_on(self):
        self.prepare()
        self.window.chk_existing_rub_marks.setChecked(True)
        self.worker.arm()
        self.window.sync_rub_cycle()
        self.window.chk_existing_rub_marks.setChecked(False)
        self.assertIsNone(self.window.rub_display_result)
        self.assertIsNone(self.window.rub_preview.pixmap())

    def test_stale_reference_result_cannot_enable_a_reset_batch(self):
        self.window.rub_guidance_tick()
        self.worker.arm()
        self.worker.stone_box.return_value = None
        self.window.rub_guidance_tick()
        self.assertFalse(self.worker.rub_reference_ready)
        self.assertIsNone(self.window.rub_cycle.before_result)

    def test_acid_cannot_start_without_a_confirmed_after_mask(self):
        tr = pending_track()
        self.worker.tracks, self.worker.acid_pending_id = [tr], 1
        with self.assertRaisesRegex(RuntimeError, 'new rubbed mask'):
            self.worker.begin_acid_test(False)
        self.assertIsNone(self.worker.acid_token)

    def test_reference_is_required_before_rubbing_credit(self):
        tr = pending_track()
        tr.rubbed = False
        self.worker.tracks, self.worker.rub_track = [tr], tr
        ev = dict(mask=np.ones((20,20),np.uint8), origin=(50,50), box=(50,50,70,70), stone=(0,0,512,512))
        self.worker.rubber = SimpleNamespace(snapshot=lambda:(ev,True,100.),active=True)
        self.worker.rub_spot_image = Mock(return_value=None)
        self.worker.update_rubbing(100.)
        self.assertFalse(tr.rubbed)
        self.worker.rub_reference_ready = True
        self.worker.update_rubbing(100.)
        self.assertTrue(tr.rubbed)


class TargetReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QtWidgets.QApplication.instance() or app.QtWidgets.QApplication([])

    def test_alignment_retry_keeps_ph_and_publishes_fresh_roi(self):
        frame = np.full((100,160,3), 80, np.uint8)
        mask = np.zeros(frame.shape[:2],np.uint8)
        mask[30:60,40:70] = 255
        located = dict(roi_mask=mask,stone=(10,10,150,90),streak=(40,30,70,60))
        target = Mock()
        target.locate.side_effect = [RuntimeError('RETAKE - alignment unclear'), located]
        context = dict(token='ready',jewel_id=1,stone=(10,10,150,90),streak=(40,30,70,60),
                       grabber=SimpleNamespace(latest=lambda:(frame,time.time())),
                       stone_getter=lambda:(10,10,150,90),rub_target=target)
        thread = AcidTestThread(context)
        proof = dict(result='ACIDIC_REACTION_VERIFIED')
        thread.ph_verification, thread.ph_frame_shape = proof, frame.shape
        thread.progress.connect(lambda _:thread.confirm_stone_ready())
        results = []
        thread.target_ready.connect(results.append)
        self.assertTrue(thread.wait_for_stone_ready())
        self.assertIs(proof, thread.ph_verification)
        self.assertEqual(2, target.locate.call_count)
        self.assertEqual(1, len(results))
        np.testing.assert_array_equal(mask, thread.context['roi_mask'])

    def test_archive_contains_confirmed_mask_and_before_after_evidence(self):
        frame = np.full((100,160,3),80,np.uint8)
        mask = np.zeros(frame.shape[:2],np.uint8)
        mask[30:60,40:70] = 255
        seq = AcidSequence((10,10,150,90),(40,30,70,60),.1,baseline_seconds=0,
                           clock=lambda:1.,streak_mask=mask)
        seq.push(frame,0.)
        for i in range(3):
            seq.accept_detections([Detection((42,32,68,58),.95)],.1*i)
        seq.push(frame,1.2)
        target = SimpleNamespace(metadata=dict(selected_zone='Z1'), evidence=dict(before_rubbing=frame, new_rub_mask=mask))
        context = dict(token='save',jewel_id=1,batch_id='test',rub_target=target,ph_verification_enabled=False)
        thread = AcidTestThread(context)
        thread.ph_verification = dict(result='SKIPPED_BY_OPERATOR')
        thread.models.detector_sha256 = 'test-hash'
        prediction = dict(prediction='non_gold',confidence=.9)
        with tempfile.TemporaryDirectory() as directory, patch('acid_test.integration.LAB_ROOT',Path(directory)):
            result = thread.save_capture(seq,prediction)
            root = Path(result['record_path']).parent
            saved = json.loads(Path(result['record_path']).read_text())
            np.testing.assert_array_equal(cv2.imread(str(root/saved['new_rub_roi']),0),mask)
            self.assertTrue((root/'before_rubbing.png').is_file())
            self.assertFalse(saved['acid_roi_rule']['center_inside_mask'])
            self.assertFalse(saved['acid_roi_rule']['center_inside_streak'])
            self.assertTrue(saved['acid_roi_rule']['center_inside_stone'])
            self.assertEqual('detector_bbox', saved['acid_roi_rule']['region'])
            self.assertEqual(.5, saved['acid_roi_rule']['min_overlap_of_smaller'])

    def test_complete_target_flow_rejects_wrong_acid_then_finishes_with_ph_on_or_off(self):
        for enabled in (True,False):
            with self.subTest(ph_enabled=enabled):
                frame,stone,old=photos()
                models=Mock(photo=Mock(return_value=(stone,old,DEST)))
                cycle=RubCycle(models)
                before=cycle.before(frame)
                x,y=(180,180)
                after,marks=frame.copy(),old.copy()
                cv2.circle(after,(x,y),12,(220,220,220),-1)
                cv2.circle(marks,(x,y),12,255,-1)
                models.photo.return_value=(stone,marks,DEST)
                models.surface.return_value=(stone,DEST)
                _,target=cycle.after(after)
                self.assertEqual(0,target.roi_mask[y+10,x+10])
                wrong=Detection((x+8,y+8,x+12,y+12),.95)
                correct=Detection((x-14,y-14,x+60,y+14),.95)
                self.assertEqual(0,target.roi_mask[y,x+23])
                detector=Mock()
                count=[0]
                def predict(*args):
                    count[0]+=1
                    return ([wrong if count[0]<=3 else correct],1.)
                detector.predict.side_effect=predict
                classifier=Mock()
                classifier.predict.return_value=dict(prediction='non_gold',confidence=.9)
                cache=SimpleNamespace(get=Mock(return_value=(detector,classifier)),detector_sha256='test')
                context=dict(token='run',jewel_id=1,batch_id='test',stone=target.stone,streak=target.streak,
                             roi_mask=target.roi_mask,rub_target=target,ph_verification_enabled=enabled,
                             grabber=SimpleNamespace(latest=lambda:(after.copy(),time.time())),
                             stone_getter=lambda:target.stone)
                thread=AcidTestThread(context,.12,models=cache)
                events,completed,failed=[],[],[]
                def verify(owner):
                    owner.ph_frame_shape=after.shape
                    owner.ph_verification=dict(result='ACIDIC_REACTION_VERIFIED')
                    return True
                def progress(payload):
                    events.append(payload)
                    if payload['phase']=='WAIT_FOR_STONE_READY':thread.confirm_stone_ready()
                thread.progress.connect(progress)
                thread.completed.connect(completed.append)
                thread.failed.connect(failed.append)
                with tempfile.TemporaryDirectory() as directory, \
                     patch('acid_test.integration.LAB_ROOT',Path(directory)), \
                     patch('acid_test.integration.AcidSequence',side_effect=lambda a,b,c:AcidSequence(a,b,.12,.1)), \
                     patch.object(AcidTestThread,'verify_ph_paper',autospec=True,side_effect=verify) as ph:
                    thread.start()
                    deadline=time.monotonic()+6
                    while thread.isRunning() and time.monotonic()<deadline:
                        self.qt.processEvents();time.sleep(.01)
                    thread.cancel()
                    self.assertTrue(thread.wait(2000))
                    self.qt.processEvents()
                    self.assertFalse(failed)
                    self.assertEqual(1,len(completed))
                    self.assertGreaterEqual(count[0],6)
                    self.assertTrue(any('outside' in event['message'] for event in events))
                    self.assertEqual(int(enabled),ph.call_count)
                    root=Path(completed[0]['result']['record_path']).parent
                    self.assertTrue((root/'new_rub_roi.png').is_file())


if __name__ == '__main__':
    unittest.main()
