import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Boxes

import jewel_tracker_5 as app


def polygon(x=80, y=80, width=20):
    return np.array([(x, y), (x+width, y), (x+width, y+20), (x, y+20)], np.float32)


class FullFrameMaskTests(unittest.TestCase):
    def test_one_unmodified_frame_supplies_tracking_and_stone_mask(self):
        with patch.object(app, 'JewelIdModel', return_value=None):
            worker = app.Worker(0, [(40, 40), (220, 40), (220, 220), (40, 220)])
        frame = np.full((300, 400, 3), 230, np.uint8)
        frame[80:101, 80:101] = (0, 150, 200)
        frame[0, 0] = (10, 20, 30)  # outside ROI must reach inference unchanged
        poly = polygon()
        model = Mock()
        model.predict.return_value = [SimpleNamespace(
            boxes=Boxes(torch.tensor([[80., 80., 100., 100., .9, 0.]]), frame.shape[:2]),
            masks=SimpleNamespace(xy=[poly]))]
        kept, _ = worker.detect(model, frame, worker.roi, captured_at=time.time())
        observer = app.RubbingObserver(lambda: (60, 60, 160, 160))
        observer.observe(*worker.gold_observation, frame.shape, time.time(), (60, 60, 160, 160), 0)
        model.predict.assert_called_once()
        self.assertIs(frame, model.predict.call_args.args[0])
        self.assertEqual(1, len(kept))
        self.assertIsNotNone(observer.snapshot()[0])

    def test_mask_appears_on_entry_without_model_call_or_motion(self):
        stone = (60, 60, 160, 160)
        observer = app.RubbingObserver(lambda: stone)
        observer.observe([polygon(200, 80)], [.9], (300, 400, 3), time.time(), stone, 0)
        self.assertIsNone(observer.snapshot()[0])
        observer.observe([polygon()], [.9], (300, 400, 3), time.time(), stone, 0)
        self.assertIsNotNone(observer.snapshot()[0])
        self.assertFalse(observer.snapshot()[1])

    def test_overlap_and_confidence_checks_still_apply(self):
        stone = (60, 60, 160, 160)
        self.assertIsNone(app.rubbing_evidence([polygon()], [.14], (300, 400, 3), stone))
        self.assertIsNotNone(app.rubbing_evidence([polygon()], [.15], (300, 400, 3), stone))
        self.assertIsNone(app.rubbing_evidence([polygon(155)], [.9], (300, 400, 3), stone))
        ev = app.rubbing_evidence([polygon(149)], [.9], (300, 400, 3), stone)
        self.assertIsNotNone(ev)
        self.assertLessEqual(ev['box'][2], stone[2])

    def test_disappearance_clears_unfinished_credit_but_preserves_confirmed_rubbing(self):
        with patch.object(app, 'JewelIdModel', return_value=None):
            worker = app.Worker(0, [(0, 0), (99, 0), (99, 99)])
        worker.stone_box = Mock(return_value=None)
        worker.rubber = app.RubbingObserver(worker.stone_box)
        unfinished = SimpleNamespace(rubbed=False, rub_time=.4, rub_last=100.,
                                     rub_boxes=[(1, 1, 5, 5)], rub_masks=[1], rub_marks=[1])
        confirmed = SimpleNamespace(rubbed=True, rub_time=.7, rub_last=100.)
        worker.tracks = [unfinished, confirmed]
        worker.update_rubbing(100.1)
        self.assertEqual(0., unfinished.rub_time)
        self.assertEqual([], unfinished.rub_marks)
        self.assertEqual([], unfinished.rub_masks)
        self.assertEqual([], unfinished.rub_boxes)
        self.assertEqual(.7, confirmed.rub_time)

    def test_saved_streak_follows_stone_when_it_is_put_down_elsewhere(self):
        mask = np.ones((20, 20), np.uint8)
        tr = SimpleNamespace(rub_marks=[(mask, (100, 100), (40, 40, 240, 240))])
        boxes, masks = app.projected_rub_geometry(tr, (240, 140, 640, 540))
        self.assertEqual([(360, 260, 400, 300)], boxes)
        self.assertEqual((40, 40), masks[0][0].shape)
        self.assertEqual((360, 260), masks[0][1])
        # Projection always starts from the original mask, avoiding cumulative drift.
        boxes, masks = app.projected_rub_geometry(tr, (40, 40, 240, 240))
        self.assertEqual([(100, 100, 120, 120)], boxes)
        self.assertEqual((20, 20), masks[0][0].shape)


class RubbingPhotoTests(unittest.TestCase):
    def test_confirmation_freezes_current_crop_without_masks_or_old_positions(self):
        with patch.object(app, 'JewelIdModel', return_value=None):
            worker = app.Worker(0, [(0, 0), (299, 0), (299, 299), (0, 299)])
        worker.stone_box = Mock(return_value=(60, 60, 240, 240))
        worker.cur_frame = np.random.default_rng(9).integers(
            0, 256, (300, 300, 3), dtype=np.uint8)
        expected = worker.cur_frame[80:180, 80:180].copy()
        tr = SimpleNamespace(id=1, rubbed=False, acid_done=False, rub_time=.6,
                             rub_last=99.9, rub_boxes=deque([(0, 0, 20, 20)]),
                             rub_masks=deque(), rub_marks=deque())
        worker.tracks, worker.rub_track = [tr], tr
        worker.rub_reference_ready = True
        worker.emit_jewel_table = Mock()
        ev = dict(box=(120, 120, 140, 140), origin=(120, 120),
                  mask=np.ones((20, 20), np.uint8), stone=(60, 60, 240, 240))
        worker.rubber = SimpleNamespace(snapshot=lambda: (ev, True, 100.), active=True)

        worker.update_rubbing(100.)

        self.assertTrue(tr.rubbed)
        self.assertEqual(1, worker.acid_pending_id)
        np.testing.assert_array_equal(expected, tr.rub_img)
        self.assertFalse(np.shares_memory(worker.cur_frame, tr.rub_img))
        worker.cur_frame[:] = 0
        worker.stone_box.return_value = (0, 0, 100, 100)
        worker.update_rubbing(103.)
        np.testing.assert_array_equal(expected, tr.rub_img)


class RubbingContinuityTests(unittest.TestCase):
    def setUp(self):
        self.stone = (40, 40, 240, 240)
        self.generation = 0
        self.observer = app.RubbingObserver(lambda: self.stone, lambda: self.generation)
        self.observer.active = True
        self.clock = 100.

    def observe(self, x=100, y=100, captured_stone=None, age=0.):
        self.clock += .1
        captured = self.stone if captured_stone is None else captured_stone
        with patch.object(app.time, 'time', return_value=self.clock+age):
            self.observer.observe([polygon(x, y)], [.9], (700, 900, 3), self.clock, captured, self.generation)

    def moving_history(self):
        for x in [100, 104, 108, 112]:
            self.observe(x)
        self.assertTrue(self.observer.rubbing)

    def test_one_pixel_jitter_does_not_discard_mask_or_count_as_rubbing(self):
        for offset in [0, 1, -1, 1, 0, -1]:
            old = self.stone
            self.stone = (40+offset, 40, 240+offset, 240)
            self.observe(captured_stone=old)
            self.assertIsNotNone(self.observer.ev)
        self.assertFalse(self.observer.rubbing)

    def test_large_inflight_movement_skips_sample_without_resetting_history(self):
        self.moving_history()
        before = list(self.observer.moves)
        old = self.stone
        self.stone = (240, 40, 440, 240)
        self.observe(300, captured_stone=old)
        self.assertEqual(before, list(self.observer.moves))
        self.assertIsNone(self.observer.ev)
        self.assertFalse(self.observer.rubbing)
        self.observe(300)
        self.assertIsNotNone(self.observer.ev)

    def test_old_result_is_skipped_without_erasing_progress(self):
        self.moving_history()
        before = list(self.observer.moves)
        self.observe(age=app.LIVE_RESULT_MAX_AGE_S+.1)
        self.assertEqual(before, list(self.observer.moves))
        self.assertIsNone(self.observer.ev)
        self.assertFalse(self.observer.rubbing)

    def test_no_stone_at_capture_waits_for_a_fresh_observation(self):
        self.observer.observe([polygon()], [.9], (700, 900, 3), time.time(), None, 0)
        self.assertIsNone(self.observer.ev)

    def test_stone_translation_with_jewel_does_not_count_as_rubbing(self):
        for offset in [0, 20, 40, 60, 80]:
            self.stone = (40+offset, 40, 240+offset, 240)
            self.observe(100+offset)
        self.assertFalse(self.observer.rubbing)
        self.assertTrue(all(distance < .01 for _, distance in self.observer.moves))

    def test_jewel_can_rub_while_stone_is_carried(self):
        for index, offset in enumerate([0, 20, 40, 60, 80]):
            self.stone = (40+offset, 40, 240+offset, 240)
            self.observe(100+offset+4*index)
        self.assertTrue(self.observer.rubbing)

    def test_uniform_stone_and_jewel_scale_change_does_not_count_as_rubbing(self):
        for size in [200, 220, 240, 260, 280]:
            self.stone = (40, 40, 40+size, 40+size)
            # Build an evidence centroid at a constant relative position.
            centre = np.array([40+.35*size, 40+.35*size])
            self.clock += .1
            self.observer.judge({'c': centre, 'stone': self.stone}, self.clock)
        self.assertFalse(self.observer.rubbing)

    def test_multiple_gold_misses_preserve_history_but_never_show_old_mask(self):
        self.moving_history()
        before = list(self.observer.moves)
        for _ in range(4):
            self.clock += .1
            with patch.object(app.time, 'time', return_value=self.clock):
                self.observer.observe([], [], (700, 900, 3), self.clock, self.stone, 0)
            self.assertIsNone(self.observer.ev)
            self.assertFalse(self.observer.rubbing)
        self.assertEqual(before, list(self.observer.moves))
        self.observe(140)
        self.assertFalse(self.observer.rubbing)

    def test_disappearing_stone_resets_history(self):
        self.moving_history()
        old = self.stone
        self.stone = None
        self.observe(captured_stone=old)
        self.assertFalse(self.observer.moves)
        self.assertIsNone(self.observer.ev)
        self.assertIsNone(self.observer.prev)

    def test_previous_batch_result_cannot_change_new_batch(self):
        self.moving_history()
        self.generation = 1
        self.observer.reset()
        self.observer.observe([polygon()], [.9], (700, 900, 3), time.time(), self.stone, 0)
        self.assertIsNone(self.observer.ev)
        self.assertFalse(self.observer.moves)


if __name__ == '__main__':
    unittest.main()
