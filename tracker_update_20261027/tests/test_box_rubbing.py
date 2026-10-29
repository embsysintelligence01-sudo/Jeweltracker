import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jewel_tracker_5 as tracker

STONE = (100, 100, 400, 400)


def make_worker(move_px=300):
    with patch.object(tracker, 'IdentityMatcher', return_value=Mock()), patch.object(tracker, 'JewelIdModel', return_value=Mock()):
        worker = tracker.Worker(0, [(0, 0), (999, 0), (999, 799), (0, 799)])
    det = SimpleNamespace(box=np.array([200., 200., 260., 260.]), c=np.array([230., 230.]), area=3600.0,
                          mask=np.ones((60, 60), np.uint8), origin=(200, 200), app=None)
    worker.tracks = [tracker.Track(1, det, 0.0), tracker.Track(2, det, 0.0)]
    worker.stone_box = Mock(return_value=STONE)
    worker.rubber = SimpleNamespace(active=True, reset=Mock())
    worker.rub_spot_image = Mock(return_value=None)
    worker.rub_move_px = move_px
    return worker


def see(worker, tr, cx, cy, t, size=60):
    """The jewel's box detected this frame, centred on (cx, cy)."""
    box = np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2])
    tr.det_seen = tr.last_seen = t
    tr.det_box = box
    tr.det_mask, tr.det_origin = np.ones((size, size), np.uint8), (int(box[0]), int(box[1]))
    worker.update_box_rubbing(t)


def rub(worker, tr, frames, t0=0.0, dt=0.1, stroke=20, cx=250, cy=250):
    """Back-and-forth strokes of `stroke` px per frame on the stone."""
    t = t0
    for k in range(frames):
        t += dt
        see(worker, tr, cx + (stroke if k % 2 else -stroke) / 2, cy, t)
    return t


class BoxRubbingTests(unittest.TestCase):
    def test_box_moving_on_the_stone_completes_the_rubbing(self):
        w = make_worker(300)
        tr = w.tracks[0]
        rub(w, tr, 10)
        self.assertFalse(tr.rubbed)  # 200 px so far
        rub(w, tr, 10, t0=1.0)
        self.assertTrue(tr.rubbed)
        self.assertEqual(1, w.acid_pending_id)
        self.assertFalse(w.rubber.active)

    def test_rubbed_region_is_where_the_box_moved_on_the_stone(self):
        w = make_worker(100)
        tr = w.tracks[0]
        rub(w, tr, 30, stroke=20, cx=380)  # at the stone's right edge: the box pokes out past it
        self.assertTrue(tr.rubbed)
        (x1, y1, x2, y2), = list(tr.rub_boxes)
        self.assertLessEqual(x2, STONE[2])  # clipped to the stone
        self.assertGreaterEqual(x1, STONE[0])
        self.assertTrue(STONE[0] <= y1 < y2 <= STONE[3])
        self.assertGreater(len(tr.rub_masks), 0)

    def test_jitter_is_not_rubbing(self):
        w = make_worker(100)
        tr = w.tracks[0]
        rub(w, tr, 200, stroke=2)  # a still jewel whose box wobbles by 2 px for 20 s
        self.assertFalse(tr.rubbed)
        self.assertEqual(0.0, tr.box_rub_px)

    def test_box_jumps_are_not_rubbing(self):
        w = make_worker(100)
        tr = w.tracks[0]
        rub(w, tr, 30, stroke=400)
        self.assertFalse(tr.rubbed)

    def test_moving_off_the_stone_is_not_rubbing(self):
        w = make_worker(100)
        tr = w.tracks[0]
        rub(w, tr, 60, stroke=20, cx=700, cy=600)  # lots of movement, but on the bench
        self.assertFalse(tr.rubbed)

    def test_carried_across_the_stone_is_not_rubbing(self):
        w = make_worker(100)
        tr = w.tracks[0]
        t = 0.0
        for k in range(8):  # 0.8 s on the stone, moving fast: less than the dwell time
            t += 0.1
            see(w, tr, 120 + 40 * k, 250, t)
        self.assertFalse(tr.rubbed)

    def test_slider_sets_how_much_movement_is_needed(self):
        for move_px, expect in ((100, True), (1500, False)):
            w = make_worker(move_px)
            tr = w.tracks[0]
            rub(w, tr, 30)  # 600 px of movement
            self.assertEqual(expect, tr.rubbed, move_px)

    def test_leaving_the_stone_for_too_long_starts_the_count_again(self):
        w = make_worker(500)
        tr = w.tracks[0]
        t = rub(w, tr, 15)  # 300 px
        self.assertGreater(tr.box_rub_px, 0)
        for k in range(30):  # lifted off for 3 s
            t += 0.1
            see(w, tr, 700, 600, t)
        self.assertEqual(0.0, tr.box_rub_px)
        self.assertIsNone(tr.stone_since)

    def test_a_short_hand_cover_keeps_the_count(self):
        w = make_worker(500)
        tr = w.tracks[0]
        t = rub(w, tr, 15)
        before = tr.box_rub_px
        w.update_box_rubbing(t + 1.0)  # not detected for 1 s (hand over it)
        t = rub(w, tr, 5, t0=t + 1.1)
        self.assertGreater(tr.box_rub_px, before)

    def test_only_one_jewel_owns_the_next_acid_test(self):
        w = make_worker(100)
        a, b = w.tracks
        t = rub(w, a, 30)
        self.assertTrue(a.rubbed)
        rub(w, b, 30, t0=t)
        self.assertFalse(b.rubbed)
        self.assertEqual(1, w.acid_pending_id)


class SliderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])

    def test_slider_updates_the_worker_and_label_and_is_remembered(self):
        settings = os.path.join(tempfile.mkdtemp(), 'ui_settings.json')
        with patch.object(tracker.QtCore.QTimer, 'singleShot'), patch.object(tracker, 'write_log'),                 patch.object(tracker, 'UI_SETTINGS_FILE', settings):
            window = tracker.MainWindow()
            self.assertEqual(tracker.RUB_BOX_MOVE_PX, window.rub_slider.value())
            window.worker = SimpleNamespace(rub_move_px=0)
            window.rub_slider.setValue(450)
            self.assertEqual(450, window.worker.rub_move_px)
            self.assertIn('450', window.lbl_rub_move.text())
            window.worker = None
            window.close()
            self.assertEqual(450, tracker.MainWindow.load_rub_move(window))  # read back from the settings file


if __name__ == '__main__':
    unittest.main()
