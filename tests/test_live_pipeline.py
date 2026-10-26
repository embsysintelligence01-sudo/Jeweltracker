import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import jewel_tracker_5 as tracker
from live_pipeline import InferenceDispatcher, StaleInference, LatestPreview


class ScheduleTests(unittest.TestCase):
    def test_one_inference_owner_prioritizes_bench_over_pending_stone(self):
        entered, release = threading.Event(), threading.Event()
        calls, owners = [], []
        def predict(source, **kwargs):
            calls.append(source)
            owners.append(threading.get_ident())
            if source == 'first':
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test did not release inference')
            return source
        dispatcher = InferenceDispatcher(predict)
        self.addCleanup(dispatcher.close)
        first = dispatcher.submit('stone', 'first', {})
        self.assertTrue(entered.wait(2))
        stone = dispatcher.submit('stone', 'second stone', {})
        bench = dispatcher.submit('bench', 'bench', {})
        self.assertEqual(2, len(dispatcher.pending))
        release.set()
        self.assertEqual('first', first.result(2))
        self.assertEqual('bench', bench.result(2))
        self.assertEqual('second stone', stone.result(2))
        self.assertEqual(['first', 'bench', 'second stone'], calls)
        self.assertEqual(1, len(set(owners)))
        self.assertNotEqual(threading.get_ident(), owners[0])

    def test_expired_work_is_dropped_without_running_the_model(self):
        predict = Mock()
        dispatcher = InferenceDispatcher(predict)
        self.addCleanup(dispatcher.close)
        expired = dispatcher.submit('bench', 'old frame', {}, time.monotonic()-1)
        with self.assertRaises(StaleInference):
            expired.result(2)
        predict.assert_not_called()
        dispatcher.close()
        self.assertFalse(dispatcher.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            dispatcher.submit('bench', 'new frame', {})

    def test_model_failure_is_returned_and_the_next_job_can_still_run(self):
        predict = Mock(side_effect=[ValueError('bad input'), 'ok'])
        dispatcher = InferenceDispatcher(predict)
        self.addCleanup(dispatcher.close)
        with self.assertRaisesRegex(ValueError, 'bad input'):
            dispatcher.submit('stone', None, {}).result(2)
        self.assertEqual('ok', dispatcher.submit('bench', None, {}).result(2))


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])

    def test_fresh_camera_pixels_are_used_and_old_or_wrong_size_overlays_expire(self):
        preview = LatestPreview()
        overlay = np.zeros((10, 10, 3), np.uint8)
        overlay[1, 1] = (0, 255, 0)
        preview.publish(overlay, 10)
        fresh = np.full_like(overlay, 70)
        result = preview.render(fresh.copy(), 10.1)
        np.testing.assert_array_equal(result[1, 1], (0, 255, 0))
        np.testing.assert_array_equal(result[5, 5], (70, 70, 70))
        np.testing.assert_array_equal(preview.render(fresh.copy(), 11), fresh)
        small = fresh[:5, :5].copy()
        np.testing.assert_array_equal(preview.render(small.copy(), 10.1), small)
        preview.clear()
        np.testing.assert_array_equal(preview.render(fresh.copy(), 10.1), fresh)

    def test_worker_overlay_survives_slow_inference_until_observation_expires(self):
        with patch.object(tracker, 'JewelIdModel', return_value=None):
            worker = tracker.Worker(0, [(0, 0), (9, 0), (9, 9)])
        self.assertEqual(tracker.LIVE_RESULT_MAX_AGE_S, worker.preview.max_age)
        overlay = np.zeros((10, 10, 3), np.uint8)
        overlay[1, 1] = (0, 255, 0)
        worker.preview.publish(overlay, 10.)
        frame = np.full_like(overlay, 70)
        # 200 ms inference plus the next 200 ms observation gap remains visible.
        np.testing.assert_array_equal(worker.preview.render(frame.copy(), 10.4)[1, 1], (0, 255, 0))
        np.testing.assert_array_equal(worker.preview.render(frame.copy(), 10.51), frame)
        worker.arm()
        np.testing.assert_array_equal(worker.preview.render(frame.copy(), 10.41), frame)

    def test_gui_preview_does_not_wait_for_tracking_or_model_locks(self):
        with patch.object(tracker.QtCore.QTimer, 'singleShot'), patch.object(tracker, 'write_log'):
            window = tracker.MainWindow()
        model_lock, track_lock = threading.Lock(), threading.Lock()
        model_lock.acquire()
        track_lock.acquire()
        try:
            frame = np.full((100, 160, 3), 80, np.uint8)
            grabber = SimpleNamespace(latest=lambda **kwargs: (frame.copy(), time.time()))
            window.worker = SimpleNamespace(grabber=grabber, preview=LatestPreview(),
                                            lock=track_lock, model=SimpleNamespace(lock=model_lock))
            window.video.set_frame = Mock()
            window.refresh_preview()
            window.refresh_preview()
            self.assertEqual(2, window.video.set_frame.call_count)
        finally:
            model_lock.release()
            track_lock.release()
            window.worker = None
            window.close()

    def test_downscaled_preview_keeps_original_camera_coordinates(self):
        label = tracker.VideoLabel()
        label.set_frame(tracker.to_qimage(np.zeros((90, 160, 3), np.uint8)),
                        source_size=(2560, 1440))
        self.assertEqual((2560, 1440), label.frame_size)
        clicks = []
        label.clicked.connect(lambda x, y, left: clicks.append((x, y, left)))
        event = tracker.QtGui.QMouseEvent(tracker.QtCore.QEvent.MouseButtonPress,
                                         tracker.QtCore.QPointF(label.width()/2, label.height()/2),
                                         tracker.QtCore.Qt.LeftButton, tracker.QtCore.Qt.LeftButton,
                                         tracker.QtCore.Qt.NoModifier)
        label.mousePressEvent(event)
        self.assertEqual([(1280, 720, True)], clicks)

    def test_grabber_does_not_copy_a_frame_that_was_already_consumed(self):
        grabber = tracker.RTSPGrabber(0)
        grabber.frame, grabber.ts = Mock(), 10
        self.assertEqual((None, 10), grabber.latest(after_ts=10))
        grabber.frame.copy.assert_not_called()
        frame, ts = grabber.latest(after_ts=9)
        self.assertIs(frame, grabber.frame.copy.return_value)
        self.assertEqual(10, ts)


class GuardingTests(unittest.TestCase):
    def setUp(self):
        with patch.object(tracker, 'JewelIdModel', return_value=None):
            self.worker = tracker.Worker(0, [(0, 0), (399, 0), (399, 299), (0, 299)])
        self.worker.mode = 'LOCKED'
        self.worker.stone_box = Mock(return_value=None)
        self.events = []
        self.worker.event.connect(lambda msg, level: self.events.append((msg, level)))

    @staticmethod
    def detection(x=80):
        return tracker.Det(np.array([x-10, 60, x+10, 80]), np.array([float(x), 70.]),
                           400., .95, mask=np.ones((20, 20), np.uint8), origin=(x-10, 60))

    def test_jewel_is_followed_outside_roi_and_keeps_id_when_returning(self):
        tr = tracker.Track(1, self.detection(360), 100)
        self.worker.tracks = [tr]
        frame = np.full((300, 500, 3), 180, np.uint8)
        for k, x in enumerate([375, 390] + [410] * (tracker.OUT_FRAMES + 3), 1):
            self.worker.update_tracks([self.detection(x)], 100 + k * .15, frame)
        self.assertEqual('LEFT_ROI', tr.state)
        self.assertTrue(any('left the ROI' in msg for msg, _ in self.events))
        self.worker.update_tracks([self.detection(300)], tr.last_seen + .15, frame)
        self.assertEqual('OK', tr.state)
        self.assertEqual(1, tr.id)
        self.assertTrue(any('back on the bench' in msg for msg, _ in self.events))

    def test_removed_jewel_alert_and_return_are_preserved_during_rubbing(self):
        tr = tracker.Track(1, self.detection(), 100)
        tr.spot, tr.removed_since = 'removed', 100.5
        self.worker.tracks = [tr]
        self.worker.still_since = 100.5
        self.worker.rubbing_now = True  # another jewel can be rubbed
        self.worker.evaluate(tr, False, 103)
        self.assertEqual('MISSING', tr.state)
        self.assertTrue(any('removed from the bench' in msg for msg, _ in self.events))
        tr.correct(self.detection(), 103.2)
        self.worker.evaluate(tr, True, 103.2)
        self.assertEqual('OK', tr.state)
        self.assertEqual(1, tr.id)

    def test_hand_occlusion_and_recent_rubbing_do_not_raise_missing(self):
        tr = tracker.Track(1, self.detection(), 100)
        tr.spot = 'hand'
        self.worker.still_since = 100
        self.worker.evaluate(tr, False, 104)
        self.assertEqual('OCCLUDED', tr.state)
        tr.rub_last = 104
        self.worker.evaluate(tr, False, 104.2)
        self.assertEqual('RUBBING', tr.state)
        self.assertFalse(self.events)

    def test_reset_invalidates_pending_results_and_preview(self):
        revision = self.worker.flow_revision
        self.worker.preview.publish(np.ones((10, 10, 3), np.uint8), 10)
        self.worker.arm()
        self.assertGreater(self.worker.flow_revision, revision)
        self.assertIsNone(self.worker.preview.overlay)


class WorkerPipelineTests(unittest.TestCase):
    def test_late_bench_result_cannot_change_tracking_or_a_reset_batch(self):
        for reset in (False, True):
            with self.subTest(reset=reset), patch.object(tracker, 'JewelIdModel', return_value=None):
                worker = tracker.Worker(0, [(0, 0), (159, 0), (159, 99), (0, 99)])
                worker.mode = 'LOCKED'
                clock = [100.]
                calls = [0]
                frame = np.full((100, 160, 3), 80, np.uint8)
                def latest(**kwargs):
                    calls[0] += 1
                    if calls[0] > 1:
                        worker.running = False
                        return None, clock[0]
                    return frame, clock[0]
                def detect(*args, **kwargs):
                    if reset:
                        worker.arm()
                    else:
                        clock[0] += tracker.LIVE_RESULT_MAX_AGE_S + .1
                    return [], []
                grabber = Mock()
                grabber.latest.side_effect = latest
                grabber.af_busy_until = 0
                grabber.problem = ""
                grabber.connected = True
                grabber.is_alive.return_value = False
                stone, rubber = Mock(), Mock(active=False)
                stone.is_alive.return_value = rubber.is_alive.return_value = False
                worker.detect = Mock(side_effect=detect)
                worker.update_tracks, worker.try_lock = Mock(), Mock()
                with patch.object(tracker, 'SharedGoldModel'), patch.object(tracker, 'StoneOnnxModel'), \
                     patch.object(tracker, 'RTSPGrabber', return_value=grabber), \
                     patch.object(tracker, 'StoneWatcher', return_value=stone), \
                     patch.object(tracker, 'RubbingObserver', return_value=rubber), \
                     patch.object(tracker.time, 'time', side_effect=lambda: clock[0]):
                    worker.run()
                worker.detect.assert_called_once()
                worker.update_tracks.assert_not_called()
                worker.try_lock.assert_not_called()

    def test_bench_only_updates_on_completed_observations(self):
        with patch.object(tracker, 'JewelIdModel', return_value=None):
            worker = tracker.Worker(0, [(0, 0), (159, 0), (159, 99), (0, 99)])
        clock = [100.]
        jobs = []
        frame = np.full((100, 160, 3), 80, np.uint8)
        def latest(**kwargs):
            clock[0] += .05
            if clock[0] > 103:
                worker.running = False
            return frame.copy(), clock[0]
        grabber = Mock()
        grabber.latest.side_effect = latest
        grabber.af_busy_until = 0
        grabber.is_alive.return_value = False
        stone = Mock()
        stone.is_alive.return_value = False
        rubber = Mock(active=True)
        rubber.is_alive.return_value = False
        def detect(*args, **kwargs):
            jobs.append('bench')
            clock[0] += .06
            return [], []
        def step(*args):
            jobs.append('stone')
            clock[0] += .06
        rubber.step.side_effect = step
        worker.detect = Mock(side_effect=detect)
        worker.mode = 'LOCKED'
        worker.update_tracks = Mock()
        worker.draw = lambda image, *args: image
        with patch.object(tracker, 'SharedGoldModel'), patch.object(tracker, 'StoneOnnxModel'), \
             patch.object(tracker, 'RTSPGrabber', return_value=grabber), \
             patch.object(tracker, 'StoneWatcher', return_value=stone), \
             patch.object(tracker, 'RubbingObserver', return_value=rubber), \
             patch.object(tracker.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(tracker.time, 'monotonic', side_effect=lambda: clock[0]):
            worker.run()
        self.assertIn('bench', jobs)
        self.assertGreater(worker.update_tracks.call_count, 5)
        # Only completed bench observations reach the guarding state machine.
        self.assertLessEqual(worker.update_tracks.call_count, jobs.count('bench'))
        self.assertGreaterEqual(worker.update_tracks.call_count, jobs.count('bench') - 1)
        self.assertGreater(rubber.observe.call_count, 0)



if __name__ == '__main__':
    unittest.main()
