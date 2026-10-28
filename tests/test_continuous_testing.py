import json
from pathlib import Path
import tempfile
import time
import unittest
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from continuous_testing import (ContinuousMonitor, StonePose, RubReference,
                                choose_hand_jewel, visibility, inside_roi, workflow_lines)
from session_evidence import FrameJournal, prediction_text, write_report
from acid_freshness import FreshAcidGate


STONE = (80, 60, 220, 200)
QUAD = np.float32([[80, 60], [219, 60], [219, 199], [80, 199]])


def scene(mark=False, acid=False):
    frame = np.full((300, 400, 3), 220, np.uint8)
    frame[60:200, 80:220] = 65
    if mark:
        frame[110:135, 130:140] = 225
    if acid:
        frame[107:139, 128:143] = (0, 255, 0)
    return frame


def track(jid=1, state='OK', spot='jewel', seen=0, in_hand=False):
    return dict(id=jid, state=state, spot=spot, det_seen=seen, acid_done=False,
                in_hand=in_hand,
                pos=np.array([300., 100.]), thumb=scene()[60:100, 280:320])


class Photos:
    def photo(self, frame):
        stone = np.zeros(frame.shape[:2], np.uint8)
        stone[60:200, 80:220] = 255
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marks = ((gray > 180) & (stone > 0)).astype(np.uint8)*255
        return stone, marks, QUAD.copy()

    def marks(self, frame):
        return self.photo(frame)[1]


class Detector:
    def predict(self, frame, confidence):
        found = (frame[:, :, 1] > 240) & (frame[:, :, 0] < 20)
        ys, xs = np.where(found)
        detections = [] if not len(xs) else [SimpleNamespace(
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)), confidence=.94)]
        return detections, 0.


class EvidenceTests(unittest.TestCase):
    def test_hand_evidence_does_not_treat_absence_as_in_hand(self):
        missed = track(state='MISSING', spot='removed')
        under = track(2, state='OCCLUDED', spot='hand')
        absent = track(3, state='OCCLUDED', spot='removed')
        self.assertEqual('MISSED', visibility(missed, 5, True))
        self.assertEqual('NOT_SEEN', visibility(absent, 5, False))
        self.assertIsNone(choose_hand_jewel([missed, under], 5, False, {}))
        self.assertIsNone(choose_hand_jewel([missed, absent], 5, False, {}))
        self.assertIsNone(choose_hand_jewel([under, track(4, 'OCCLUDED', 'hand')], 5, True, {}))
        held = dict(track(seen=5), in_hand=True)
        self.assertEqual('IN_HAND', visibility(held, 5, False))
        self.assertEqual(1, choose_hand_jewel([under, held], 5, False, {}))

    def test_recent_hand_association_has_two_second_grace(self):
        self.assertEqual(1, choose_hand_jewel([track(seen=10)], 10, False, {1: 8.1}))
        self.assertIsNone(choose_hand_jewel([track(seen=10.2)], 10.2, False, {1: 8.1}))

    def test_corner_pose_retains_mask_position_with_uncertain_visibility(self):
        pose = StonePose()
        self.assertTrue(pose.update(scene(), 0, STONE))
        old = pose.quad.copy()
        covered = scene()
        covered[60:200, 80:220] = 180
        self.assertFalse(pose.update(covered, .2, STONE))
        self.assertTrue(pose.uncertain)
        np.testing.assert_array_equal(old, pose.quad)
        self.assertTrue(inside_roi(old, np.float32([[60,40],[240,40],[240,220],[60,220]])))

    def test_new_mark_excludes_reference_marks_and_unchanged_frame(self):
        models = Photos()
        ref = RubReference(models, scene(), 1, QUAD)
        self.assertNotEqual('NEW_RUB', ref.after(models, scene(), 2, QUAD)['status'])
        result = ref.after(models, scene(mark=True), 2, QUAD)
        self.assertEqual('NEW_RUB', result['status'])
        self.assertGreater(np.count_nonzero(result['roi_mask']), 0)
        ref = RubReference(models, scene(mark=True), 1, QUAD)
        self.assertNotEqual('NEW_RUB', ref.after(models, scene(mark=True), 2, QUAD)['status'])

    def test_journal_preserves_source_timestamps_and_writes_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = FrameJournal(tmp)
            for i in range(8):
                journal.offer(scene(), 100+i*.1)
            journal.close()
            clip = journal.clip(100.1, 100.6, Path(tmp)/'clip.avi')
            self.assertTrue(Path(clip['path']).is_file())
            self.assertGreater(clip['frames'], 3)
            self.assertEqual(clip['timestamps'], sorted(clip['timestamps']))
            saved = json.loads((Path(tmp)/'clip.json').read_text())
            self.assertEqual(100.1, saved['requested_start'])
            self.assertAlmostEqual(.5, saved['playback_seconds'], places=3)

    def test_report_keeps_model_result_despite_missing_tracking_and_mask(self):
        prediction = dict(prediction='gold_22k', confidence=.97, probabilities={'gold_22k': .97})
        record = dict(jewel_id=1, prediction=prediction, rubbing_status='UNCONFIRMED',
                      mask_status='UNCONFIRMED', mask_reason='Hand covered the stone',
                      identity_status='MISSED_DURING_PROCESS', issues=['Tracking missed for 2 seconds'])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(tmp, [dict(id=1)], [record])
            self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))
            saved = json.loads((Path(tmp)/'session.json').read_text())
            self.assertEqual(prediction, saved['tests'][0]['prediction'])
            self.assertEqual('22K GOLD (97.0%)', prediction_text(prediction))
            self.assertNotIn('risk', saved['tests'][0])
            self.assertEqual('INVALID (acid model) (70.0%)', prediction_text(dict(prediction='invalid', confidence=.7)))


class AutomaticFlowTests(unittest.TestCase):
    def test_covering_and_uncovering_alone_never_starts_a_jewel_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[]
            monitor=ContinuousMonitor(tmp,lambda:None,Photos(),Mock(),lambda kind,**kw:events.append((kind,kw)))
            try:
                for i in range(25):
                    ts=100+i*.1
                    covered=2<=i<18
                    image=scene(mark=i>=18)
                    monitor.journal.offer(image,ts)
                    monitor.step(dict(frame=image,timestamp=ts,mode='LOCKED',revision=0,
                        stone=STONE,placement=None,bench_moving=covered,rubbing=covered,
                        tracks=[track(1,'OCCLUDED' if covered else 'OK','hand' if covered else 'jewel',ts)]))
                self.assertIsNone(monitor.active)
                self.assertEqual({},monitor.recent_hand)
                self.assertFalse(any(kind in ('begin','mask','captured') for kind,_ in events))
                self.assertEqual([],list(Path(tmp).glob('test_*')))
                monitor.journal.queue.join()
                self.assertGreater(len(monitor.journal.entries),0)
                self.assertTrue(any(v['status']=='UNDER_HAND' for v in monitor.jewels[1]['visibility']))
            finally:
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_actual_pickup_starts_cycle_and_later_coverage_does_not_end_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[]
            monitor=ContinuousMonitor(tmp,lambda:None,Photos(),Mock(),lambda kind,**kw:events.append((kind,kw)))
            try:
                for ts,held in [(100,True),(101,False),(102,False)]:
                    monitor.step(dict(frame=scene(),timestamp=ts,mode='LOCKED',revision=0,
                        stone=STONE,placement=None,bench_moving=True,rubbing=False,
                        tracks=[track(1,'OK' if held else 'OCCLUDED','jewel' if held else 'hand',ts,in_hand=held),
                                track(2,'OCCLUDED','hand',ts)]))
                self.assertEqual(1,monitor.active['jewel_id'])
                self.assertEqual(100,monitor.recent_hand[1])
                self.assertNotIn(2,monitor.recent_hand)
                self.assertEqual(1,sum(kind=='begin' for kind,_ in events))
                self.assertFalse(any(kind=='captured' for kind,_ in events))
            finally:
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_auto_position_includes_full_stone_during_setup_but_never_moves_in_test(self):
        explicit = np.float32([[50,30],[250,30],[250,180],[50,180]])
        for mode, configured, expected_inside in [('ARMING', None, True),
                                                   ('LOCKED', None, False),
                                                   ('ARMING', explicit, False)]:
            with self.subTest(mode=mode, explicit=configured is not None), tempfile.TemporaryDirectory() as tmp:
                monitor = ContinuousMonitor(tmp, lambda: None, Photos(), Mock(), lambda *a, **kw: None)
                try:
                    partial = scene()
                    partial[160:200, 80:220] = 220
                    for timestamp, frame, stone in [(100, partial, (80,60,220,160)),
                                                     (101, scene(), STONE)]:
                        monitor.step(dict(frame=frame, timestamp=timestamp, mode=mode, revision=0,
                            stone=stone, placement=configured, bench_moving=False, rubbing=False, tracks=[]))
                    self.assertEqual(expected_inside, inside_roi(QUAD, monitor.placement))
                    if configured is not None:
                        np.testing.assert_array_equal(configured, monitor.placement)
                finally:
                    monitor.pool.shutdown(wait=True)
                    monitor.detector_pool.shutdown(wait=True)
                    monitor.finish_pool.shutdown(wait=True)
                    monitor.journal.close()

    def test_prepared_mask_must_persist_after_the_held_jewel_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = ContinuousMonitor(tmp, lambda: None, Photos(), Mock(), lambda *a, **k: None)
            try:
                ref = RubReference(Photos(), scene(), 1, QUAD)
                candidate = ref.after(Photos(), scene(mark=True), 2, QUAD)
                monitor.pose.quad = QUAD
                active = dict(mask_candidate=candidate, mask=None, returned=3, acid_onset=None,
                              jewel_id=1, id='test')
                monitor._confirm_candidate(active, scene(), 3)
                self.assertIsNone(active['mask'])
                self.assertIsNone(active['mask_candidate'])
                active['mask_candidate'] = candidate
                monitor._confirm_candidate(active, scene(mark=True), 3.1)
                self.assertEqual('CONFIRMED_LIVE', active['mask_status'])
            finally:
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_one_positive_source_frame_triggers_without_three_confirmations(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = ContinuousMonitor(tmp, lambda: None, Photos(), Mock(), lambda *a, **k: None)
            try:
                monitor.last_ts = 101
                monitor.active = dict(id='test', jewel_id=1, returned=100, baseline_at=100,
                                      acid_onset=None, mask=None, started=99,
                                      fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD))
                monitor.acid_context = (scene(acid=True), QUAD, True, True)
                future = Future()
                future.set_result((100.2, 'test', [dict(bbox=(130,110,145,140), confidence=.9)]))
                monitor.acid_job = future
                monitor.step(dict(timestamp=101, frame=scene(), mode='LOCKED'))
                self.assertEqual(100.2, monitor.active['acid_onset'])
            finally:
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_before_stone_is_saved_before_ids_are_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            monitor = ContinuousMonitor(tmp, lambda: None, Photos(), Mock(), lambda kind, **kw: events.append((kind, kw)))
            monitor.detector = Detector()
            try:
                for i in range(30):
                    ts = 100+i*.1
                    monitor.step(dict(frame=scene(), timestamp=ts, mode='ARMING', revision=0,
                                      stone=STONE, placement=None, bench_moving=False, rubbing=False, tracks=[]))
                    time.sleep(.003)
                if monitor.reference_job is not None:
                    monitor.reference_job.result(timeout=5)
                monitor.step(dict(frame=scene(), timestamp=103., mode='ARMING', revision=0,
                                  stone=STONE, placement=None, bench_moving=False, rubbing=False, tracks=[]))
                self.assertIsNotNone(monitor.reference)
                self.assertTrue(Path(monitor.reference.image_path).is_file())
                self.assertTrue(any(kind == 'reference' for kind, _ in events))
                self.assertEqual([], monitor.tests)
            finally:
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_no_acid_keeps_original_session_active_past_timeouts_and_next_jewel(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            monitor = ContinuousMonitor(tmp, lambda: None, Photos(), Mock(), lambda kind, **kw: events.append((kind, kw)))
            monitor.detector = Detector()
            try:
                for i in range(170):
                    ts = 100+i*.5
                    held1, held2 = 4 <= i < 9, 45 <= i < 52
                    tracks = [track(1, 'OCCLUDED' if held1 else 'OK', 'hand' if held1 else 'jewel', ts, in_hand=held1),
                              track(2, 'OCCLUDED' if held2 else 'OK', 'hand' if held2 else 'jewel', ts, in_hand=held2)]
                    monitor.journal.offer(scene(), ts)
                    monitor.step(dict(frame=scene(), timestamp=ts, mode='LOCKED', revision=0 if i < 80 else 1,
                                      stone=STONE, placement=None, bench_moving=held1 or held2,
                                      rubbing=False, tracks=tracks))
                    time.sleep(.003)
                self.assertIsNotNone(monitor.active)
                self.assertEqual(1, monitor.active['jewel_id'])
                self.assertIsNone(monitor.active['acid_onset'])
                self.assertIsNotNone(monitor.active['baseline_at'])
                self.assertEqual('WAIT_FOR_ACID', monitor.active['phase'])
                self.assertEqual([], monitor.tests)
                self.assertFalse(any(kind == 'captured' for kind, _ in events))
                phases = [payload['phase'] for kind, payload in events if kind == 'phase']
                self.assertLess(phases.index('MASK_CHECK'), phases.index('BASELINE'))
                self.assertLess(phases.index('BASELINE'), phases.index('WAIT_FOR_ACID'))
            finally:
                if monitor.active:
                    monitor.active['stop_search'].set()
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_development_overlay_shows_mask_acid_and_completed_jewels(self):
        lines = workflow_lines(dict(phase='WAIT_FOR_ACID', jewel_id=2, mask_state='NOT DETECTED',
                                    acid_state='NOT DETECTED - session active'), [1])
        self.assertIn('J2', lines[0])
        self.assertIn('WAITING FOR ACID', lines[0])
        self.assertIn('NEW RUB: NOT DETECTED', lines[1])
        self.assertEqual('DONE: J1', lines[2])
        uncertain = workflow_lines(dict(phase='REACTION', mask_state='DETECTED', mask_uncertain=True))
        self.assertTrue(any('VISIBILITY UNCERTAIN' in line for line in uncertain))

    def test_live_misses_are_recovered_from_pre_acid_and_reaction_recording(self):
        class ClipOnlyPhotos(Photos):
            def photo(self, frame):
                if threading.current_thread().name.startswith('live-rub-photo') and frame[110,130,0] > 180:
                    raise RuntimeError('Live mask observation missed')
                return super().photo(frame)

        with tempfile.TemporaryDirectory() as tmp:
            classifier = Mock()
            classifier.predict.return_value = dict(prediction='gold_22k', confidence=.97)
            monitor = ContinuousMonitor(tmp, lambda: None, ClipOnlyPhotos(), Mock(), lambda *a, **kw: None)
            monitor.detector, monitor.classifier = Detector(), classifier
            detect = monitor._detect
            def clip_only(frame, ts, stone, cycle_id):
                if threading.current_thread().name.startswith('live-acid-roi'):
                    return ts, cycle_id, []
                return detect(frame, ts, stone, cycle_id)
            monitor._detect = clip_only
            try:
                for i in range(150):
                    ts = 100+i*.1
                    in_hand = 20 <= i < 35
                    current = track(state='OCCLUDED' if in_hand else 'OK',
                                    spot='hand' if in_hand else 'jewel', seen=ts if not in_hand else 101.9, in_hand=in_hand)
                    current['acid_done'] = i >= 140
                    frame = scene(mark=i >= 35, acid=48 <= i < 110)
                    monitor.journal.offer(frame, ts)
                    monitor.step(dict(frame=frame, timestamp=ts, mode='LOCKED', revision=0,
                                      stone=STONE, placement=None, bench_moving=in_hand,
                                      rubbing=False, tracks=[current]))
                    time.sleep(.004)
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()
                self.assertEqual(1, len(monitor.tests))
                record = monitor.tests[0]
                self.assertEqual('RECOVERED_FROM_CLIP', record['acid_status'])
                self.assertEqual('RECOVERED_FROM_CLIP', record['mask_status'])
                self.assertEqual('gold_22k', record['prediction']['prediction'])
                self.assertAlmostEqual(7., record['acid_clip']['requested_end']-record['acid_clip']['requested_start'])
                self.assertGreaterEqual(record['frame_counts']['reaction'], 20)
            finally:
                monitor.pool.shutdown(wait=True, cancel_futures=True)
                monitor.detector_pool.shutdown(wait=True, cancel_futures=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_no_mask_or_live_rubbing_does_not_block_single_hit_acid_and_next_jewel(self):
        with tempfile.TemporaryDirectory() as tmp:
            emitted = []
            classifier = Mock()
            classifier.predict.return_value = dict(prediction='non_gold', confidence=.96,
                                                   probabilities={'non_gold': .96})
            monitor = ContinuousMonitor(tmp, lambda: None, Photos(),
                                        Mock(), lambda kind, **kw: emitted.append((kind, kw)))
            monitor.detector, monitor.classifier = Mock(wraps=Detector()), classifier
            reaction_counts = []
            try:
                for i in range(120):
                    ts = 100+i*.1
                    in_hand = 20 <= i < 35
                    current = track(state='OCCLUDED' if in_hand else 'OK',
                                    spot='hand' if in_hand else 'jewel', seen=ts if not in_hand else 101.9, in_hand=in_hand)
                    current['acid_done'] = i >= 105
                    # Intentionally no visible rub mask, but a visible acid event.
                    frame = scene(acid=i >= 48)
                    monitor.journal.offer(frame, ts)
                    monitor.step(dict(frame=frame, timestamp=ts, mode='LOCKED', revision=0,
                        stone=STONE, placement=None, bench_moving=in_hand,
                        rubbing=False, tracks=[current]))
                    if monitor.active and monitor.active['acid_onset'] is not None:
                        reaction_counts.append(monitor.detector.predict.call_count)
                    time.sleep(.005)
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()
                self.assertEqual(1, len(monitor.tests))
                record = monitor.tests[0]
                self.assertEqual('non_gold', record['prediction']['prediction'])
                self.assertEqual('UNCONFIRMED', record['mask_status'])
                self.assertEqual('UNCONFIRMED', record['rubbing_status'])
                self.assertIn(record['acid_status'], ('DETECTED_LIVE_ONE_HIT', 'RECOVERED_FROM_CLIP'))
                self.assertIsNone(monitor.active)
                self.assertTrue(any(kind == 'captured' for kind, _ in emitted))
                self.assertTrue((Path(tmp)/'report.pdf').is_file())
                self.assertGreater(len(reaction_counts), 20)
                self.assertEqual(1, len(set(reaction_counts)), 'Acid inference must stop throughout reaction capture')
            finally:
                monitor.pool.shutdown(wait=True, cancel_futures=True)
                monitor.detector_pool.shutdown(wait=True, cancel_futures=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()


if __name__ == '__main__':
    unittest.main()
