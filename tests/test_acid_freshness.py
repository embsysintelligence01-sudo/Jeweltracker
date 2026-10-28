import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from acid_freshness import AcidHistory, FreshAcidGate, footprint
from acid_test.runtime import classify_probabilities
from continuous_testing import ContinuousMonitor
from rubbing_recovery import recover_contact
from session_evidence import FrameJournal
from test_continuous_testing import scene, track, Photos, Detector, STONE, QUAD

OLD = dict(bbox=(128,107,143,139), confidence=.94)
NEW = dict(bbox=(178,107,193,139), confidence=.94)


def two_patches():
    image = scene(acid=True)
    image[107:139,178:193] = (0,255,0)
    return image


class FreshnessTests(unittest.TestCase):
    def test_faint_new_application_is_not_permanently_learned_as_old_acid(self):
        gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
        faint=scene();faint[107:110,128:133]=(0,255,0)
        self.assertIsNone(gate.check(faint,QUAD,[OLD],1)[0])
        self.assertFalse(np.any(gate.old_mask))
        self.assertIsNotNone(gate.check(scene(acid=True),QUAD,[OLD],1.2)[0])
    def test_previous_patch_survives_stone_translation_and_new_patch_can_trigger_once(self):
        history = AcidHistory()
        history.remember([OLD], QUAD)
        gate = FreshAcidGate(history.mask, scene(acid=True), QUAD)
        shifted_quad = QUAD+np.float32([30,20])
        shifted = cv2.warpAffine(scene(acid=True), np.float32([[1,0,30],[0,1,20]]), (400,300))
        moved_old = dict(OLD,bbox=(158,127,173,159))
        best, reason = gate.check(shifted, shifted_quad, [moved_old], 10)
        self.assertIsNone(best)
        self.assertEqual('PREVIOUS_ACID_IGNORED',reason)
        # Even obvious changes inside the old footprint do not prove a new dose.
        changed = scene(acid=True)
        changed[107:139,128:143] = (255,255,255)
        self.assertIsNone(gate.check(changed,QUAD,[OLD],11)[0])
        best, _ = gate.check(two_patches(), QUAD, [OLD, NEW], 12)
        self.assertEqual(NEW['bbox'],best['bbox'])

    def test_baseline_acid_global_light_and_unknown_geometry_do_not_trigger(self):
        empty = np.zeros((256,256),np.uint8)
        gate = FreshAcidGate(empty,scene(acid=True),QUAD)
        self.assertIsNone(gate.check(scene(acid=True),QUAD,[OLD],1)[0])
        gate = FreshAcidGate(empty,scene(),QUAD)
        brighter = np.clip(scene().astype(int)+35,0,255).astype(np.uint8)
        self.assertIsNone(gate.check(brighter,QUAD,[OLD],2)[0])
        gate = FreshAcidGate(empty,scene(),QUAD)
        self.assertIsNone(gate.check(scene(acid=True),QUAD,[OLD],3,False)[0])
        self.assertIsNotNone(gate.check(scene(acid=True),QUAD,[OLD],4,True)[0])

    def test_occlusion_and_detector_misses_never_clear_history(self):
        history = AcidHistory()
        history.observe_idle(scene(),QUAD,[],0,True)
        history.observe_idle(scene(acid=True),QUAD,[OLD],1,True)
        for ts in (2,3,4):
            history.observe_idle(scene(),QUAD,[],ts,False)
            history.observe_idle(scene(acid=True),QUAD,[],ts+.1,True)
        self.assertTrue(np.any(history.mask))
        for ts in (5,5.4,5.8):
            history.observe_idle(scene(),QUAD,[],ts,True)
        self.assertFalse(np.any(history.mask))
        self.assertEqual('VISUALLY_CLEARED',history.events[-1]['status'])

    def test_live_clip_and_final_export_all_reject_old_acid(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = ContinuousMonitor(tmp,lambda:None,Photos(),Mock(),lambda *a,**k:None)
            monitor.detector=Detector()
            monitor.classifier=Mock()
            try:
                monitor.acid_history.remember([OLD],QUAD)
                for i in range(100):
                    ts=100+i*.1
                    held=10<=i<20
                    image=scene(acid=True)
                    monitor.journal.offer(image,ts)
                    monitor.step(dict(frame=image,timestamp=ts,mode='LOCKED',revision=0,
                        stone=STONE,placement=None,bench_moving=held,rubbing=False,
                        tracks=[track(state='OCCLUDED' if held else 'OK',spot='hand' if held else 'jewel',seen=ts,in_hand=held)]))
                    time.sleep(.003)
                a=monitor.active
                self.assertIsNotNone(a)
                self.assertIsNone(a['acid_onset'])
                monitor.journal.queue.join()
                recovery=dict(a,fresh_gate=a['fresh_gate'].fork(),stop_search=threading.Event())
                self.assertIsNone(monitor._search_recording(recovery,a['returned'],monitor.last_ts)[1])
                monitor._finish(monitor.last_ts,'Explicit session stop for verification.')
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                record=monitor.tests[0]
                self.assertFalse(record['capture_complete'])
                self.assertIsNone(record['acid_onset'])
                self.assertIsNone(record['prediction'])
                self.assertTrue(record['fresh_acid_rejections'])
                monitor.classifier.predict.assert_not_called()
            finally:
                monitor.pool.shutdown(wait=True,cancel_futures=True)
                monitor.detector_pool.shutdown(wait=True,cancel_futures=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()

    def test_new_acid_elsewhere_keeps_model_output_and_saves_both_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            classifier=Mock()
            classifier.predict.return_value=dict(prediction='non_gold',confidence=.95)
            monitor=ContinuousMonitor(tmp,lambda:None,Photos(),Mock(),lambda *a,**k:None)
            monitor.detector,monitor.classifier=Detector(),classifier
            monitor.acid_history.remember([OLD],QUAD)
            try:
                for i in range(135):
                    ts=100+i*.1
                    held=20<=i<35
                    image=scene(mark=i>=35)
                    if i>=48:image[107:139,178:193]=(0,255,0)
                    monitor.journal.offer(image,ts)
                    monitor.step(dict(frame=image,timestamp=ts,mode='LOCKED',revision=0,
                        stone=STONE,placement=None,bench_moving=held,rubbing=False,
                        tracks=[dict(track(state='OCCLUDED' if held else 'OK',spot='hand' if held else 'jewel',seen=ts,in_hand=held),
                                     acid_done=i>=120)]))
                    time.sleep(.004)
                monitor.pool.shutdown(wait=True)
                monitor.detector_pool.shutdown(wait=True)
                monitor.finish_pool.shutdown(wait=True)
                r=monitor.tests[0]
                self.assertEqual('non_gold',r['prediction']['prediction'])
                self.assertTrue(r['capture_complete'])
                self.assertEqual('ACID_AT_DIFFERENT_LOCATION',r['acid_target_status'])
                self.assertLess(r['acid_new_mask_overlap'],.5)
                self.assertGreater(r['rub_previous_acid_overlap'],0)
                self.assertTrue(Path(r['acid_rub_comparison_image']).is_file())
                self.assertTrue(any('contamination possible' in note for note in r['issues']))
                saved=json.loads(Path(r['record_path']).read_text())
                self.assertEqual('non_gold',saved['prediction']['prediction'])
                self.assertEqual('ACID_AT_DIFFERENT_LOCATION',saved['acid_target_status'])
            finally:
                monitor.pool.shutdown(wait=True,cancel_futures=True)
                monitor.detector_pool.shutdown(wait=True,cancel_futures=True)
                monitor.finish_pool.shutdown(wait=True)
                monitor.journal.close()


class InvalidReasonTests(unittest.TestCase):
    def test_threshold_explanations_preserve_existing_material_decisions(self):
        result=classify_probabilities([.002,.637,.361])
        self.assertEqual(('invalid','non_gold'),result[:2])
        self.assertIn('65%',result[3])
        result=classify_probabilities([.84,.10,.06])
        self.assertEqual(('invalid','gold_22k'),result[:2])
        self.assertIn('85%',result[3])
        result=classify_probabilities([.05,.1,.85])
        self.assertIn('does not identify a specific physical cause',result[3])
        self.assertEqual(('non_gold','non_gold'),classify_probabilities([.05,.9,.05])[:2])


class ContactRecoveryTests(unittest.TestCase):
    def test_contact_photo_is_from_this_interval_and_cropped_to_stone(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            def observer(frame, stone):
                gold=(frame[:,:,2]==250)&(frame[:,:,1]==180)
                ys,xs=np.where(gold)
                if not len(xs):return None
                x,y=int(xs.min()),int(ys.min())
                return dict(c=np.array([x+5,y+5]),box=(x,y,x+10,y+10),
                            mask=np.ones((10,10),np.uint8),origin=(x,y))
            try:
                observations=[]
                for i,x in enumerate([100,130,100,130,100,130,100,130]):
                    ts=99+i*.2
                    image=scene()
                    image[100:110,x:x+10]=(0,180,250)
                    journal.offer(image,ts)
                    observations.append((ts,STONE))
                journal.close()
                cycle=dict(started=99.4,returned=100.3,acid_onset=100.5,finished=106,
                           directory=Path(tmp),stone_observations=observations)
                def save(directory,name,frame):
                    path=directory/(name+'.jpg');cv2.imwrite(str(path),frame);return str(path)
                r=recover_contact(journal,cycle,observer,save)
                self.assertEqual('CONTACT_AND_MOTION_RECOVERED',r['rubbing_contact_status'])
                self.assertTrue(99.4<=r['rub_timestamp']<=100.3)
                self.assertLess(cv2.imread(r['rub_image']).shape[1],200)
                self.assertTrue(all(99.4<=p['timestamp']<=100.3 for p in r['rubbing_contact_samples']))
                r=recover_contact(journal,cycle,lambda *a:None,save)
                self.assertEqual('CONTACT_NOT_VISIBLE',r['rubbing_contact_status'])
                self.assertNotIn('CONFIRMED',r['rubbing_contact_status'])
                self.assertTrue(99.4<=r['rub_timestamp']<=100.3)
            finally:journal.close()
