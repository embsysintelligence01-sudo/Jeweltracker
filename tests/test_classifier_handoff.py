import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np

from acid_test.flow import pad_bbox
from acid_freshness import FreshAcidGate
from continuous_testing import RubReference, choose_hand_jewel
from test_acid_application import DETECTION
import test_contact_gate as contact_tests
from test_continuous_testing import scene,QUAD,STONE,Photos,track
from test_sop_revision import monitor_case,snap,contact


class ClassifierHandoffTests(unittest.TestCase):
    def test_reference_sample_padding_and_both_orientations(self):
        # Exact boxes in the supplied lab sample's manifest. Padding is per
        # axis, and transposing the detector must transpose the padded crop.
        detector=(1079,976,1137,1095)
        expected=(1053,922,1163,1149)
        self.assertEqual(expected,pad_bbox(detector,.45,2560,1440))
        x1,y1,x2,y2=detector
        rotated=pad_bbox((y1,x1,y2,x2),.45,1440,2560)
        self.assertEqual((expected[1],expected[0],expected[3],expected[2]),rotated)
        self.assertEqual((110,227),(expected[2]-expected[0],expected[3]-expected[1]))

    def test_first_two_seconds_and_pouring_frames_feed_legacy_crop(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a.update(returned=100.1,return_frame=scene(),baseline_at=100.2)
            a['reference']=RubReference(Photos(),scene(),99,QUAD)
            a['poses']=[(100+i/15,QUAD.copy()) for i in range(160)]
            # Visible colour sentinel separates the early baseline from the
            # later waiting/pouring footage inside the actual detector crop.
            for i in range(160):
                ts=100+i/15;frame=scene(acid=ts>=105)
                if 100.2<=ts<102.2:frame[110:130,130:140]=(85,85,85)
                if 102.2<=ts<105:frame[110:130,130:140]=(120,120,120)
                m.journal.offer(frame,ts);m.journal.queue.join()
            m.classifier=Mock()
            from test_continuous_testing import Detector
            m.detector=Detector()
            m.classifier.predict.return_value=dict(prediction='gold_22k',confidence=.97)
            m._set_acid(a,105,dict(DETECTION,source_frame=scene(acid=True),quad=QUAD))
            m._finish(110.1);m.drain_finalization()
            record=m.tests[0]
            self.assertEqual(list(pad_bbox(record['classification_detector_bbox'],.45,400,300)),record['reaction_bbox'])
            self.assertEqual([100.2,102.2],record['classification_baseline_window'])
            self.assertEqual([102.2,105],record['classification_transition_window'])
            baseline,pouring,reaction=m.classifier.predict.call_args.args
            self.assertGreaterEqual(len(pouring),30)
            self.assertLessEqual(len(pouring),225)
            self.assertGreaterEqual(len(reaction),24)
            box=record['reaction_bbox'];x,y=135-box[0],120-box[1]
            self.assertTrue(all(abs(int(frame[y,x,0])-85)<8 for frame in baseline))
            self.assertTrue(any(abs(int(frame[y,x,0])-120)<8 for frame in pouring))
            self.assertEqual('gold_22k',record['prediction']['prediction'])

    def test_next_pickup_during_reaction_reopens_live_segmentation_for_j2(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            w=contact_tests.ContactWorkerGateTests().worker()
            original=m._emit_callback
            def dispatch(kind,**data):
                original(kind,**data);w.continuous_event(kind,**data)
            m._emit_callback=dispatch
            m._recover_and_save=Mock()
            m.step(snap(100,held=1));old=m.active
            m._set_acid(old,100.5,dict(DETECTION,quad=QUAD))
            self.assertIsNone(w.continuous_contact_token)
            m.step(snap(101,held=2))
            m.step(snap(101.2,held=2))
            m.step(snap(101.4,held=2))
            current=m.active
            self.assertEqual(2,current['jewel_id'])
            self.assertEqual('NEXT_JEWEL_PICKUP',old['close_reason'])
            self.assertTrue(m._acid_cancelled(old['id']))
            self.assertEqual(current['id'],w.continuous_contact_token)
            w._observe_stone_contact(scene(),time.time(),STONE,w.flow_revision,current['id'])
            self.assertIsNone(w.rubber.ev)  # the first overlap cannot arm rubbing
            w._observe_stone_contact(scene(),time.time(),STONE,w.flow_revision,current['id'])
            self.assertIsNotNone(w.rubber.ev)
            self.assertEqual(current['id'],w.rubber.ev['cycle_token'])
            # An old phase/result cannot close the new segmentation gate.
            w.continuous_event('phase',token=old['id'],jewel_id=1,phase='DONE')
            self.assertEqual(current['id'],w.continuous_contact_token)

    def test_no_final_acid_recovery_or_model_call_after_handoff(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            entered=threading.Event();release=threading.Event()
            save=m._save
            def delayed(a):entered.set();release.wait(3);save(a)
            m._save=delayed
            m.step(snap(100,held=1));old=m.active
            old.update(returned=100.1,return_frame=scene(),baseline_at=100.2)
            old['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
            old['poses']=[(100,QUAD.copy())]
            m._search_recording=Mock(side_effect=AssertionError('Old acid recovery must not restart'))
            m.step(snap(101,held=2));m.step(snap(101.2,held=2));m.step(snap(101.4,held=2))
            self.assertTrue(entered.wait(1))
            m.detector=Mock()
            _,_,found=m._detect(scene(acid=True),102,STONE,old['id'])
            self.assertEqual([],found)
            m.detector.predict.assert_not_called()
            release.set();m.drain_finalization()
            m._search_recording.assert_not_called()
            self.assertIsNone(old['acid_onset'])
            self.assertEqual(2,m.active['jewel_id'])

    def test_current_held_jewel_wins_over_previous_recent_hand_memory(self):
        previous=track(1,seen=100)
        current=track(2,seen=100,in_hand=True)
        self.assertEqual(2,choose_hand_jewel([previous,current],100,True,{1:99.8,2:100}))

    def test_repeated_detector_timestamp_cannot_confirm_handoff(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));old=m.active
            for ts in (101,101.2,101.4):
                sample=snap(ts,held=2);sample['tracks'][1]['det_seen']=101
                m.step(sample)
            self.assertIs(old,m.active)


if __name__=='__main__':unittest.main()
