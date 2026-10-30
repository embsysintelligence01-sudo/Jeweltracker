import time
import unittest
from unittest.mock import Mock
import numpy as np
import test_contact_gate as contact_tests
from test_continuous_testing import scene,STONE
from test_sop_revision import monitor_case,snap,contact

class OverlapTriggerTests(unittest.TestCase):
    def worker(self):
        w=contact_tests.ContactWorkerGateTests().worker()
        w.continuous_event('begin',jewel_id=1,token='j1',timestamp=time.time()-.1)
        return w

    def observe(self,w,ts=None):
        w._observe_stone_contact(scene(),time.time() if ts is None else ts,STONE,w.flow_revision,'j1')

    def test_two_distinct_overlap_frames_required_not_duplicate_snapshot(self):
        w=self.worker();ts=time.time()
        self.observe(w,ts)
        self.assertIsNone(w.rubber.ev)
        self.observe(w,ts)
        self.assertIsNone(w.rubber.ev)
        self.observe(w)
        self.assertIsNotNone(w.rubber.ev)
        self.assertEqual(2,w.continuous_overlap_trigger['frames'])
        self.assertFalse(w.rubber.rubbing)

    def test_no_jewel_or_border_only_mask_does_not_trigger(self):
        for poly in (None,np.float32([[218,65],[224,65],[224,195],[218,195]])):
            with self.subTest(poly=poly is None):
                w=self.worker()
                w.gold_observation=([],[]) if poly is None else ([poly],[.99])
                for _ in range(3):self.observe(w)
                self.assertIsNone(w.rubber.ev)
                self.assertIsNone(w.continuous_overlap_trigger)

    def test_a_missed_detection_breaks_consecutive_overlap(self):
        w=self.worker();gold=w.gold_observation
        self.observe(w);w.gold_observation=([],[]);self.observe(w)
        w.gold_observation=gold;self.observe(w)
        self.assertIsNone(w.continuous_overlap_trigger)
        self.observe(w);self.assertIsNotNone(w.continuous_overlap_trigger)

    def test_pickup_or_false_return_cannot_start_rubbing_or_baseline_before_overlap(self):
        with monitor_case() as (m,events):
            s=snap(100,held=1);s['overlap_trigger_required']=True
            m.step(s);a=m.active
            for ts in (100.2,100.4,100.8):
                s=snap(ts,contact=contact(ts,confirmed=True));s['overlap_trigger_required']=True
                m.step(s)
            self.assertIsNone(a['contact']);self.assertIsNone(a['baseline_at'])
            self.assertIsNone(a['returned'])
            self.assertFalse(any(k=='contact' for k,_ in events))
            self.assertEqual('WAIT_FOR_CONTACT',m.phase[1])

    def test_matching_cycle_trigger_arms_possible_contact_and_wrong_token_does_not(self):
        with monitor_case() as (m,events):
            s=snap(100,held=1);s['overlap_trigger_required']=True;m.step(s);a=m.active
            for token,ts in (('old',100.2),(a['id'],100.4)):
                s=snap(ts,held=1,contact=contact(ts,confirmed=False))
                s.update(overlap_trigger_required=True,rubbing_trigger=dict(timestamp=100.1,confirmed_at=ts,frames=2,cycle_token=token))
                m.step(s)
                if token=='old':self.assertIsNone(a['contact'])
            self.assertEqual(100.1,a['rubbing_started_at'])
            self.assertIsNotNone(a['contact']);self.assertFalse(a['live_rub'])

class MotionEdgeRejectionTests(unittest.TestCase):
    def test_thin_boundary_movement_is_not_contact_evidence(self):
        from sop_evidence import local_motion
        old=scene();new=scene();new[60:200,215:220]=220
        previous,_=local_motion(None,old,STONE)
        _,motion=local_motion(previous,new,STONE)
        self.assertIsNone(motion)

    def test_no_overlap_trigger_disables_motion_only_recorded_recovery(self):
        import tempfile
        from pathlib import Path
        from session_evidence import FrameJournal
        from rubbing_recovery import recover_contact
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            for i in range(4):journal.offer(scene(),100+i*.1)
            journal.close();observer=Mock()
            cycle=dict(started=100,returned=100.3,finished=101,acid_onset=None,
                overlap_trigger_required=True,rubbing_started_at=None,
                directory=Path(tmp),stone_observations=[(100,STONE)])
            result=recover_contact(journal,cycle,observer,Mock())
            self.assertEqual('CONTACT_NOT_VISIBLE',result['rubbing_contact_status'])
            observer.assert_not_called()

if __name__=='__main__':unittest.main()
