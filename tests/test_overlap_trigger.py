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

    def test_first_contact_is_retained_without_confirming_rubbing(self):
        w=self.worker();ts=time.time()
        self.observe(w,ts)
        self.assertIsNotNone(w.rubber.ev)
        self.assertFalse(w.rubber.rubbing)
        self.observe(w,ts)
        self.assertFalse(w.rubber.rubbing)
        self.assertIsNone(w.continuous_overlap_trigger)

    def test_no_jewel_or_border_only_mask_does_not_trigger(self):
        for poly in (None,np.float32([[218,65],[224,65],[224,195],[218,195]])):
            with self.subTest(poly=poly is None):
                w=self.worker()
                w.gold_observation=([],[]) if poly is None else ([poly],[.99])
                for _ in range(3):self.observe(w)
                self.assertIsNone(w.rubber.ev)
                self.assertIsNone(w.continuous_overlap_trigger)

    def test_missed_detection_does_not_require_a_new_two_frame_overlap(self):
        w=self.worker();gold=w.gold_observation
        self.observe(w);w.gold_observation=([],[]);self.observe(w)
        w.gold_observation=gold;self.observe(w)
        self.assertIsNotNone(w.rubber.ev)
        self.assertIsNone(w.continuous_overlap_trigger)
        self.assertFalse(w.rubber.rubbing)

    def test_old_overlap_flag_does_not_disable_contact_or_reliable_return(self):
        with monitor_case() as (m,events):
            s=snap(100,held=1);s['overlap_trigger_required']=True
            m.step(s);a=m.active
            m.step(snap(100.1,held=1,contact=contact(100.1,confirmed=False)))
            self.assertIsNotNone(a['contact']);self.assertFalse(a['live_rub'])
            self.assertIsNone(a['baseline_at'])
            for ts in (100.4,100.6,101,101.8):m.step(snap(ts))
            self.assertIsNotNone(a['returned']);self.assertIsNotNone(a['baseline_at'])

    def test_wrong_contact_token_is_rejected_without_an_overlap_gate(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            for token,ts in (('old',100.2),(a['id'],100.4)):
                c=dict(contact(ts,confirmed=False),cycle_token=token)
                m.step(snap(ts,held=1,contact=c))
                if token=='old':self.assertIsNone(a['contact'])
            self.assertEqual(100.4,a['rubbing_started_at'])
            self.assertIsNotNone(a['contact']);self.assertFalse(a['live_rub'])

class MotionEdgeRejectionTests(unittest.TestCase):
    def test_thin_boundary_movement_is_not_contact_evidence(self):
        from sop_evidence import local_motion
        old=scene();new=scene();new[60:200,215:220]=220
        previous,_=local_motion(None,old,STONE)
        _,motion=local_motion(previous,new,STONE)
        self.assertIsNone(motion)

    def test_missed_live_overlap_still_allows_unverified_recorded_recovery(self):
        import tempfile
        from pathlib import Path
        from session_evidence import FrameJournal
        from rubbing_recovery import recover_contact
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            for i in range(4):journal.offer(scene(),100+i*.1)
            journal.close();observer=Mock(return_value=None)
            cycle=dict(started=100,returned=100.3,finished=101,acid_onset=None,
                overlap_trigger_required=True,rubbing_started_at=None,
                directory=Path(tmp),stone_observations=[(100,STONE)])
            result=recover_contact(journal,cycle,observer,Mock())
            self.assertEqual('CONTACT_NOT_VISIBLE',result['rubbing_contact_status'])
            self.assertGreater(observer.call_count,0)
            self.assertFalse(result['rubbing_verified'])

if __name__=='__main__':unittest.main()
