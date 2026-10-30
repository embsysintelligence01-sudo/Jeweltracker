import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

import jewel_tracker_5 as app
from continuous_testing import RubReference
from rubbing_recovery import recover_contact
from session_evidence import FrameJournal
from test_continuous_testing import scene, STONE, QUAD, Photos
from test_sop_revision import monitor_case, snap, contact


class ContactWorkerGateTests(unittest.TestCase):
    def worker(self):
        w=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        w.mode='LOCKED'
        w.stone_box=lambda:STONE
        w.rubber=app.RubbingObserver(lambda:STONE,lambda:w.flow_revision)
        w.gold_observation=([np.float32([[125,105],[145,105],[145,140],[125,140]])],np.array([.9]))
        return w

    def observe(self,w,token=None,ts=None):
        supplied=ts is not None
        ts=time.time() if ts is None else ts
        w._observe_stone_contact(scene(),ts,STONE,w.flow_revision,token)
        w._observe_stone_contact(scene(),ts+.0001 if supplied else time.time(),STONE,w.flow_revision,token)

    def test_static_bed_does_not_interpret_stone_segmentation(self):
        w=self.worker()
        w.rubber.observe=Mock(wraps=w.rubber.observe)
        self.observe(w)
        self.assertIsNone(w.rubber.ev)
        self.assertFalse(w.rubber.active)
        w.rubber.observe.assert_not_called()

    def test_only_active_cycle_accepts_contact_and_new_cycle_resets_motion(self):
        w=self.worker();start=time.time()
        w.continuous_event('begin',jewel_id=1,token='j1',timestamp=start)
        self.observe(w,'j1')
        self.assertEqual('j1',w.rubber.ev['cycle_token'])
        w.rubber.moves.extend([1.,2.,3.])
        w.continuous_event('contact_gate',jewel_id=1,token='j1',open=False,timestamp=time.time())
        self.assertIsNone(w.rubber.ev)
        self.observe(w,'j1')
        self.assertIsNone(w.rubber.ev)
        w.continuous_event('begin',jewel_id=2,token='j2',timestamp=start)
        self.assertEqual([],list(w.rubber.moves))
        self.observe(w,'j1')  # inference begun in J1 cannot become J2 contact
        self.assertIsNone(w.rubber.ev)
        self.observe(w,'j2')
        self.assertEqual('j2',w.rubber.ev['cycle_token'])
        w.continuous_event('contact_gate',jewel_id=1,token='j1',open=False)
        self.assertEqual('j2',w.continuous_contact_token)

    def test_early_source_duplicate_begin_and_roi_reset_cannot_reopen_contact(self):
        w=self.worker();start=time.time()
        w.continuous_event('begin',jewel_id=1,token='j1',timestamp=start)
        self.observe(w,'j1',start-.01)
        self.assertIsNone(w.rubber.ev)
        w.continuous_event('phase',jewel_id=1,token='j1',phase='BASELINE')
        self.assertIsNone(w.continuous_contact_token)
        updates=[]
        w.workflow.connect(lambda payload:updates.append(payload['kind']))
        w.continuous_event('contact',jewel_id=1,token='j1',photo=scene(),timestamp=time.time())
        self.assertNotIn('contact',updates)
        w.continuous_event('begin',jewel_id=1,token='j1',timestamp=start)
        self.assertIsNone(w.continuous_contact_token)
        w.continuous_event('begin',jewel_id=2,token='j2',timestamp=start)
        w.set_zone('roi',[(0,0),(399,0),(399,299),(0,299)])
        self.observe(w,'j2')
        self.assertIsNone(w.rubber.ev)


class ContactMonitorGateTests(unittest.TestCase):
    def test_stone_motion_or_rubbing_signal_without_pickup_cannot_start_recording(self):
        with monitor_case() as (m,events):
            for ts in (100,100.2,100.4):
                s=snap(ts,contact=contact(ts,confirmed=True));s['rubbing']=True;s['bench_moving']=True
                m.step(s)
            self.assertIsNone(m.active)
            self.assertFalse(any(k=='contact' for k,_ in events))

    def test_two_jewels_only_pickup_owner_accepts_contact_then_return_freezes_it(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            first=contact(100.1,confirmed=True)
            m.step(snap(100.1,held=1,contact=first))
            saved=a['contact']
            self.assertEqual(1,a['jewel_id'])
            m.step(snap(100.2,contact=contact(100.2,confirmed=True)))
            self.assertIsNone(a['returned'])  # first return observation is not enough
            m.step(snap(100.4,contact=contact(100.4,confirmed=True)))
            self.assertIsNotNone(a['returned'])
            self.assertEqual('JEWEL_RETURNED',a['contact_close_reason'])
            later=contact(100.6,confirmed=True);later['area']=100000
            m.step(snap(100.6,contact=later))
            self.assertIs(saved,a['contact'])
            self.assertEqual(1,sum(k=='contact' for k,_ in events))

    def test_mask_or_failed_mask_baseline_closes_contact_and_preserves_photo(self):
        for mode in ('mask','baseline'):
            with self.subTest(mode=mode),monitor_case() as (m,events):
                m.step(snap(100,held=1));a=m.active
                m._contact(a,contact(100.1,confirmed=True));saved=a['contact']
                if mode=='mask':
                    a['returned']=100.2
                    ref=RubReference(Photos(),scene(),99,QUAD)
                    a['mask_candidate']=ref.after(Photos(),scene(mark=True),100.25,QUAD)
                    a['contact_region']=a['mask_candidate']['new_mask'].copy()
                    m.pose.quad=QUAD
                    for ts in (100.3,100.4,100.5):m._confirm_candidate(a,scene(mark=True),ts)
                    self.assertIsNotNone(a['mask'])
                    self.assertEqual('NEW_MASK_CONFIRMED',a['contact_close_reason'])
                else:
                    a['returned']=100.2;a['return_frame']=scene();m.pose.quad=QUAD
                    m._baseline(a,scene(),100.4)
                    self.assertEqual('BASELINE_STARTED',a['contact_close_reason'])
                late=contact(100.6,confirmed=True);late['area']=100000
                m._contact(a,late)
                self.assertIs(saved,a['contact'])

    def test_wrong_cycle_contact_cannot_confirm_next_jewel(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));old=m.active['id']
            m._finish(100.2)
            m.step(snap(100.4,held=2));a=m.active
            stale=dict(contact(100.5,confirmed=True),cycle_token=old)
            m._contact(a,stale)
            self.assertIsNone(a['contact'])
            m._contact(a,dict(contact(100.6,confirmed=True),cycle_token=a['id']))
            self.assertEqual(100.6,a['contact']['timestamp'])

    def test_recovery_cannot_replace_frozen_contact_with_post_gate_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp);journal.offer(scene(),100.1);journal.close()
            late=contact(101,confirmed=True)
            cycle=dict(started=100,returned=100.2,finished=107,acid_onset=102,
                       contact_closed_at=100.4,contact=late,directory=Path(tmp),
                       stone_observations=[(100.1,STONE)])
            def save(d,n,img):p=d/(n+'.png');cv2.imwrite(str(p),img);return str(p)
            result=recover_contact(journal,cycle,None,save)
            self.assertFalse(result['rubbing_verified'])
            self.assertEqual('CONTACT_NOT_VISIBLE',result['rubbing_contact_status'])
            self.assertLessEqual(result['rubbing_evidence_interval'][1],100.4)


if __name__=='__main__':unittest.main()
