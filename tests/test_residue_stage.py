import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from acid_freshness import AcidHistory, FreshAcidGate, normalized, footprint
from continuous_testing import draw_workflow, RubReference
from test_continuous_testing import scene,QUAD,Photos
from test_acid_application import DETECTION
from test_sop_revision import monitor_case,snap


class ResidueStageTests(unittest.TestCase):
    def test_cleaned_stone_with_remaining_rub_mark_clears_history(self):
        history=AcidHistory()
        history.observe_idle(scene(),QUAD,[],0,True)
        dry=scene(mark=True)
        history.remember([DETECTION],QUAD,baseline=normalized(dry,QUAD))
        history.observe_idle(scene(mark=True,acid=True),QUAD,[DETECTION],1,True)
        self.assertEqual('PREVIOUS_ACID_VISIBLE',history.snapshot(1)['status'])
        for ts in (2,2.4,2.8):history.observe_idle(dry,QUAD,[],ts,True)
        self.assertFalse(np.any(history.mask))
        self.assertEqual('VISUALLY_CLEARED',history.snapshot(2.8)['status'])
        gate=FreshAcidGate(history.mask,dry,QUAD)
        self.assertIsNotNone(gate.check(scene(mark=True,acid=True),QUAD,[DETECTION],3)[0])

    def test_covered_or_missed_acid_is_not_declared_clean(self):
        history=AcidHistory()
        history.observe_idle(scene(),QUAD,[],0,True)
        history.remember([DETECTION],QUAD,baseline=normalized(scene(),QUAD))
        for ts in (1,1.4,1.8):history.observe_idle(scene(acid=True),QUAD,[],ts,True)
        self.assertTrue(np.any(history.mask))
        for ts in (2,2.4,2.8):history.observe_idle(scene(),QUAD,[],ts,False)
        self.assertTrue(np.any(history.mask))
        self.assertEqual('PREVIOUS_ACID_UNCERTAIN',history.snapshot(2.8)['status'])

    def test_next_pickup_continues_with_warning_and_residue_does_not_trigger_timer(self):
        with monitor_case() as (m,events):
            h=m.acid_history
            h.observe_idle(scene(),QUAD,[],98,True)
            h.observe_idle(scene(acid=True),QUAD,[DETECTION],99,True)
            m.step(snap(100,held=1,frame=scene(acid=True)))
            a=m.active
            self.assertIsNotNone(a)
            self.assertEqual('PREVIOUS_ACID_VISIBLE',a['previous_acid_state']['status'])
            self.assertIn('contamination possible',a['residue_warning'])
            gate=FreshAcidGate(h.mask,scene(acid=True),QUAD)
            self.assertIsNone(gate.check(scene(acid=True),QUAD,[DETECTION],101)[0])
            self.assertIsNone(a['acid_onset'])

    def test_after_pickup_rubbing_or_baseline_cannot_clear_or_pollute_residue_history(self):
        with monitor_case() as (m,events):
            m.acid_history.observe_idle(scene(),QUAD,[],98,True)
            m.acid_history.remember([DETECTION],QUAD,baseline=normalized(scene(),QUAD))
            m.step(snap(100,held=1));a=m.active
            old=m.acid_history.mask.copy()
            a['returned']=100.1;a['baseline_at']=100.2
            a['fresh_gate']=FreshAcidGate(old,scene(),QUAD)
            for ts in (100.3,100.7,101.1):
                m._acid_observation(ts,a['id'],[],(scene(),QUAD,True,True,False))
            np.testing.assert_array_equal(old,m.acid_history.mask)
            mark=dict(DETECTION,bbox=(180,105,195,140))
            m._acid_observation(101.2,a['id'],[mark],(scene(mark=True),QUAD,True,True,True))
            np.testing.assert_array_equal(old,m.acid_history.mask)
            self.assertTrue(np.any(a['fresh_gate'].old_mask & footprint(mark['bbox'],QUAD)))

    def test_late_clear_frame_before_pickup_updates_frozen_warning(self):
        with monitor_case() as (m,events):
            m.acid_history.observe_idle(scene(),QUAD,[],95,True)
            m.acid_history.remember([DETECTION],QUAD,baseline=normalized(scene(),QUAD))
            for ts in (98.1,98.5):m.acid_history.observe_idle(scene(),QUAD,[],ts,True)
            m.step(snap(99,held=1));a=m.active
            self.assertIsNotNone(a['residue_warning'])
            m._acid_observation(98.9,None,[],(scene(),QUAD,True,True,False))
            self.assertIsNone(a['residue_warning'])
            self.assertEqual('VISUALLY_CLEARED',a['previous_acid_state']['status'])

    def test_initial_detection_is_not_drawn_as_measured_acid_area(self):
        base=np.zeros((300,400,3),np.uint8)
        state=dict(phase='REACTION',jewel_id=1)
        initial=dict(detected=True,bbox=(310,210,350,250),confidence=.95)
        vis=draw_workflow(base.copy(),state,acid=initial)
        self.assertFalse(np.any(vis[210:251,310:351]))
        initial['area_status']='ESTIMATED_PERSISTENT_FOOTPRINT'
        vis=draw_workflow(base.copy(),state,acid=initial)
        self.assertTrue(np.any(vis[210:251,310:351]))

    def test_unavailable_area_report_has_no_small_cyan_fallback(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['returned']=100.1;a['return_frame']=scene()
            a['reference']=RubReference(Photos(),scene(),99,QUAD)
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(62)]
            for i in range(62):
                m.journal.offer(scene(acid=i>=10),100+i*.1);m.journal.queue.join()
            m._set_acid(a,101,dict(DETECTION,source_frame=scene(acid=True),quad=QUAD))
            m._finish(106.1);m.drain_finalization()
            r=m.tests[0]
            self.assertEqual('AREA_UNAVAILABLE',r['acid_area_status'])
            image=cv2.imread(r['acid_rub_comparison_image'])
            cyan=(image[:,:,0]>180)&(image[:,:,1]>180)&(image[:,:,2]<80)
            self.assertFalse(np.any(cyan))
            self.assertEqual('COMPARISON_UNAVAILABLE',r['acid_target_status'])


if __name__=='__main__':unittest.main()
