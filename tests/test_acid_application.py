import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from acid_application import AcidApplicationVerifier
from acid_freshness import FreshAcidGate, normalized, footprint
from continuous_testing import reaction_covered, reaction_region_covered
from test_continuous_testing import scene, QUAD, STONE, Photos
from continuous_testing import RubReference
from test_sop_revision import monitor_case, snap

DETECTION = dict(bbox=(125, 100, 148, 145), confidence=.94,
                 freshness='NEW_APPEARANCE_OUTSIDE_PREVIOUS_ACID')


def needle(patch=False):
    frame=scene(acid=patch)
    cv2.line(frame,(137,75),(137,160),(205,205,205),2)
    return frame


def verifier(old=None):
    return AcidApplicationVerifier(normalized(scene(),QUAD),
        np.zeros((256,256),np.uint8) if old is None else old)


class ApplicationVerificationTests(unittest.TestCase):
    def test_full_detector_roi_is_preserved_instead_of_tight_changed_patch(self):
        v=verifier();image=scene(acid=True)
        detection=dict(DETECTION,bbox=(110,85,165,160))
        v.offer(image,QUAD,detection,10)
        self.assertIsNone(v.observe(image,QUAD,10))
        _,best=v.observe(image,QUAD,10.04)
        self.assertEqual(detection['bbox'],best['bbox'])
        x1,y1,x2,y2=best['verified_patch_bbox']
        self.assertLess((x2-x1)*(y2-y1),55*75)

    def test_full_detector_roi_follows_stone_before_first_clear_frame(self):
        v=verifier();detection=dict(DETECTION,bbox=(110,85,165,160))
        v.offer(needle(),QUAD,detection,10)
        self.assertIsNone(v.observe(needle(),QUAD,10))
        moved=cv2.warpAffine(scene(acid=True),np.float32([[1,0,20],[0,1,15]]),(400,300))
        self.assertIsNone(v.observe(moved,QUAD+[20,15],10.5))
        onset,best=v.observe(moved,QUAD+[20,15],10.54)
        self.assertEqual(10.5,onset)
        self.assertEqual((130,100,185,175),best['bbox'])
        np.testing.assert_array_equal(moved,best['source_frame'])

    def test_reaction_coverage_follows_a_moved_stone_and_rejects_unknown_pose(self):
        frame=np.full((300,400,3),65,np.uint8)
        skin=cv2.cvtColor(np.uint8([[[10,80,180]]]),cv2.COLOR_HSV2BGR)[0,0]
        frame[85:165,190:250]=skin
        box=(128,107,143,139)
        self.assertFalse(reaction_covered(frame,box))
        self.assertTrue(reaction_region_covered(frame,box,QUAD,QUAD+[80,0],True))
        self.assertTrue(reaction_region_covered(scene(acid=True),box,QUAD,QUAD,False))

    def test_two_distinct_clear_frames_confirm_without_repeated_model_detections(self):
        v=verifier();image=scene(acid=True)
        v.offer(image,QUAD,DETECTION,10)
        self.assertIsNone(v.observe(image,QUAD,10))
        for _ in range(10):self.assertIsNone(v.observe(image,QUAD,10))
        onset,best=v.observe(image,QUAD,10.04)
        self.assertEqual(10,onset)
        self.assertEqual(2,best['clear_patch_frames'])
        self.assertEqual('VISIBLE_PATCH_ONSET',best['application_timing_status'])
        self.assertEqual(10.04,best['application_verified_at'])
        self.assertFalse(v.pending)

    def test_stationary_needle_is_not_confirmed_just_because_motion_stopped(self):
        v=verifier();image=needle()
        v.offer(image,QUAD,DETECTION,10)
        for t in (10,10.04,10.2,11,12,15):
            self.assertIsNone(v.observe(image,QUAD,t))
        self.assertTrue(v.pending)
        self.assertEqual('APPLICATOR_LIKE_CHANGE_REMAINS',v.pending[0]['reason'])
        self.assertIsNone(v.observe(image,QUAD,15.1))
        self.assertFalse(v.pending)
        self.assertEqual('APPLICATION_OBSCURED_OR_NOT_PERSISTENT',v.events[-1]['status'])

    def test_tool_disappearing_without_a_patch_is_rejected(self):
        v=verifier();v.offer(needle(),QUAD,DETECTION,10)
        v.observe(needle(),QUAD,10)
        self.assertIsNone(v.observe(scene(),QUAD,10.2))
        self.assertFalse(v.pending)
        self.assertEqual('CANDIDATE_DISAPPEARED',v.events[-1]['status'])

    def test_patch_after_needle_withdrawal_keeps_supported_onset_and_timing_warning(self):
        v=verifier();v.offer(needle(),QUAD,DETECTION,10)
        self.assertIsNone(v.observe(needle(),QUAD,10))
        self.assertIsNone(v.observe(scene(acid=True),QUAD,10.5))
        onset,best=v.observe(scene(acid=True),QUAD,10.54)
        self.assertEqual(10.5,onset)
        self.assertEqual(10,best['application_candidate_at'])
        self.assertEqual('ONSET_OBSCURED_USING_FIRST_CLEAR_PATCH',best['application_timing_status'])
        np.testing.assert_array_equal(scene(acid=True),best['source_frame'])

    def test_unknown_pose_coverage_and_frame_gaps_cannot_confirm(self):
        v=verifier();image=scene(acid=True)
        v.offer(image,QUAD,DETECTION,10)
        for t in (10,10.04):self.assertIsNone(v.observe(image,QUAD,t,clear=False))
        self.assertIsNone(v.observe(image,QUAD,10.1,geometry_known=False))
        self.assertIsNone(v.observe(image,QUAD,10.2))
        self.assertIsNone(v.observe(image,QUAD,11))
        self.assertEqual(1,v.pending[0]['count'])
        self.assertIsNotNone(v.observe(image,QUAD,11.04))

    def test_local_movement_resets_the_supported_onset_until_patch_settles(self):
        v=verifier();image=scene(acid=True)
        v.offer(image,QUAD,DETECTION,10)
        self.assertIsNone(v.observe(image,QUAD,10))
        moved=scene();moved[107:139,134:149]=(0,255,0)
        self.assertIsNone(v.observe(moved,QUAD,10.04))
        onset,_=v.observe(moved,QUAD,10.08)
        self.assertEqual(10.04,onset)

    def test_candidate_memory_is_bounded_and_disjoint_disappearances_are_safe(self):
        v=verifier()
        for i,x in enumerate((125,175,85)):
            v.offer(scene(),QUAD,dict(DETECTION,bbox=(x,105,x+20,140)),10+i*.04)
        self.assertEqual(2,len(v.pending))
        self.assertEqual('CANDIDATE_REPLACED_UNVERIFIED',v.events[-2]['status'])
        self.assertIsNone(v.observe(scene(),QUAD,10.4))
        self.assertFalse(v.pending)

    def test_stone_translation_does_not_create_motion_or_move_candidate_to_another_spot(self):
        v=verifier();image=scene(acid=True)
        v.offer(image,QUAD,DETECTION,10)
        self.assertIsNone(v.observe(image,QUAD,10))
        moved=cv2.warpAffine(image,np.float32([[1,0,20],[0,1,15]]),(400,300))
        onset,best=v.observe(moved,QUAD+[20,15],10.04)
        self.assertEqual(10,onset)
        self.assertTrue(best['bbox'][0]<150)  # original supported source, not the moved frame
        self.assertIsNone(verifier().observe(moved,QUAD+[20,15],10.04))

    def test_baseline_patch_can_confirm_but_unrelated_patch_cannot(self):
        v=AcidApplicationVerifier(normalized(scene(acid=True),QUAD),footprint(DETECTION['bbox'],QUAD))
        v.offer(scene(acid=True),QUAD,DETECTION,10)
        self.assertIsNone(v.observe(scene(acid=True),QUAD,10))
        self.assertIsNotNone(v.observe(scene(acid=True),QUAD,10.04))
        v=verifier();v.offer(needle(),QUAD,DETECTION,10)
        other=scene();other[105:135,180:195]=(0,255,0)
        for t in (10.1,10.2):self.assertIsNone(v.observe(other,QUAD,t))


class ApplicationFlowTests(unittest.TestCase):
    def test_mark_from_between_tool_candidate_and_clear_patch_cannot_be_used_as_pre_acid_mask(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['mask']=dict(timestamp=101)
            m._set_acid(a,102,dict(DETECTION,application_candidate_at=100.5,quad=QUAD))
            self.assertIsNone(a['mask'])
            self.assertEqual('REJECTED_ACID_OVERLAP',a['mask_evidence_status'])
            self.assertTrue(any(k=='clear_mask' for k,_ in events))

    def test_export_rechecks_aligned_reaction_frames_even_if_live_coverage_was_missed(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['returned']=100.1;a['return_frame']=scene()
            a['reference']=RubReference(Photos(),scene(),99,QUAD)
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(62)]
            skin=cv2.cvtColor(np.uint8([[[10,80,180]]]),cv2.COLOR_HSV2BGR)[0,0]
            for i in range(62):
                frame=scene(acid=i>=10)
                if i>=55:frame[85:165,110:170]=skin
                m.journal.offer(frame,100+i*.1)
            m.classifier=Mock()
            m._set_acid(a,101,dict(DETECTION,bbox=(128,107,143,139)))
            m._finish(106.1);m.drain_finalization()
            record=m.tests[0]
            self.assertEqual('PARTIAL',record['reaction_coverage_status'])
            self.assertTrue(record['reaction_rejected_frames'])
            self.assertLess(record['reaction_last_timestamp'],105.5)
            self.assertIsNone(record['prediction'])
            m.classifier.predict.assert_not_called()

    def test_unverified_needle_is_not_classified_by_final_recorded_recovery(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['returned']=100.1;a['return_frame']=scene();a['baseline_at']=100.1
            a['acid_baseline_checked']=True
            a['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
            narrow=dict(DETECTION,bbox=(134,80,141,160))
            m.detector=m.classifier=Mock()
            m._detect=lambda frame,ts,stone,cid:(ts,cid,[narrow])
            image=needle()
            for i in range(31):
                ts=100.2+i*.2
                m.journal.offer(image,ts);a['poses'].append((ts,QUAD.copy()))
            m._acid_observation(100.2,a['id'],[narrow],(image,QUAD,True,True,False))
            for i in range(31):
                m._verify_application(a,image,QUAD,100.2+i*.2,True,True)
            self.assertIsNone(a['acid_onset'])
            m._finish(106.2);m.drain_finalization()
            record=m.tests[0]
            self.assertIsNone(record['prediction'])
            self.assertIsNone(record['acid_onset'])
            self.assertFalse(record['capture_complete'])
            self.assertTrue(any(e['status']=='APPLICATION_OBSCURED_OR_NOT_PERSISTENT'
                                for e in record['acid_candidate_events']))
            m.classifier.predict.assert_not_called()

    def test_live_candidate_does_not_latch_until_fresh_camera_frames_verify_it(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            a=m.active;a['returned']=100.1;a['baseline_at']=100.2
            a['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
            context=(scene(acid=True),QUAD,True,True,False)
            m._acid_observation(100.3,a['id'],[DETECTION],context)
            self.assertIsNone(a['acid_onset'])
            self.assertFalse(a['stop_search'].is_set())
            self.assertTrue(any(e.get('provisional') for k,e in events if k=='acid_detection'))
            m.step(snap(100.34,frame=scene(acid=True)))
            self.assertEqual(100.3,a['acid_onset'])
            self.assertTrue(a['stop_search'].is_set())
            self.assertEqual(1,sum(e.get('detected',False) for k,e in events if k=='acid_detection'))
            m.step(snap(100.38,frame=scene(acid=True)))
            self.assertEqual(1,sum(e.get('detected',False) for k,e in events if k=='acid_detection'))

    def test_recorded_recovery_rejects_needle_only_and_accepts_remaining_patch(self):
        for remaining_patch in (False,True):
            with self.subTest(remaining_patch=remaining_patch), monitor_case() as (m,events):
                m.step(snap(100,held=1));a=m.active
                a['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
                a['poses']=[]
                for i in range(12):
                    ts=100.2+i*.2
                    image=needle() if i<3 else scene(acid=remaining_patch)
                    m.journal.offer(image,ts);a['poses'].append((ts,QUAD.copy()))
                m.journal.queue.join()
                m._detect=lambda frame,ts,stone,cid:(ts,cid,[DETECTION])
                recovery=dict(a,stop_search=threading.Event())
                observed,best,_,_=m._search_recording(recovery,100.2,102.4)
                if remaining_patch:
                    self.assertIsNotNone(best)
                    self.assertGreaterEqual(observed,100.8)
                    self.assertEqual('DETECTOR_AND_VISIBLE_PERSISTENT_PATCH',best['application_verification'])
                else:
                    self.assertIsNone(best)


if __name__=='__main__':unittest.main()
