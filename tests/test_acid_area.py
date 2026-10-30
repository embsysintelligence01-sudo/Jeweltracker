import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from acid_area import AcidAreaMeasurement, compare_area, region_box
from acid_freshness import normalized, footprint
from test_continuous_testing import scene, QUAD, Photos
from test_acid_application import DETECTION, needle
from test_sop_revision import monitor_case, snap


def measurement(old=None):
    return AcidAreaMeasurement(normalized(scene(),QUAD),
        np.zeros((256,256),np.uint8) if old is None else old,10,DETECTION,QUAD)


class AcidAreaTests(unittest.TestCase):
    def test_background_snapshot_cannot_mutate_live_accumulation_or_source_image(self):
        live=measurement();image=scene(acid=True)
        live.observe(image,QUAD,[DETECTION],10)
        background=live.fork()
        self.assertTrue(background.observe(image,QUAD,[DETECTION],10.1))
        self.assertFalse(np.any(live.mask))
        self.assertTrue(np.any(background.mask))
        self.assertIs(image,background.source_frame)
        self.assertEqual(10.1,background.updated_at)
        self.assertIsNone(live.source_frame)

    def test_dark_elongated_wet_streak_is_not_treated_as_a_bright_needle(self):
        area=measurement();image=scene()
        image[80:160,134:141]=15
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10))
        self.assertTrue(area.observe(image,QUAD,[DETECTION],10.1))
        self.assertTrue(np.any(area.mask))
        box=region_box(area.mask,QUAD,image.shape)
        self.assertLess(box[1],90)
        self.assertGreater(box[3],150)

    def test_export_keeps_classifier_crop_independent_of_expanded_display_area(self):
        from acid_freshness import FreshAcidGate
        from continuous_testing import RubReference
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['returned']=100.1;a['return_frame']=scene()
            a['reference']=RubReference(Photos(),scene(),99,QUAD)
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(62)]
            a['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
            m.classifier=Mock()
            from test_continuous_testing import Detector
            m.detector=Detector()
            m.classifier.predict.return_value=dict(prediction='non_gold',confidence=.95)
            for i in range(62):
                m.journal.offer(scene(acid=i>=10),100+i*.1)
                m.journal.queue.join()
            m._set_acid(a,101,dict(DETECTION,source_frame=scene(acid=True),quad=QUAD))
            larger=dict(DETECTION,bbox=(120,95,155,150))
            context=(scene(acid=True),QUAD,True,True,False)
            for ts in (101.2,101.4):m._acid_observation(ts,a['id'],[larger],context)
            m._finish(106.1);m.drain_finalization()
            record=m.tests[0]
            self.assertEqual(101,record['acid_onset'])
            self.assertEqual(list(DETECTION['bbox']),record['detector_bbox'])
            from acid_test.flow import pad_bbox
            self.assertEqual(list(pad_bbox(record['classification_detector_bbox'],.45,400,300)),record['reaction_bbox'])
            self.assertGreaterEqual(record['acid_area_bbox'][2],154)
            self.assertEqual('COMPARISON_UNAVAILABLE',record['acid_target_status'])
            self.assertEqual('ESTIMATED_PERSISTENT_VISUAL_ACID_AREA',record['acid_area_status'])
            self.assertTrue(Path(record['acid_area_mask']).is_file())
            self.assertEqual('DISABLED_UNMASKED_SOURCE',record['acid_visual_overlay'])
            import cv2
            display=cv2.imread(record['acid_rub_comparison_image'])
            cyan=(display[:,:,0]>180)&(display[:,:,1]>180)&(display[:,:,2]<80)
            self.assertFalse(np.any(cyan))
            self.assertGreater(record['acid_placement_timestamp'],record['acid_onset'])
            self.assertEqual('non_gold',record['prediction']['prediction'])

    def test_later_larger_detection_expands_only_after_clear_persistence(self):
        area=measurement();image=scene(acid=True)
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10))
        self.assertTrue(area.observe(image,QUAD,[DETECTION],10.1))
        original=area.mask.copy()
        larger=dict(DETECTION,bbox=(120,95,155,150))
        self.assertFalse(area.observe(image,QUAD,[larger],10.3))
        np.testing.assert_array_equal(original,area.mask)
        self.assertTrue(area.observe(image,QUAD,[larger],10.5))
        self.assertGreater(np.count_nonzero(area.mask),np.count_nonzero(original))
        box=region_box(area.mask,QUAD,image.shape)
        self.assertLessEqual(box[0],121)
        self.assertGreaterEqual(box[2],154)
        self.assertEqual(10,area.onset)

    def test_old_acid_history_does_not_reject_visible_area_after_application(self):
        area=measurement(footprint(DETECTION['bbox'],QUAD));image=scene(acid=True)
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10))
        self.assertTrue(area.observe(image,QUAD,[DETECTION],10.1))
        self.assertTrue(np.any(area.mask))
        self.assertNotIn('PREVIOUS_ACID_OVERLAP',area.rejections)

    def test_tool_and_covered_frames_cannot_expand_area(self):
        for old,image,clear in ((None,needle(),True),(None,scene(acid=True),False)):
            with self.subTest(clear=clear,old=old is not None):
                area=measurement(old)
                for ts in (10,10.1,10.3):
                    self.assertFalse(area.observe(image,QUAD,[DETECTION],ts,clear=clear))
                self.assertFalse(np.any(area.mask))

    def test_unknown_pose_duplicate_and_outside_window_rejected(self):
        area=measurement();image=scene(acid=True)
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10,geometry_known=False))
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10.1))
        self.assertFalse(area.observe(image,QUAD,[DETECTION],10.1))
        self.assertFalse(area.observe(image,QUAD,[DETECTION],15.1))
        self.assertFalse(np.any(area.mask))

    def test_area_is_stone_relative_and_unrelated_change_is_excluded(self):
        area=measurement();image=scene(acid=True)
        area.observe(image,QUAD,[DETECTION],10)
        moved=cv2.warpAffine(image,np.float32([[1,0,20],[0,1,15]]),(400,300))
        detection=dict(DETECTION,bbox=tuple(np.array(DETECTION['bbox'])+[20,15,20,15]))
        self.assertTrue(area.observe(moved,QUAD+[20,15],[detection],10.1))
        before=region_box(area.mask,QUAD,image.shape)
        after=region_box(area.mask,QUAD+[20,15],image.shape)
        np.testing.assert_array_equal(np.array(before)+[20,15,20,15],after)
        other=scene();other[105:135,185:200]=(0,255,0)
        unrelated=dict(DETECTION,bbox=(185,105,200,135))
        for ts in (10.3,10.5):self.assertFalse(area.observe(other,QUAD,[unrelated],ts))
        self.assertEqual(before,region_box(area.mask,QUAD,image.shape))

    def test_missing_partial_full_and_different_locations_are_separate(self):
        rub=np.zeros((256,256),np.uint8);rub[50:100,50:80]=255
        self.assertEqual('COMPARISON_UNAVAILABLE',compare_area(None,rub)['acid_target_status'])
        self.assertEqual('ACID_OVERLAPS_NEW_RUB',compare_area(rub,rub)['acid_target_status'])
        partial=rub.copy();partial[75:]=0
        result=compare_area(partial,rub)
        self.assertEqual('ACID_PARTIALLY_OVERLAPS_NEW_RUB',result['acid_target_status'])
        self.assertEqual(.5,result['acid_estimated_rub_coverage'])
        self.assertEqual('ACID_AT_DIFFERENT_LOCATION',compare_area(np.roll(rub,70,axis=1),rub)['acid_target_status'])

    def test_measurement_updates_do_not_rearm_timer_or_accept_other_cycles(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            from acid_freshness import FreshAcidGate
            a['fresh_gate']=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
            m._set_acid(a,101,dict(DETECTION,source_frame=scene(acid=True),quad=QUAD))
            larger=dict(DETECTION,bbox=(120,95,155,150))
            context=(scene(acid=True),QUAD,True,True,False)
            for ts in (101.2,101.4):m._acid_observation(ts,a['id'],[larger],context)
            self.assertTrue(np.any(a['acid_area'].mask))
            self.assertEqual(101,a['acid_onset'])
            self.assertEqual(DETECTION['bbox'],a['detector_bbox'])
            self.assertTrue(a['stop_search'].is_set())
            self.assertEqual(1,sum(k=='acid_detection' and p.get('detected',False) for k,p in events))
            mask=a['acid_area'].mask.copy()
            m._acid_observation(101.6,'other-cycle',[larger],context)
            np.testing.assert_array_equal(mask,a['acid_area'].mask)
            a['reaction_end']=101.5
            m._acid_observation(101.8,a['id'],[larger],context)
            self.assertEqual(101.4,a['acid_area'].updated_at)


if __name__=='__main__':unittest.main()
