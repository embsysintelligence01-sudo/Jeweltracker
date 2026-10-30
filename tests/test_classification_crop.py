import unittest
import numpy as np
import cv2

from classification_crop import ClassificationCrop,extract_crop
from acid_freshness import normalized,footprint
from test_continuous_testing import scene,QUAD,STONE
from test_acid_application import DETECTION
from test_sop_revision import monitor_case


class VerifiedCropTests(unittest.TestCase):
    def test_crop_alignment_follows_stone_without_masking_or_changing_dimensions(self):
        image=scene(acid=True);box=(115,90,160,155)
        expected=extract_crop(image,box,QUAD,QUAD)
        moved=cv2.warpAffine(image,np.float32([[1,0,30],[0,1,20]]),(400,300))
        actual=extract_crop(moved,box,QUAD,QUAD+[30,20])
        self.assertEqual((65,45,3),actual.shape)
        np.testing.assert_array_equal(expected,actual)

    def test_full_box_with_surrounding_stone_is_not_rejected_as_a_tight_patch(self):
        verifier=ClassificationCrop(normalized(scene(),QUAD))
        image=scene(acid=True)
        detection=dict(DETECTION,bbox=(110,85,170,160))
        verifier.observe(image,QUAD,[detection],10)
        verifier.observe(image,QUAD,[detection],10.2)
        self.assertIsNotNone(verifier.best)

    def test_unverified_crop_never_reaches_material_classifier(self):
        from unittest.mock import Mock
        with monitor_case() as (m,_):
            from test_sop_revision import snap
            m.step(snap(100,held=1));a=m.active
            a.update(returned=100.1,baseline_at=100.2,return_frame=scene())
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(62)]
            for i in range(62):
                m.journal.offer(scene(acid=i>=10),100+i*.1);m.journal.queue.join()
            m.classifier=Mock()
            m._set_acid(a,101,dict(DETECTION,source_frame=scene(acid=True),quad=QUAD))
            m._finish(106.1);m.drain_finalization()
            r=m.tests[0]
            self.assertEqual('UNVERIFIED_APPLICATION_REGION',r['classification_roi_status'])
            self.assertIsNone(r['prediction'])
            m.classifier.predict.assert_not_called()

    def test_full_frame_coordinates_are_not_offset_by_stone_origin(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        with monitor_case() as (m,_):
            m.detector=Mock()
            m.detector.predict.return_value=([SimpleNamespace(**DETECTION)],0)
            image=scene(acid=True)
            _,_,found=m._detect(image,100,STONE,None)
            self.assertIs(image,m.detector.predict.call_args.args[0])
            self.assertEqual(DETECTION['bbox'],found[0]['bbox'])

    def test_wrong_early_box_cannot_own_the_verified_classifier_region(self):
        verifier=ClassificationCrop(normalized(scene(),QUAD))
        wrong=dict(DETECTION,bbox=(180,80,200,100))
        image=scene(acid=True)
        verifier.observe(image,QUAD,[wrong],10)
        verifier.observe(image,QUAD,[wrong],10.2)
        self.assertIsNone(verifier.best)
        verifier.observe(image,QUAD,[DETECTION],10.4)
        verifier.observe(image,QUAD,[DETECTION],10.6)
        self.assertEqual(DETECTION['bbox'],verifier.best['bbox'])

    def test_movement_uses_stone_relative_box_and_two_distinct_observations(self):
        verifier=ClassificationCrop(normalized(scene(),QUAD))
        image=scene(acid=True)
        verifier.observe(image,QUAD,[DETECTION],10)
        verifier.observe(image,QUAD,[DETECTION],10)
        self.assertIsNone(verifier.best)
        moved=cv2.warpAffine(image,np.float32([[1,0,20],[0,1,15]]),(400,300))
        detection=dict(DETECTION,bbox=tuple(np.array(DETECTION['bbox'])+[20,15,20,15]))
        verifier.observe(moved,QUAD+[20,15],[detection],10.2)
        self.assertEqual(detection['bbox'],verifier.best['bbox'])

    def test_missing_geometry_and_unchanged_residue_fail(self):
        for clear,image,target in ((False,scene(acid=True),None),(True,scene(),None)):
            verifier=ClassificationCrop(normalized(scene(),QUAD),target)
            verifier.observe(image,QUAD,[DETECTION],10,clear)
            verifier.observe(image,QUAD,[DETECTION],10.2,clear)
            self.assertIsNone(verifier.best)


if __name__=='__main__':unittest.main()
