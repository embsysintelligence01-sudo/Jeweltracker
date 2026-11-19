import unittest
import numpy as np
from acid_application import AcidApplicationVerifier
from acid_freshness import FreshAcidGate, normalized
from classification_crop import ClassificationCrop
from test_acid_application import DETECTION
from test_continuous_testing import scene, QUAD, Detector
from test_sop_revision import monitor_case,snap

class FreshnessAdvisoryFlowTests(unittest.TestCase):
    def test_before_photo_refreshes_without_acid_detector_or_clearance(self):
        with monitor_case() as (m,events):
            m.acid_history.mask[:]=255
            m.acid_history.status='PREVIOUS_ACID_UNCERTAIN'
            m.step(snap(100,frame=scene(acid=True)))
            self.assertIsNotNone(m.reference_job)
            m.reference_job.result(timeout=5)
            m._collect_jobs()
            first=m.reference
            self.assertEqual(100,first.timestamp)
            self.assertIsNone(m.detector)
            # Wiping changes the next idle photo without waiting for a negative
            # detector result or a historical dry-reference comparison.
            m.step(snap(102.1,frame=scene()))
            m.reference_job.result(timeout=5)
            m._collect_jobs()
            self.assertGreater(m.reference.timestamp,first.timestamp)
            np.testing.assert_array_equal(scene(),m.reference.frame)
            self.assertEqual(2,len([e for e in events if e[0]=='reference']))

    def test_existing_acid_in_baseline_does_not_start_reaction(self):
        with monitor_case() as (m,events):
            image=scene(acid=True)
            m.step(snap(100,held=1,frame=image));a=m.active
            a.update(returned=100.1,baseline_at=100.2,return_frame=image,
                     fresh_gate=FreshAcidGate(np.full((256,256),255,np.uint8),image,QUAD))
            m._acid_observation(100.3,a['id'],[DETECTION],(image,QUAD,True,True,False))
            m._verify_application(a,image,QUAD,100.34,True,True)
            self.assertIsNone(a['acid_onset'])
            self.assertEqual('NO_NEW_PATCH_SINCE_BASELINE',a['application_verifier'].pending[0]['reason'])

    def test_weak_baseline_change_confirms_clear_patch_without_freshness_retry(self):
        faint=scene();faint[107:110,128:133]=(0,255,0)
        gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD)
        best,_=gate.check(faint,QUAD,[DETECTION],10)
        self.assertTrue(best['freshness_warning'])
        verifier=AcidApplicationVerifier(gate.baseline,gate.old_mask)
        verifier.offer(faint,QUAD,best,10)
        self.assertIsNone(verifier.observe(faint,QUAD,10))
        onset,result=verifier.observe(faint,QUAD,10.04)
        self.assertEqual(10,onset)
        self.assertEqual('PATCH_OBSERVATION_TIME_FRESH_APPLICATION_UNVERIFIED',result['application_timing_status'])

    def test_classifier_crop_is_not_blocked_by_unchanged_baseline_patch(self):
        image=scene(acid=True)
        crop=ClassificationCrop(normalized(image,QUAD))
        crop.observe(image,QUAD,[DETECTION],10)
        crop.observe(image,QUAD,[DETECTION],10.2)
        self.assertIsNotNone(crop.best)
        self.assertEqual(DETECTION['bbox'],crop.best['bbox'])

    def test_recorded_recovery_rejects_unchanged_baseline_patch(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active;m.detector=Detector()
            image=scene(acid=True)
            a.update(returned=100.1,baseline_at=100.2,return_frame=image,
                     fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),image,QUAD),
                     poses=[(100+i*.1,QUAD.copy()) for i in range(25)])
            for i in range(25):m.journal.offer(image,100+i*.1)
            m.journal.queue.join()
            onset,best,_,_=m._search_recording(a,100.2,102.3)
            self.assertIsNone(onset)
            self.assertIsNone(best)


if __name__=='__main__':unittest.main()
