import unittest
import numpy as np
from acid_application import AcidApplicationVerifier
from acid_freshness import FreshAcidGate, normalized
from classification_crop import ClassificationCrop
from test_acid_application import DETECTION
from test_continuous_testing import scene, QUAD, Detector
from test_sop_revision import monitor_case,snap

class FreshnessAdvisoryFlowTests(unittest.TestCase):
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

    def test_recorded_recovery_allows_existing_patch_and_preserves_freshness_warning(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active;m.detector=Detector()
            image=scene(acid=True)
            a.update(returned=100.1,baseline_at=100.2,return_frame=image,
                     fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),image,QUAD),
                     poses=[(100+i*.1,QUAD.copy()) for i in range(25)])
            for i in range(25):m.journal.offer(image,100+i*.1)
            m.journal.queue.join()
            onset,best,_,_=m._search_recording(a,100.2,102.3)
            self.assertIsNotNone(onset)
            self.assertTrue(best['freshness_warning'])
            m._set_acid(a,onset,best,recovered=True)
            self.assertTrue(a['stop_search'].is_set())
            self.assertTrue(any('Freshness checking is advisory' in s for s in a['issues']))

if __name__=='__main__':unittest.main()
