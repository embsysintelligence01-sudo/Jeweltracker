import unittest
from unittest.mock import Mock
import numpy as np
from acid_freshness import FreshAcidGate
from acid_test.flow import pad_bbox
from test_continuous_testing import scene,QUAD
from test_acid_application import DETECTION
from test_sop_revision import monitor_case,snap

PREDICTION=dict(prediction='non_gold',confidence=.91,probabilities={'non_gold':.91})

class ConfirmedAcidCropTests(unittest.TestCase):
    def prepare(self,m,confirmed=True):
        m.step(snap(100,held=1));a=m.active
        a.update(returned=100.1,baseline_at=100.2,return_frame=scene(),
                 fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD),
                 poses=[(100+i*.1,QUAD.copy()) for i in range(71)])
        for i in range(71):m.journal.offer(scene(acid=i>=10),100+i*.1)
        m.journal.queue.join()
        best=dict(DETECTION,source_frame=scene(acid=True),quad=QUAD.copy())
        if confirmed:best['application_verification']='DETECTOR_AND_VISIBLE_PERSISTENT_PATCH'
        m._set_acid(a,101,best)
        m.classifier=Mock();m.classifier.predict.return_value=PREDICTION
        return a,best

    def test_confirmed_region_is_copied_once_and_never_follows_later_box(self):
        with monitor_case() as (m,_):
            a,best=self.prepare(m)
            frozen=a['confirmed_classification_region']
            best['quad'][:]+=20
            a['detector_bbox']=(180,80,200,100)
            m._set_acid(a,102,dict(best,bbox=(180,80,200,100)))
            self.assertEqual(list(DETECTION['bbox']),frozen['bbox'])
            np.testing.assert_array_equal(QUAD,np.asarray(frozen['quad']))
            self.assertEqual(101,frozen['timestamp'])

    def test_next_pickup_search_cancellation_does_not_block_complete_classification(self):
        with monitor_case() as (m,_):
            a,_=self.prepare(m)
            m.detector=Mock()
            m.acid_cancellations[a['id']].set()
            m._finish(106.1);m.drain_finalization()
            r=m.tests[0]
            self.assertEqual(PREDICTION,r['prediction'])
            self.assertEqual('COMPLETE',r['reaction_coverage_status'])
            self.assertEqual('CONFIRMED_APPLICATION_DETECTOR_BOX',r['classification_roi_status'])
            self.assertEqual('SEARCH_CANCELLED_AFTER_NEXT_PICKUP',r['classification_advisory_status'])
            self.assertEqual(list(pad_bbox(DETECTION['bbox'],.45,400,300)),r['reaction_bbox'])
            m.detector.predict.assert_not_called()
            m.classifier.predict.assert_called_once()

    def test_missing_optional_detector_pair_does_not_block_complete_classification(self):
        with monitor_case() as (m,_):
            a,_=self.prepare(m)
            m._classification_crop=Mock(side_effect=AssertionError('must not reverify confirmed source'))
            m._finish(106.1);m.drain_finalization()
            self.assertEqual(PREDICTION,m.tests[0]['prediction'])
            m.classifier.predict.assert_called_once()
            self.assertEqual([],m.tests[0]['classification_roi_checks'])

    def test_unconfirmed_box_is_not_promoted_to_confirmed_source(self):
        with monitor_case() as (m,_):
            a,_=self.prepare(m,False)
            self.assertNotIn('confirmed_classification_region',a)
            m._finish(106.1);m.drain_finalization()
            self.assertIsNone(m.tests[0]['prediction'])
            m.classifier.predict.assert_not_called()

    def test_cleaning_interruption_still_blocks_incomplete_reaction(self):
        with monitor_case() as (m,_):
            a,_=self.prepare(m)
            a['reaction_end']=102
            m._finish(106.1);m.drain_finalization()
            self.assertEqual('PARTIAL',m.tests[0]['reaction_coverage_status'])
            self.assertIsNone(m.tests[0]['prediction'])
            m.classifier.predict.assert_not_called()

if __name__=='__main__':unittest.main()
