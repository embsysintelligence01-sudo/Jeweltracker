import unittest
import numpy as np
from acid_freshness import FreshAcidGate, footprint
from test_continuous_testing import scene, QUAD, Detector
from test_acid_application import DETECTION, needle
from test_sop_revision import monitor_case, snap
from audit_report import event_rows

class OverlapAdvisoryTests(unittest.TestCase):
    def cycle(self,m):
        m.acid_history.remember([DETECTION],QUAD)
        m.step(snap(100,held=1));a=m.active
        a.update(returned=100.1,baseline_at=100.2,return_frame=scene(),
                 fresh_gate=FreshAcidGate(m.acid_history.mask,scene(),QUAD),
                 poses=[(100+i*.1,QUAD.copy()) for i in range(31)])
        return a

    def test_live_same_location_application_starts_once_and_warns_in_pdf_table(self):
        with monitor_case() as (m,events):
            a=self.cycle(m);m.last_ts=100.5
            m._acid_observation(100.5,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertIsNone(a['acid_onset'])
            m._verify_application(a,scene(acid=True),QUAD,100.55,True,True)
            self.assertEqual(100.5,a['acid_onset'])
            self.assertTrue(a['fresh_acid_evidence']['previous_acid_overlap_warning'])
            notes=[s for s in a['issues'] if 'Application overlaps' in s]
            self.assertEqual(1,len(notes));self.assertIn('may be inaccurate',notes[0])
            for ts in (100.6,100.8):
                m._acid_observation(ts,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertEqual(1,len([e for k,e in events if k=='acid_detection' and e.get('detected')]))
            self.assertTrue(any('stored location may be inaccurate' in row[-1] for row in event_rows(a)))

    def test_recorded_recovery_accepts_same_location_and_retains_warning_metadata(self):
        with monitor_case() as (m,events):
            a=self.cycle(m);m.detector=Detector()
            for i in range(20):
                ts=100.2+i*.1
                m.journal.offer(scene(acid=i>=3),ts)
            m.journal.queue.join()
            onset,best,_,_=m._search_recording(a,100.2,102.1)
            self.assertIsNotNone(onset);self.assertIsNotNone(best)
            self.assertTrue(best['previous_acid_overlap_warning'])
            m._set_acid(a,onset,best,recovered=True)
            self.assertTrue(a['fresh_acid_evidence']['previous_acid_overlap_warning'])
            self.assertTrue(any('Application overlaps' in note for note in a['issues']))

    def test_unchanged_residue_is_a_candidate_with_freshness_warning(self):
        gate=FreshAcidGate(footprint(DETECTION['bbox'],QUAD),scene(acid=True),QUAD)
        for ts in (1,2,3):
            best,reason=gate.check(scene(acid=True),QUAD,[DETECTION],ts)
            self.assertIsNotNone(best);self.assertEqual('ACID_DETECTOR_CANDIDATE',reason)
            self.assertTrue(best['freshness_warning'])

    def test_needle_at_remembered_location_is_still_unverified(self):
        with monitor_case() as (m,events):
            a=self.cycle(m)
            m._acid_observation(100.5,a['id'],[DETECTION],(needle(),QUAD,True,True,False))
            for ts in (100.55,100.6):m._verify_application(a,needle(),QUAD,ts,True,True)
            self.assertIsNone(a['acid_onset'])

if __name__=='__main__':unittest.main()
