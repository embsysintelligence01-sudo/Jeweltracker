import tempfile
import unittest
from pathlib import Path
import numpy as np
from audit_report import event_rows, render_report
from continuous_testing import visibility
from acid_freshness import FreshAcidGate
from test_continuous_testing import scene, QUAD, track
from test_acid_application import DETECTION
from test_sop_revision import monitor_case, snap

class EarlyAcidAuditTests(unittest.TestCase):
    def test_baseline_false_positive_does_not_ban_early_application(self):
        with monitor_case() as (m, events):
            m.step(snap(100,held=1));a=m.active
            a.update(returned=100.1,return_frame=scene(),baseline_at=100.2,
                     fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(),QUAD))
            m._acid_observation(100.1,a['id'],[DETECTION],(scene(),QUAD,True,True,True))
            self.assertFalse(np.any(a['fresh_gate'].old_mask));self.assertIsNone(a['acid_onset'])
            m.last_ts=100.5
            m._acid_observation(100.5,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertIsNone(a['acid_onset'])
            m._verify_application(a,scene(acid=True),QUAD,100.55,True,True)
            self.assertEqual(100.5,a['acid_onset'])
            self.assertLess(a['acid_onset'],a['baseline_at']+2)
            self.assertTrue(a['stop_search'].is_set())
            m._acid_observation(101,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertEqual(100.5,a['acid_onset'])

    def test_existing_baseline_patch_proceeds_with_freshness_warning(self):
        with monitor_case() as (m, events):
            m.step(snap(100,held=1));a=m.active
            a.update(returned=100.1,baseline_at=100.2,
                     fresh_gate=FreshAcidGate(np.zeros((256,256),np.uint8),scene(acid=True),QUAD))
            for ts in (100.3,100.6,101):
                m._acid_observation(ts,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertIsNotNone(a['acid_onset'])
            self.assertTrue(a['fresh_acid_evidence']['freshness_warning'])
            self.assertTrue(any('not a verified chemical application time' in note for note in a['issues']))

    def test_hand_evidence_precedes_missing_state(self):
        self.assertEqual('IN_HAND',visibility(track(state='MISSING',in_hand=True),10,False))
        self.assertEqual('UNDER_HAND',visibility(track(state='MISSING',spot='hand'),10,False))
        self.assertEqual('MISSED',visibility(track(state='MISSING',spot='removed'),10,False))

    def test_per_jewel_table_only_actual_missing_and_interruption(self):
        r=dict(jewel_id=1,baseline_at=100,acid_onset=100.6,application_candidate_at=100.5,
               finished=105.6,reaction_end=102,coverage_since=102.001,
               usable_reaction_seconds=1.4,visibility=[
                   dict(jewel_id=1,status='IN_HAND',start=99,end=100),
                   dict(jewel_id=1,status='UNDER_HAND',start=100,end=101),
                   dict(jewel_id=1,status='MISSED',start=101,end=101.2),
                   dict(jewel_id=2,status='MISSED',start=99,end=103)])
        rows=event_rows(r);missed=[row for row in rows if row[0]=='Jewel genuinely missed']
        self.assertEqual(1,len(missed));self.assertEqual('0.20 s',missed[0][3])
        self.assertTrue(any('early application' in row[-1] for row in rows))
        self.assertTrue(any(row[0]=='Possible cleaning / coverage' for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.pdf'
            r['issues']=['Evidence note '+str(i)+' '+('detail '*30) for i in range(70)]
            render_report(path,Path(tmp),[dict(id=1)],[r],[],lambda p:'Unavailable')
            self.assertTrue(path.read_bytes().startswith(b'%PDF'))

if __name__=='__main__':unittest.main()
