import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reportlab.pdfgen.canvas import Canvas

from audit_report import GREEN, render_report, session_period, when


class ReportLayoutTests(unittest.TestCase):
    def test_verification_is_inside_evidence_panels_and_overlap_is_green(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'report.pdf'
            spy=Mock(wraps=Canvas(str(output)))
            record=dict(jewel_id=1,rubbing_verified=True,mask_evidence_status='VERIFIED',
                        acid_target_status='ACID_OVERLAPS_NEW_RUB')
            with patch('audit_report.canvas.Canvas',return_value=spy):
                render_report(output,root,[dict(id=1)],[record],[],lambda p:'Unavailable')
            strings=[call.args[2] for call in spy.drawString.call_args_list]
            self.assertNotIn('RUBBING CONTACT',strings)
            self.assertNotIn('NEW RUBBED-MARK DETECTION',strings)
            self.assertNotIn('ACID PLACEMENT',strings)
            self.assertIn('VERIFIED - OVERLAPPING',strings)
            color=None;green_statuses=[]
            for call in spy.method_calls:
                if call[0]=='setFillColor':color=call.args[0]
                if call[0]=='drawString' and call.args[2] in ('VERIFIED','VERIFIED - OVERLAPPING'):
                    green_statuses.append(color)
            self.assertEqual([GREEN,GREEN,GREEN],green_statuses)

    def test_summary_has_per_jewel_clip_links_times_and_three_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rub=root/'rub clip.avi';rub.touch()
            acid=root/'acid clip.avi';acid.touch()
            records=[dict(jewel_id=1,started=200,finished=220,rubbing_clip={'path':str(rub)},
                          acid_clip={'path':str(acid)}),dict(jewel_id=2,started=100,finished=150)]
            output=root/'report.pdf'
            spy=Mock(wraps=Canvas(str(output)))
            with patch('audit_report.canvas.Canvas',return_value=spy):
                render_report(output,root,[dict(id=1),dict(id=2)],records,[],lambda p:'Unavailable')
            strings=[call.args[2] for call in spy.drawString.call_args_list]
            self.assertNotIn('VERIFIED SOP',strings)
            self.assertIn('JEWELS',strings)
            self.assertIn('TEST CYCLES',strings)
            self.assertIn('ACID CAPTURES',strings)
            self.assertIn(session_period(records),strings)
            self.assertIn('Start: '+when(200)+' | End: '+when(220)+' IST',strings)
            self.assertIn('Open rubbing clip',strings)
            self.assertIn('Open acid clip',strings)
            self.assertIn('rubbing unavailable',strings)
            self.assertIn('acid unavailable',strings)
            urls=[call.args[0] for call in spy.linkURL.call_args_list]
            self.assertIn(rub.as_uri(),urls)
            self.assertIn(acid.as_uri(),urls)
            self.assertTrue(output.read_bytes().startswith(b'%PDF'))

    def test_period_uses_first_start_last_finish_and_handles_no_tests(self):
        records=[dict(started=200,finished=220),dict(started=100,finished=300)]
        self.assertEqual(session_period(records),session_period([dict(started=100,finished=300)]))
        self.assertEqual('All jewels | Start: Not captured | End: Not captured | IST',session_period([]))

    def test_archive_links_target_final_assets_and_not_staging_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            staged=root/'stage.avi';staged.touch()
            final=root/'archives'/'rubbing.avi'
            output=root/'report.pdf'
            spy=Mock(wraps=Canvas(str(output)))
            with patch('audit_report.canvas.Canvas',return_value=spy):
                render_report(output,root,[dict(id=1)],
                    [dict(jewel_id=1,rubbing_clip={'path':str(final)})],[],lambda p:'Unavailable',
                    asset_resolver=lambda p:staged if Path(p)==final else Path(p))
            urls=[call.args[0] for call in spy.linkURL.call_args_list]
            self.assertIn(final.as_uri(),urls)
            self.assertNotIn(staged.as_uri(),urls)
