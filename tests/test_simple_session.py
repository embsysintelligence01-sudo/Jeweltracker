import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence_archive import RetentionPolicy, encoder_available, tree_bytes
from session_evidence import write_report
from simple_session import finalize_simple_session, convert_simple_session_to_avi, _decoded_digest
from compact_evidence import media_info, run_encoder
import test_evidence_retention as fixtures


@unittest.skipUnless(encoder_available(), 'FFmpeg/libx264 required')
class SimpleSessionTests(unittest.TestCase):
    def fixture(self, root):
        fixtures.EvidenceRetentionTests().fixture(root)
        return RetentionPolicy(require_server_ack=False, simple_output=True, target_test_bytes=3_000_000)

    def test_only_report_and_two_playable_clips_remain_with_portable_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            result = finalize_simple_session(root, policy)
            self.assertEqual('cleaned', result['status'])
            files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.name.startswith('.')}
            self.assertEqual({'report.pdf', 'J1/rubbing.avi', 'J1/acid.avi'}, files)
            self.assertLessEqual(tree_bytes(root), 3_000_000)
            pdf = (root / 'report.pdf').read_bytes()
            self.assertIn(b'J1/rubbing.avi', pdf)
            self.assertIn(b'J1/acid.avi', pdf)
            self.assertNotIn(b'/URI (file:', pdf)
            self.assertNotIn(b'/URI (session.json)', pdf)
            before = (root / 'report.pdf').read_bytes()
            write_report(root, [], [])
            self.assertEqual(before, (root / 'report.pdf').read_bytes())
            self.assertFalse((root / 'session.json').exists())
            self.assertEqual('already_cleaned', finalize_simple_session(root, policy)['status'])

    def test_failed_encoding_keeps_sources_and_original_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            before = (root / 'report.pdf').read_bytes()
            with patch('simple_session.encode_budgeted', side_effect=OSError('encoding failed')):
                with self.assertRaisesRegex(OSError, 'encoding failed'):
                    finalize_simple_session(root, policy)
            self.assertTrue((root / 'frames').is_dir())
            self.assertTrue((root / 'test_1/pickup_rubbing_return.avi').is_file())
            self.assertEqual(before, (root / 'report.pdf').read_bytes())

    def test_missing_clip_does_not_delete_any_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            data = json.loads((root / 'session.json').read_text())
            data['tests'][0]['acid_clip'] = None
            (root / 'session.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'no acid_clip'):
                finalize_simple_session(root, policy)
            self.assertTrue((root / 'frames').is_dir())

    def test_testing_cleanup_preserves_concurrent_packet_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            recording = root / 'packet_sealing' / '.working' / 'recording.avi'
            recording.parent.mkdir(parents=True)
            recording.write_bytes(b'active packet recording')
            finalize_simple_session(root, policy)
            self.assertEqual(b'active packet recording', recording.read_bytes())

    def test_cleanup_interruption_resumes_after_session_json_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.fixture(root)
            remove = shutil.rmtree

            def interrupt(path, *args, **kwargs):
                if Path(path).name == 'frames':
                    (root / 'session.json').unlink(missing_ok=True)
                    raise OSError('interrupted cleanup')
                return remove(path, *args, **kwargs)

            with patch('simple_session.shutil.rmtree', side_effect=interrupt):
                with self.assertRaisesRegex(OSError, 'interrupted cleanup'):
                    finalize_simple_session(root, policy)
            self.assertTrue((root / '.simple-staging/commit.json').is_file())
            self.assertEqual('cleaned', finalize_simple_session(root, policy)['status'])
            self.assertFalse((root / 'frames').exists())
            self.assertFalse((root / '.simple-staging').exists())

    def make_mp4_session(self, root):
        policy = self.fixture(root)
        finalize_simple_session(root, policy)
        for source in (root / 'J1').glob('*.avi'):
            run_encoder([encoder_available(), '-v', 'error', '-y', '-i', str(source),
                         '-c:v', 'copy', str(source.with_suffix('.mp4'))])
            source.unlink()
        pdf = root / 'report.pdf'
        pdf.write_bytes(pdf.read_bytes().replace(b'.avi)', b'.mp4)'))
        return policy

    def test_mp4_conversion_preserves_pixels_timing_size_and_pdf_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.make_mp4_session(root)
            before = {p.stem: (_decoded_digest(p), media_info(p)) for p in (root / 'J1').glob('*.mp4')}
            result = convert_simple_session_to_avi(root, policy)
            self.assertEqual('cleaned', result['status'])
            self.assertFalse(list(root.rglob('*.mp4')))
            self.assertLessEqual(tree_bytes(root), 3_000_000)
            for p in (root / 'J1').glob('*.avi'):
                self.assertEqual(before[p.stem][0], _decoded_digest(p))
                self.assertAlmostEqual(before[p.stem][1]['duration_seconds'], media_info(p)['duration_seconds'], places=2)
            self.assertIn(b'J1/rubbing.avi', (root / 'report.pdf').read_bytes())
            self.assertNotIn(b'.mp4)', (root / 'report.pdf').read_bytes())

    def test_mp4_conversion_failure_preserves_mp4_and_original_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.make_mp4_session(root)
            pdf = (root / 'report.pdf').read_bytes()
            with patch('simple_session._decoded_digest', side_effect=[(10, 'a'), (10, 'b')]):
                with self.assertRaisesRegex(OSError, 'pixels changed'):
                    convert_simple_session_to_avi(root, policy)
            self.assertEqual(2, len(list((root / 'J1').glob('*.mp4'))))
            self.assertEqual(pdf, (root / 'report.pdf').read_bytes())
