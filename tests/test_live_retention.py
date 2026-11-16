import copy
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from evidence_archive import RetentionPolicy, SessionGuard, encoder_available, tree_bytes
from live_retention import LiveRetention, compact_test, enqueue_test, finalize_live_session, job_path
from session_evidence import FrameJournal, atomic_json, write_report
import test_evidence_retention as fixtures


@unittest.skipUnless(encoder_available(), 'FFmpeg/libx264 required')
class LiveRetentionTests(unittest.TestCase):
    def policy(self):
        return RetentionPolicy(require_server_ack=False, simple_output=True,
                               idle_buffer_seconds=5, target_test_bytes=3_000_000)

    def fixture(self, root):
        journal, record = fixtures.EvidenceRetentionTests().fixture(root)
        jewel = dict(id=1, image=str(root / 'J1.jpg'))
        enqueue_test(root, record, jewel)
        return journal, record, jewel

    def test_camera_keeps_writing_while_completed_test_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = self.policy()
            journal = FrameJournal(root, fps=10, policy=policy)
            folder = root / 'test_1'
            folder.mkdir()
            image = np.full((64, 96, 3), 110, np.uint8)
            journal.pin_interval('test_1', 100, 120)
            journal.pin_interval('other-reader', 104, 105)
            for i in range(201):
                timestamp = 100 + i * .1
                journal.processed_through(timestamp)
                journal.offer(image, timestamp)
                journal.queue.join()
            record = dict(cycle_id='test_1', jewel_id=1, started=104, finished=115,
                          capture_complete=True, close_reason='REACTION_COMPLETE',
                          record_path=str(folder / 'manifest.json'), issues=[],
                          rubbing_clip=journal.clip(103, 106, folder / 'rubbing.avi'),
                          acid_clip=journal.clip(108, 115, folder / 'acid.avi'))
            atomic_json(folder / 'manifest.json', record)
            jewel = dict(id=1)
            write_report(root, [jewel], [record])
            worker = LiveRetention(root, policy, journal, Mock())
            try:
                worker.start()
                worker.submit(record, jewel)
                deadline = time.monotonic() + 20
                tick = 0
                while time.monotonic() < deadline:
                    tick += 1
                    journal.processed_through(200 + tick)
                    journal.offer(image, 200 + tick)
                    path = job_path(root, 'test_1')
                    if (path.exists() and json.loads(path.read_text())['status'] == 'CLEANED'
                            and 'test_1' not in journal.pins):
                        break
                    time.sleep(.05)
                self.assertEqual('CLEANED', json.loads(path.read_text())['status'])
                self.assertFalse(journal.closed)
                self.assertTrue(worker.is_alive())
                self.assertFalse(folder.exists())
                journal.processed_through(1000 + tick)
                journal.offer(image, 1000 + tick)
                journal.queue.join()
                self.assertTrue((root / 'frames/104.000000.jpg').exists())
                self.assertFalse((root / 'frames/110.000000.jpg').exists())
                self.assertNotIn('test_1', journal.pins)
                self.assertIn('other-reader', journal.pins)
                output = root / 'J1/test_1'
                self.assertEqual({'report.pdf', 'rubbing.avi', 'acid.avi'}, {p.name for p in output.iterdir()})
                # A delayed report request with old source paths must keep the final links.
                write_report(root, [jewel], [record])
                saved = json.loads((root / 'session.json').read_text())['tests'][0]
                self.assertEqual(str(output / 'report.pdf'), saved['compact_report'])
                self.assertIn(b'J1/test_1/rubbing.avi', (root / 'report.pdf').read_bytes())
                self.assertIn(b'/URI (rubbing.avi)', (output / 'report.pdf').read_bytes())
                self.assertLessEqual(tree_bytes(output) + (root / 'report.pdf').stat().st_size, 3_000_000)
            finally:
                worker.close()
                journal.close()

    def test_failure_is_saved_and_a_new_worker_retries_without_losing_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, record, _ = self.fixture(root)
            journal = Mock()
            worker = LiveRetention(root, self.policy(), journal, Mock())
            with self.assertLogs('live_retention', level='ERROR'), \
                    patch('live_retention.encode_budgeted', side_effect=OSError('disk temporarily unavailable')):
                worker.process_pending(force=True)
            state = json.loads(job_path(root, 'test_1').read_text())
            self.assertEqual('FAILED', state['status'])
            self.assertIn('disk temporarily unavailable', state['error'])
            self.assertGreater(state['next_retry'], time.time())
            self.assertTrue(Path(record['rubbing_clip']['path']).is_file())
            journal.release_interval.assert_not_called()
            with patch('live_retention.encode_budgeted') as encoder:
                worker.process_pending()  # A saved retry deadline must be respected.
                encoder.assert_not_called()
            restarted = LiveRetention(root, self.policy(), journal, Mock())
            with patch('live_retention.time.time', return_value=state['next_retry'] + 1):
                restarted.process_pending()  # Ordinary background polling resumes it when due.
            self.assertEqual('CLEANED', json.loads(job_path(root, 'test_1').read_text())['status'])
            journal.release_interval.assert_called_with('test_1')

    def test_interrupted_deletion_retries_verified_output_without_encoding_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, record, jewel = self.fixture(root)
            remove = shutil.rmtree

            def fail_raw(path, *args, **kwargs):
                if Path(path) == root / 'test_1':
                    raise OSError('temporary cleanup error')
                return remove(path, *args, **kwargs)

            worker = LiveRetention(root, self.policy(), Mock(), Mock())
            with self.assertLogs('live_retention', level='ERROR'), \
                    patch('live_retention.shutil.rmtree', side_effect=fail_raw):
                worker.process_pending(force=True)
            state = json.loads(job_path(root, 'test_1').read_text())
            self.assertTrue(state['published'])
            self.assertEqual('FAILED', state['status'])
            write_report(root, [jewel], [record])
            self.assertIn('compact_report', json.loads((root / 'session.json').read_text())['tests'][0])
            with patch('live_retention.encode_budgeted', side_effect=AssertionError('must resume publication')):
                worker.process_pending(force=True)
            self.assertFalse((root / 'test_1').exists())
            self.assertEqual('CLEANED', json.loads(job_path(root, 'test_1').read_text())['status'])

    def test_restart_skips_another_live_session_and_recovers_it_after_unlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            old = parent / 'old'
            old.mkdir()
            self.fixture(old)
            (old / 'session_closed.json').unlink()  # Simulate an abruptly terminated app.
            current = parent / 'current'
            current.mkdir()
            worker = LiveRetention(current, self.policy(), Mock(), Mock())
            with SessionGuard(old):
                worker.process_pending(force=True)
                self.assertTrue((old / 'test_1').exists())
                self.assertEqual('QUEUED', json.loads(job_path(old, 'test_1').read_text())['status'])
            worker.process_pending(force=True)
            self.assertFalse((old / 'frames').exists())
            self.assertFalse((old / 'session.json').exists())
            self.assertTrue((old / 'report.pdf').exists())
            self.assertTrue((old / 'J1/test_1/report.pdf').exists())
            self.assertLessEqual(tree_bytes(old), 3_000_000)

    def test_restart_keeps_unfinished_tests_and_their_raw_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            old = parent / 'old'
            old.mkdir()
            self.fixture(old)
            (old / 'session_closed.json').unlink()
            unfinished = old / 'test_2'
            unfinished.mkdir()
            (unfinished / 'pickup.jpg').write_bytes(b'unfinished evidence')
            current = parent / 'current'
            current.mkdir()
            worker = LiveRetention(current, self.policy(), Mock(), Mock())
            worker.process_pending(force=True)
            self.assertTrue((old / 'J1/test_1/report.pdf').exists())
            self.assertTrue((unfinished / 'pickup.jpg').exists())
            self.assertTrue((old / 'frames').exists())
            self.assertFalse((old / 'session_closed.json').exists())

    def test_missing_second_clip_does_not_delete_or_release_any_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, record = fixtures.EvidenceRetentionTests().fixture(root)
            record['acid_clip'] = None
            enqueue_test(root, record, dict(id=1))
            release = Mock()
            with self.assertRaisesRegex(OSError, 'Missing acid_clip'):
                compact_test(root, job_path(root, 'test_1'), self.policy(), release)
            self.assertTrue((root / 'frames').exists())
            self.assertTrue((root / 'test_1').exists())
            release.assert_not_called()

    def test_normal_session_close_keeps_only_reports_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            compact_test(root, job_path(root, 'test_1'), self.policy())
            finalize_live_session(root, self.policy())
            files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.name.startswith('.')}
            self.assertEqual({'report.pdf', 'J1/test_1/report.pdf', 'J1/test_1/rubbing.avi', 'J1/test_1/acid.avi'}, files)

    def test_monitor_queues_automatic_cleanup_and_drains_it_on_stop(self):
        from continuous_testing import ContinuousMonitor
        from test_continuous_testing import Photos
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, record = fixtures.EvidenceRetentionTests().fixture(root)
            monitor = ContinuousMonitor(root, lambda: None, Photos(), Mock(), Mock())
            monitor.acid_models.get.return_value = (Mock(), Mock())
            monitor.jewels[1] = dict(id=1, image=str(root / 'J1.jpg'))
            try:
                monitor.start()
                deadline = time.monotonic() + 5
                while monitor.storage_worker is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNotNone(monitor.storage_worker)
                monitor._commit(record)
            finally:
                monitor.stop()
                monitor.join(20)
            self.assertFalse(monitor.is_alive())
            self.assertFalse((root / 'storage_error.json').exists())
            self.assertFalse((root / 'frames').exists())
            self.assertFalse((root / 'test_1').exists())
            self.assertTrue((root / 'J1/test_1/report.pdf').exists())
            self.assertTrue((root / 'J1/test_1/acid.avi').exists())
