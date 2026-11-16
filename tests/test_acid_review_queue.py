import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acid_review_queue import preserve_review_sample


class ReviewQueueTests(unittest.TestCase):
    def test_pending_frames_survive_cleanup_and_are_not_recreated_after_review(self):
        from simple_session import _publish_and_clean
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'session'
            raw = root / 'test_1'
            raw.mkdir(parents=True)
            export = {}
            for phase in ('baseline', 'transition', 'observation'):
                path = raw / 'training_sequence' / phase / '0.png'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'lossless frame')
                export[phase + '_frames'] = [dict(path=str(path.relative_to(raw)))]
            manifest = raw / 'manifest.json'
            manifest.write_text(json.dumps(dict(temporal_training_capture=export)))
            queued = Path(preserve_review_sample(manifest))
            self.assertEqual(root / 'pending_review/test_1/manifest.json', queued)
            self.assertFalse(json.loads(queued.read_text()).get('review_capture_path'))
            moved = Path(tmp) / 'reviewed'
            queued.parent.rename(moved)
            preserve_review_sample(manifest)
            self.assertFalse(queued.exists())
            moved.rename(queued.parent)
            stage = root / '.simple-staging'
            stage.mkdir()
            (root / 'report.pdf').write_bytes(b'saved report')
            with patch('simple_session._validate_files'):
                _publish_and_clean(root, stage, dict(files={'report.pdf': {}}, previous_bytes=0, jewel_sizes=[]))
            self.assertFalse(raw.exists())
            self.assertTrue(queued.exists())
            self.assertEqual(b'lossless frame', (queued.parent / 'training_sequence/baseline/0.png').read_bytes())
            self.assertEqual(b'saved report', (root / 'report.pdf').read_bytes())

    def test_copy_failure_preserves_original_without_marking_it_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'test_1'
            raw.mkdir()
            (raw / 'frame.png').write_bytes(b'frame')
            export = {phase + '_frames': [dict(path='frame.png')]
                      for phase in ('baseline', 'transition', 'observation')}
            manifest = raw / 'manifest.json'
            manifest.write_text(json.dumps(dict(temporal_training_capture=export)))
            before = manifest.read_bytes()
            with patch('acid_review_queue.shutil.copyfile', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    preserve_review_sample(manifest)
            self.assertEqual(before, manifest.read_bytes())
            self.assertEqual([], list((raw.parent / 'pending_review').iterdir()))
