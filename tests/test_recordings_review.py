import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import jewel_tracker_5 as app


class RecordingsReviewTests(unittest.TestCase):
    def test_main_app_starts_camera_free_reviewer_with_both_recording_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracker = root / 'jewel_tracker/jewel_tracker'
            lab = root / 'puritytesting/acid_temporal_lab'
            python = lab / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
            python.parent.mkdir(parents=True)
            python.touch()
            (lab / 'review_import.py').touch()
            owner = SimpleNamespace(recordings_review_process=None, add_log=Mock())
            with patch.object(app, 'SCRIPT_DIR', str(tracker)), patch.object(app.QtCore, 'QProcess') as process_class:
                process = process_class.return_value
                app.MainWindow.review_acid_recordings(owner)
                process.start.assert_called_once_with(str(python), [str(lab / 'review_import.py'),
                    '--captures', str(tracker), '--dataset', str(lab / 'dataset')])
                environment = process.setProcessEnvironment.call_args.args[0]
                self.assertFalse(environment.contains('QT_QPA_PLATFORM_PLUGIN_PATH'))
                self.assertIs(process, owner.recordings_review_process)
                process.state.return_value = 2
                process_class.NotRunning = 0
                app.MainWindow.review_acid_recordings(owner)
                self.assertEqual(1, process.start.call_count)

    def test_missing_lab_explains_location_and_does_not_start_process(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(app, 'SCRIPT_DIR', temporary), patch.object(
            app.QtWidgets.QMessageBox, 'information'
        ) as message, patch.object(app.QtCore, 'QProcess') as process:
            owner = SimpleNamespace(recordings_review_process=None, add_log=Mock())
            app.MainWindow.review_acid_recordings(owner)
            message.assert_called_once()
            self.assertIn('Your recordings are saved at', message.call_args.args[2])
            process.assert_not_called()
