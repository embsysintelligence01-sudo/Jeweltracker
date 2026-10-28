import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

import jewel_tracker_5 as app


class ContinuousUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        with patch.object(QtCore.QTimer, 'singleShot'):
            self.window = app.MainWindow()

    def tearDown(self):
        self.window.close()

    def test_main_window_has_no_per_test_retakes_or_manual_handoff(self):
        self.window.show()
        self.qt.processEvents()
        for widget in (self.window.btn_rub_reference, self.window.btn_start_acid,
                       self.window.btn_stone_ready, self.window.btn_cancel_acid):
            self.assertFalse(widget.isVisible())
        self.assertTrue(self.window.btn_test.isVisible())

    def test_missing_mask_note_does_not_replace_acid_result_in_table(self):
        record = dict(prediction=dict(prediction='non_gold', confidence=.94),
                      rubbing_status='UNCONFIRMED', mask_status='UNCONFIRMED',
                      record_path='example/manifest.json')
        self.window.show_table([(1, None, False, None, True, True, record, False, None)])
        self.assertIn('NON-GOLD', self.window.table.item(0, 5).text())
        self.assertEqual('UNCONFIRMED', self.window.table.item(0, 2).text())

    def test_visible_jewel_requires_actual_hand_colour_contact_for_in_hand(self):
        frame = np.full((100, 100, 3), 220, np.uint8)
        gold = cv2.cvtColor(np.uint8([[[25, 140, 200]]]), cv2.COLOR_HSV2BGR)[0,0]
        skin = cv2.cvtColor(np.uint8([[[10, 180, 180]]]), cv2.COLOR_HSV2BGR)[0,0]
        frame[30:60, 30:60] = gold
        track = SimpleNamespace(det_mask=np.ones((30,30), np.uint8), det_origin=(30,30))
        self.assertFalse(app.Worker.hand_contact(track, frame))
        frame[30:60, 55:70] = skin
        self.assertTrue(app.Worker.hand_contact(track, frame))

    def test_first_jewel_done_remains_visible_while_second_is_rubbing(self):
        worker = app.Worker(0, [(0,0),(399,0),(399,299),(0,299)], continuous=True)
        self.window.worker = worker
        worker.workflow.connect(self.window.continuous_update)
        worker.jewel_table.connect(self.window.show_table)
        for jid in (1, 2):
            box = np.array([30*jid, 30., 30*jid+20, 60.])
            worker.tracks.append(app.Track(jid, app.Det(box, np.array([30*jid+10,45.]), 600, .9), 0))
        worker.continuous_event('begin', jewel_id=2, token='second')
        worker.continuous_event('phase', jewel_id=2, token='second', phase='RUBBING', message='Rubbing J2')
        worker.continuous_event('result', record=dict(jewel_id=1, capture_complete=True,
            prediction=dict(prediction='non_gold', confidence=.94), rubbing_status='CONFIRMED_LIVE',
            record_path='test/manifest.json'))
        self.assertIn('J1 DONE', self.window.done_label.text())
        self.assertIn('J2', self.window.phase_label.text())
        self.assertIn('RUBBING', self.window.phase_label.text())
        self.assertEqual('DONE', self.window.table.item(0,4).text())
        self.window.worker = None


if __name__ == '__main__':
    unittest.main()
