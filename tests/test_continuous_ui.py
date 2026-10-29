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

    def test_provisional_acid_is_displayed_as_possible_not_confirmed(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        self.window.worker=worker
        worker.workflow.connect(self.window.continuous_update)
        worker.continuous_event('begin',jewel_id=1,token='candidate-cycle')
        worker.continuous_event('acid_detection',jewel_id=1,token='candidate-cycle',
                               detected=False,provisional=True,timestamp=100,bbox=(128,107,143,139))
        self.assertIn('POSSIBLE ACID',self.window.acid_instruction.text())
        self.assertFalse(worker.continuous_acid['detected'])
        worker.continuous_event('acid_detection',jewel_id=1,token='candidate-cycle',
                               detected=True,timestamp=100.1,bbox=(128,107,143,139))
        self.assertIn('ACID VERIFIED',self.window.acid_instruction.text())
        self.window.worker=None

    def test_visible_jewel_requires_actual_hand_colour_contact_for_in_hand(self):
        frame = np.full((100, 100, 3), 220, np.uint8)
        gold = cv2.cvtColor(np.uint8([[[25, 140, 200]]]), cv2.COLOR_HSV2BGR)[0,0]
        skin = cv2.cvtColor(np.uint8([[[10, 180, 180]]]), cv2.COLOR_HSV2BGR)[0,0]
        frame[30:60, 30:60] = gold
        track = SimpleNamespace(det_mask=np.ones((30,30), np.uint8), det_origin=(30,30))
        self.assertFalse(app.Worker.hand_contact(track, frame))
        frame[30:60, 55:70] = skin
        self.assertTrue(app.Worker.hand_contact(track, frame))

    def test_area_updates_change_overlay_without_accepting_a_stale_cycle(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        self.window.worker=worker
        worker.workflow.connect(self.window.continuous_update)
        worker.continuous_event('begin',jewel_id=1,token='area-cycle')
        worker.continuous_event('acid_area',jewel_id=1,token='area-cycle',detected=True,
                               timestamp=101,bbox=(110,90,160,160))
        self.assertEqual((110,90,160,160),worker.continuous_acid['bbox'])
        self.assertIn('original 5-second timer continues',self.window.acid_instruction.text())
        worker.continuous_event('acid_area',jewel_id=1,token='old-cycle',detected=True,
                               timestamp=102,bbox=(0,0,10,10))
        self.assertEqual((110,90,160,160),worker.continuous_acid['bbox'])
        self.window.worker=None

    def test_stale_begin_and_result_are_rejected_when_reset_races_with_emission(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        worker.flow_revision=1
        worker.continuous_event('begin',jewel_id=1,token='late-old',revision=0)
        self.assertIsNone(worker.continuous_cycle_id)
        box=np.array([30.,30.,60.,60.])
        worker.tracks.append(app.Track(1,app.Det(box,np.array([45.,45.]),900,.9),0))
        worker.continuous_event('begin',jewel_id=1,token='new',revision=1)
        worker.continuous_event('result',record=dict(jewel_id=1,cycle_id='new',revision=0,
            capture_complete=True,prediction=dict(prediction='gold_22k',confidence=.98),
            rubbing_status='CONFIRMED_LIVE',record_path='old/manifest.json'))
        self.assertFalse(worker.tracks[0].done)
        self.assertEqual('Recording',worker.tracks[0].acid_result['rubbing_status'])

    def test_contact_and_new_mark_have_separate_images(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        self.window.worker=worker
        worker.workflow.connect(self.window.continuous_update)
        box=np.array([30.,30.,60.,60.])
        worker.tracks.append(app.Track(1,app.Det(box,np.array([45.,45.]),900,.9),0))
        worker.continuous_event('begin',jewel_id=1,token='cycle')
        contact=np.full((100,100,3),80,np.uint8)
        mark=np.full((100,100,3),160,np.uint8)
        worker.continuous_event('contact',jewel_id=1,token='cycle',photo=contact,timestamp=100,
                                box=(30,30,60,60),confirmed=False)
        worker.continuous_event('mask',jewel_id=1,token='cycle',photo=mark,mask=np.ones((100,100),np.uint8),stone=(0,0,100,100))
        np.testing.assert_array_equal(contact,worker.tracks[0].rub_img)
        np.testing.assert_array_equal(mark,worker.tracks[0].new_rub_img)
        self.assertFalse(self.window.contact_preview.pixmap().isNull())
        self.assertFalse(self.window.rub_preview.pixmap().isNull())
        self.assertIn('unverified',self.window.contact_instruction.text())
        self.window.worker=None

    def test_old_cycle_results_cannot_cross_a_reset_with_reused_jewel_ids(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        box=np.array([30.,30.,60.,60.])
        worker.tracks.append(app.Track(1,app.Det(box,np.array([45.,45.]),900,.9),0))
        worker.continuous_event('begin',jewel_id=1,token='old')
        worker.flow_revision+=1
        worker.continuous_event('begin',jewel_id=1,token='new')
        worker.continuous_event('result',record=dict(jewel_id=1,cycle_id='old',prediction=dict(prediction='gold_22k',confidence=.98),
            capture_complete=True,rubbing_status='CONFIRMED_LIVE',record_path='old/manifest.json'))
        self.assertEqual('Recording',worker.tracks[0].acid_result['rubbing_status'])
        self.assertFalse(worker.tracks[0].done)
        self.assertEqual('new',worker.continuous_cycle_id)

    def test_late_old_result_does_not_replace_latest_finished_jewels_display(self):
        worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        self.window.worker=worker
        worker.workflow.connect(self.window.continuous_update)
        for jid in (1,2):
            box=np.array([30.*jid,30.,30.*jid+20,60.])
            worker.tracks.append(app.Track(jid,app.Det(box,np.array([30.*jid+10,45.]),600,.9),0))
        worker.continuous_event('begin',jewel_id=1,token='first')
        worker.continuous_event('captured',jewel_id=1,token='first')
        worker.continuous_event('begin',jewel_id=2,token='second')
        worker.continuous_event('captured',jewel_id=2,token='second')
        worker.continuous_event('result',record=dict(jewel_id=2,cycle_id='second',capture_complete=True,
            prediction=dict(prediction='non_gold',confidence=.95),rubbing_status='UNCONFIRMED',record_path='second/manifest.json'))
        text=self.window.phase_label.text()
        worker.continuous_event('result',record=dict(jewel_id=1,cycle_id='first',capture_complete=False,
            prediction=None,rubbing_status='UNCONFIRMED',close_reason='NEXT_JEWEL_PICKUP',record_path='first/manifest.json'))
        self.assertEqual(text,self.window.phase_label.text())
        self.assertIn('J2',text)
        self.assertEqual('NEXT_JEWEL_PICKUP',worker.tracks[0].acid_result['close_reason'])
        self.window.worker=None

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
