"""Testing-to-packing handoff retains identity and waits for native workers."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
from PyQt5 import QtCore, QtWidgets
import jewel_tracker_5 as original
from packet_stage import ready_for_packet, packet_module, testing_snapshot


class PacketStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.packet = packet_module(original)

    def track(self, jid=7):
        return SimpleNamespace(id=jid, done=True, acid_result=dict(prediction=dict(
            prediction='gold_22k', confidence=.98)), pos=np.array([60., 60.]),
            box=np.array([40., 40., 80., 80.]), det_seen=time.time(), rubbed=True, acid_done=True)

    def test_gate_requires_every_result_and_no_active_cycle(self):
        worker = SimpleNamespace(isRunning=lambda: True, mode='LOCKED',
                                 id_model=SimpleNamespace(ready=True), tracks=[self.track(), self.track(9)])
        self.assertTrue(ready_for_packet(worker))
        worker.tracks[1].acid_result = None
        self.assertFalse(ready_for_packet(worker))
        worker.tracks[1] = self.track(9)
        worker.continuous_cycle_id = 'running'
        self.assertFalse(ready_for_packet(worker))
        worker.continuous_cycle_id = None
        worker.tracks = []
        self.assertFalse(ready_for_packet(worker))

    def test_bridge_reuses_ids_gallery_and_gold_backend(self):
        from standalone.tracking import TrackingBridge
        backend = object()
        previous = SimpleNamespace(tracks=[self.track(7), self.track(9)],
            model=SimpleNamespace(backend=backend), id_model=SimpleNamespace(ready=True),
            identity_extractor=object(), frame_wh=(640, 480))
        with patch('standalone.tracking.OpenVINOYolo', side_effect=AssertionError('Must reuse gold')):
            bridge = TrackingBridge([(0, 0), (640, 0), (640, 480)], previous=previous)
        self.assertIs(bridge.tracks, previous.tracks)
        self.assertIs(bridge.id_model, previous.id_model)
        self.assertIs(bridge.model.model, previous.model)
        self.assertEqual([7, 9], list(bridge.bench_anchors))
        self.assertEqual('LOCKED', bridge.mode)
        bridge.close()

    def test_transition_waits_then_unloads_testing_caches(self):
        with patch.object(QtCore.QTimer, 'singleShot'):
            window = original.MainWindow()
        busy = [True]
        previous = SimpleNamespace(isRunning=lambda: busy[0], mode='LOCKED',
            id_model=SimpleNamespace(ready=True), tracks=[self.track()], identity_slic=True,
            continuous_monitor=SimpleNamespace(directory=Path('/tmp/test-evidence')),
            stone=SimpleNamespace(model=object()), rubber=object(),
            roi=np.array([(0, 0), (100, 0), (100, 100)]), source=0, model=Mock())
        previous.stop = lambda: None
        window.worker = previous
        stage = window.packet_stage
        with patch('packet_stage.packet_module', return_value=self.packet):
            stage.begin()
        self.assertIs(stage.pending, previous)
        self.assertFalse(window.btn_conn.isEnabled())
        stage.tick()
        self.assertIsNone(stage.window)
        busy[0] = False
        with patch.object(self.packet.MainWindow, 'connect_camera') as connect:
            def setup_camera():
                stage.window.worker = SimpleNamespace(exported=SimpleNamespace(connect=lambda slot: None),
                                                      isRunning=lambda: False)
            connect.side_effect = setup_camera
            with patch.object(window.acid_models, 'close', wraps=window.acid_models.close) as acid_close, \
                 patch.object(window.rub_models, 'close', wraps=window.rub_models.close) as rub_close:
                stage.tick()
                acid_close.assert_called_once()
                rub_close.assert_called_once()
        self.assertIsNone(previous.stone)
        self.assertIsNone(previous.continuous_monitor)
        self.assertIs(stage.window.handoff['worker'], previous)
        self.assertEqual(7, stage.window.handoff['testing']['jewels'][0]['id'])
        self.assertIsNone(window.worker)
        window.close()

    def test_incomplete_and_processing_attempts_enable_packing(self):
        tr = self.track()
        tr.done = tr.acid_done = False
        tr.acid_result = dict(prediction=None, close_reason='REACTION_COMPLETE')
        worker = SimpleNamespace(isRunning=lambda: True, mode='LOCKED',
            id_model=SimpleNamespace(ready=True), tracks=[tr])
        self.assertTrue(ready_for_packet(worker))
        tr.acid_result = None
        tr.test_captured = True
        self.assertTrue(ready_for_packet(worker))
        worker.continuous_monitor = SimpleNamespace(active={'jewel_id': 7})
        self.assertFalse(ready_for_packet(worker))

    def test_incomplete_snapshot_retains_missing_prediction(self):
        tr = self.track()
        tr.done = tr.acid_done = False
        tr.acid_result = dict(prediction=None, close_reason='PARTIAL_REACTION')
        result = testing_snapshot(SimpleNamespace(tracks=[tr]), '/tmp/session')['jewels'][0]
        self.assertIsNone(result['prediction'])
        self.assertEqual('INCOMPLETE', result['test_status'])
        self.assertEqual('PARTIAL_REACTION', result['close_reason'])

    def test_packet_window_starts_before_analysis_drains_and_models_stay_loaded(self):
        import threading
        with patch.object(QtCore.QTimer, 'singleShot'):
            window = original.MainWindow()
        busy = [True]
        tr = self.track()
        tr.test_captured = True
        tr.acid_result = None
        previous = SimpleNamespace(isRunning=lambda: busy[0], mode='LOCKED',
            id_model=SimpleNamespace(ready=True), tracks=[tr], identity_slic=True,
            continuous_monitor=SimpleNamespace(directory=Path('/tmp/test-evidence'), active=None),
            stone=SimpleNamespace(model=object()), rubber=object(),
            packet_capture_stopped=threading.Event(), model=Mock(),
            roi=np.array([(0, 0), (100, 0), (100, 100)]), source=0, stop=lambda: None)
        window.worker = previous
        stage = window.packet_stage
        with patch('packet_stage.packet_module', return_value=self.packet):
            stage.begin()
        previous.packet_capture_stopped.set()
        with patch.object(self.packet.MainWindow, 'connect_camera') as connect:
            def setup_camera():
                stage.window.worker = SimpleNamespace(isRunning=lambda: False)
            connect.side_effect = setup_camera
            with patch.object(window.acid_models, 'close', wraps=window.acid_models.close) as close:
                stage.tick()
                self.assertIsNotNone(stage.window)
                close.assert_not_called()
                self.assertFalse(stage.testing_ready.is_set())
                self.assertIsNone(stage.testing['jewels'][0]['prediction'])
                self.assertFalse(stage.close())
                tr.acid_result = dict(prediction=None, close_reason='PARTIAL_REACTION')
                busy[0] = False
                stage.tick()
                close.assert_called_once()
                self.assertTrue(stage.testing_ready.is_set())
                self.assertEqual('INCOMPLETE', stage.testing['jewels'][0]['test_status'])
        window.close()

    def test_acid_cache_releases_all_models(self):
        cache = original.AcidModelCache()
        cache._models = (object(), object())
        cache._ph_model = object()
        cache.close()
        self.assertIsNone(cache._models)
        self.assertIsNone(cache._ph_model)


if __name__ == '__main__':
    unittest.main()
