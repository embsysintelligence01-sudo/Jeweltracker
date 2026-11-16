"""Handoff the completed testing session to packet sealing and number OCR."""
import gc
import importlib
import os
from pathlib import Path
import sys
import time
import threading
import copy

from PyQt5 import QtCore


def packet_wait_reason(worker):
    if worker is None or not worker.isRunning():
        return 'Connect the camera and complete a test attempt for each jewel.'
    tracks = getattr(worker, 'tracks', ())
    if not tracks or getattr(worker, 'mode', None) != 'LOCKED':
        return 'Waiting for jewel assignment.'
    if (getattr(worker, 'continuous_cycle_id', None) or getattr(worker, 'acid_token', None)
            or getattr(getattr(worker, 'continuous_monitor', None), 'active', None) is not None):
        return 'Waiting for active rubbing/acid evidence capture to finish.'
    if not getattr(getattr(worker, 'id_model', None), 'ready', False):
        return 'Preparing jewel identities.'
    missing = [tr.id for tr in tracks if not (getattr(tr, 'test_captured', False)
        or (getattr(tr, 'acid_result', None) and (tr.acid_result.get('prediction')
                                                or tr.acid_result.get('close_reason'))))]
    if missing:
        return 'Waiting for a completed test attempt: '+', '.join(f'J{jid}' for jid in missing)+'.'
    return ''


def ready_for_packet(worker):
    return not packet_wait_reason(worker)


def testing_snapshot(worker, directory, final=False):
    jewels = []
    for tr in worker.tracks:
        record = getattr(tr, 'acid_result', None) or {}
        prediction = copy.deepcopy(record.get('prediction'))
        reason = record.get('close_reason') or ('Analysis finished without a material result.' if final and not prediction else None)
        jewels.append(dict(id=tr.id, rubbed=bool(tr.rubbed), acid_done=bool(tr.acid_done),
            prediction=prediction, close_reason=reason,
            test_status='COMPLETE' if prediction and tr.acid_done else
                        'INCOMPLETE' if reason else 'PROCESSING',
            capture_finished=bool(getattr(tr, 'test_captured', False) or record.get('close_reason') or prediction)))
    return dict(directory=str(directory), jewels=jewels)


def packet_module(original):
    root = Path(os.environ.get('JEWEL_PACKET_DIR', Path(original.SCRIPT_DIR).parents[1] / 'packet_sealing'))
    required = ('bagnewmodel.onnx', 'fp_filter_resnet18_bag.onnx', 'stripmodel.onnx',
                'hsv_fp_filter_strip.pt', 'handseg model.onnx', 'pcnumber.onnx')
    missing = [name for name in required if not (root / 'model onnx' / name).is_file()]
    if missing:
        raise RuntimeError('Packet model files missing in '+str(root)+': '+', '.join(missing))
    if not (root / 'standalone' / 'app.py').is_file():
        raise RuntimeError('Packet application missing: '+str(root))
    # Running this script as __main__ must not import a second copy of the tracker.
    sys.modules.setdefault('jewel_tracker_5', original)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module('standalone.app')
    module.ffmpeg_path()  # Fail before stopping testing if video export is unavailable.
    return module


class PacketStage(QtCore.QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.window = None
        self.pending = None
        self.draining = None
        self.gold_owner = None
        self.testing = None
        self.testing_ready = threading.Event()
        self.module = None
        self.reset_requested = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    @property
    def active(self):
        return self.pending is not None or self.window is not None or self.draining is not None

    def begin(self):
        owner = self.owner
        if self.window is not None:
            self.window.show()
            self.window.raise_()
            return
        if self.pending is not None or not ready_for_packet(owner.worker):
            return
        worker = owner.worker
        if owner.acid_thread is not None or owner.rub_job is not None or owner.reset_timer.isActive():
            return
        if any(time.time()-tr.det_seen > 2.5 for tr in worker.tracks):
            owner.add_log('Return all tested jewels separately to the bench before starting packet sealing.', 'info')
            return
        try:
            self.module = packet_module(sys.modules[owner.__class__.__module__])
            if not worker.identity_slic:
                raise RuntimeError('Packet tracking needs SLIC. Start with run_tracker_slic.sh.')
        except Exception as exc:
            owner.add_log('Packet sealing unavailable: '+str(exc), 'alert')
            return
        self.pending = worker
        self.gold_owner = worker
        worker.preserve_gold_model = True
        self.testing_ready.clear()
        owner.btn_packet.setEnabled(False)
        owner.rub_timer.stop()
        self.lock_controls()
        owner.lbl_status.setText('Finishing camera handoff; jewel analysis will continue in the background…')
        worker.stop()

    def lock_controls(self):
        owner = self.owner
        for control in (owner.btn_conn, owner.cam_mode, owner.btn_reset, owner.btn_roi,
                        owner.btn_stone, owner.btn_test, owner.btn_clear, owner.btn_af):
            control.setEnabled(False)

    def tick(self):
        owner = self.owner
        if self.draining is not None and not self.draining.isRunning():
            self.finish_analysis()
        capture_stopped = (self.pending is not None and
            (getattr(self.pending, 'packet_capture_stopped', threading.Event()).is_set()
             or not self.pending.isRunning()))
        if capture_stopped:
            previous, self.pending = self.pending, None
            self.draining = previous
            try:
                self.testing = testing_snapshot(previous, previous.continuous_monitor.directory)
                testing = self.testing
                self.window = self.module.MainWindow(handoff=dict(worker=previous, testing=testing, testing_ready=self.testing_ready))
                self.window.roi = previous.roi.tolist()
                self.window.roi_override = True
                self.window.source.setText(str(previous.source))
                self.window.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
                self.window.session_exported.connect(self.exported)
                self.window.show()
                self.window.connect_camera()
                if not previous.isRunning():
                    self.finish_analysis()
                if self.draining is None:
                    owner.worker = None
                owner.lbl_status.setText('Packet sealing / OCR active. Pending jewel analysis continues; incomplete tests stay incomplete.')
                owner.phase_label.setText('PROCESS: Packet sealing → packet number OCR')
                owner.btn_reset.setEnabled(True)
            except Exception as exc:
                if self.window is not None and (self.window.worker is None or not self.window.worker.isRunning()):
                    self.window.close()
                    self.window.deleteLater()
                    self.window = None
                if self.draining is None:
                    owner.worker = None
                owner.set_running(False)
                owner.btn_conn.setEnabled(self.draining is None)
                owner.cam_mode.setEnabled(self.draining is None)
                owner.btn_af.setEnabled(self.draining is None)
                if self.draining is None:
                    owner.rub_timer.start(200)
                owner.add_log('Packet handoff failed; testing evidence retained: '+str(exc), 'alert')
        if self.reset_requested and self.window is not None:
            worker = self.window.worker
            if (worker is None or not worker.isRunning()) and self.draining is None:
                self.release_gold()
                self.window.close()
                self.window.deleteLater()
                self.window = None
                self.reset_requested = False
                owner.set_running(False)
                owner.btn_conn.setEnabled(True)
                owner.cam_mode.setEnabled(True)
                owner.btn_af.setEnabled(True)
                owner.rub_timer.start(200)
                owner.toggle_connect()
        reason = packet_wait_reason(owner.worker)
        owner.packet_status.setText(('Packet stage active; pending jewel analysis continues.' if self.draining is not None else
                                    'Packet stage active; jewel outcomes retained.') if self.window is not None else
                                   'Switching camera; background results are retained.' if self.pending is not None else
                                   reason or 'Ready for packing. Incomplete outcomes are allowed and remain recorded.')
        owner.btn_packet.setToolTip(reason or 'Capture attempts finished. Packing may start while result processing continues; incomplete outcomes are retained.')
        owner.btn_packet.setText('Open packet sealing / OCR' if self.window is not None else 'Packet Sealing + OCR')
        owner.btn_packet.setEnabled(self.window is not None or (self.pending is None
            and not owner.reset_timer.isActive() and ready_for_packet(owner.worker)))

    def finish_analysis(self):
        previous = self.draining
        if previous is None:
            return
        owner = self.owner
        if self.testing is not None:
            self.testing.update(testing_snapshot(previous, self.testing['directory'], final=True))
        previous.continuous_monitor = None
        if previous.stone is not None:
            previous.stone.model = None
        previous.stone = previous.rubber = None
        owner.rub_models.close()
        owner.acid_models.close()
        owner.rub_cycle = owner.rub_cleanup = owner.rub_job = None
        owner.rub_cycle_key = owner.rub_display_result = None
        self.draining = None
        if owner.worker is previous:
            owner.worker = None
        self.testing_ready.set()
        if self.window is None and self.pending is None:
            self.release_gold()
            owner.set_running(False)
            owner.btn_conn.setEnabled(True)
            owner.cam_mode.setEnabled(True)
            owner.btn_af.setEnabled(True)
            owner.rub_timer.start(200)
        gc.collect()

    def release_gold(self):
        if self.gold_owner is not None:
            self.gold_owner.model.close()
            self.gold_owner = None

    def exported(self, result):
        owner = self.owner
        from PyQt5 import QtCore
        path = str(Path(result['directory']) / result['report'])
        url = QtCore.QUrl.fromLocalFile(path).toString()
        owner.report_label.setText(f'<a href="{url}">Jewel testing + packet sealing + OCR report</a>')
        number = result.get('packet_number', {})
        sealing = 'Sealed (inferred)' if result['sealed'] else 'Sealing incomplete'
        label = 'Packet number: '+number['text'] if number.get('status') == 'confirmed' else 'Packet number not confirmed'
        owner.phase_label.setText(sealing+' | '+label)
        owner.add_log('Packet evidence saved: '+str(result['directory']), 'info')

    def reset(self):
        if self.pending is not None:
            return
        if self.window is not None:
            self.reset_requested = True
            self.owner.btn_reset.setEnabled(False)
            self.window.finish_session()

    def close(self):
        if self.pending is not None:
            return False
        if self.window is not None:
            worker = self.window.worker
            if worker is not None and worker.isRunning():
                self.window.finish_session()
                self.owner.lbl_status.setText('Saving packet evidence. Close again after export finishes.')
                return False
            self.window.close()
        if self.draining is not None:
            self.owner.lbl_status.setText('Waiting for pending jewel analysis and evidence to finish…')
            return False
        self.release_gold()
        return True
