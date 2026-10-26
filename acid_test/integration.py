"""Qt5 adapter sharing Jewel Tracker's camera, never opening a second camera."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from PyQt5 import QtCore

from .runtime import TemporalMobileNet, latest_temporal_model_path, DETECTOR_PATH, LAB_ROOT, _sha256
from .onnx_detector import OnnxAcidDetector
from .sequence import AcidSequence
from .flow import bbox_iou
from acidphintegration.auto_sop import (
    AutoSOP, analyze_paper, plausible_match, PAPER_CLASS_ID, CONFIDENCE,
)
from acidphintegration.openvino_obb import OpenVINOOBB

PH_MODEL_PATH = Path(__file__).resolve().parents[1] / 'acidphintegration/acidph_openvino/acidph.xml'


class AcidModelCache:
    """Window-owned models: lazy initialization, one warm-up, no idle inference."""
    def __init__(self):
        self._lock = threading.Lock()
        self._models = None
        self._ph_model = None
        self.detector_sha256 = None

    def get_ph(self):
        with self._lock:
            if self._ph_model is None:
                try:
                    model = OpenVINOOBB(PH_MODEL_PATH)
                except ModuleNotFoundError as exc:
                    if exc.name != 'openvino':
                        raise
                    raise RuntimeError('OpenVINO is missing in this Python environment; install requirements.txt using this app\'s Python interpreter') from exc
                model.detect(np.zeros((320, 320, 3), dtype=np.uint8), PAPER_CLASS_ID, CONFIDENCE)
                self._ph_model = model
            return self._ph_model

    def get(self):
        with self._lock:
            if self._models is None:
                detector = OnnxAcidDetector(DETECTOR_PATH)
                classifier = TemporalMobileNet(latest_temporal_model_path())
                # Synthetic warm-up only. Live frames are used after the
                # operator starts a confirmed rubbed jewel's capture.
                blank = np.zeros((320, 320, 3), dtype=np.uint8)
                detector.predict(blank, 0.60)
                classifier.predict([blank], [blank], [blank])
                self.detector_sha256 = _sha256(DETECTOR_PATH)
                self._models = detector, classifier
            return self._models


def acid_crop(seq):
    """The detector's acid box within the latest padded observation crop."""
    if not seq.observation_frames:
        return None
    rx, ry, _, _ = seq.reaction_bbox
    x1, y1, x2, y2 = seq.detector_bbox
    frame = seq.observation_frames[-1]
    crop = frame[max(0, y1 - ry):y2 - ry, max(0, x1 - rx):x2 - rx]
    return crop.copy() if crop.size else None


class AcidTestThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(object)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)
    crop_ready = QtCore.pyqtSignal(object)
    target_ready = QtCore.pyqtSignal(object)

    def __init__(self, context, observation_seconds=5, acid_grade='', models=None):
        super().__init__()
        self.context = context
        self.token = context['token']
        self.jewel_id = context['jewel_id']
        self.duration, self.acid_grade = observation_seconds, acid_grade
        self.cancelled = threading.Event()
        self.models = models if models is not None else AcidModelCache()
        self.ph_verification_enabled = context.get('ph_verification_enabled', True)
        # Verification belongs to this attempt, never to the cached model.
        self.ph_verification = None
        self.ph_images = {}
        self.ph_frame_shape = None
        self._stone_ready = threading.Event()
        self._waiting_for_stone = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def confirm_stone_ready(self):
        """Accept one operator confirmation after pH verification or its explicit bypass."""
        if (self.cancelled.is_set() or not self._waiting_for_stone.is_set()
                or self._stone_ready.is_set()):
            return False
        self._stone_ready.set()
        return True

    def payload(self, **values):
        return dict(token=self.token, jewel_id=self.jewel_id, **values)

    def verify_ph_paper(self):
        self.progress.emit(self.payload(phase='PH_LOADING', message='Loading pH-paper model (OpenVINO CPU)...'))
        loading_started = time.perf_counter()
        model = self.models.get_ph()
        self.progress.emit(self.payload(
            phase='PH_READY', model_path=str(PH_MODEL_PATH),
            message=f'pH-paper model ready on CPU ({time.perf_counter() - loading_started:.1f}s).',
        ))
        sop = AutoSOP()
        started = last_fresh = time.monotonic()
        last_ts = None
        shape = None
        self.progress.emit(self.payload(phase='PH_FRESH', message='Show unused yellow pH paper. Keep acid OFF the stone.'))
        while not self.cancelled.wait(0.1):
            frame, ts = self.context['grabber'].latest()
            now = time.monotonic()
            if now - started > 120:
                raise RuntimeError('pH-paper reaction was not verified within 120 seconds; retry with fresh yellow paper')
            if frame is None or ts == last_ts or time.time() - ts > 3:
                if now - last_fresh > 3:
                    raise RuntimeError('Camera stream lost; pH check incomplete. Restore the camera and retry')
                continue
            last_ts, last_fresh = ts, now
            if shape is not None and frame.shape != shape:
                raise RuntimeError('Camera resolution changed; repeat the pH check')
            shape = frame.shape
            candidates = model.detect(frame, PAPER_CLASS_ID, CONFIDENCE)
            matched = [d for d in candidates if plausible_match(sop.prev_bbox, d['bbox'])]
            detection = max(matched, key=lambda d: d['score'], default=None)
            metrics = analyze_paper(detection['crop'], detection['mask']) if detection else None
            passed = sop.update(detection, metrics, time.monotonic())
            if detection is not None:
                crop = detection['crop']
                scale = min(1.0, 224 / max(crop.shape[:2]))
                if scale < 1:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                self.crop_ready.emit(self.payload(crop=crop, kind='ph_paper'))
            if passed:
                self.ph_frame_shape = frame.shape
                self.ph_verification = dict(
                    result='ACIDIC_REACTION_VERIFIED', reagent_identity_verified=False,
                    physical_liquid_contact_proven=False, bbox=list(detection['bbox']),
                    detail=sop.end_detail, model=PH_MODEL_PATH.name,
                    model_sha256=_sha256(PH_MODEL_PATH),
                    weights_sha256=_sha256(PH_MODEL_PATH.with_suffix('.bin')),
                )
                self.ph_images = {
                    'baseline': sop.baseline_image.copy(), 'reaction': detection['crop'].copy(),
                    'reaction_mask': metrics['overlay'].copy(),
                }
                self.progress.emit(self.payload(phase='PH_VERIFIED', message='pH reaction verified. Remove paper/hands, then click Ready for stone test.'))
                return True
            phase = 'PH_FRESH' if sop.state == 'WAIT_FRESH' else 'PH_REACTION'
            instruction = ('Show unused yellow pH paper.' if phase == 'PH_FRESH'
                           else 'Add acid to the SAME pH paper and keep it visible.')
            self.progress.emit(self.payload(phase=phase, message=f'{instruction} {sop.last_diagnostic}. Keep acid OFF the stone.'))
        return False

    def wait_for_stone_ready(self):
        if self.ph_verification is None or self.cancelled.is_set():
            return False
        self._waiting_for_stone.set()
        try:
            self.progress.emit(self.payload(
                phase='WAIT_FOR_STONE_READY',
                message=('pH verified. ' if self.ph_verification_enabled else 'pH verification OFF. ')
                        + 'Remove paper/hands from the stone. Click Ready for stone test when ready; keep acid OFF the stone.',
            ))
            last_ts, last_fresh, last_hidden_message = None, time.monotonic(), -1e9
            # pH verification survives a hidden or repositioned stone.
            while not self.cancelled.wait(0.1):
                frame, ts = self.context['grabber'].latest()
                now = time.monotonic()
                if frame is None or ts == last_ts or time.time() - ts > 3:
                    if now - last_fresh > 3:
                        raise RuntimeError('Camera stream lost while waiting for stone readiness; retry the test')
                    continue
                last_ts, last_fresh = ts, now
                if frame.shape != self.ph_frame_shape:
                    raise RuntimeError('Camera resolution changed; retry the test')
                stone = self.context['stone_getter']()
                if stone is None:
                    self._stone_ready.clear()
                    if now-last_hidden_message >= .5:
                        self.progress.emit(self.payload(phase='WAIT_FOR_STONE_READY',
                            message='Stone is not visible. Uncover it and click Ready. The pH result is retained.'))
                        last_hidden_message = now
                    continue
                if self._stone_ready.is_set():
                    target = self.context.get('rub_target')
                    if target is not None:
                        try:
                            located = target.locate(frame)
                        except RuntimeError as exc:
                            self._stone_ready.clear()
                            self.progress.emit(self.payload(phase='WAIT_FOR_STONE_READY',
                                message=f'{exc}. pH progress is retained; keep acid OFF the stone.'))
                            continue
                        self.context = dict(self.context, **located)
                        self.target_ready.emit(self.payload(**located))
                        return not self.cancelled.is_set()
                    previous = np.asarray(self.context['stone'], float)
                    current = np.asarray(stone, float)
                    scale = (current[2:]-current[:2])/np.maximum(previous[2:]-previous[:2], 1.)
                    streak = np.asarray(self.context['streak'], float).reshape(2, 2)
                    streak = np.rint(current[:2]+(streak-previous[:2])*scale).astype(int)
                    self.context = dict(self.context, stone=tuple(map(int, stone)),
                                        streak=tuple(map(int, streak.ravel())))
                    return not self.cancelled.is_set()
            return False
        finally:
            self._waiting_for_stone.clear()

    def run(self):
        try:
            if self.ph_verification_enabled:
                if not self.verify_ph_paper() or self.cancelled.is_set():
                    return
            else:
                frame, ts = self.context['grabber'].latest()
                if frame is None or time.time() - ts > 3:
                    raise RuntimeError('Camera is unavailable or stale; retry the test')
                self.ph_frame_shape = frame.shape
                self.ph_verification = dict(result='SKIPPED_BY_OPERATOR', enabled=False)
                self.progress.emit(self.payload(phase='PH_SKIPPED', message='pH verification OFF for this test.'))
            if self.cancelled.is_set():
                return
            if not self.wait_for_stone_ready():
                return
            self.progress.emit(self.payload(message='Preparing cached acid models...', phase='LOADING'))
            detector, classifier = self.models.get()
            if self.cancelled.is_set():
                return
            seq = AcidSequence(self.context['stone'], self.context['streak'], self.duration)
            if self.context.get('roi_mask') is not None:
                seq.streak_mask = (self.context['roi_mask'] > 0).astype(np.uint8)*255
            last_ts, last_fresh, last_detection, last_progress = None, time.monotonic(), -1e9, -1e9
            future = None
            last_phase = None
            preview_count = 0
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='acid-detector') as pool:
                while not self.cancelled.is_set():
                    # latest() copies a full camera frame. Poll at the pouring
                    # capture rate rather than copying 1440p frames in a busy loop.
                    if self.cancelled.wait(1 / 15):
                        return
                    frame, ts = self.context['grabber'].latest()
                    now = time.monotonic()
                    if frame is None or ts == last_ts:
                        if now - last_fresh > 3:
                            raise RuntimeError('Camera stream lost; acid test incomplete. Restore the camera and retry')
                        self.cancelled.wait(0.01)
                        continue
                    last_ts, last_fresh = ts, now
                    stone = self.context['stone_getter']()
                    if stone is None or bbox_iou(stone, self.context['stone']) < 0.7:
                        raise RuntimeError('Stone moved or disappeared; repeat the acid test with a still stone')
                    seq.push(frame, now)
                    if len(seq.observation_frames) != preview_count:
                        preview_count = len(seq.observation_frames)
                        crop = acid_crop(seq)
                        if crop is not None:
                            scale = min(1.0, 224 / max(crop.shape[:2]))
                            if scale < 1:
                                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                            self.crop_ready.emit(self.payload(crop=crop))
                    if future is not None and future.done():
                        detections, _ = future.result()
                        seq.accept_detections(detections, now)
                        # Buffer decoding can take seconds. Progress must use
                        # the same fresh time base as the observation window.
                        now = time.monotonic()
                        future = None
                    if seq.state == 'WAIT_FOR_ACID' and future is None and now - last_detection >= 0.2:
                        future = pool.submit(detector.predict, frame.copy(), 0.60)
                        last_detection = now
                    if seq.state != last_phase or now - last_progress >= 0.1:
                        self.progress.emit(self.payload(phase=seq.state, message=seq.instruction(now)))
                        last_progress = now
                        last_phase = seq.state
                    if seq.state == 'PROCESSING':
                        prediction = classifier.predict(seq.baseline_frames, seq.transition_frames, seq.observation_frames)
                        if self.cancelled.is_set():
                            return
                        result = self.save_capture(seq, prediction)
                        if not self.cancelled.is_set():
                            self.completed.emit(self.payload(result=result))
                        return
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(self.payload(message=str(exc), traceback=traceback.format_exc()))

    def save_capture(self, seq, prediction):
        if (self.ph_verification is None or (self.ph_verification_enabled
                and self.ph_verification.get('result') != 'ACIDIC_REACTION_VERIFIED')):
            raise RuntimeError('pH-paper verification is required before saving an acid result')
        capture_id = f'{self.context["batch_id"]}_J{self.jewel_id}_{uuid.uuid4().hex[:12]}'
        root = LAB_ROOT / 'captures'
        root.mkdir(exist_ok=True)
        temporary, final = root / ('.' + capture_id + '.tmp'), root / capture_id
        temporary.mkdir()
        record = {
            'dataset_version': 'jewel_acid_prediction_v1',
            'created_at': datetime.now().astimezone().isoformat(),
            'batch_id': self.context['batch_id'], 'jewel_id': self.jewel_id,
            'prediction': prediction, 'actual_label': None,
            'acid_grade': self.acid_grade, 'observation_seconds': self.duration,
            'baseline_seconds': seq.baseline_seconds, 'capture_fps': 5, 'transition_fps': 15,
            'camera_shape': list(seq.shape), 'stone_bbox': list(seq.stone), 'streak_bbox': list(seq.streak),
            'detector_model': DETECTOR_PATH.name,
            'detector_sha256': self.models.detector_sha256 or _sha256(DETECTOR_PATH),
            'detector_bbox': list(seq.detector_bbox), 'reaction_bbox': list(seq.reaction_bbox),
            'detector_confidence': seq.detector_confidence,
            'record_path': str(final / 'manifest.json'),
            'ph_verification': dict(self.ph_verification),
            'ph_verification_enabled': self.ph_verification_enabled,
        }
        try:
            if seq.streak_mask is not None:
                if not cv2.imwrite(str(temporary / 'new_rub_roi.png'), seq.streak_mask):
                    raise RuntimeError('Could not save the confirmed new-rub ROI')
                record['new_rub_roi'] = 'new_rub_roi.png'
                record['acid_roi_rule'] = dict(center_inside_mask=False, center_inside_streak=False,
                                               center_inside_stone=True, region='detector_bbox',
                                               min_overlap_of_smaller=.5)
            target = self.context.get('rub_target')
            if target is not None:
                record['rub_preparation'] = dict(target.metadata)
                record['rub_preparation']['images'] = {}
                for name, image in target.evidence.items():
                    filename = name + '.png'
                    if not cv2.imwrite(str(temporary / filename), image):
                        raise RuntimeError('Could not save before/after rubbing evidence')
                    record['rub_preparation']['images'][name] = filename
            ph_files = {}
            for name, image in self.ph_images.items():
                filename = f'ph_{name}.png'
                if not cv2.imwrite(str(temporary / filename), image):
                    raise RuntimeError('Could not save pH-paper evidence')
                ph_files[name] = filename
            record['ph_verification']['images'] = ph_files
            crop = acid_crop(seq)
            if crop is not None:
                if not cv2.imwrite(str(temporary / 'acid_preview.png'), crop):
                    raise RuntimeError('Could not save acid preview')
                record['acid_preview'] = 'acid_preview.png'
            for name, frames, offsets in (
                ('baseline', seq.baseline_frames, None),
                ('transition', seq.transition_frames, seq.transition_offsets),
                ('observation', seq.observation_frames, seq.observation_offsets),
            ):
                (temporary / name).mkdir()
                entries = []
                for i, frame in enumerate(frames):
                    if self.cancelled.is_set():
                        raise RuntimeError('Acid test cancelled')
                    relative = f'{name}/frame_{i:04d}.png'
                    if not cv2.imwrite(str(temporary / relative), frame):
                        raise RuntimeError('Could not save acid capture')
                    entry = {'path': relative}
                    if offsets is not None:
                        entry['offset_seconds'] = offsets[i]
                    entries.append(entry)
                record[name + '_frames'] = entries
            (temporary / 'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            if self.cancelled.is_set():
                raise RuntimeError('Acid test cancelled')
            temporary.replace(final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return record
