"""Qt5 adapter sharing Jewel Tracker's camera, never opening a second camera."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from PyQt5 import QtCore

from .runtime import TemporalMobileNet, latest_temporal_model_path, DETECTOR_PATH, LAB_ROOT, _sha256
from .onnx_detector import OnnxAcidDetector
from .sequence import AcidSequence
from .flow import bbox_iou


class AcidModelCache:
    """Window-owned models: lazy initialization, one warm-up, no idle inference."""
    def __init__(self):
        self._lock = threading.Lock()
        self._models = None
        self.detector_sha256 = None

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

    def __init__(self, context, observation_seconds=5, acid_grade='', models=None):
        super().__init__()
        self.context = context
        self.token = context['token']
        self.jewel_id = context['jewel_id']
        self.duration, self.acid_grade = observation_seconds, acid_grade
        self.cancelled = threading.Event()
        self.models = models if models is not None else AcidModelCache()

    def cancel(self):
        self.cancelled.set()

    def payload(self, **values):
        return dict(token=self.token, jewel_id=self.jewel_id, **values)

    def run(self):
        try:
            self.progress.emit(self.payload(message='Preparing cached acid models...', phase='LOADING'))
            detector, classifier = self.models.get()
            if self.cancelled.is_set():
                return
            seq = AcidSequence(self.context['stone'], self.context['streak'], self.duration)
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
                self.failed.emit(self.payload(message=str(exc)))

    def save_capture(self, seq, prediction):
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
        }
        try:
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
