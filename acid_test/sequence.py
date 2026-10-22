"""Temporal capture flow adapted from the local app.py, independent of the GUI."""
from __future__ import annotations

from collections import deque
import time
import cv2
import numpy as np

from .flow import bbox_iou, pad_bbox


class AcidSequence:
    def __init__(self, stone, streak, observation_seconds=5.0, baseline_seconds=2.0, clock=None):
        self.stone, self.streak = tuple(stone), tuple(streak)
        self.clock = clock if clock is not None else time.monotonic
        self.duration, self.baseline_seconds = observation_seconds, baseline_seconds
        self.state = 'BASELINE'
        self.started = self.prompted = self.detected = None
        self.shape = None
        self.baseline_full = deque(maxlen=20)
        self.transition_buffer = deque(maxlen=225)
        self.baseline_frames, self.transition_frames, self.observation_frames = [], [], []
        self.transition_offsets, self.observation_offsets = [], []
        self.last_sample = self.last_transition = -1e9
        self.previous = None
        self.hits = 0
        self.detector_bbox = self.reaction_bbox = None
        self.detector_confidence = 0.0

    def push(self, frame, now):
        if self.started is None:
            self.started, self.shape = now, frame.shape
        if frame.shape != self.shape:
            raise RuntimeError('Camera resolution changed; repeat the acid test')
        if self.state == 'BASELINE':
            if now - self.last_sample >= 0.2 - 1e-6:
                self.baseline_full.append(frame.copy())
                self.last_sample = now
            if now - self.started >= self.baseline_seconds:
                self.state, self.prompted = 'WAIT_FOR_ACID', now
                self.last_sample = -1e9
        if self.state == 'WAIT_FOR_ACID':
            if now - self.prompted > 120:
                raise RuntimeError('Acid was not confirmed within 120 seconds; retry the test')
            if now - self.last_transition >= 1 / 15 - 1e-6:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError('Could not preserve the pouring sequence')
                self.transition_buffer.append((now - self.prompted, encoded))
                self.last_transition = now
        elif self.state == 'OBSERVING':
            if now - self.last_sample >= 0.2 - 1e-6:
                x1, y1, x2, y2 = self.reaction_bbox
                self.observation_frames.append(frame[y1:y2, x1:x2].copy())
                self.observation_offsets.append(now - self.detected)
                self.last_sample = now
            if now - self.detected >= self.duration:
                self.state = 'PROCESSING'

    def accept_detections(self, detections, now):
        if self.state != 'WAIT_FOR_ACID':
            return
        sx1, sy1, sx2, sy2 = self.stone
        # Restrict the model to the rubbed jewel's streak; old acid elsewhere
        # on the stone must not complete this jewel's test.
        rx1, ry1, rx2, ry2 = pad_bbox(self.streak, 0.45, self.shape[1], self.shape[0])
        valid = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if (det.confidence >= 0.60 and sx1 <= cx <= sx2 and sy1 <= cy <= sy2
                    and rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                valid.append(det)
        best = max(valid, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) * d.confidence, default=None)
        if best is None:
            self.previous, self.hits = None, 0
            return
        self.hits = self.hits + 1 if self.previous is not None and bbox_iou(self.previous, best.bbox) >= 0.45 else 1
        self.previous = best.bbox
        if self.hits < 3:
            return
        self.detector_bbox, self.detector_confidence = best.bbox, best.confidence
        self.reaction_bbox = pad_bbox(best.bbox, 0.45, self.shape[1], self.shape[0])
        x1, y1, x2, y2 = self.reaction_bbox
        self.baseline_frames = [f[y1:y2, x1:x2].copy() for f in self.baseline_full]
        for offset, encoded in self.transition_buffer:
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError('Could not recover a pouring frame')
            self.transition_frames.append(frame[y1:y2, x1:x2].copy())
            self.transition_offsets.append(offset)
        if not self.baseline_frames or not self.transition_frames:
            raise RuntimeError('Baseline and pouring frames are required; repeat the acid test')
        self.baseline_full.clear()
        self.transition_buffer.clear()
        # Match the standalone lab: decoding is preparation, not observation.
        # Start a full observation window only after all crops are ready.
        self.state, self.detected, self.last_sample = 'OBSERVING', self.clock(), -1e9

    def instruction(self, now):
        if self.state == 'BASELINE':
            return f'BASELINE: do not apply acid. Keep still ({max(0, self.baseline_seconds - (now - self.started)):.1f}s)'
        if self.state == 'WAIT_FOR_ACID':
            return f'NOW APPLY ACID to this jewel\'s streak; remove hands. Detection {self.hits}/3'
        if self.state == 'OBSERVING':
            return f'ACID CONFIRMED: observing reaction ({max(0, self.duration - (now - self.detected)):.1f}s)'
        return 'Classifying the reaction and saving the capture...'
