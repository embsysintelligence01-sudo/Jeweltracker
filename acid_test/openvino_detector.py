"""CPU acid detection using the supplied two-class INT8 segmentation export."""
from pathlib import Path
import hashlib
import json
import threading
import time

import cv2
import numpy as np

from .onnx_detector import Detection


class OpenVINOAcidDetector:
    def __init__(self, path):
        import openvino as ov

        self.path = Path(path)
        binary = self.path.with_suffix('.bin')
        for required in (self.path, binary):
            if not required.is_file():
                raise FileNotFoundError(required)
        metadata = json.loads(self.path.with_suffix('.json').read_text())
        self.names = dict(enumerate(metadata['names']))
        if metadata['task'] != 'segment' or self.names != {0: 'Acid-flow', 1: 'rubbed_Mask'}:
            raise ValueError('Expected the Acid-flow / rubbed_Mask segmentation export')
        core = ov.Core()
        model = core.read_model(str(self.path))
        self.quantized = any(op.get_type_name() == 'FakeQuantize' for op in model.get_ops())
        if not self.quantized:
            raise ValueError('Expected the selected INT8 acid export')
        if tuple(model.input().shape) != (1, 3, 416, 416) or [tuple(x.shape) for x in model.outputs] != [(1, 38, 3549), (1, 32, 104, 104)]:
            raise ValueError('Expected the two-class 416x416 segmentation output contract')
        self.compiled = core.compile_model(model, 'CPU', {
            'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
            'NUM_STREAMS': 1, 'ENABLE_CPU_PINNING': False,
        })
        self.output = self.compiled.output(0)
        self.lock = threading.Lock()
        digest = hashlib.sha256()
        for artifact in (self.path, binary):
            with artifact.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(chunk)
        self.model_sha256 = digest.hexdigest()

    @staticmethod
    def preprocess(frame):
        height, width = frame.shape[:2]
        scale = min(416 / width, 416 / height)
        rw, rh = round(width * scale), round(height * scale)
        left = round((416 - rw) / 2 - .1)
        top = round((416 - rh) / 2 - .1)
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, top, 416-rh-top, left, 416-rw-left,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        return tensor, scale, left, top

    @staticmethod
    def postprocess(raw, shape, scale, left, top, confidence):
        rows = raw[0].T
        # Channels 4 and 5 are class scores; the remaining 32 are mask coefficients.
        # A rubbed-mark prediction must never become an acid candidate.
        valid = np.isfinite(rows[:, :6]).all(axis=1)
        rows = rows[valid & (rows[:, 4] >= confidence) & (rows[:, 4] > rows[:, 5])
                    & (rows[:, 2] > 0) & (rows[:, 3] > 0)]
        if not len(rows):
            return []
        boxes = rows[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), rows[:, 4].tolist(), confidence, .45)
        height, width = shape[:2]
        found = []
        for index in np.asarray(selected).reshape(-1):
            x, y, w, h = boxes[index]
            box = (int(np.clip(round((x-left)/scale), 0, width)),
                   int(np.clip(round((y-top)/scale), 0, height)),
                   int(np.clip(round((x+w-left)/scale), 0, width)),
                   int(np.clip(round((y+h-top)/scale), 0, height)))
            if box[2] > box[0] and box[3] > box[1]:
                found.append(Detection(box, float(rows[index, 4])))
        return found

    def predict(self, frame, confidence):
        started = time.perf_counter()
        tensor, scale, left, top = self.preprocess(frame)
        with self.lock:
            raw = self.compiled([tensor])[self.output].copy()
        return self.postprocess(raw, frame.shape, scale, left, top, confidence), (time.perf_counter()-started)*1000
