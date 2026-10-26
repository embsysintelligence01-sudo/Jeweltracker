"""CPU inference for the supplied YOLOv8 OBB OpenVINO IR export."""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np


class OpenVINOOBB:
    """Keep preprocessing, inference and rotated boxes independent of Qt/Torch."""

    def __init__(self, model_path):
        import openvino as ov

        path = Path(model_path)
        for required in (path, path.with_suffix('.bin')):
            if not required.is_file():
                raise FileNotFoundError(f'Model file not found: {required}')
        core = ov.Core()
        model = core.read_model(str(path))
        self.task = model.get_rt_info(['framework', 'task']).value
        if self.task != 'obb':
            raise ValueError(f'Expected an OBB model, got {self.task!r}.')
        self.names = ast.literal_eval(model.get_rt_info(['framework', 'names']).value)
        shape = tuple(model.input().shape)
        if len(shape) != 4 or shape[:2] != (1, 3):
            raise ValueError(f'Expected fixed [1, 3, height, width] input, got {shape}.')
        self.height, self.width = shape[2:]
        output_shape = tuple(model.output().shape)
        expected_channels = 5 + len(self.names)  # xywh, class scores, angle
        if len(output_shape) != 3 or output_shape[:2] != (1, expected_channels):
            raise ValueError(f'Expected raw YOLO OBB output, got {output_shape}.')
        self.device = core.get_property('CPU', 'FULL_DEVICE_NAME')
        self.compiled = core.compile_model(model, 'CPU', {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2, 'NUM_STREAMS': 1})
        self.output = self.compiled.output()

    def preprocess(self, image):
        """YOLO letterbox with fixed export size, BGR to RGB, NCHW float32."""
        h, w = image.shape[:2]
        gain = min(self.width / w, self.height / h)
        resized_w, resized_h = round(w * gain), round(h * gain)
        left = round((self.width - resized_w) / 2 - 0.1)
        top = round((self.height - resized_h) / 2 - 0.1)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, top, self.height - resized_h - top,
            left, self.width - resized_w - left, cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        return tensor, gain, (left, top)

    def detect(self, image, class_id=0, confidence=0.35):
        if class_id not in self.names:
            raise ValueError(f'Class {class_id} is not in model classes: {self.names}')
        tensor, gain, padding = self.preprocess(image)
        prediction = self.compiled([tensor])[self.output]
        return self.postprocess(prediction, image, gain, padding, class_id, confidence)

    def postprocess(self, prediction, image, gain, padding, class_id, confidence):
        """Decode xywh + scores + radians, suppress duplicate rotated boxes."""
        rows = prediction[0].T
        scores = rows[:, 4:4 + len(self.names)]
        keep = (scores.argmax(axis=1) == class_id) & (scores[:, class_id] > confidence)
        keep &= np.isfinite(rows).all(axis=1) & (rows[:, 2:4] > 0).all(axis=1)
        rows = rows[keep]
        if not len(rows):
            return []
        # Keep NMS work bounded even for a noisy frame.
        rows = rows[np.argsort(-rows[:, 4 + class_id])[:30000]]
        scores = rows[:, 4 + class_id]
        rectangles = [
            ((float(row[0]), float(row[1])), (float(row[2]), float(row[3])),
             float(np.degrees(row[-1])))
            for row in rows
        ]
        selected = cv2.dnn.NMSBoxesRotated(rectangles, scores.tolist(), confidence, 0.7)
        h, w = image.shape[:2]
        found = []
        for index in np.asarray(selected).reshape(-1)[:300]:
            quad = (cv2.boxPoints(rectangles[index]) - np.asarray(padding)) / gain
            x1, y1 = np.maximum(quad.min(axis=0).astype(int), (0, 0))
            x2, y2 = np.minimum(quad.max(axis=0).astype(int), (w, h))
            if x2 - x1 < 12 or y2 - y1 < 8:
                continue
            quad = quad.astype(np.int32)
            # Rasterize locally to avoid a full-camera-sized mask per candidate.
            mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillConvexPoly(mask, quad - (x1, y1), 255)
            found.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'quad': quad,
                'score': float(scores[index]),
                'crop': image[y1:y2, x1:x2].copy(),
                'mask': mask,
                'mode': 'obb',
            })
        return found
