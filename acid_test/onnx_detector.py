"""CPU acid detection for the supplied single-class YOLOv8-seg export."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class OnnxAcidDetector:
    def __init__(self, path: Path):
        self.path = Path(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 320, 320] or inputs[0].type != "tensor(float)":
            raise RuntimeError("Acid detector requires a float32 [1,3,320,320] input")
        # 4 decoded xywh coordinates, 1 class score, and 32 mask coefficients.
        if len(outputs) != 2 or outputs[0].shape != [1, 37, 2100] or outputs[1].shape != [1, 32, 80, 80]:
            raise RuntimeError("Acid detector requires the single-class YOLOv8-seg output contract")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @staticmethod
    def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Acid detector requires a nonempty BGR image")
        height, width = frame.shape[:2]
        scale = min(320 / width, 320 / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        left = round((320 - resized_w) / 2 - 0.1)
        top = round((320 - resized_h) / 2 - 0.1)
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, top, 320 - resized_h - top, left, 320 - resized_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return tensor, scale, left, top

    def predict(self, frame_bgr: np.ndarray, confidence: float) -> tuple[list[Detection], float]:
        started = time.perf_counter()
        tensor, scale, left, top = self.preprocess(frame_bgr)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        candidates = raw[0].T
        candidates = candidates[np.isfinite(candidates[:, :5]).all(axis=1)]
        candidates = candidates[
            (candidates[:, 4] >= confidence) & (candidates[:, 2] > 0) & (candidates[:, 3] > 0)
        ]
        detections = []
        if len(candidates):
            boxes = candidates[:, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] / 2  # center xywh -> top-left xywh
            scores = candidates[:, 4]
            selected = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence, 0.45)
            frame_h, frame_w = frame_bgr.shape[:2]
            for index in np.asarray(selected).reshape(-1):
                x, y, width, height = boxes[index]
                mapped = (
                    int(np.clip(round((x - left) / scale), 0, frame_w)),
                    int(np.clip(round((y - top) / scale), 0, frame_h)),
                    int(np.clip(round((x + width - left) / scale), 0, frame_w)),
                    int(np.clip(round((y + height - top) / scale), 0, frame_h)),
                )
                if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
                    detections.append(Detection(mapped, float(scores[index])))
        return detections, (time.perf_counter() - started) * 1000.0
