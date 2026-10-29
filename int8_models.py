"""Bounded OpenVINO execution for the supplied jewellery and mark exports."""
from pathlib import Path
import threading

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops

ROOT = Path(__file__).resolve().parent
JEWEL_MODEL = ROOT / 'int_8_GOLD_JEWEL/gold_int8.xml'
RUBBED_MODEL = ROOT / 'INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.xml'


class OpenVINOYolo:
    """Single-class fixed-shape YOLO, returning the existing Results interface.

    Copy request buffers under the lock: sparse recovery and live work may
    share the jewellery instance. Neither model creates an extra live pass.
    """
    def __init__(self, path, task='segment', imgsz=320, names=None):
        import openvino as ov
        if task not in ('segment', 'detect'):
            raise ValueError(f'Unsupported YOLO task: {task}')
        self.path = str(Path(path).resolve())
        self.task, self.imgsz = task, imgsz
        self.names = names or {0: 'gold' if task == 'segment' else 'rubbedmask'}
        core = ov.Core()
        model = core.read_model(self.path)
        if tuple(model.input().shape) != (1, 3, imgsz, imgsz):
            raise ValueError(f'Model must have fixed [1, 3, {imgsz}, {imgsz}] input')
        shapes = [tuple(output.shape) for output in model.outputs]
        anchors = sum((imgsz // stride)**2 for stride in (8, 16, 32))
        expected = ([(1, 37, anchors), (1, 32, imgsz//4, imgsz//4)]
                    if task == 'segment' else [(1, 5, anchors)])
        if shapes != expected:
            raise ValueError(f'Unexpected {task} outputs: {shapes}; expected {expected}')
        self.quantized = any(op.get_type_name() == 'FakeQuantize' for op in model.get_ops())
        if not self.quantized:
            raise ValueError(f'Expected the supplied quantized INT8 model: {self.path}')
        self.compiled = core.compile_model(model, 'CPU', {
            'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
            'NUM_STREAMS': 1, 'ENABLE_CPU_PINNING': False,
        })
        self.outputs = list(self.compiled.outputs)
        self.letterbox = LetterBox((imgsz, imgsz), auto=False)
        self.lock = threading.Lock()

    def predict(self, source, imgsz=None, conf=.25, device='cpu', verbose=False,
                stream=False, iou=.7, max_det=300):
        if device != 'cpu' or stream or (imgsz is not None and imgsz != self.imgsz):
            raise ValueError(f'INT8 execution requires CPU, stream=False, imgsz={self.imgsz}')
        padded = self.letterbox(image=source)
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.
        with self.lock:
            outputs = self.compiled([tensor])
            raw = torch.from_numpy(outputs[self.outputs[0]].copy())
            proto = (torch.from_numpy(outputs[self.outputs[1]][0].copy())
                     if self.task == 'segment' else None)
        pred = nms.non_max_suppression(raw, conf, iou, nc=1, max_det=max_det)[0]
        masks = None
        if len(pred):
            if proto is not None:
                masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4],
                                         (self.imgsz, self.imgsz), upsample=True)
                keep = masks.amax((-2, -1)) > 0
                pred, masks = pred[keep], masks[keep]
            pred[:, :4] = ops.scale_boxes((self.imgsz, self.imgsz), pred[:, :4], source.shape)
        return [Results(source, path=self.path, names=self.names, boxes=pred[:, :6], masks=masks)]


class RubbedMarkDetector:
    """Candidate boxes, NOT segmentation; RubCheck refines them by scene change."""
    evidence_kind = 'DETECTION_BOX_WITH_BEFORE_AFTER_CHANGE'

    def __init__(self, path=RUBBED_MODEL):
        self.model = OpenVINOYolo(path, task='detect', imgsz=640)

    def regions(self, frame):
        result = self.model.predict(frame, conf=.5, iou=.45, max_det=30)[0]
        mask = np.zeros(frame.shape[:2], np.uint8)
        h, w = mask.shape
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        return mask
