"""Execute the original stone ONNX export with a bounded OpenVINO CPU budget."""
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class StoneOnnxModel:
    """Same fixed-640 model and YOLO decoding, without a second YOLO predictor.

    OpenVINO reads the supplied .onnx directly. Float32 execution preserves
    the original weights and precision; one stream with two CPU threads
    leaves the rest of the CPU available for jewellery guarding and the GUI.
    """

    names = {0: 'St'}

    def __init__(self, path):
        import openvino as ov

        self.path = str(Path(path).resolve())
        core = ov.Core()
        model = core.read_model(self.path)
        if tuple(model.input().shape) != (1, 3, 640, 640):
            raise ValueError('Stone model must have fixed [1, 3, 640, 640] input')
        shapes = [tuple(output.shape) for output in model.outputs]
        if shapes != [(1, 37, 8400), (1, 32, 160, 160)]:
            raise ValueError(f'Unexpected single-class stone segmentation outputs: {shapes}')
        self.compiled = core.compile_model(model, 'CPU', {
            'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
            'NUM_STREAMS': 1, 'INFERENCE_PRECISION_HINT': 'f32',
            'ENABLE_CPU_PINNING': False,
        })
        self.box_output, self.mask_output = self.compiled.outputs
        self.letterbox = LetterBox((640, 640), auto=False)

    def predict(self, source, imgsz=640, conf=0.20, device='cpu', verbose=False):
        if imgsz != 640 or device != 'cpu':
            raise ValueError('Stone ONNX execution requires imgsz=640 and device=cpu')
        padded = self.letterbox(image=source)
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        outputs = self.compiled([tensor])
        # Copy because NMS edits boxes in place and inference owns output memory.
        raw = torch.from_numpy(outputs[self.box_output].copy())
        pred = nms.non_max_suppression(raw, conf, 0.7, nc=1)[0]
        if len(pred):
            proto = torch.from_numpy(outputs[self.mask_output][0].copy())
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], (640, 640), upsample=True)
            pred = pred[masks.amax((-2, -1)) > 0]
            pred[:, :4] = ops.scale_boxes((640, 640), pred[:, :4], source.shape)
        return [Results(source, path=self.path, names=self.names, boxes=pred[:, :6])]
