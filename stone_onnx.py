"""Execute fixed-640 stone ONNX or OpenVINO IR models with a bounded CPU budget."""
from pathlib import Path
import threading

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class StoneOnnxModel:
    """Fixed-640 stone model and YOLO decoding, without a second YOLO predictor.

    The legacy class name also accepts quantized OpenVINO .xml/.bin models.
    One stream and two CPU threads leave CPU capacity for jewellery and the GUI.
    """

    names = {0: 'St'}

    def __init__(self, path):
        import openvino as ov

        self.path = str(Path(path).resolve())
        core = ov.Core()
        model = core.read_model(self.path)
        self.quantized = any(op.get_type_name() == 'FakeQuantize' for op in model.get_ops())
        if tuple(model.input().shape) != (1, 3, 640, 640):
            raise ValueError('Stone model must have fixed [1, 3, 640, 640] input')
        shapes = [tuple(output.shape) for output in model.outputs]
        if shapes != [(1, 37, 8400), (1, 32, 160, 160)]:
            raise ValueError(f'Unexpected single-class stone segmentation outputs: {shapes}')
        config = {
            'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
            'NUM_STREAMS': 1, 'ENABLE_CPU_PINNING': False,
        }
        if not self.quantized:
            config['INFERENCE_PRECISION_HINT'] = 'f32'
        self.compiled = core.compile_model(model, 'CPU', config)
        self.box_output, self.mask_output = self.compiled.outputs
        self.letterbox = LetterBox((640, 640), auto=False)
        self.lock = threading.Lock()

    def predict(self, source, imgsz=640, conf=0.20, device='cpu', verbose=False,
                return_masks=False):
        if imgsz != 640 or device != 'cpu':
            raise ValueError('Stone execution requires imgsz=640 and device=cpu')
        padded = self.letterbox(image=source)
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        # Live detection and sparse reference captures share this model. OpenVINO
        # owns the output buffers, so copy both before another request can run.
        with self.lock:
            outputs = self.compiled([tensor])
            raw = torch.from_numpy(outputs[self.box_output].copy())
            proto = torch.from_numpy(outputs[self.mask_output][0].copy())
        pred = nms.non_max_suppression(raw, conf, 0.7, nc=1)[0]
        full_masks = None
        if len(pred):
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], (640, 640), upsample=True)
            keep = masks.amax((-2, -1)) > 0
            pred, masks = pred[keep], masks[keep]
            pred[:, :4] = ops.scale_boxes((640, 640), pred[:, :4], source.shape)
            if return_masks and len(pred):
                full_masks = ops.scale_masks(masks[None].float(), source.shape[:2])[0] > .5
        return [Results(source, path=self.path, names=self.names, boxes=pred[:, :6], masks=full_masks)]
