"""Run one rubbed-mask ONNX inference on a saved photo, then exit.

This version uses ONNX Runtime, NumPy, and OpenCV only; it does not import
PyTorch or Ultralytics. It expects a standard YOLOv8-seg ONNX export with
one prediction output and one mask-prototype output.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL = Path(__file__).resolve().parent / "models" / "rubbedmask.onnx"
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 30


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def xywh_to_xyxy(boxes):
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def crop_masks(masks, boxes):
    """Crop prototype-resolution masks to boxes expressed at that resolution."""
    count, height, width = masks.shape
    x1 = np.clip(np.floor(boxes[:, 0]).astype(int), 0, width)
    y1 = np.clip(np.floor(boxes[:, 1]).astype(int), 0, height)
    x2 = np.clip(np.ceil(boxes[:, 2]).astype(int), 0, width)
    y2 = np.clip(np.ceil(boxes[:, 3]).astype(int), 0, height)

    xs = np.arange(width)[None, None, :]
    ys = np.arange(height)[None, :, None]
    keep = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )
    return masks * keep


def decode_outputs(outputs):
    """Identify prediction and prototype tensors in a standard YOLOv8-seg export."""
    arrays = [np.asarray(output) for output in outputs]
    proto_index = next((i for i, arr in enumerate(arrays) if arr.ndim == 4), None)
    pred_index = next((i for i, arr in enumerate(arrays) if arr.ndim == 3), None)
    if proto_index is None or pred_index is None:
        shapes = [arr.shape for arr in arrays]
        raise RuntimeError(
            "Unsupported rubbed-mask ONNX outputs. Expected YOLO segmentation "
            f"prediction and prototype tensors; received {shapes}."
        )

    proto = arrays[proto_index][0]
    # Standard prototype layout is [mask_channels, mask_h, mask_w].
    if proto.ndim != 3:
        raise RuntimeError(f"Unexpected prototype tensor shape: {proto.shape}")
    if proto.shape[0] > 256 and proto.shape[-1] <= 256:
        proto = np.transpose(proto, (2, 0, 1))

    pred = arrays[pred_index][0]
    # Standard raw YOLO output is [attributes, anchors]. Convert to [anchors, attributes].
    # Attributes are 4 box values + class scores + mask coefficients.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    mask_channels = proto.shape[0]
    if pred.shape[1] <= 4 + mask_channels:
        raise RuntimeError(
            f"Unexpected prediction shape {pred.shape} with {mask_channels} mask channels."
        )

    class_count = pred.shape[1] - 4 - mask_channels
    if class_count < 1:
        raise RuntimeError("Could not determine class-score columns in ONNX predictions.")
    return pred.astype(np.float32, copy=False), proto.astype(np.float32, copy=False), class_count


def detect_mask(frame, session):
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (INPUT_SIZE - resized_width) // 2
    top = (INPUT_SIZE - resized_height) // 2

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        INPUT_SIZE - resized_height - top,
        left,
        INPUT_SIZE - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    prediction, prototypes, class_count = decode_outputs(outputs)

    mask_channels, proto_h, proto_w = prototypes.shape
    class_scores = prediction[:, 4:4 + class_count]
    scores = class_scores.max(axis=1)
    coeffs = prediction[:, 4 + class_count:4 + class_count + mask_channels]

    valid = np.isfinite(scores) & (scores >= CONFIDENCE_THRESHOLD)
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint8)

    boxes = xywh_to_xyxy(prediction[valid, :4])
    scores = scores[valid]
    coeffs = coeffs[valid]

    # Clip boxes to the model input and remove degenerate candidates.
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, INPUT_SIZE)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, INPUT_SIZE)
    good = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, coeffs = boxes[good], scores[good], coeffs[good]
    if len(boxes) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    nms_boxes = [
        [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
        for box in boxes
    ]
    indices = cv2.dnn.NMSBoxes(
        nms_boxes, scores.astype(float).tolist(),
        CONFIDENCE_THRESHOLD, IOU_THRESHOLD
    )
    if indices is None or len(indices) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    indices = np.asarray(indices).reshape(-1)[:MAX_DETECTIONS]

    # Combine the retained mask coefficients with the prototype tensor.
    proto_flat = prototypes.reshape(mask_channels, -1)
    mask_logits = coeffs[indices] @ proto_flat
    masks = sigmoid(mask_logits.reshape(-1, proto_h, proto_w))

    # Crop masks using boxes scaled from 640x640 to prototype resolution.
    proto_boxes = boxes[indices].copy()
    proto_boxes[:, [0, 2]] *= proto_w / INPUT_SIZE
    proto_boxes[:, [1, 3]] *= proto_h / INPUT_SIZE
    masks = crop_masks(masks, proto_boxes)

    # Upsample to model input size, threshold, and merge all detections.
    combined = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    for mask in masks:
        full = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        combined |= (full > 0.5).astype(np.uint8) * 255

    # Remove letterbox padding and restore the original photo dimensions.
    unpadded = combined[top:top + resized_height, left:left + resized_width]
    if unpadded.size == 0:
        raise RuntimeError("Mask became empty after removing letterbox padding.")
    return cv2.resize(unpadded, (width, height), interpolation=cv2.INTER_NEAREST)


def create_session(model_path=MODEL):
    """Load the rubbed-mask model once for reuse by a host application's worker."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Rubbed-mask ONNX model not found: {model_path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one ONNX input, found {len(inputs)}.")
    input_shape = inputs[0].shape
    if len(input_shape) != 4 or input_shape[1:] != [3, INPUT_SIZE, INPUT_SIZE]:
        raise RuntimeError(
            f"Rubbed-mask ONNX input contract changed: expected [1, 3, 640, 640], "
            f"received {input_shape}."
        )
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()

    frame = cv2.imread(str(args.photo), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read captured photo: {args.photo}")
    session = create_session(args.model)

    mask = detect_mask(frame, session)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.mask), mask):
        raise OSError(f"Cannot save rubbed-mask result: {args.mask}")


if __name__ == "__main__":
    main()
