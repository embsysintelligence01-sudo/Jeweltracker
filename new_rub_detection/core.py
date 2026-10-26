"""GUI-independent detection functions extracted unchanged from the working App.py."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import DEST, SIZE


class OnnxStoneModel:
    """CPU ONNX Runtime wrapper for a YOLOv8-style instance-segmentation ONNX."""

    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: "
                "python -m pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        self.input = inputs[0]
        shape = self.input.shape
        if len(shape) != 4:
            raise RuntimeError(f"Expected a 4D image input, got {shape}")
        # Most YOLO ONNX exports are NCHW; also accept NHWC exports.
        if shape[1] == 3 or shape[1] == "3":
            self.nchw = True
            self.input_h = int(shape[2]) if isinstance(shape[2], int) else 640
            self.input_w = int(shape[3]) if isinstance(shape[3], int) else 640
        elif shape[3] == 3 or shape[3] == "3":
            self.nchw = False
            self.input_h = int(shape[1]) if isinstance(shape[1], int) else 640
            self.input_w = int(shape[2]) if isinstance(shape[2], int) else 640
        else:
            raise RuntimeError(f"Could not identify 3-channel image layout in {shape}")
        self.input_dtype = np.float32 if "float" in self.input.type else np.uint8
        self.output_names = [item.name for item in self.session.get_outputs()]

    def close(self):
        # ONNX Runtime releases its session resources when this object is collected.
        self.session = None


def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def infer_stone(model, frame, confidence_threshold=0.25, iou_threshold=0.45):
    """Return (bbox, confidence, 0/255 mask) detections from YOLOv8-seg ONNX.

    Expects the common Ultralytics segmentation export: a prediction tensor
    [1, 4 + classes + mask_coefficients, anchors] (or transposed), plus a
    prototype tensor [1, mask_coefficients, mask_h, mask_w].
    """
    original_h, original_w = frame.shape[:2]
    target_w, target_h = model.input_w, model.input_h
    scale = min(target_w / original_w, target_h / original_h)
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if model.input_dtype == np.float32:
        tensor = rgb.astype(np.float32) / 255.0
    else:
        tensor = rgb
    if model.nchw:
        tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(np.ascontiguousarray(tensor), axis=0)

    outputs = model.session.run(model.output_names, {model.input.name: tensor})
    proto_index = next((i for i, out in enumerate(outputs) if np.asarray(out).ndim == 4), None)
    pred_index = next((i for i, out in enumerate(outputs) if np.asarray(out).ndim == 3), None)
    if proto_index is None or pred_index is None:
        shapes = [np.asarray(out).shape for out in outputs]
        raise RuntimeError(
            "Unsupported ONNX outputs; expected YOLO segmentation predictions and "
            f"prototypes, got {shapes}. A detection-only ONNX cannot provide stone masks."
        )
    proto = np.asarray(outputs[proto_index])[0]
    if proto.shape[0] > proto.shape[-1] and proto.shape[-1] <= 256:
        # Handle NHWC prototype tensors if an exporter produced them.
        proto = np.transpose(proto, (2, 0, 1))
    mask_channels, proto_h, proto_w = proto.shape
    pred = np.asarray(outputs[pred_index])[0]
    # Convert to [anchors, attributes]. Attributes = xywh + class scores + mask coefficients.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    expected_min = 4 + mask_channels + 1
    if pred.shape[1] < expected_min:
        raise RuntimeError(
            f"Unexpected prediction shape {pred.shape} for {mask_channels} mask channels; "
            "confirm this is a YOLO segmentation ONNX model."
        )
    class_count = pred.shape[1] - 4 - mask_channels
    class_scores = pred[:, 4:4 + class_count]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep = scores >= confidence_threshold
    if not np.any(keep):
        return []
    pred = pred[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    coeffs = pred[:, 4 + class_count:]

    # YOLO predicts center-x, center-y, width, height in model-input pixels.
    boxes_xywh = pred[:, :4].copy()
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy]
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(),
                               confidence_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        return []
    indices = np.asarray(indices).reshape(-1)[:10]

    proto_flat = proto.reshape(mask_channels, -1)
    result = []
    for idx in indices:
        box = boxes_xyxy[idx]
        x1, y1, x2, y2 = box
        # Project mask coefficients onto the prototype tensor.
        logits = coeffs[idx] @ proto_flat
        mask_prob = _sigmoid(logits.reshape(proto_h, proto_w))
        mask_input = cv2.resize(mask_prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        ix1, iy1 = int(np.clip(np.floor(x1), 0, target_w)), int(np.clip(np.floor(y1), 0, target_h))
        ix2, iy2 = int(np.clip(np.ceil(x2), 0, target_w)), int(np.clip(np.ceil(y2), 0, target_h))
        cropped_mask = np.zeros_like(mask_input, dtype=np.float32)
        if ix2 > ix1 and iy2 > iy1:
            cropped_mask[iy1:iy2, ix1:ix2] = mask_input[iy1:iy2, ix1:ix2]
        unpadded = cropped_mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        if unpadded.size == 0:
            continue
        frame_mask = cv2.resize(unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        frame_mask = (frame_mask > 0.5).astype(np.uint8) * 255

        fx1 = int(np.clip(round((x1 - pad_x) / scale), 0, original_w - 1))
        fy1 = int(np.clip(round((y1 - pad_y) / scale), 0, original_h - 1))
        fx2 = int(np.clip(round((x2 - pad_x) / scale), 0, original_w - 1))
        fy2 = int(np.clip(round((y2 - pad_y) / scale), 0, original_h - 1))
        if fx2 <= fx1 or fy2 <= fy1 or not np.any(frame_mask):
            continue
        result.append(((fx1, fy1, fx2, fy2), float(scores[idx]), frame_mask))
    return result


def merged_mark_regions(mask, stone_region):
    """Merge touching predictions and ignore tiny model specks on the stone."""
    joined = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))
    joined[~stone_region] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    merged = np.zeros(mask.shape, np.uint8)
    regions = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 80:
            merged[labels == index] = 255
            regions += 1
    coverage = np.count_nonzero(merged) / max(1, np.count_nonzero(stone_region))
    return merged, regions, coverage


def new_mark_regions(mask, old_mask, stone_region):
    """Keep whole new regions; tolerate small shifts in old model masks."""
    current, _, _ = merged_mark_regions(mask, stone_region)
    near_old = cv2.dilate(old_mask, np.ones((25, 25), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(current, 8)
    new = np.zeros_like(current)
    for index in range(1, count):
        region = labels == index
        if np.count_nonzero(near_old & region) < 0.5 * stats[index, cv2.CC_STAT_AREA]:
            new[region] = 255
    return new


def new_mark_locations(mask, frame_to_stone):
    """Return each new region's center in saved-stone and current-frame coordinates."""
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    if count == 1:
        return []
    order = sorted(range(1, count), key=lambda index: stats[index, cv2.CC_STAT_AREA], reverse=True)
    stone_points = np.float32([centers[index] for index in order])
    frame_points = cv2.perspectiveTransform(stone_points[None],
                                            np.linalg.inv(frame_to_stone))[0]
    return [(tuple(np.rint(stone).astype(int)), tuple(np.rint(frame).astype(int)))
            for stone, frame in zip(stone_points, frame_points)]


def stone_region(stone_mask, transform):
    region = cv2.warpPerspective(stone_mask, transform, (SIZE, SIZE),
                                 flags=cv2.INTER_NEAREST)
    return cv2.erode(region, np.ones((11, 11), np.uint8)) > 0


def draw_stone_grid(image, quad, labels=True):
    """Draw a perspective-aware 2x2 Z1-Z4 grid inside the stone quadrilateral.

    `quad` is expected in perimeter order: top-left, top-right,
    bottom-right, bottom-left, as returned by stone_quad().
    """
    if quad is None:
        return image
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        return image

    matrix = cv2.getPerspectiveTransform(DEST, pts)
    half = SIZE // 2

    def project(points):
        return cv2.perspectiveTransform(np.float32([points]), matrix)[0]

    # Draw a subtle dark underlay, then bright grid lines for visibility.
    grid_lines = [project([(half, 0), (half, SIZE - 1)]),
                  project([(0, half), (SIZE - 1, half)])]
    for p1, p2 in grid_lines:
        a, b = tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int))
        cv2.line(image, a, b, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(image, a, b, (220, 220, 220), 1, cv2.LINE_AA)

    # Use the same perspective mapping as zone analysis and the rubbing target.
    centers = dict(zip(("Z1", "Z2", "Z3", "Z4"), project(
        [(half / 2, half / 2), (half * 1.5, half / 2),
         (half / 2, half * 1.5), (half * 1.5, half * 1.5)])))
    for name, point in centers.items():
        if labels:
            px, py = np.rint(point).astype(int)
            text_size, baseline = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            tx = max(0, min(image.shape[1] - text_size[0] - 2, int(px - text_size[0] / 2)))
            ty = max(text_size[1] + 2, min(image.shape[0] - baseline - 2, int(py + text_size[1] / 2)))
            cv2.putText(image, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(image, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def marked_photo(frame, bbox, quad, tag, full_mask, stone_mask, *, show_grid=True,
                 grid_quad=None):
    shown = frame.copy()
    cv2.rectangle(shown, bbox[:2], bbox[2:], (0, 220, 0), 2)
    cv2.polylines(shown, [quad.astype(np.int32)], True, (255, 170, 0), 2)
    if show_grid:
        draw_stone_grid(shown, quad if grid_quad is None else grid_quad)
    if tag is not None:
        cv2.polylines(shown, [tag.astype(np.int32)], True, (0, 255, 255), 2)
    on_stone = (full_mask > 0) & (stone_mask > 0)
    green = np.full_like(shown, (0, 255, 0))
    blended = cv2.addWeighted(shown, 0.50, green, 0.50, 0)
    shown[on_stone] = blended[on_stone]
    return shown
